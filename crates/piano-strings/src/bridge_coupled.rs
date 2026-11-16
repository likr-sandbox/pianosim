//! Stage 2 <-> Stage 3 closed-loop continuous SIMD stepping with per-sample WDF
//! bridge port coupling, `1x` unilateral dampers, key-batched `CHUNK = 128` tails,
//! 2-string paired SAV drift regulation, and zero `ModalSoa` touches in steady state.

use piano_spec::{LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_damp_kernels::refresh_chunk_damp_y;
use crate::bridge_diag::{flush_bridge_subnormals_core, update_sav_dormancy};
use crate::bridge_kernels::step_uncoupled_prefixes_fast;
use crate::bridge_port::{BridgePortDriver, BridgePorts};
use crate::bridge_prefix_step::step_coupled_prefixes;
use crate::bridge_sav::step_sav_scheduled;
use crate::bridge_tails::{
    prepare_sample_tails_and_hammers, sav_long_vecs, step_uncoupled_tails, BridgeRun, ChunkPlan,
    HammerStepForce, KeyCat, SavLongSlice, TailDampPartial, SUB,
};
use crate::choir_map::ChoirMap;
use crate::contact::{step_clearance_subblock, KeyActionDriver};
use crate::contact_ports::{
    accum_prefix_damp_vel_bp, refresh_string_ports_bp, ContactPorts, DAMPER_CLEAR_THRESHOLD_M,
};
use crate::longitudinal::LongitudinalBank;
use crate::sav_drift::SavBank;

#[inline]
fn step_all_strings_prefix_wdf(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    ts: usize,
) {
    let n_cp = plan.n_coupled_strs;
    let (ul, ur) =
        step_uncoupled_prefixes_fast(r.bridge_ports, r.ports, plan, ts, a_waves, b_waves);
    out_pair.0[ts] += ul;
    out_pair.1[ts] += ur;

    if n_cp > 0 {
        step_coupled_prefixes(r, driver, plan, n_cp, a_waves, b_waves, out_pair, ts);
    }

    if r.sav_on && plan.n_sav > 0 {
        step_sav_scheduled(
            r.bridge_ports,
            r.sav,
            &plan.sav_strings[..plan.n_sav],
            r.long_on,
        );
    }
    for idx in 0..plan.n_a {
        let k = usize::from(plan.active_keys[idx]);
        let has_hammer = plan.key_has_hammer[k];
        let sav_key = r.sav_on && k < LONGITUDINAL_MAX_KEY_IDX;
        for s_idx in r.keys[k].string_range.clone() {
            if !r.ports.may_be_nonzero[s_idx] {
                continue;
            }
            if has_hammer {
                refresh_string_ports_bp(
                    r.bridge_ports,
                    r.ports,
                    s_idx,
                    r.strings[s_idx].has_damper,
                );
            } else if sav_key
                && r.ports.struck_active[s_idx]
                && !r.ports.sav_dormant[s_idx]
                && r.sav.string_n_vh(s_idx) > 0
            {
                let m = &r.bridge_ports.meta[s_idx];
                let tp = plan.tail_partial_damp[s_idx];
                let vdv_br =
                    accum_prefix_damp_vel_bp(r.bridge_ports, r.ports, m.beta_v_off, m.v_br.clone());
                let vdh_br =
                    accum_prefix_damp_vel_bp(r.bridge_ports, r.ports, m.beta_h_off, m.h_br.clone());
                let alpha = r.ports.alpha_vel[s_idx];
                r.ports.v_dv[s_idx] = alpha * (vdv_br + tp.vdv_raw);
                r.ports.v_dh[s_idx] = alpha * (vdh_br + tp.vdh_raw);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_subblock_bridge_coupled_core(
    ports: &mut ContactPorts,
    bridge_ports: &mut BridgePorts,
    long_bank: &LongitudinalBank,
    sav: &mut SavBank,
    map: &ChoirMap,
    driver: &mut impl KeyActionDriver,
    bridge: &mut impl BridgePortDriver,
    long_on: bool,
    sav_on: bool,
    lossless: bool,
    flush_end: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
    mut zone_frames: Option<&mut [[f32; 8]]>,
) {
    let total = out_l
        .len()
        .min(out_r.len())
        .min(zone_frames.as_deref().map_or(usize::MAX, <[[f32; 8]]>::len));
    out_l.fill(0.0);
    out_r.fill(0.0);
    if total == 0 {
        return;
    }
    let zone_weights = *bridge.zone_weights();
    let mut r = BridgeRun {
        ports,
        bridge_ports,
        long_bank,
        sav,
        keys: map.keys(),
        strings: map.strings(),
        zone_weights,
        long_on,
        sav_on,
        lossless,
    };
    let (mut chunk_base, mut a_waves, mut b_waves, mut zone_out, mut dup_zone) = (
        0usize,
        [0.0_f32; TOTAL_STRINGS],
        [0.0_f32; TOTAL_STRINGS],
        [0.0_f32; 8],
        [0.0_f32; 8],
    );
    let mut plan = ChunkPlan {
        key_cat: [KeyCat::Uncoupled; NUM_KEYS],
        key_has_hammer: [false; NUM_KEYS],
        key_has_mid_ev: [false; NUM_KEYS],
        str_coupled: [false; TOTAL_STRINGS],
        was_nz: [false; TOTAL_STRINGS],
        was_tail_nz: [false; TOTAL_STRINGS],
        active_keys: [0; NUM_KEYS],
        coupled_strs: [0; TOTAL_STRINGS],
        sav_long_slices: [SavLongSlice::ZERO; TOTAL_STRINGS],
        sav_strings: [0; TOTAL_STRINGS],
        hammer_forces: [HammerStepForce::ZERO; TOTAL_STRINGS],
        tail_partial_damp: [TailDampPartial::ZERO; TOTAL_STRINGS],
        symp_gv: [0.0; TOTAL_STRINGS],
        symp_gh: [0.0; TOTAL_STRINGS],
        symp_mask: [0.0; TOTAL_STRINGS],
        symp_has_gains: [false; TOTAL_STRINGS],
        zone_tail_n: [[0.0; 8]; SUB],
        n_a: 0,
        n_coupled_strs: 0,
        n_sl: 0,
        n_sav: 0,
    };

    while chunk_base < total {
        let sub = (total - chunk_base).min(SUB);
        let (sub_l, sub_r) = (
            &mut out_l[chunk_base..chunk_base + sub],
            &mut out_r[chunk_base..chunk_base + sub],
        );
        plan.was_nz = r.ports.may_be_nonzero;
        plan.was_tail_nz = r.ports.tail_nonzero;
        r.bridge_ports.compute_a_waves(&plan.was_nz, &mut a_waves);
        plan.key_cat.fill(KeyCat::Uncoupled);
        plan.key_has_hammer.fill(false);
        plan.key_has_mid_ev.fill(false);
        plan.str_coupled.fill(false);
        plan.symp_has_gains.fill(false);
        plan.symp_gv.fill(0.0);
        plan.symp_gh.fill(0.0);
        plan.symp_mask.fill(0.0);
        (plan.n_a, plan.n_coupled_strs, plan.n_sl, plan.n_sav) = (0, 0, 0, 0);
        plan.zone_tail_n[..sub].fill([0.0; 8]);

        for (k, key) in r.keys.iter().enumerate() {
            let sr = key.string_range.clone();
            let mid_ev = driver.has_mid_block_event(k, chunk_base, sub);
            let hammer = driver.is_hammer_active(k);
            let any_nz = plan.was_nz[sr.clone()].iter().any(|&b| b);
            if !mid_ev && !hammer && !any_nz {
                if key.has_damper {
                    step_clearance_subblock(driver, k, sub, None);
                }
                continue;
            }
            let min_c = if mid_ev {
                0.0
            } else if key.has_damper {
                step_clearance_subblock(
                    driver,
                    k,
                    sub,
                    Some(&mut r.ports.y_clear_scratch[k][..sub]),
                )
            } else {
                f32::MAX
            };
            let any_struck = r.ports.struck_active[sr.clone()].iter().any(|&b| b);
            if mid_ev || hammer || (any_struck && min_c < DAMPER_CLEAR_THRESHOLD_M) {
                plan.key_cat[k] = KeyCat::ActiveContact;
                plan.key_has_hammer[k] = hammer;
                plan.key_has_mid_ev[k] = mid_ev;
                plan.active_keys[plan.n_a] = k as u8;
                plan.n_a += 1;
                for s_idx in sr.clone() {
                    plan.push_coupled(s_idx);
                }
            } else if min_c < DAMPER_CLEAR_THRESHOLD_M {
                if !any_struck && key.has_damper && min_c >= 2.5e-5 {
                    let mut all_clear = true;
                    for s_idx in sr.clone() {
                        if !plan.was_nz[s_idx] {
                            continue;
                        }
                        if r.ports.tail_nonzero[s_idx] {
                            all_clear = false;
                            break;
                        }
                        if !r.ports.ports_synced[s_idx] {
                            crate::contact_ports::refresh_prefix_damp_ports_bp(
                                r.bridge_ports,
                                r.ports,
                                s_idx,
                            );
                            r.ports.ports_synced[s_idx] = true;
                        }
                        let lim = 0.5 * min_c;
                        let bound_v =
                            r.ports.y_dv[s_idx].abs() + 4.0e-3 * r.ports.v_dv[s_idx].abs();
                        let bound_h =
                            r.ports.y_dh[s_idx].abs() + 4.0e-3 * r.ports.v_dh[s_idx].abs();
                        if bound_v > lim || bound_h > lim {
                            all_clear = false;
                            break;
                        }
                    }
                    if all_clear {
                        continue;
                    }
                }
                plan.key_cat[k] = KeyCat::DampedSymp;
                if key.has_damper {
                    let yc0 = r.ports.y_clear_scratch[k][0];
                    let yc_end = r.ports.y_clear_scratch[k][sub - 1];
                    let const_closed = yc0 <= -5.0e-5
                        && r.ports.y_clear_scratch[k][..sub].iter().all(|&c| c == yc0);
                    let closed_chunk = yc0 <= -5.0e-5 && yc_end <= -5.0e-5;
                    for s_idx in sr.clone() {
                        if plan.was_nz[s_idx] {
                            let mut routed_fast = false;
                            if closed_chunk {
                                if !r.ports.ports_synced[s_idx] {
                                    crate::contact_ports::refresh_prefix_damp_ports_bp(
                                        r.bridge_ports,
                                        r.ports,
                                        s_idx,
                                    );
                                    r.ports.ports_synced[s_idx] = true;
                                }
                                let (yv, yh) =
                                    (r.ports.y_dv[s_idx].abs(), r.ports.y_dh[s_idx].abs());
                                if yv <= 6.5e-6 && yh <= 6.5e-6 {
                                    let rv = r.bridge_ports.r_damp_prefix_v[s_idx];
                                    let rh = r.bridge_ports.r_damp_prefix_h[s_idx];
                                    let gains_opt = if const_closed {
                                        driver
                                            .symp_linear_gains_if_settled(k, s_idx, rv, rh, yc0)
                                            .or_else(|| {
                                                driver.symp_linear_gains_for_chunk(
                                                    k, s_idx, rv, rh, yc0, sub,
                                                )
                                            })
                                    } else {
                                        driver.symp_linear_gains_for_chunk(
                                            k,
                                            s_idx,
                                            rv,
                                            rh,
                                            0.5 * (yc0 + yc_end),
                                            sub,
                                        )
                                    };
                                    if let Some((gv, gh)) = gains_opt {
                                        plan.symp_gv[s_idx] = gv;
                                        plan.symp_gh[s_idx] = gh;
                                        plan.symp_mask[s_idx] = 1.0;
                                        plan.symp_has_gains[s_idx] = true;
                                        routed_fast = true;
                                    }
                                }
                            }
                            if !routed_fast {
                                plan.push_coupled(s_idx);
                            }
                        }
                    }
                }
            }
            if plan.key_cat[k] != KeyCat::ActiveContact {
                for s_idx in sr.clone() {
                    let nl = sav_long_vecs(&r, s_idx);
                    if nl > 0 {
                        let start = r.bridge_ports.meta[s_idx].tl_l.start;
                        plan.sav_long_slices[plan.n_sl] = SavLongSlice {
                            s_idx: s_idx as u8,
                            start: start as u16,
                            end: (start + nl) as u16,
                        };
                        plan.n_sl += 1;
                    }
                }
            }
            if r.sav_on && k < LONGITUDINAL_MAX_KEY_IDX {
                for s_idx in sr {
                    if !r.lossless && !hammer && !mid_ev && r.ports.struck_active[s_idx] {
                        update_sav_dormancy(r.ports, r.bridge_ports, r.sav, s_idx);
                    }
                    if r.ports.struck_active[s_idx]
                        && !r.ports.sav_dormant[s_idx]
                        && r.sav.string_n_vh(s_idx) > 0
                    {
                        plan.sav_strings[plan.n_sav] = s_idx as u8;
                        plan.n_sav += 1;
                    }
                }
            }
        }

        step_uncoupled_tails(&mut r, &mut plan, sub_l, sub_r);
        for ts in 0..sub {
            prepare_sample_tails_and_hammers(
                &mut r, driver, &mut plan, chunk_base, ts, sub_l, sub_r,
            );
            let (dup_l, dup_r) = bridge.step_bridge_sample_with_duplex_zones(
                &a_waves,
                &plan.zone_tail_n[ts],
                &mut b_waves,
                &mut zone_out,
                &mut dup_zone,
            );
            if let Some(zf) = zone_frames.as_deref_mut() {
                for (dst, (&zo, &dz)) in zf[chunk_base + ts]
                    .iter_mut()
                    .zip(zone_out.iter().zip(dup_zone.iter()))
                {
                    *dst = zo + dz;
                }
            }
            if dup_l.is_finite() {
                sub_l[ts] += dup_l;
            }
            if dup_r.is_finite() {
                sub_r[ts] += dup_r;
            }
            step_all_strings_prefix_wdf(
                &mut r,
                driver,
                &mut plan,
                &mut a_waves,
                &b_waves,
                (sub_l, sub_r),
                ts,
            );
        }
        refresh_chunk_damp_y(r.bridge_ports, r.ports, &plan.symp_has_gains);
        for s_idx in 0..TOTAL_STRINGS {
            if !plan.str_coupled[s_idx]
                && !plan.symp_has_gains[s_idx]
                && r.ports.may_be_nonzero[s_idx]
            {
                r.ports.ports_synced[s_idx] = false;
            }
        }
        chunk_base += sub;
    }

    if !flush_end {
        for (s_idx, entry) in r.strings.iter().enumerate() {
            if r.ports.may_be_nonzero[s_idx]
                && !plan.key_has_hammer[usize::from(entry.key_idx)]
                && !plan.symp_has_gains[s_idx]
                && !plan.str_coupled[s_idx]
            {
                r.ports.ports_synced[s_idx] = false;
            }
        }
        return;
    }
    let mut flush_closed_damp = [false; NUM_KEYS];
    for (k, key) in r.keys.iter().enumerate() {
        if !r.lossless
            && key.has_damper
            && !plan.key_has_hammer[k]
            && driver.damper_clearance_m(k) <= -1.0e-5
        {
            flush_closed_damp[k] = true;
        }
    }
    flush_bridge_subnormals_core(
        r.ports,
        r.bridge_ports,
        r.sav,
        r.strings,
        &plan.key_has_hammer,
        &flush_closed_damp,
        r.lossless,
    );
}
