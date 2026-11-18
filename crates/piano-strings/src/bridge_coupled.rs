//! Stage 2 <-> Stage 3 closed-loop continuous SIMD stepping with per-sample WDF
//! bridge port coupling, `1x` unilateral dampers, key-batched `CHUNK = 128` tails,
//! 2-string paired SAV drift regulation, and zero `ModalSoa` touches in steady state.

use piano_spec::{DAMPED_KEYS, LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_damp_kernels::refresh_chunk_damp_y;
use crate::bridge_diag::{flush_bridge_subnormals_core, update_sav_dormancy};
use crate::bridge_kernels::step_uncoupled_prefixes_fast;
use crate::bridge_port::{BridgePortDriver, BridgePorts};
use crate::bridge_prefix_step::{
    step_clearance_subblock_with_peel, step_coupled_prefixes, step_idle_clearance_check_peel,
};
use crate::bridge_sav::step_sav_scheduled;
use crate::bridge_tails::{
    key_clear_of_open_damper, key_needs_active_tail, prepare_sample_tails_and_hammers,
    route_damped_symp_key, sav_long_vecs, step_uncoupled_tails, BridgeRun, ChunkPlan,
    HammerStepForce, KeyCat, SavLongSlice, TailDampPartial, SUB,
};
use crate::choir_map::ChoirMap;
use crate::contact::KeyActionDriver;
use crate::contact_ports::{
    accum_prefix_damp_vel_bp, refresh_string_ports_bp, ContactPorts, DAMPER_CLEAR_THRESHOLD_M,
};
use crate::longitudinal::LongitudinalBank;
use crate::sav_drift::SavBank;

const MAX_PEEL_SLOTS: usize = 24;

#[inline]
#[allow(clippy::too_many_arguments)]
fn step_all_strings_prefix_wdf(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    ts: usize,
    peel_slot: &[u8; DAMPED_KEYS],
    peel_buf: &[[f32; SUB]; MAX_PEEL_SLOTS],
) {
    let n_cp = plan.n_coupled_strs;
    let (ul, ur) =
        step_uncoupled_prefixes_fast(r.bridge_ports, r.ports, plan, ts, a_waves, b_waves);
    out_pair.0[ts] += ul;
    out_pair.1[ts] += ur;

    if n_cp > 0 {
        step_coupled_prefixes(
            r, driver, plan, n_cp, a_waves, b_waves, out_pair, ts, peel_slot, peel_buf,
        );
    }

    if r.sav_on && plan.n_sav > 0 {
        step_sav_scheduled(
            r.bridge_ports,
            r.sav,
            &plan.sav_strings[..plan.n_sav],
            r.long_on,
        );
        for &s_u8 in &plan.sav_strings[..plan.n_sav] {
            let s_idx = usize::from(s_u8);
            if !r.ports.may_be_nonzero[s_idx] || !r.strings[s_idx].has_damper {
                continue;
            }
            let k = usize::from(r.strings[s_idx].key_idx);
            if plan.key_has_hammer[k] {
                continue;
            }
            if plan.str_coupled[s_idx] || plan.symp_has_gains[s_idx] {
                let m = &r.bridge_ports.meta[s_idx];
                let vdv_br =
                    accum_prefix_damp_vel_bp(r.bridge_ports, r.ports, m.beta_v_off, m.v_br.clone());
                let vdh_br =
                    accum_prefix_damp_vel_bp(r.bridge_ports, r.ports, m.beta_h_off, m.h_br.clone());
                let (tv, th) = if plan.key_cat[k] == KeyCat::ActiveContact {
                    let tp = plan.tail_partial_damp[s_idx];
                    (tp.vdv_raw, tp.vdh_raw)
                } else {
                    (0.0, 0.0)
                };
                let alpha = r.ports.alpha_vel[s_idx];
                r.ports.v_dv[s_idx] = alpha * (vdv_br + tv);
                r.ports.v_dh[s_idx] = alpha * (vdh_br + th);
            }
        }
    }
    for idx in 0..plan.n_a {
        let k = usize::from(plan.active_keys[idx]);
        if plan.key_has_hammer[k] {
            let alpha_w = (driver.hammer_compression_norm(k) / 0.65).clamp(0.0, 1.0);
            for s_idx in r.keys[k].string_range.clone() {
                if r.ports.may_be_nonzero[s_idx] {
                    refresh_string_ports_bp(
                        r.bridge_ports,
                        r.ports,
                        s_idx,
                        r.strings[s_idx].has_damper,
                        alpha_w,
                    );
                }
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
        symp_y_lim: [0.0; TOTAL_STRINGS],
        symp_has_gains: [false; TOTAL_STRINGS],
        zone_tail_n: [[0.0; 8]; SUB],
        n_a: 0,
        n_coupled_strs: 0,
        n_sl: 0,
        n_sav: 0,
    };

    let mut peel_buf = [[0.0_f32; SUB]; MAX_PEEL_SLOTS];
    let mut peel_slot = [u8::MAX; DAMPED_KEYS];

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
        plan.symp_y_lim.fill(0.0);
        peel_slot.fill(u8::MAX);
        let mut n_peel_slots = 0usize;
        (plan.n_a, plan.n_coupled_strs, plan.n_sl, plan.n_sav) = (0, 0, 0, 0);
        plan.zone_tail_n[..sub].fill([0.0; 8]);

        for (k, key) in r.keys.iter().enumerate() {
            let sr = key.string_range.clone();
            let mid_ev = driver.has_mid_block_event(k, chunk_base, sub);
            let hammer = driver.is_hammer_active(k);
            let any_nz = plan.was_nz[sr.clone()].iter().any(|&b| b);
            let (min_c, has_peel) = if mid_ev {
                (0.0, false)
            } else if !hammer && !any_nz {
                if key.has_damper {
                    if n_peel_slots < MAX_PEEL_SLOTS {
                        let (mc, ap) = step_idle_clearance_check_peel(
                            driver,
                            k,
                            sub,
                            &mut r.ports.y_clear_scratch[k][..sub],
                            &mut peel_buf[n_peel_slots][..sub],
                        );
                        if !ap {
                            continue;
                        }
                        peel_slot[k] = n_peel_slots as u8;
                        n_peel_slots += 1;
                        (mc, true)
                    } else {
                        let _ = crate::contact::step_clearance_subblock(driver, k, sub, None);
                        continue;
                    }
                } else {
                    continue;
                }
            } else if key.has_damper {
                if n_peel_slots < MAX_PEEL_SLOTS {
                    let (mc, ap) = step_clearance_subblock_with_peel(
                        driver,
                        k,
                        sub,
                        Some(&mut r.ports.y_clear_scratch[k][..sub]),
                        &mut peel_buf[n_peel_slots][..sub],
                    );
                    if ap {
                        peel_slot[k] = n_peel_slots as u8;
                        n_peel_slots += 1;
                    }
                    (mc, ap)
                } else {
                    let mc = crate::contact::step_clearance_subblock(
                        driver,
                        k,
                        sub,
                        Some(&mut r.ports.y_clear_scratch[k][..sub]),
                    );
                    (mc, false)
                }
            } else {
                (f32::MAX, false)
            };
            if r.sav_on && k < LONGITUDINAL_MAX_KEY_IDX && !r.lossless && !hammer && !mid_ev {
                for s_idx in sr.clone() {
                    if r.ports.struck_active[s_idx] {
                        update_sav_dormancy(r.ports, r.bridge_ports, r.sav, s_idx);
                    }
                }
            }
            if mid_ev || hammer {
                plan.key_cat[k] = KeyCat::ActiveContact;
                plan.key_has_hammer[k] = hammer;
                plan.key_has_mid_ev[k] = mid_ev;
                plan.active_keys[plan.n_a] = k as u8;
                plan.n_a += 1;
                for s_idx in sr.clone() {
                    plan.push_coupled(s_idx);
                }
            } else if has_peel || min_c < DAMPER_CLEAR_THRESHOLD_M {
                let open_clear = !has_peel
                    && key.has_damper
                    && key_clear_of_open_damper(&r, &plan.was_nz, sr.clone(), min_c);
                if open_clear {
                    for s_idx in sr.clone() {
                        if plan.was_nz[s_idx] {
                            driver.relax_open_gap_chunk(s_idx, sub);
                        }
                    }
                } else if key.has_damper && key_needs_active_tail(&r, sr.clone(), min_c) {
                    plan.key_cat[k] = KeyCat::ActiveContact;
                    plan.active_keys[plan.n_a] = k as u8;
                    plan.n_a += 1;
                    for s_idx in sr.clone() {
                        plan.push_coupled(s_idx);
                    }
                } else {
                    plan.key_cat[k] = KeyCat::DampedSymp;
                    if key.has_damper {
                        route_damped_symp_key(
                            &mut r,
                            driver,
                            &mut plan,
                            k,
                            sr.clone(),
                            sub,
                            has_peel,
                        );
                    }
                }
            }
            if plan.key_cat[k] != KeyCat::ActiveContact
                || (!plan.key_has_hammer[k] && !plan.key_has_mid_ev[k])
            {
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
                &peel_slot,
                &peel_buf,
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
