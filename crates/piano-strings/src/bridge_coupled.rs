//! Stage 2 <-> Stage 3 closed-loop continuous SIMD stepping with per-sample WDF
//! bridge port coupling, `1x` unilateral dampers, key-batched `CHUNK = 128` tails,
//! 2-string paired SAV drift regulation, and zero `ModalSoa` touches in steady state.

use piano_spec::{NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_damp_kernels::refresh_chunk_damp_y;
use crate::bridge_flush::{flush_bridge_subnormals_core, update_sav_dormancy};
use crate::bridge_port::{BridgePortDriver, BridgePorts};
use crate::bridge_prefix_step::{
    step_clearance_subblock_with_peel, step_idle_clearance_check_peel,
};
use crate::bridge_scratch::{step_all_strings_prefix_wdf, BridgeScratch, MAX_PEEL_SLOTS};
use crate::bridge_tails::{
    key_clear_of_open_damper, key_needs_active_tail, prepare_sample_tails_and_hammers,
    route_damped_symp_key, sav_long_vecs, step_uncoupled_tails, BridgeRun, KeyCat, SavLongSlice,
    SUB,
};
use crate::choir_map::ChoirMap;
use crate::contact::KeyActionDriver;
use crate::contact_ports::{refresh_string_ports_bp, ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::longitudinal::LongitudinalBank;
use crate::sav_drift::SavBank;

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_subblock_bridge_coupled_core(
    ports: &mut ContactPorts,
    bridge_ports: &mut BridgePorts,
    long_bank: &LongitudinalBank,
    sav: &mut SavBank,
    map: &ChoirMap,
    scratch: &mut BridgeScratch,
    driver: &mut impl KeyActionDriver,
    bridge: &mut impl BridgePortDriver,
    passive_v_vecs: &[u16; TOTAL_STRINGS],
    passive_h_vecs: &[u16; TOTAL_STRINGS],
    sav_max_key_idx: usize,
    sav_dormancy_ratio: f64,
    energy_horizon_sq: f32,
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
    bridge_ports.sync_a_waves(&ports.may_be_nonzero);
    let mut a_waves = bridge_ports.a_waves;
    let mut r = BridgeRun {
        ports,
        bridge_ports,
        long_bank,
        sav,
        keys: map.keys(),
        strings: map.strings(),
        zone_weights,
        passive_v_vecs,
        passive_h_vecs,
        sav_max_key_idx,
        sav_dormancy_ratio,
        long_on,
        sav_on,
        lossless,
    };
    let plan = &mut scratch.plan;
    let peel_buf = &mut scratch.peel_buf;
    let peel_slot = &mut scratch.peel_slot;
    let b_waves = &mut scratch.b_waves;
    let zone_out = &mut scratch.zone_out;
    let dup_zone = &mut scratch.dup_zone;
    let acc_th = &mut scratch.acc_th;
    let acc_l = &mut scratch.acc_l;
    let mut chunk_base = 0usize;

    while chunk_base < total {
        let sub = (total - chunk_base).min(SUB);
        let (sub_l, sub_r) = (
            &mut out_l[chunk_base..chunk_base + sub],
            &mut out_r[chunk_base..chunk_base + sub],
        );
        plan.was_nz = r.ports.may_be_nonzero;
        plan.was_tail_nz = r.ports.tail_nonzero;
        plan.key_cat.fill(KeyCat::Uncoupled);
        plan.key_has_hammer.fill(false);
        plan.key_has_mid_ev.fill(false);
        for &s_u8 in &plan.coupled_strs[..plan.n_coupled_strs] {
            plan.str_coupled[usize::from(s_u8)] = false;
        }
        for &s_u8 in &plan.symp_dirty_strs[..plan.n_symp_dirty] {
            let s = usize::from(s_u8);
            plan.symp_has_gains[s] = false;
            plan.symp_gv[s] = 0.0;
            plan.symp_gh[s] = 0.0;
            plan.symp_mask[s] = 0.0;
            plan.symp_y_lim[s] = 0.0;
        }
        peel_slot.fill(u8::MAX);
        let mut n_peel_slots = 0usize;
        (
            plan.n_a,
            plan.n_coupled_strs,
            plan.n_symp_dirty,
            plan.n_sl,
            plan.n_sav,
            plan.n_sav_cd,
        ) = (0, 0, 0, 0, 0, 0);
        plan.zone_tail_n[..sub].fill([0.0; 8]);

        for (k, key) in r.keys.iter().enumerate() {
            let sr = key.string_range.clone();
            let mid_ev = driver.has_mid_block_event(k, chunk_base, sub);
            let hammer = driver.is_hammer_active(k);
            let any_nz = plan.was_nz[sr.clone()].iter().any(|&b| b);
            let (min_c, has_peel) = if mid_ev {
                (0.0, false)
            } else if let Some(cur_c) = driver
                .is_damper_settled_open(k, DAMPER_CLEAR_THRESHOLD_M)
                .filter(|&c| c >= DAMPER_CLEAR_THRESHOLD_M)
            {
                if !hammer && !any_nz {
                    continue;
                }
                if hammer {
                    r.ports.y_clear_scratch[k][..sub].fill(cur_c);
                }
                (cur_c, false)
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
            if r.sav_on && k < r.sav_max_key_idx && !r.lossless && !hammer && !mid_ev {
                for s_idx in sr.clone() {
                    if r.ports.struck_active[s_idx] {
                        update_sav_dormancy(
                            r.ports,
                            r.bridge_ports,
                            r.sav,
                            s_idx,
                            r.sav_dormancy_ratio,
                        );
                    }
                }
            }
            if !hammer
                && any_nz
                && key.has_damper
                && (mid_ev || has_peel || min_c < DAMPER_CLEAR_THRESHOLD_M)
            {
                for s_idx in sr.clone() {
                    if r.ports.may_be_nonzero[s_idx] && !r.ports.damp_ports_synced[s_idx] {
                        refresh_string_ports_bp(r.bridge_ports, r.ports, s_idx, true, 0.0);
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
                        route_damped_symp_key(&mut r, driver, plan, k, sr.clone(), sub, has_peel);
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
            if r.sav_on && k < r.sav_max_key_idx {
                let damp_cand =
                    key.has_damper && (!plan.key_has_hammer[k] || plan.key_has_mid_ev[k]);
                for s_idx in sr {
                    if r.ports.struck_active[s_idx]
                        && !r.ports.sav_dormant[s_idx]
                        && r.sav.string_n_vh(s_idx) > 0
                    {
                        plan.sav_strings[plan.n_sav] = s_idx as u8;
                        plan.n_sav += 1;
                        if damp_cand && (plan.str_coupled[s_idx] || plan.symp_has_gains[s_idx]) {
                            plan.sav_coupled_damp_strs[plan.n_sav_cd] = s_idx as u8;
                            plan.n_sav_cd += 1;
                        }
                    }
                }
            }
        }

        step_uncoupled_tails(&mut r, plan, acc_th, acc_l, sub_l, sub_r);
        for ts in 0..sub {
            prepare_sample_tails_and_hammers(&mut r, driver, plan, chunk_base, ts, sub_l, sub_r);
            let (dup_l, dup_r) = bridge.step_bridge_sample_with_duplex_zones(
                &a_waves,
                &plan.zone_tail_n[ts],
                b_waves,
                zone_out,
                dup_zone,
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
                plan,
                &mut a_waves,
                b_waves,
                (sub_l, sub_r),
                ts,
                peel_slot,
                peel_buf,
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
    r.bridge_ports.a_waves = a_waves;

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
        energy_horizon_sq,
        r.lossless,
    );
}
