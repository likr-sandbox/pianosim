//! Stage 2 <-> Stage 3 closed-loop continuous SIMD stepping with per-sample WDF
//! bridge port coupling (`< 5,500 Hz` prefix), `TILE = 8` unilateral dampers,
//! `CHUNK = 128` linear tails, SAV drift regulation, and longitudinal precursors.

use piano_simd::F32x4;
use piano_spec::{LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::{
    step_pol_prefix_wdf, step_pol_prefix_wdf_inj, BridgePortDriver, BridgePorts,
};
use crate::bridge_tails::{
    prepare_sample_tails_and_hammers, sav_long_vecs, step_damped_symp_tile, step_damped_tile,
    step_uncoupled_tails, BridgeRun, ChunkPlan, HammerStepForce, KeyCat, SUB,
};
use crate::choir_map::ChoirMap;
use crate::contact::{step_clearance_subblock, KeyActionDriver};
use crate::contact_ports::{refresh_string_ports, ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::{flush_subnormals_range, TILE};

fn step_all_strings_prefix_wdf(
    r: &mut BridgeRun<'_>,
    plan: &ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    t: usize,
) {
    let beta_buf = r.bridge_ports.beta_br.as_f32x4_slice();
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for s_idx in 0..TOTAL_STRINGS {
        let entry = &r.strings[s_idx];
        let k = usize::from(entry.key_idx);
        let m = &r.bridge_ports.meta[s_idx];
        let v_br = 0.5 * (a_waves[s_idx] + b_waves[s_idx]);
        let neg_vbr = F32x4::splat(-v_br);
        let is_hammer = plan.key_cat[k] == KeyCat::HammerActive;
        let (cv, br_v, ch, br_h, sav_active) = if is_hammer {
            let hf = plan.hammer_forces[s_idx];
            if hf.fh != 0.0 {
                r.ports.struck_active[s_idx] = true;
            }
            let nz_now = r.ports.may_be_nonzero[s_idx]
                || hf.fh != 0.0
                || hf.fd_v != 0.0
                || hf.fd_h != 0.0
                || v_br != 0.0;
            let (cv, br_v) = step_pol_prefix_wdf_inj(
                r.soa,
                r.ports,
                beta_buf,
                m.v_br.clone(),
                m.beta_v_off,
                neg_vbr,
                hf.fh,
                hf.fd_v,
                entry.has_damper,
            );
            let (ch, br_h) = step_pol_prefix_wdf_inj(
                r.soa,
                r.ports,
                beta_buf,
                m.h_br.clone(),
                m.beta_h_off,
                neg_vbr,
                hf.fh,
                hf.fd_h,
                entry.has_damper,
            );
            r.ports.may_be_nonzero[s_idx] = nz_now;
            (cv, br_v, ch, br_h, r.ports.struck_active[s_idx])
        } else {
            let (cv, br_v) =
                step_pol_prefix_wdf(r.soa, beta_buf, m.v_br.clone(), m.beta_v_off, neg_vbr);
            let (ch, br_h) =
                step_pol_prefix_wdf(r.soa, beta_buf, m.h_br.clone(), m.beta_h_off, neg_vbr);
            if v_br != 0.0 {
                r.ports.may_be_nonzero[s_idx] = true;
                r.ports.ports_synced[s_idx] = false;
            }
            (cv, br_v, ch, br_h, r.ports.struck_active[s_idx])
        };
        if r.sav_on && sav_active && k < LONGITUDINAL_MAX_KEY_IDX && r.sav.string_n_vh(s_idx) > 0 {
            r.sav.step_string_sav(s_idx, entry, r.soa, r.long_on);
        }
        if is_hammer && r.ports.may_be_nonzero[s_idx] {
            refresh_string_ports(r.soa, r.ports, s_idx, entry);
        } else if matches!(plan.key_cat[k], KeyCat::DampedStruck | KeyCat::DampedSymp) {
            r.ports.ports_synced[s_idx] = false;
        }
        a_waves[s_idx] = m.a_wave_gain * (br_v + br_h);
        let term = cv + ch;
        sum_l += term * entry.pan_l;
        sum_r += term * entry.pan_r;
    }
    out_pair.0[t] += sum_l;
    out_pair.1[t] += sum_r;
}

/// Executes one closed-loop Stage 1 <-> Stage 2 <-> Stage 3 audio block (`<= SUB` chunked)
/// driving all 88 keys / 230 strings / ~26k modes unconditionally with zero heap allocations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_subblock_bridge_coupled_core(
    soa: &mut ModalSoa,
    ports: &mut ContactPorts,
    bridge_ports: &BridgePorts,
    long_bank: &LongitudinalBank,
    sav: &mut SavBank,
    map: &ChoirMap,
    driver: &mut impl KeyActionDriver,
    bridge: &mut impl BridgePortDriver,
    long_on: bool,
    sav_on: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    out_l.fill(0.0);
    out_r.fill(0.0);
    let zone_weights = *bridge.zone_weights();
    let mut r = BridgeRun {
        soa,
        ports,
        bridge_ports,
        long_bank,
        sav,
        keys: map.keys(),
        strings: map.strings(),
        zone_weights,
        long_on,
        sav_on,
    };
    let (mut chunk_base, mut a_waves, mut b_waves, mut zone_out) = (
        0usize,
        [0.0_f32; TOTAL_STRINGS],
        [0.0_f32; TOTAL_STRINGS],
        [0.0_f32; 8],
    );
    let mut plan = ChunkPlan {
        key_cat: [KeyCat::Uncoupled; NUM_KEYS],
        was_nz: [false; TOTAL_STRINGS],
        hammer_keys: [0; NUM_KEYS],
        damped_keys: [0; NUM_KEYS],
        damped_symp_keys: [0; NUM_KEYS],
        sav_long_strings: [0; TOTAL_STRINGS],
        hammer_forces: [HammerStepForce::ZERO; TOTAL_STRINGS],
        zone_tail_n: [[0.0; 8]; SUB],
        n_h: 0,
        n_d: 0,
        n_ds: 0,
        n_sl: 0,
    };

    while chunk_base < total {
        let sub = (total - chunk_base).min(SUB);
        let (sub_l, sub_r) = (
            &mut out_l[chunk_base..chunk_base + sub],
            &mut out_r[chunk_base..chunk_base + sub],
        );
        plan.was_nz = r.ports.may_be_nonzero;
        r.bridge_ports
            .compute_a_waves(r.soa, &plan.was_nz, &mut a_waves);
        plan.key_cat.fill(KeyCat::Uncoupled);
        (plan.n_h, plan.n_d, plan.n_ds, plan.n_sl) = (0, 0, 0, 0);
        plan.zone_tail_n[..sub].fill([0.0; 8]);

        for (k, key) in r.keys.iter().enumerate() {
            let sr = key.string_range.clone();
            let hammer = driver.is_hammer_active(k);
            let any_nz = plan.was_nz[sr.clone()].iter().any(|&b| b);
            if !hammer && !any_nz {
                if key.has_damper {
                    let _ = step_clearance_subblock(driver, k, sub, None);
                }
                continue;
            }
            let min_c = if key.has_damper {
                step_clearance_subblock(
                    driver,
                    k,
                    sub,
                    Some(&mut r.ports.y_clear_scratch[k][..sub]),
                )
            } else {
                f32::MAX
            };
            if hammer {
                plan.key_cat[k] = KeyCat::HammerActive;
                plan.hammer_keys[plan.n_h] = k as u8;
                plan.n_h += 1;
            } else if min_c < DAMPER_CLEAR_THRESHOLD_M {
                let any_struck = r.ports.struck_active[sr.clone()].iter().any(|&b| b);
                if any_struck {
                    plan.key_cat[k] = KeyCat::DampedStruck;
                    plan.damped_keys[plan.n_d] = k as u8;
                    plan.n_d += 1;
                } else {
                    plan.key_cat[k] = KeyCat::DampedSymp;
                    plan.damped_symp_keys[plan.n_ds] = k as u8;
                    plan.n_ds += 1;
                }
            }
            if !hammer {
                for s_idx in sr {
                    if sav_long_vecs(&r, s_idx) > 0 {
                        plan.sav_long_strings[plan.n_sl] = s_idx as u8;
                        plan.n_sl += 1;
                    }
                }
            }
        }

        step_uncoupled_tails(&mut r, &mut plan, sub_l, sub_r);
        let mut t = 0usize;
        while t < sub {
            let t_len = (sub - t).min(TILE);
            if plan.n_d > 0 {
                step_damped_tile(&mut r, driver, &mut plan, t, t_len, sub_l, sub_r);
            }
            if plan.n_ds > 0 {
                step_damped_symp_tile(&mut r, driver, &plan, t, t_len);
            }
            for off in 0..t_len {
                let ts = t + off;
                prepare_sample_tails_and_hammers(&mut r, driver, &mut plan, ts, sub_l, sub_r);
                let (dup_l, dup_r) = bridge.step_bridge_sample(
                    &a_waves,
                    &plan.zone_tail_n[ts],
                    &mut b_waves,
                    &mut zone_out,
                );
                if dup_l.is_finite() {
                    sub_l[ts] += dup_l;
                }
                if dup_r.is_finite() {
                    sub_r[ts] += dup_r;
                }
                step_all_strings_prefix_wdf(
                    &mut r,
                    &plan,
                    &mut a_waves,
                    &b_waves,
                    (sub_l, sub_r),
                    ts,
                );
            }
            t += t_len;
        }
        for (s_idx, entry) in r.strings.iter().enumerate() {
            if r.ports.may_be_nonzero[s_idx] && !flush_subnormals_range(r.soa, entry.all_vecs()) {
                r.ports.clear_string_state(s_idx);
                r.sav.reset_string(s_idx);
            }
        }
        chunk_base += sub;
    }
}
