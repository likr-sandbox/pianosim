//! Stage 2 <-> Stage 3 closed-loop continuous SIMD stepping with per-sample WDF
//! bridge port coupling (`< 5,500 Hz` prefix), `1x` per-sample unilateral dampers,
//! `CHUNK = 128` linear tails, SAV drift regulation, and longitudinal precursors.

use piano_simd::F32x4;
use piano_spec::{LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::{
    step_pol_prefix_wdf, step_pol_prefix_wdf_inj, step_pol_prefix_wdf_symp_damp, BridgePortDriver,
    BridgePorts,
};
use crate::bridge_tails::{
    prepare_sample_tails_and_hammers, sav_long_vecs, step_uncoupled_tails, BridgeRun, ChunkPlan,
    HammerStepForce, KeyCat, SUB,
};
use crate::choir_map::ChoirMap;
use crate::contact::{step_clearance_subblock, KeyActionDriver};
use crate::contact_ports::{
    refresh_prefix_damp_ports, refresh_string_ports, ContactPorts, DAMPER_CLEAR_THRESHOLD_M,
};
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::flush_subnormals_range;

fn step_all_strings_prefix_wdf(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    ts: usize,
) {
    let beta_buf = r.bridge_ports.beta_br.as_f32x4_slice();
    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for s_idx in 0..TOTAL_STRINGS {
        let entry = &r.strings[s_idx];
        let k = usize::from(entry.key_idx);
        let m = &r.bridge_ports.meta[s_idx];
        let v_br = 0.5 * (a_waves[s_idx] + b_waves[s_idx]);
        let neg_vbr = F32x4::splat(-v_br);
        let cat = plan.key_cat[k];
        let (cv, br_v, ch, br_h) = if cat == KeyCat::ActiveContact {
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
            (cv, br_v, ch, br_h)
        } else if cat == KeyCat::DampedSymp && entry.has_damper && r.ports.may_be_nonzero[s_idx] {
            let yc = r.ports.y_clear_scratch[k][ts];
            if yc < DAMPER_CLEAR_THRESHOLD_M {
                if !r.ports.ports_synced[s_idx] {
                    refresh_prefix_damp_ports(
                        r.soa,
                        r.ports,
                        s_idx,
                        m.v_br.clone(),
                        m.h_br.clone(),
                    );
                }
                let raw_fv = driver.step_damper_sample(
                    k,
                    s_idx * 2,
                    r.ports.y_dv[s_idx],
                    r.ports.v_dv[s_idx],
                    r.bridge_ports.r_damp_prefix_v[s_idx],
                    yc,
                );
                let raw_fh = driver.step_damper_sample(
                    k,
                    s_idx * 2 + 1,
                    r.ports.y_dh[s_idx],
                    r.ports.v_dh[s_idx],
                    r.bridge_ports.r_damp_prefix_h[s_idx],
                    yc,
                );
                let fd_v = if raw_fv.is_finite() { raw_fv } else { 0.0 };
                let fd_h = if raw_fh.is_finite() { raw_fh } else { 0.0 };
                let (cv, br_v, ydv, vdv_raw) = step_pol_prefix_wdf_symp_damp(
                    r.soa,
                    r.ports,
                    beta_buf,
                    m.v_br.clone(),
                    m.beta_v_off,
                    neg_vbr,
                    fd_v,
                );
                let (ch, br_h, ydh, vdh_raw) = step_pol_prefix_wdf_symp_damp(
                    r.soa,
                    r.ports,
                    beta_buf,
                    m.h_br.clone(),
                    m.beta_h_off,
                    neg_vbr,
                    fd_h,
                );
                let alpha = r.ports.alpha_vel[s_idx];
                r.ports.y_dv[s_idx] = ydv;
                r.ports.v_dv[s_idx] = alpha * vdv_raw;
                r.ports.y_dh[s_idx] = ydh;
                r.ports.v_dh[s_idx] = alpha * vdh_raw;
                r.ports.ports_synced[s_idx] = true;
                (cv, br_v, ch, br_h)
            } else {
                let (cv, br_v) =
                    step_pol_prefix_wdf(r.soa, beta_buf, m.v_br.clone(), m.beta_v_off, neg_vbr);
                let (ch, br_h) =
                    step_pol_prefix_wdf(r.soa, beta_buf, m.h_br.clone(), m.beta_h_off, neg_vbr);
                r.ports.ports_synced[s_idx] = false;
                (cv, br_v, ch, br_h)
            }
        } else {
            let (cv, br_v) =
                step_pol_prefix_wdf(r.soa, beta_buf, m.v_br.clone(), m.beta_v_off, neg_vbr);
            let (ch, br_h) =
                step_pol_prefix_wdf(r.soa, beta_buf, m.h_br.clone(), m.beta_h_off, neg_vbr);
            if r.ports.may_be_nonzero[s_idx] || v_br != 0.0 {
                r.ports.may_be_nonzero[s_idx] = true;
                r.ports.ports_synced[s_idx] = false;
            }
            (cv, br_v, ch, br_h)
        };
        if r.sav_on
            && r.ports.struck_active[s_idx]
            && k < LONGITUDINAL_MAX_KEY_IDX
            && r.sav.string_n_vh(s_idx) > 0
        {
            r.sav.step_string_sav(s_idx, entry, r.soa, r.long_on);
        }
        if cat == KeyCat::ActiveContact && r.ports.may_be_nonzero[s_idx] {
            refresh_string_ports(r.soa, r.ports, s_idx, entry);
        }
        a_waves[s_idx] = m.a_wave_gain * (br_v + br_h);
        let term = cv + ch;
        sum_l += term * entry.pan_l;
        sum_r += term * entry.pan_r;
    }
    out_pair.0[ts] += sum_l;
    out_pair.1[ts] += sum_r;
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
        key_has_hammer: [false; NUM_KEYS],
        was_nz: [false; TOTAL_STRINGS],
        active_keys: [0; NUM_KEYS],
        sav_long_strings: [0; TOTAL_STRINGS],
        hammer_forces: [HammerStepForce::ZERO; TOTAL_STRINGS],
        zone_tail_n: [[0.0; 8]; SUB],
        n_a: 0,
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
        plan.key_has_hammer.fill(false);
        (plan.n_a, plan.n_sl) = (0, 0);
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
            let any_struck = r.ports.struck_active[sr.clone()].iter().any(|&b| b);
            if hammer || (any_struck && min_c < DAMPER_CLEAR_THRESHOLD_M) {
                plan.key_cat[k] = KeyCat::ActiveContact;
                plan.key_has_hammer[k] = hammer;
                plan.active_keys[plan.n_a] = k as u8;
                plan.n_a += 1;
            } else if min_c < DAMPER_CLEAR_THRESHOLD_M {
                plan.key_cat[k] = KeyCat::DampedSymp;
            }
            if plan.key_cat[k] != KeyCat::ActiveContact {
                for s_idx in sr {
                    if sav_long_vecs(&r, s_idx) > 0 {
                        plan.sav_long_strings[plan.n_sl] = s_idx as u8;
                        plan.n_sl += 1;
                    }
                }
            }
        }

        step_uncoupled_tails(&mut r, &mut plan, sub_l, sub_r);
        for ts in 0..sub {
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
                driver,
                &plan,
                &mut a_waves,
                &b_waves,
                (sub_l, sub_r),
                ts,
            );
        }
        for (s_idx, entry) in r.strings.iter().enumerate() {
            if r.ports.may_be_nonzero[s_idx] {
                if !flush_subnormals_range(r.soa, entry.all_vecs()) {
                    r.ports.clear_string_state(s_idx);
                    r.sav.reset_string(s_idx);
                } else if plan.key_cat[usize::from(entry.key_idx)] == KeyCat::DampedSymp {
                    // DampedSymp updates only prefix damper ports (`y_h`/`v_h` and tails need sync if promoted).
                    r.ports.ports_synced[s_idx] = false;
                }
            }
        }
        chunk_base += sub;
    }
}
