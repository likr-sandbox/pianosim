//! Coupled-string prefix WDF + hammer/damper stepping kernel (`no_std`).

use piano_simd::F32x4;
use piano_spec::{LONGITUDINAL_MAX_KEY_IDX, TOTAL_STRINGS};

use crate::bridge_kernels::{
    step_pol_prefix_wdf_inj, step_pol_prefix_wdf_symp_damp, step_string_prefix_wdf_br,
};
use crate::bridge_port::{C_SIGN_H_ARR, C_SIGN_V_ARR};
use crate::bridge_tails::{BridgeRun, ChunkPlan, KeyCat};
use crate::contact::KeyActionDriver;
use crate::contact_ports::{refresh_prefix_damp_ports_bp, DAMPER_CLEAR_THRESHOLD_M};

#[inline(never)]
pub(crate) fn step_coupled_prefixes(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    ts: usize,
) {
    let c_sign_v = F32x4::from_array(C_SIGN_V_ARR);
    let c_sign_h = F32x4::from_array(C_SIGN_H_ARR);
    let (mut sum_l_vec, mut sum_r_vec) = (F32x4::zero(), F32x4::zero());
    for i in 0..plan.n_coupled_strs {
        let s_idx = usize::from(plan.coupled_strs[i]);
        let entry = &r.strings[s_idx];
        let k = usize::from(entry.key_idx);
        let m = &r.bridge_ports.meta[s_idx];
        let (v_br_r, h_br_r) = (m.v_br.clone(), m.h_br.clone());
        let (beta_v_off, beta_h_off, a_wave_gain) = (m.beta_v_off, m.beta_h_off, m.a_wave_gain);
        let (r_dv, r_dh) = (
            r.bridge_ports.r_damp_prefix_v[s_idx],
            r.bridge_ports.r_damp_prefix_h[s_idx],
        );
        let (nv, nh) = (v_br_r.len(), h_br_r.len());
        let v_br = 0.5 * (a_waves[s_idx] + b_waves[s_idx]);
        let neg_vbr = F32x4::splat(-v_br);
        let cat = plan.key_cat[k];
        let (acc_c, acc_br) = if cat == KeyCat::ActiveContact {
            let hf = plan.hammer_forces[s_idx];
            if hf.fh != 0.0 && !r.ports.struck_active[s_idx] {
                r.ports.struck_active[s_idx] = true;
                if r.sav_on && k < LONGITUDINAL_MAX_KEY_IDX && r.sav.string_n_vh(s_idx) > 0 {
                    plan.sav_strings[plan.n_sav] = s_idx as u8;
                    plan.n_sav += 1;
                    plan.sav_sched_dirty = true;
                }
            }
            let nz_now = r.ports.may_be_nonzero[s_idx]
                || hf.fh != 0.0
                || hf.fd_v != 0.0
                || hf.fd_h != 0.0
                || v_br != 0.0;
            r.ports.may_be_nonzero[s_idx] = nz_now;
            if plan.key_has_hammer[k] {
                let (cv_v, br_v_v) = step_pol_prefix_wdf_inj(
                    r.ports,
                    r.bridge_ports,
                    v_br_r,
                    beta_v_off,
                    c_sign_v,
                    neg_vbr,
                    hf.fh,
                    hf.fd_v,
                    entry.has_damper,
                );
                let (ch_v, br_h_v) = step_pol_prefix_wdf_inj(
                    r.ports,
                    r.bridge_ports,
                    h_br_r,
                    beta_h_off,
                    c_sign_h,
                    neg_vbr,
                    hf.fh,
                    hf.fd_h,
                    entry.has_damper,
                );
                (cv_v + ch_v, br_v_v + br_h_v)
            } else {
                let (cv_v, br_v_v, ydv_br, vdv_br_raw) = step_pol_prefix_wdf_symp_damp(
                    r.ports,
                    r.bridge_ports,
                    v_br_r,
                    beta_v_off,
                    c_sign_v,
                    neg_vbr,
                    hf.fd_v,
                );
                let (ch_v, br_h_v, ydh_br, vdh_br_raw) = step_pol_prefix_wdf_symp_damp(
                    r.ports,
                    r.bridge_ports,
                    h_br_r,
                    beta_h_off,
                    c_sign_h,
                    neg_vbr,
                    hf.fd_h,
                );
                let tp = plan.tail_partial_damp[s_idx];
                r.ports.y_dv[s_idx] = ydv_br + tp.ydv;
                r.ports.y_dh[s_idx] = ydh_br + tp.ydh;
                if !(r.sav_on
                    && r.ports.struck_active[s_idx]
                    && k < LONGITUDINAL_MAX_KEY_IDX
                    && r.sav.string_n_vh(s_idx) > 0)
                {
                    let alpha = r.ports.alpha_vel[s_idx];
                    r.ports.v_dv[s_idx] = alpha * (vdv_br_raw + tp.vdv_raw);
                    r.ports.v_dh[s_idx] = alpha * (vdh_br_raw + tp.vdh_raw);
                }
                r.ports.ports_synced[s_idx] = true;
                (cv_v + ch_v, br_v_v + br_h_v)
            }
        } else if cat == KeyCat::DampedSymp && entry.has_damper && r.ports.may_be_nonzero[s_idx] {
            let yc = r.ports.y_clear_scratch[k][ts];
            if yc < DAMPER_CLEAR_THRESHOLD_M {
                if !r.ports.ports_synced[s_idx] {
                    refresh_prefix_damp_ports_bp(r.bridge_ports, r.ports, s_idx);
                }
                let raw_fv = driver.step_damper_sample(
                    k,
                    s_idx * 2,
                    r.ports.y_dv[s_idx],
                    r.ports.v_dv[s_idx],
                    r_dv,
                    yc,
                );
                let raw_fh = driver.step_damper_sample(
                    k,
                    s_idx * 2 + 1,
                    r.ports.y_dh[s_idx],
                    r.ports.v_dh[s_idx],
                    r_dh,
                    yc,
                );
                let fd_v = if raw_fv.is_finite() { raw_fv } else { 0.0 };
                let fd_h = if raw_fh.is_finite() { raw_fh } else { 0.0 };
                let (cv_v, br_v_v, ydv, vdv_raw) = step_pol_prefix_wdf_symp_damp(
                    r.ports,
                    r.bridge_ports,
                    v_br_r,
                    beta_v_off,
                    c_sign_v,
                    neg_vbr,
                    fd_v,
                );
                let (ch_v, br_h_v, ydh, vdh_raw) = step_pol_prefix_wdf_symp_damp(
                    r.ports,
                    r.bridge_ports,
                    h_br_r,
                    beta_h_off,
                    c_sign_h,
                    neg_vbr,
                    fd_h,
                );
                let alpha = r.ports.alpha_vel[s_idx];
                r.ports.y_dv[s_idx] = ydv;
                r.ports.v_dv[s_idx] = alpha * vdv_raw;
                r.ports.y_dh[s_idx] = ydh;
                r.ports.v_dh[s_idx] = alpha * vdh_raw;
                r.ports.ports_synced[s_idx] = true;
                (cv_v + ch_v, br_v_v + br_h_v)
            } else {
                let (acc_c, acc_br) =
                    step_string_prefix_wdf_br(r.bridge_ports, beta_v_off, nv, nh, neg_vbr);
                r.ports.ports_synced[s_idx] = false;
                (acc_c, acc_br)
            }
        } else {
            let (acc_c, acc_br) =
                step_string_prefix_wdf_br(r.bridge_ports, beta_v_off, nv, nh, neg_vbr);
            if r.ports.may_be_nonzero[s_idx] || v_br != 0.0 {
                r.ports.may_be_nonzero[s_idx] = true;
                r.ports.ports_synced[s_idx] = false;
            }
            (acc_c, acc_br)
        };
        a_waves[s_idx] = a_wave_gain * acc_br.reduce_sum();
        sum_l_vec = acc_c.mul_add(F32x4::splat(entry.pan_l), sum_l_vec);
        sum_r_vec = acc_c.mul_add(F32x4::splat(entry.pan_r), sum_r_vec);
    }
    out_pair.0[ts] += sum_l_vec.reduce_sum();
    out_pair.1[ts] += sum_r_vec.reduce_sum();
}
