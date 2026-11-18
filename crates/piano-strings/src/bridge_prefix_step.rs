//! Coupled-string prefix WDF + hammer/damper stepping kernel (`no_std`).

use piano_simd::F32x4;
use piano_spec::{DAMPED_KEYS, LONGITUDINAL_MAX_KEY_IDX, TOTAL_STRINGS};

use crate::bridge_kernels::{
    step_pol_prefix_wdf_inj, step_string_prefix_wdf_br, step_string_prefix_wdf_symp_damp_vh,
};
use crate::bridge_port::ALT_SIGN_ARR;
use crate::bridge_tails::{BridgeRun, ChunkPlan, KeyCat, SUB};
use crate::contact::KeyActionDriver;
use crate::contact_ports::{refresh_prefix_damp_ports_bp, DAMPER_CLEAR_THRESHOLD_M};
use crate::zoh_kernel::TILE;

#[inline]
pub(crate) fn step_clearance_subblock_with_peel(
    driver: &mut impl KeyActionDriver,
    k: usize,
    sub: usize,
    mut dst_clear: Option<&mut [f32]>,
    dst_peel: &mut [f32],
) -> (f32, bool) {
    let (mut min_c, mut any_peel, mut b) = (f32::MAX, false, 0usize);
    let (mut tmp_c, mut tmp_p) = ([0.0_f32; TILE], [0.0_f32; TILE]);
    while b < sub {
        let t_len = (sub - b).min(TILE);
        let (mc, tile_peel) =
            driver.step_key_clearance_tile_with_peel(k, t_len, &mut tmp_c, &mut tmp_p);
        min_c = min_c.min(mc);
        if let Some(ref mut out) = dst_clear {
            out[b..b + t_len].copy_from_slice(&tmp_c[..t_len]);
        }
        if tile_peel {
            if !any_peel && b > 0 {
                dst_peel[..b].fill(0.0);
            }
            dst_peel[b..b + t_len].copy_from_slice(&tmp_p[..t_len]);
            any_peel = true;
        } else if any_peel {
            dst_peel[b..b + t_len].fill(0.0);
        }
        b += t_len;
    }
    (min_c, any_peel)
}

#[inline]
pub(crate) fn step_idle_clearance_check_peel(
    driver: &mut impl KeyActionDriver,
    k: usize,
    sub: usize,
    dst_clear: &mut [f32],
    dst_peel: &mut [f32],
) -> (f32, bool) {
    let (mut min_c, mut any_peel, mut b) = (f32::MAX, false, 0usize);
    let (mut tmp_c, mut tmp_p) = ([0.0_f32; TILE], [0.0_f32; TILE]);
    while b < sub {
        let t_len = (sub - b).min(TILE);
        let (mc, tile_peel) =
            driver.step_key_clearance_tile_with_peel(k, t_len, &mut tmp_c, &mut tmp_p);
        min_c = min_c.min(mc);
        if tile_peel {
            if !any_peel {
                if b > 0 {
                    dst_clear[..b].fill(tmp_c[0]);
                    dst_peel[..b].fill(0.0);
                }
                any_peel = true;
            }
            dst_clear[b..b + t_len].copy_from_slice(&tmp_c[..t_len]);
            dst_peel[b..b + t_len].copy_from_slice(&tmp_p[..t_len]);
        } else if any_peel {
            dst_clear[b..b + t_len].copy_from_slice(&tmp_c[..t_len]);
            dst_peel[b..b + t_len].fill(0.0);
        }
        b += t_len;
    }
    (min_c, any_peel)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_coupled_prefixes(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    n_coupled: usize,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    ts: usize,
    peel_slot: &[u8; DAMPED_KEYS],
    peel_buf: &[[f32; SUB]],
) {
    let alt_sign = F32x4::from_array(ALT_SIGN_ARR);
    let (mut sum_l_vec, mut sum_r_vec) = (F32x4::zero(), F32x4::zero());
    for i in 0..n_coupled {
        let s_idx = usize::from(plan.coupled_strs[i]);
        let c_sign_v = F32x4::splat(r.bridge_ports.c_vert[s_idx]) * alt_sign;
        let c_sign_h = F32x4::splat(r.bridge_ports.c_horiz[s_idx]) * alt_sign;
        let entry = &r.strings[s_idx];
        let k = usize::from(entry.key_idx);
        let f_peel = if !entry.has_damper {
            0.0
        } else if plan.key_has_mid_ev[k] {
            let f = driver.damper_peel_force_n(k);
            if f.is_finite() {
                f
            } else {
                0.0
            }
        } else {
            let slot = usize::from(peel_slot[k]);
            if slot < peel_buf.len() {
                peel_buf[slot][ts]
            } else {
                0.0
            }
        };
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
                }
            }
            let nz_now = r.ports.may_be_nonzero[s_idx]
                || hf.fh != 0.0
                || hf.fd_v != 0.0
                || hf.fd_h != 0.0
                || f_peel != 0.0
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
                    hf.alpha_w,
                    hf.fd_v + f_peel,
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
                    hf.alpha_w,
                    hf.fd_h,
                    entry.has_damper,
                );
                (cv_v + ch_v, br_v_v + br_h_v)
            } else {
                let (acc_c, acc_br, [ydv_br, vdv_br_raw, ydh_br, vdh_br_raw]) =
                    step_string_prefix_wdf_symp_damp_vh(
                        r.ports,
                        r.bridge_ports,
                        v_br_r,
                        h_br_r,
                        beta_v_off,
                        beta_h_off,
                        c_sign_v,
                        c_sign_h,
                        neg_vbr,
                        hf.fd_v + f_peel,
                        hf.fd_h,
                    );
                let tp = plan.tail_partial_damp[s_idx];
                r.ports.y_dv[s_idx] = ydv_br + tp.ydv;
                r.ports.y_dh[s_idx] = ydh_br + tp.ydh;
                if !(r.sav_on
                    && r.ports.struck_active[s_idx]
                    && !r.ports.sav_dormant[s_idx]
                    && k < LONGITUDINAL_MAX_KEY_IDX
                    && r.sav.string_n_vh(s_idx) > 0)
                {
                    let alpha = r.ports.alpha_vel[s_idx];
                    r.ports.v_dv[s_idx] = alpha * (vdv_br_raw + tp.vdv_raw);
                    r.ports.v_dh[s_idx] = alpha * (vdh_br_raw + tp.vdh_raw);
                }
                r.ports.ports_synced[s_idx] = true;
                (acc_c, acc_br)
            }
        } else if cat == KeyCat::DampedSymp
            && entry.has_damper
            && (r.ports.may_be_nonzero[s_idx] || f_peel != 0.0)
        {
            if f_peel != 0.0 {
                r.ports.may_be_nonzero[s_idx] = true;
            }
            let y_lim = plan.symp_y_lim[s_idx];
            if plan.symp_has_gains[s_idx]
                && f_peel == 0.0
                && r.ports.y_dv[s_idx].abs() <= y_lim
                && r.ports.y_dh[s_idx].abs() <= y_lim
            {
                let (gv, gh) = (plan.symp_gv[s_idx], plan.symp_gh[s_idx]);
                let fd_v = gv * r.ports.v_dv[s_idx];
                let fd_h = gh * r.ports.v_dh[s_idx];
                let (acc_c, acc_br, [ydv, vdv_raw, ydh, vdh_raw]) =
                    step_string_prefix_wdf_symp_damp_vh(
                        r.ports,
                        r.bridge_ports,
                        v_br_r,
                        h_br_r,
                        beta_v_off,
                        beta_h_off,
                        c_sign_v,
                        c_sign_h,
                        neg_vbr,
                        fd_v,
                        fd_h,
                    );
                let alpha = r.ports.alpha_vel[s_idx];
                r.ports.y_dv[s_idx] = ydv;
                r.ports.v_dv[s_idx] = alpha * vdv_raw;
                r.ports.y_dh[s_idx] = ydh;
                r.ports.v_dh[s_idx] = alpha * vdh_raw;
                r.ports.ports_synced[s_idx] = true;
                (acc_c, acc_br)
            } else {
                plan.symp_has_gains[s_idx] = false;
                let yc = r.ports.y_clear_scratch[k][ts];
                if yc < DAMPER_CLEAR_THRESHOLD_M {
                    if !r.ports.ports_synced[s_idx] {
                        refresh_prefix_damp_ports_bp(r.bridge_ports, r.ports, s_idx);
                    }
                    if f_peel == 0.0
                        && yc > 2.0e-7
                        && !r.ports.tail_nonzero[s_idx]
                        && r.ports.y_dv[s_idx].abs() <= 0.25 * yc
                        && r.ports.y_dh[s_idx].abs() <= 0.25 * yc
                    {
                        let (acc_c, acc_br) = step_string_prefix_wdf_br(
                            r.bridge_ports,
                            beta_v_off,
                            nv,
                            nh,
                            c_sign_v,
                            c_sign_h,
                            neg_vbr,
                        );
                        r.ports.ports_synced[s_idx] = false;
                        (acc_c, acc_br)
                    } else {
                        let (raw_fv, raw_fh) = driver.step_damper_pair(
                            k,
                            s_idx,
                            r.ports.y_dv[s_idx],
                            r.ports.v_dv[s_idx],
                            r_dv,
                            r.ports.y_dh[s_idx],
                            r.ports.v_dh[s_idx],
                            r_dh,
                            yc,
                        );
                        let fd_v = (if raw_fv.is_finite() { raw_fv } else { 0.0 }) + f_peel;
                        let fd_h = if raw_fh.is_finite() { raw_fh } else { 0.0 };
                        let (acc_c, acc_br, [ydv, vdv_raw, ydh, vdh_raw]) =
                            step_string_prefix_wdf_symp_damp_vh(
                                r.ports,
                                r.bridge_ports,
                                v_br_r,
                                h_br_r,
                                beta_v_off,
                                beta_h_off,
                                c_sign_v,
                                c_sign_h,
                                neg_vbr,
                                fd_v,
                                fd_h,
                            );
                        let alpha = r.ports.alpha_vel[s_idx];
                        r.ports.y_dv[s_idx] = ydv;
                        r.ports.v_dv[s_idx] = alpha * vdv_raw;
                        r.ports.y_dh[s_idx] = ydh;
                        r.ports.v_dh[s_idx] = alpha * vdh_raw;
                        r.ports.ports_synced[s_idx] = true;
                        (acc_c, acc_br)
                    }
                } else if f_peel != 0.0 {
                    let (acc_c, acc_br, [ydv, vdv_raw, ydh, vdh_raw]) =
                        step_string_prefix_wdf_symp_damp_vh(
                            r.ports,
                            r.bridge_ports,
                            v_br_r,
                            h_br_r,
                            beta_v_off,
                            beta_h_off,
                            c_sign_v,
                            c_sign_h,
                            neg_vbr,
                            f_peel,
                            0.0,
                        );
                    let alpha = r.ports.alpha_vel[s_idx];
                    r.ports.y_dv[s_idx] = ydv;
                    r.ports.v_dv[s_idx] = alpha * vdv_raw;
                    r.ports.y_dh[s_idx] = ydh;
                    r.ports.v_dh[s_idx] = alpha * vdh_raw;
                    r.ports.ports_synced[s_idx] = true;
                    (acc_c, acc_br)
                } else {
                    let (acc_c, acc_br) = step_string_prefix_wdf_br(
                        r.bridge_ports,
                        beta_v_off,
                        nv,
                        nh,
                        c_sign_v,
                        c_sign_h,
                        neg_vbr,
                    );
                    r.ports.ports_synced[s_idx] = false;
                    (acc_c, acc_br)
                }
            }
        } else {
            let (acc_c, acc_br) = step_string_prefix_wdf_br(
                r.bridge_ports,
                beta_v_off,
                nv,
                nh,
                c_sign_v,
                c_sign_h,
                neg_vbr,
            );
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

#[cfg(test)]
#[path = "bridge_prefix_step_tests.rs"]
mod tests;
