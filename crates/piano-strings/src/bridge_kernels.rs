//! L1d-packed WDF prefix ZOH kernels with 4-string SIMD transpose reduction (`no_std`).

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_damp_kernels::{step_prefix_quad_damp, step_single_prefix_damp};
use crate::bridge_open_kernels::{step_open_quad_write, step_prefix_run_k_open};
use crate::bridge_port::{BridgePorts, PrefixCoef, PrefixState, C_SIGN_V_ARR};
#[allow(unused_imports)]
pub(crate) use crate::bridge_prefix_wdf::{
    step_pol_prefix_wdf_inj, step_string_prefix_wdf_br, step_string_prefix_wdf_symp_damp_vh,
};
use crate::bridge_tails::ChunkPlan;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::{C_HORIZ, C_VERT};

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_open_quad_cold<const K: usize>(
    st4: &mut [PrefixState],
    cf4: &[PrefixCoef],
    nvp: F32x4,
    h_proj: F32x4,
    gains: F32x4,
    a_out: &mut [f32],
) -> F32x4 {
    step_open_quad_write::<K>(st4, cf4, nvp, h_proj, gains, a_out)
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_prefix_run_k<const K: usize>(
    bp: &mut BridgePorts,
    ports: &mut ContactPorts,
    p_off: usize,
    start_s: usize,
    count: usize,
    ts: usize,
    plan: &mut ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    sum_l_vec: &mut F32x4,
    sum_r_vec: &mut F32x4,
) {
    let h_proj = F32x4::splat(C_HORIZ / C_VERT);
    let c_sign_v = F32x4::from_array(C_SIGN_V_ARR);
    let neg_half = F32x4::splat(-0.5);
    let stride = 2 * K;
    let total_vecs = count * stride;
    let state = &mut bp.state[p_off..p_off + total_vecs];
    let coef = &bp.coef[p_off..p_off + total_vecs];
    let damp_bd = &bp.damp_bd[p_off..p_off + total_vecs];
    let damp_ad = &bp.damp_ad[p_off..p_off + total_vecs];
    let meta = &bp.fast_meta;
    let gains = &bp.a_wave_gains;
    let pan_l = &bp.pan_l;
    let pan_r = &bp.pan_r;
    let mut idx = 0usize;
    while idx + 4 <= count {
        let s = start_s + idx;
        if !(plan.str_coupled[s]
            | plan.str_coupled[s + 1]
            | plan.str_coupled[s + 2]
            | plan.str_coupled[s + 3])
        {
            let all_nz = ports.may_be_nonzero[s]
                & ports.may_be_nonzero[s + 1]
                & ports.may_be_nonzero[s + 2]
                & ports.may_be_nonzero[s + 3];
            let any_nz = ports.may_be_nonzero[s]
                | ports.may_be_nonzero[s + 1]
                | ports.may_be_nonzero[s + 2]
                | ports.may_be_nonzero[s + 3];
            let any_b = b_waves[s] != 0.0
                || b_waves[s + 1] != 0.0
                || b_waves[s + 2] != 0.0
                || b_waves[s + 3] != 0.0;
            if !any_nz && !any_b {
                a_waves[s..s + 4].fill(0.0);
                idx += 4;
                continue;
            }
            if !all_nz && any_b {
                for j in 0..4 {
                    if !ports.may_be_nonzero[s + j] && b_waves[s + j] != 0.0 {
                        ports.may_be_nonzero[s + j] = true;
                        ports.ports_synced[s + j] = false;
                    }
                }
            }
            let nvp = neg_half
                * (F32x4::from_slice(&a_waves[s..s + 4]) + F32x4::from_slice(&b_waves[s..s + 4]));
            let base = idx * stride;
            let st4 = &mut state[base..base + 4 * stride];
            let cf4 = &coef[base..base + 4 * stride];
            let any_damp = plan.symp_has_gains[s]
                | plan.symp_has_gains[s + 1]
                | plan.symp_has_gains[s + 2]
                | plan.symp_has_gains[s + 3];
            let ms4 = if !any_damp {
                step_open_quad_cold::<K>(
                    st4,
                    cf4,
                    nvp,
                    h_proj,
                    F32x4::from_slice(&gains[s..s + 4]),
                    &mut a_waves[s..s + 4],
                )
            } else {
                let (br4, ms4) = step_prefix_quad_damp::<K>(
                    st4,
                    cf4,
                    &damp_bd[base..base + 4 * stride],
                    &damp_ad[base..base + 4 * stride],
                    nvp,
                    h_proj,
                    s,
                    ts,
                    &plan.symp_y_lim,
                    &mut plan.str_coupled,
                    &mut plan.coupled_strs,
                    &mut plan.n_coupled_strs,
                    &mut plan.symp_has_gains,
                    &mut plan.symp_gv,
                    &mut plan.symp_gh,
                    &mut plan.symp_mask,
                    ports,
                );
                let aw4 = br4 * F32x4::from_slice(&gains[s..s + 4]);
                aw4.write_to_slice(&mut a_waves[s..s + 4]);
                ms4
            };
            *sum_l_vec = ms4.mul_add(F32x4::from_slice(&pan_l[s..s + 4]), *sum_l_vec);
            *sum_r_vec = ms4.mul_add(F32x4::from_slice(&pan_r[s..s + 4]), *sum_r_vec);
            idx += 4;
            continue;
        }
        for sub_i in 0..4 {
            let s_idx = s + sub_i;
            if plan.str_coupled[s_idx] {
                continue;
            }
            if !ports.may_be_nonzero[s_idx] {
                if b_waves[s_idx] == 0.0 {
                    a_waves[s_idx] = 0.0;
                    continue;
                }
                ports.may_be_nonzero[s_idx] = true;
                ports.ports_synced[s_idx] = false;
            }
            let off = (idx + sub_i) * stride;
            let neg_vbr = F32x4::splat(-0.5 * (a_waves[s_idx] + b_waves[s_idx]));
            let (acc_m, acc_br) = step_single_prefix_damp::<K>(
                s_idx,
                ts,
                &mut state[off..off + stride],
                &coef[off..off + stride],
                &damp_bd[off..off + stride],
                &damp_ad[off..off + stride],
                neg_vbr,
                h_proj,
                plan.symp_y_lim[s_idx],
                &mut plan.str_coupled,
                &mut plan.coupled_strs,
                &mut plan.n_coupled_strs,
                &mut plan.symp_has_gains,
                &mut plan.symp_gv,
                &mut plan.symp_gh,
                &mut plan.symp_mask,
                ports,
            );
            let fm = meta[s_idx];
            a_waves[s_idx] = fm.a_wave_gain * acc_br.reduce_sum();
            let mc = acc_m * c_sign_v;
            *sum_l_vec = mc.mul_add(F32x4::splat(fm.pan_l), *sum_l_vec);
            *sum_r_vec = mc.mul_add(F32x4::splat(fm.pan_r), *sum_r_vec);
        }
        idx += 4;
    }
    while idx < count {
        let s_idx = start_s + idx;
        if !plan.str_coupled[s_idx] {
            if !ports.may_be_nonzero[s_idx] {
                if b_waves[s_idx] == 0.0 {
                    a_waves[s_idx] = 0.0;
                    idx += 1;
                    continue;
                }
                ports.may_be_nonzero[s_idx] = true;
                ports.ports_synced[s_idx] = false;
            }
            let off = idx * stride;
            let neg_vbr = F32x4::splat(-0.5 * (a_waves[s_idx] + b_waves[s_idx]));
            let (acc_m, acc_br) = step_single_prefix_damp::<K>(
                s_idx,
                ts,
                &mut state[off..off + stride],
                &coef[off..off + stride],
                &damp_bd[off..off + stride],
                &damp_ad[off..off + stride],
                neg_vbr,
                h_proj,
                plan.symp_y_lim[s_idx],
                &mut plan.str_coupled,
                &mut plan.coupled_strs,
                &mut plan.n_coupled_strs,
                &mut plan.symp_has_gains,
                &mut plan.symp_gv,
                &mut plan.symp_gh,
                &mut plan.symp_mask,
                ports,
            );
            let fm = meta[s_idx];
            a_waves[s_idx] = fm.a_wave_gain * acc_br.reduce_sum();
            let mc = acc_m * c_sign_v;
            *sum_l_vec = mc.mul_add(F32x4::splat(fm.pan_l), *sum_l_vec);
            *sum_r_vec = mc.mul_add(F32x4::splat(fm.pan_r), *sum_r_vec);
        }
        idx += 1;
    }
}

/// Contiguously steps all uncoupled strings' `[v_br | h_br]` L1d-packed prefix vectors.
#[inline(never)]
pub(crate) fn step_uncoupled_prefixes_fast(
    bp: &mut BridgePorts,
    ports: &mut ContactPorts,
    plan: &mut ChunkPlan,
    ts: usize,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
) -> (f32, f32) {
    let (mut sum_l_vec, mut sum_r_vec) = (F32x4::zero(), F32x4::zero());
    for r_idx in 0..bp.n_prefix_runs {
        let run = bp.prefix_runs[r_idx];
        let (start_s, count, p_off) = (
            usize::from(run.start_s),
            usize::from(run.count),
            usize::from(run.p_off),
        );
        let run_has_symp = plan.symp_has_gains[start_s..start_s + count]
            .iter()
            .any(|&b| b);
        macro_rules! dispatch_open {
            ($k:literal) => {
                step_prefix_run_k_open::<$k>(
                    bp,
                    ports,
                    p_off,
                    start_s,
                    count,
                    plan,
                    a_waves,
                    b_waves,
                    &mut sum_l_vec,
                    &mut sum_r_vec,
                )
            };
        }
        macro_rules! dispatch_damp {
            ($k:literal) => {
                step_prefix_run_k::<$k>(
                    bp,
                    ports,
                    p_off,
                    start_s,
                    count,
                    ts,
                    plan,
                    a_waves,
                    b_waves,
                    &mut sum_l_vec,
                    &mut sum_r_vec,
                )
            };
        }
        match (run.n_pol, run_has_symp) {
            (0, _) => {
                for (a_wave, &coupled) in a_waves[start_s..start_s + count]
                    .iter_mut()
                    .zip(&plan.str_coupled[start_s..start_s + count])
                {
                    if !coupled {
                        *a_wave = 0.0;
                    }
                }
            }
            (1, false) => dispatch_open!(1),
            (2, false) => dispatch_open!(2),
            (3, false) => dispatch_open!(3),
            (4, false) => dispatch_open!(4),
            (5, false) => dispatch_open!(5),
            (_, false) => dispatch_open!(6),
            (1, true) => dispatch_damp!(1),
            (2, true) => dispatch_damp!(2),
            (3, true) => dispatch_damp!(3),
            (4, true) => dispatch_damp!(4),
            (5, true) => dispatch_damp!(5),
            (_, true) => dispatch_damp!(6),
        }
    }
    (sum_l_vec.reduce_sum(), sum_r_vec.reduce_sum())
}
