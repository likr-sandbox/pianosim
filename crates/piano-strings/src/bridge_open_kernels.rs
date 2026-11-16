//! Pure open-damper L1d-packed WDF prefix ZOH stepping without damper register pressure (`no_std`).

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_damp_kernels::{pair_alt_hi, pair_alt_lo, pair_sum_2, step_one_prefix};
use crate::bridge_port::{BridgePorts, PrefixCoef, PrefixState, C_SIGN_V_ARR};
use crate::bridge_tails::ChunkPlan;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::{C_HORIZ, C_VERT};

#[inline(always)]
pub(crate) fn step_open_quad<const K: usize>(
    st4: &mut [PrefixState],
    cf4: &[PrefixCoef],
    nvp: F32x4,
    h_proj: F32x4,
) -> (F32x4, F32x4) {
    let stride = 2 * K;
    let (st01, st23) = st4.split_at_mut(2 * stride);
    let (st0, st1) = st01.split_at_mut(stride);
    let (st2, st3) = st23.split_at_mut(stride);
    let (m0, br0) = step_one_prefix::<K>(st0, &cf4[0..stride], nvp.splat_lane::<0>(), h_proj);
    let (m1, br1) =
        step_one_prefix::<K>(st1, &cf4[stride..2 * stride], nvp.splat_lane::<1>(), h_proj);
    let br01 = pair_sum_2(br0, br1);
    let m01 = pair_alt_hi(m0, m1);
    let (m2, br2) = step_one_prefix::<K>(
        st2,
        &cf4[2 * stride..3 * stride],
        nvp.splat_lane::<2>(),
        h_proj,
    );
    let (m3, br3) = step_one_prefix::<K>(
        st3,
        &cf4[3 * stride..4 * stride],
        nvp.splat_lane::<3>(),
        h_proj,
    );
    (
        br01.slide2(pair_sum_2(br2, br3)),
        m01.slide2(pair_alt_lo(m2, m3)),
    )
}

#[inline]
pub(crate) fn step_open_quad_write<const K: usize>(
    st4: &mut [PrefixState],
    cf4: &[PrefixCoef],
    nvp: F32x4,
    h_proj: F32x4,
    gains4: F32x4,
    aw_out: &mut [f32],
) -> F32x4 {
    let (br4, ms4) = step_open_quad::<K>(st4, cf4, nvp, h_proj);
    (br4 * gains4).write_to_slice(aw_out);
    ms4
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_prefix_run_k_open<const K: usize>(
    bp: &mut BridgePorts,
    ports: &mut ContactPorts,
    p_off: usize,
    start_s: usize,
    count: usize,
    plan: &ChunkPlan,
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
            let (br4, ms4) = step_open_quad::<K>(st4, cf4, nvp, h_proj);
            let aw4 = br4 * F32x4::from_slice(&gains[s..s + 4]);
            aw4.write_to_slice(&mut a_waves[s..s + 4]);
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
            let (acc_m, acc_br) = step_one_prefix::<K>(
                &mut state[off..off + stride],
                &coef[off..off + stride],
                neg_vbr,
                h_proj,
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
            let (acc_m, acc_br) = step_one_prefix::<K>(
                &mut state[off..off + stride],
                &coef[off..off + stride],
                neg_vbr,
                h_proj,
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

impl BridgePorts {
    #[inline]
    pub fn reset_resting(&mut self) {
        let z = F32x4::zero();
        self.state.fill(PrefixState { u: z, v: z });
        self.tl_soa.reset_resting();
    }

    pub fn compute_a_waves(
        &self,
        may_be_nonzero: &[bool; TOTAL_STRINGS],
        a_waves: &mut [f32; TOTAL_STRINGS],
    ) {
        for (s_idx, m) in self.meta.iter().enumerate() {
            if !may_be_nonzero[s_idx] {
                a_waves[s_idx] = 0.0;
                continue;
            }
            let mut acc = F32x4::zero();
            for i in m.beta_v_off..(m.beta_v_off + m.n_vh) {
                acc = self.state[i].v.mul_add(self.coef[i].beta, acc);
            }
            a_waves[s_idx] = m.a_wave_gain * acc.reduce_sum();
        }
    }
}
