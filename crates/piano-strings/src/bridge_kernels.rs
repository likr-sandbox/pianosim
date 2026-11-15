//! L1d-packed WDF prefix ZOH kernels with 4-string SIMD transpose reduction (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_port::{
    BridgePorts, FastPrefixMeta, PrefixCoef, PrefixState, C_SIGN_H_ARR, C_SIGN_V_ARR,
};
use crate::contact_ports::ContactPorts;
use crate::modal_soa::{C_HORIZ, C_VERT};
use crate::zoh_kernel::zoh_rotate;

#[inline(always)]
fn reduce_sum_4(r0: F32x4, r1: F32x4, r2: F32x4, r3: F32x4) -> F32x4 {
    let ac = r0.unpack_lo(r2) + r0.unpack_hi(r2);
    let bd = r1.unpack_lo(r3) + r1.unpack_hi(r3);
    ac.unpack_lo(bd) + ac.unpack_hi(bd)
}

#[inline(always)]
fn step_one_prefix<const K: usize>(
    st: &mut [PrefixState],
    cf: &[PrefixCoef],
    neg_vbr: F32x4,
    h_proj: F32x4,
) -> (F32x4, F32x4) {
    let (c0, s0) = (cf[0], st[0]);
    let (un0, mut acc_v) = zoh_rotate(s0.u, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
    st[0] = PrefixState { u: un0, v: acc_v };
    let mut acc_br = acc_v * c0.beta;
    if K >= 2 {
        let (c1, s1) = (cf[1], st[1]);
        let (un1, vn1) = zoh_rotate(s1.u, c1.beta.mul_add(neg_vbr, s1.v), c1.d, c1.s);
        st[1] = PrefixState { u: un1, v: vn1 };
        acc_v = acc_v + vn1;
        acc_br = vn1.mul_add(c1.beta, acc_br);
    }
    if K >= 3 {
        let (c2, s2) = (cf[2], st[2]);
        let (un2, vn2) = zoh_rotate(s2.u, c2.beta.mul_add(neg_vbr, s2.v), c2.d, c2.s);
        st[2] = PrefixState { u: un2, v: vn2 };
        acc_v = acc_v + vn2;
        acc_br = vn2.mul_add(c2.beta, acc_br);
    }
    let (ch0, sh0) = (cf[K], st[K]);
    let (unh0, mut acc_h) = zoh_rotate(sh0.u, ch0.beta.mul_add(neg_vbr, sh0.v), ch0.d, ch0.s);
    st[K] = PrefixState { u: unh0, v: acc_h };
    acc_br = acc_h.mul_add(ch0.beta, acc_br);
    if K >= 2 {
        let (ch1, sh1) = (cf[K + 1], st[K + 1]);
        let (unh1, vnh1) = zoh_rotate(sh1.u, ch1.beta.mul_add(neg_vbr, sh1.v), ch1.d, ch1.s);
        st[K + 1] = PrefixState { u: unh1, v: vnh1 };
        acc_h = acc_h + vnh1;
        acc_br = vnh1.mul_add(ch1.beta, acc_br);
    }
    if K >= 3 {
        let (ch2, sh2) = (cf[K + 2], st[K + 2]);
        let (unh2, vnh2) = zoh_rotate(sh2.u, ch2.beta.mul_add(neg_vbr, sh2.v), ch2.d, ch2.s);
        st[K + 2] = PrefixState { u: unh2, v: vnh2 };
        acc_h = acc_h + vnh2;
        acc_br = vnh2.mul_add(ch2.beta, acc_br);
    }
    (acc_h.mul_add(h_proj, acc_v), acc_br)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_prefix_run_k<const K: usize>(
    state: &mut [PrefixState],
    coef: &[PrefixCoef],
    start_s: usize,
    count: usize,
    meta: &[FastPrefixMeta; TOTAL_STRINGS],
    gains: &[f32; TOTAL_STRINGS],
    str_coupled: &[bool; TOTAL_STRINGS],
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    sum_l_vec: &mut F32x4,
    sum_r_vec: &mut F32x4,
) {
    let h_proj = F32x4::splat(C_HORIZ / C_VERT);
    let neg_half = F32x4::splat(-0.5);
    let stride = 2 * K;
    let mut idx = 0usize;
    while idx + 4 <= count {
        let s = start_s + idx;
        if !(str_coupled[s] | str_coupled[s + 1] | str_coupled[s + 2] | str_coupled[s + 3]) {
            let nvp = neg_half
                * (F32x4::from_slice(&a_waves[s..s + 4]) + F32x4::from_slice(&b_waves[s..s + 4]));
            let base = idx * stride;
            let (st4, cf4) = (
                &mut state[base..base + 4 * stride],
                &coef[base..base + 4 * stride],
            );
            let (st01, st23) = st4.split_at_mut(2 * stride);
            let (st0, st1) = st01.split_at_mut(stride);
            let (st2, st3) = st23.split_at_mut(stride);
            let (m0, br0) =
                step_one_prefix::<K>(st0, &cf4[0..stride], nvp.splat_lane::<0>(), h_proj);
            let (m1, br1) =
                step_one_prefix::<K>(st1, &cf4[stride..2 * stride], nvp.splat_lane::<1>(), h_proj);
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
            let aw4 = reduce_sum_4(br0, br1, br2, br3) * F32x4::from_slice(&gains[s..s + 4]);
            aw4.write_to_slice(&mut a_waves[s..s + 4]);
            let (fm0, fm1, fm2, fm3) = (meta[s], meta[s + 1], meta[s + 2], meta[s + 3]);
            *sum_l_vec = m3.mul_add(
                F32x4::splat(fm3.pan_l),
                m2.mul_add(
                    F32x4::splat(fm2.pan_l),
                    m1.mul_add(
                        F32x4::splat(fm1.pan_l),
                        m0.mul_add(F32x4::splat(fm0.pan_l), *sum_l_vec),
                    ),
                ),
            );
            *sum_r_vec = m3.mul_add(
                F32x4::splat(fm3.pan_r),
                m2.mul_add(
                    F32x4::splat(fm2.pan_r),
                    m1.mul_add(
                        F32x4::splat(fm1.pan_r),
                        m0.mul_add(F32x4::splat(fm0.pan_r), *sum_r_vec),
                    ),
                ),
            );
            idx += 4;
            continue;
        }
        for sub_i in 0..4 {
            let s_idx = s + sub_i;
            if str_coupled[s_idx] {
                continue;
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
            *sum_l_vec = acc_m.mul_add(F32x4::splat(fm.pan_l), *sum_l_vec);
            *sum_r_vec = acc_m.mul_add(F32x4::splat(fm.pan_r), *sum_r_vec);
        }
        idx += 4;
    }
    while idx < count {
        let s_idx = start_s + idx;
        if !str_coupled[s_idx] {
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
            *sum_l_vec = acc_m.mul_add(F32x4::splat(fm.pan_l), *sum_l_vec);
            *sum_r_vec = acc_m.mul_add(F32x4::splat(fm.pan_r), *sum_r_vec);
        }
        idx += 1;
    }
}

/// Contiguously steps all uncoupled strings' `[v_br | h_br]` L1d-packed prefix vectors.
#[inline(never)]
pub(crate) fn step_uncoupled_prefixes_fast(
    bp: &mut BridgePorts,
    str_coupled: &[bool; TOTAL_STRINGS],
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
        match run.n_pol {
            0 => {
                for s_idx in start_s..start_s + count {
                    if !str_coupled[s_idx] {
                        a_waves[s_idx] = 0.0;
                    }
                }
            }
            1 => step_prefix_run_k::<1>(
                &mut bp.state[p_off..],
                &bp.coef[p_off..],
                start_s,
                count,
                &bp.fast_meta,
                &bp.a_wave_gains,
                str_coupled,
                a_waves,
                b_waves,
                &mut sum_l_vec,
                &mut sum_r_vec,
            ),
            2 => step_prefix_run_k::<2>(
                &mut bp.state[p_off..],
                &bp.coef[p_off..],
                start_s,
                count,
                &bp.fast_meta,
                &bp.a_wave_gains,
                str_coupled,
                a_waves,
                b_waves,
                &mut sum_l_vec,
                &mut sum_r_vec,
            ),
            _ => step_prefix_run_k::<3>(
                &mut bp.state[p_off..],
                &bp.coef[p_off..],
                start_s,
                count,
                &bp.fast_meta,
                &bp.a_wave_gains,
                str_coupled,
                a_waves,
                b_waves,
                &mut sum_l_vec,
                &mut sum_r_vec,
            ),
        }
    }
    let c_sign_v = F32x4::from_array(C_SIGN_V_ARR);
    (
        (sum_l_vec * c_sign_v).reduce_sum(),
        (sum_r_vec * c_sign_v).reduce_sum(),
    )
}

#[inline(always)]
pub(crate) fn step_string_prefix_wdf_br(
    bp: &mut BridgePorts,
    beta_off: usize,
    nv: usize,
    nh: usize,
    neg_vbr: F32x4,
) -> (F32x4, F32x4) {
    let n = nv + nh;
    let st = &mut bp.state[beta_off..beta_off + n];
    let cf = &bp.coef[beta_off..beta_off + n];
    let (mut acc_v, mut acc_h, mut acc_br) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in 0..nv {
        let (c0, s0) = (cf[i], st[i]);
        let (un, vn) = zoh_rotate(s0.u, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
        st[i] = PrefixState { u: un, v: vn };
        acc_v = acc_v + vn;
        acc_br = vn.mul_add(c0.beta, acc_br);
    }
    for i in nv..n {
        let (c0, s0) = (cf[i], st[i]);
        let (un, vn) = zoh_rotate(s0.u, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
        st[i] = PrefixState { u: un, v: vn };
        acc_h = acc_h + vn;
        acc_br = vn.mul_add(c0.beta, acc_br);
    }
    let acc_c = acc_v.mul_add(
        F32x4::from_array(C_SIGN_V_ARR),
        acc_h * F32x4::from_array(C_SIGN_H_ARR),
    );
    (acc_c, acc_br)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_inj(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    c_sign: F32x4,
    neg_vbr: F32x4,
    f_h: f32,
    f_d: f32,
    has_damper: bool,
) -> (F32x4, F32x4) {
    let n = vr.end - vr.start;
    let st = &mut bp.state[beta_off..beta_off + n];
    let cf = &bp.coef[beta_off..beta_off + n];
    let bh = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr];
    let (fh_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(f_d));
    let (mut acc_v, mut acc_br) = (F32x4::zero(), F32x4::zero());
    for i in 0..n {
        let (c0, s0) = (cf[i], st[i]);
        let u0 = if has_damper {
            fd_v.mul_add(bd[i], fh_v.mul_add(bh[i], s0.u))
        } else {
            fh_v.mul_add(bh[i], s0.u)
        };
        let (un, vn) = zoh_rotate(u0, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
        st[i] = PrefixState { u: un, v: vn };
        acc_v = acc_v + vn;
        acc_br = vn.mul_add(c0.beta, acc_br);
    }
    (acc_v * c_sign, acc_br)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_symp_damp(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    c_sign: F32x4,
    neg_vbr: F32x4,
    f_d: f32,
) -> (F32x4, F32x4, f32, f32) {
    if f_d != 0.0 {
        step_pol_prefix_wdf_symp_damp_inner::<true>(ports, bp, vr, beta_off, c_sign, neg_vbr, f_d)
    } else {
        step_pol_prefix_wdf_symp_damp_inner::<false>(ports, bp, vr, beta_off, c_sign, neg_vbr, 0.0)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_pol_prefix_wdf_symp_damp_inner<const HAS_FD: bool>(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    c_sign: F32x4,
    neg_vbr: F32x4,
    f_d: f32,
) -> (F32x4, F32x4, f32, f32) {
    let n = vr.end - vr.start;
    let st = &mut bp.state[beta_off..beta_off + n];
    let cf = &bp.coef[beta_off..beta_off + n];
    let bd = &ports.b_damp.as_f32x4_slice()[vr.clone()];
    let ad = &ports.a_damp.as_f32x4_slice()[vr];
    let fd_v = F32x4::splat(f_d);
    let (mut acc_v, mut acc_br, mut acc_yd, mut acc_vd) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in 0..n {
        let (c0, s0, bdi) = (cf[i], st[i], bd[i]);
        let u0 = if HAS_FD {
            fd_v.mul_add(bdi, s0.u)
        } else {
            s0.u
        };
        let (un, vn) = zoh_rotate(u0, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
        st[i] = PrefixState { u: un, v: vn };
        acc_v = acc_v + vn;
        acc_br = vn.mul_add(c0.beta, acc_br);
        acc_yd = vn.mul_add(ad[i], acc_yd);
        acc_vd = un.mul_add(bdi, acc_vd);
    }
    (
        acc_v * c_sign,
        acc_br,
        acc_yd.reduce_sum(),
        acc_vd.reduce_sum(),
    )
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
