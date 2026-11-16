//! Contiguous 4-string linear-equilibrium closed-damper prefix stepping (`no_std`).

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_port::{BridgePorts, PrefixCoef, PrefixState};
use crate::contact_ports::ContactPorts;
use crate::zoh_kernel::zoh_rotate;

#[inline(always)]
pub(crate) fn reduce_sum_4(r0: F32x4, r1: F32x4, r2: F32x4, r3: F32x4) -> F32x4 {
    let ac = r0.unpack_lo(r2) + r0.unpack_hi(r2);
    let bd = r1.unpack_lo(r3) + r1.unpack_hi(r3);
    ac.unpack_lo(bd) + ac.unpack_hi(bd)
}

#[inline(always)]
pub(crate) fn pair_sum_2(r0: F32x4, r1: F32x4) -> F32x4 {
    let p = r0.unpack_lo(r1) + r0.unpack_hi(r1);
    p + p.slide2(p)
}

#[inline(always)]
pub(crate) fn pair_alt_hi(r0: F32x4, r1: F32x4) -> F32x4 {
    let p = r0.unpack_lo(r1) + r0.unpack_hi(r1);
    p.slide2(p) - p
}

#[inline(always)]
pub(crate) fn pair_alt_lo(r2: F32x4, r3: F32x4) -> F32x4 {
    let p = r2.unpack_lo(r3) + r2.unpack_hi(r3);
    p - p.slide2(p)
}

impl BridgePorts {
    /// Copies `ports.b_damp` / `ports.a_damp` prefix vectors into L1d-packed prefix buffers.
    pub(crate) fn sync_damp_ports(&mut self, ports: &ContactPorts) {
        let bd_all = ports.b_damp.as_f32x4_slice();
        let ad_all = ports.a_damp.as_f32x4_slice();
        for s_idx in 0..TOTAL_STRINGS {
            let m = &self.meta[s_idx];
            let (nv, nh) = (m.v_br.len(), m.h_br.len());
            self.damp_bd[m.beta_v_off..m.beta_v_off + nv].copy_from_slice(&bd_all[m.v_br.clone()]);
            self.damp_ad[m.beta_v_off..m.beta_v_off + nv].copy_from_slice(&ad_all[m.v_br.clone()]);
            self.damp_bd[m.beta_h_off..m.beta_h_off + nh].copy_from_slice(&bd_all[m.h_br.clone()]);
            self.damp_ad[m.beta_h_off..m.beta_h_off + nh].copy_from_slice(&ad_all[m.h_br.clone()]);
        }
    }
}

#[inline(always)]
pub(crate) fn step_one_prefix<const K: usize>(
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
fn step_one_prefix_vd_only<const K: usize>(
    st: &mut [PrefixState],
    cf: &[PrefixCoef],
    bd: &[F32x4],
    fdv: F32x4,
    fdh: F32x4,
    neg_vbr: F32x4,
    h_proj: F32x4,
) -> (F32x4, F32x4, F32x4, F32x4) {
    let (c0, s0, bd0) = (cf[0], st[0], bd[0]);
    let (un0, mut acc_v) = zoh_rotate(
        fdv.mul_add(bd0, s0.u),
        c0.beta.mul_add(neg_vbr, s0.v),
        c0.d,
        c0.s,
    );
    st[0] = PrefixState { u: un0, v: acc_v };
    let mut acc_br = acc_v * c0.beta;
    let mut acc_vd_v = un0 * bd0;
    if K >= 2 {
        let (c1, s1, bd1) = (cf[1], st[1], bd[1]);
        let (un1, vn1) = zoh_rotate(
            fdv.mul_add(bd1, s1.u),
            c1.beta.mul_add(neg_vbr, s1.v),
            c1.d,
            c1.s,
        );
        st[1] = PrefixState { u: un1, v: vn1 };
        acc_v = acc_v + vn1;
        acc_br = vn1.mul_add(c1.beta, acc_br);
        acc_vd_v = un1.mul_add(bd1, acc_vd_v);
    }
    if K >= 3 {
        let (c2, s2, bd2) = (cf[2], st[2], bd[2]);
        let (un2, vn2) = zoh_rotate(
            fdv.mul_add(bd2, s2.u),
            c2.beta.mul_add(neg_vbr, s2.v),
            c2.d,
            c2.s,
        );
        st[2] = PrefixState { u: un2, v: vn2 };
        acc_v = acc_v + vn2;
        acc_br = vn2.mul_add(c2.beta, acc_br);
        acc_vd_v = un2.mul_add(bd2, acc_vd_v);
    }

    let (ch0, sh0, bdh0) = (cf[K], st[K], bd[K]);
    let (unh0, mut acc_h) = zoh_rotate(
        fdh.mul_add(bdh0, sh0.u),
        ch0.beta.mul_add(neg_vbr, sh0.v),
        ch0.d,
        ch0.s,
    );
    st[K] = PrefixState { u: unh0, v: acc_h };
    acc_br = acc_h.mul_add(ch0.beta, acc_br);
    let mut acc_vd_h = unh0 * bdh0;
    if K >= 2 {
        let (ch1, sh1, bdh1) = (cf[K + 1], st[K + 1], bd[K + 1]);
        let (unh1, vnh1) = zoh_rotate(
            fdh.mul_add(bdh1, sh1.u),
            ch1.beta.mul_add(neg_vbr, sh1.v),
            ch1.d,
            ch1.s,
        );
        st[K + 1] = PrefixState { u: unh1, v: vnh1 };
        acc_h = acc_h + vnh1;
        acc_br = vnh1.mul_add(ch1.beta, acc_br);
        acc_vd_h = unh1.mul_add(bdh1, acc_vd_h);
    }
    if K >= 3 {
        let (ch2, sh2, bdh2) = (cf[K + 2], st[K + 2], bd[K + 2]);
        let (unh2, vnh2) = zoh_rotate(
            fdh.mul_add(bdh2, sh2.u),
            ch2.beta.mul_add(neg_vbr, sh2.v),
            ch2.d,
            ch2.s,
        );
        st[K + 2] = PrefixState { u: unh2, v: vnh2 };
        acc_h = acc_h + vnh2;
        acc_br = vnh2.mul_add(ch2.beta, acc_br);
        acc_vd_h = unh2.mul_add(bdh2, acc_vd_h);
    }
    (acc_h.mul_add(h_proj, acc_v), acc_br, acc_vd_v, acc_vd_h)
}

#[inline(always)]
fn eval_pol_y(st: &[PrefixState], ad: &[F32x4], n: usize) -> f32 {
    let mut acc = F32x4::zero();
    for i in 0..n {
        acc = st[i].v.mul_add(ad[i], acc);
    }
    acc.reduce_sum()
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn clear_if_excursion(
    s_idx: usize,
    y_dv: f32,
    y_dh: f32,
    str_coupled: &mut [bool; TOTAL_STRINGS],
    coupled_strs: &mut [u8; TOTAL_STRINGS],
    n_coupled_strs: &mut usize,
    symp_has_gains: &mut [bool; TOTAL_STRINGS],
    symp_gv: &mut [f32; TOTAL_STRINGS],
    symp_gh: &mut [f32; TOTAL_STRINGS],
    symp_mask: &mut [f32; TOTAL_STRINGS],
) {
    if y_dv.abs() > 8.0e-6 || y_dh.abs() > 8.0e-6 {
        symp_has_gains[s_idx] = false;
        symp_gv[s_idx] = 0.0;
        symp_gh[s_idx] = 0.0;
        symp_mask[s_idx] = 0.0;
        if !str_coupled[s_idx] {
            str_coupled[s_idx] = true;
            coupled_strs[*n_coupled_strs] = s_idx as u8;
            *n_coupled_strs += 1;
        }
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn verify_quad_damp_y(
    k_vecs: usize,
    s: usize,
    st4: &[PrefixState],
    ad4: &[F32x4],
    str_coupled: &mut [bool; TOTAL_STRINGS],
    coupled_strs: &mut [u8; TOTAL_STRINGS],
    n_coupled_strs: &mut usize,
    symp_has_gains: &mut [bool; TOTAL_STRINGS],
    symp_gv: &mut [f32; TOTAL_STRINGS],
    symp_gh: &mut [f32; TOTAL_STRINGS],
    symp_mask: &mut [f32; TOTAL_STRINGS],
    ports: &mut ContactPorts,
) {
    let stride = 2 * k_vecs;
    for j in 0..4 {
        let s_idx = s + j;
        if !symp_has_gains[s_idx] {
            continue;
        }
        let off = j * stride;
        let y_dv = eval_pol_y(&st4[off..off + k_vecs], &ad4[off..off + k_vecs], k_vecs);
        let y_dh = eval_pol_y(
            &st4[off + k_vecs..off + stride],
            &ad4[off + k_vecs..off + stride],
            k_vecs,
        );
        ports.y_dv[s_idx] = y_dv;
        ports.y_dh[s_idx] = y_dh;
        clear_if_excursion(
            s_idx,
            y_dv,
            y_dh,
            str_coupled,
            coupled_strs,
            n_coupled_strs,
            symp_has_gains,
            symp_gv,
            symp_gh,
            symp_mask,
        );
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn verify_single_damp_y(
    k_vecs: usize,
    s_idx: usize,
    st: &[PrefixState],
    ad: &[F32x4],
    str_coupled: &mut [bool; TOTAL_STRINGS],
    coupled_strs: &mut [u8; TOTAL_STRINGS],
    n_coupled_strs: &mut usize,
    symp_has_gains: &mut [bool; TOTAL_STRINGS],
    symp_gv: &mut [f32; TOTAL_STRINGS],
    symp_gh: &mut [f32; TOTAL_STRINGS],
    symp_mask: &mut [f32; TOTAL_STRINGS],
    ports: &mut ContactPorts,
) {
    let y_dv = eval_pol_y(&st[0..k_vecs], &ad[0..k_vecs], k_vecs);
    let y_dh = eval_pol_y(&st[k_vecs..2 * k_vecs], &ad[k_vecs..2 * k_vecs], k_vecs);
    ports.y_dv[s_idx] = y_dv;
    ports.y_dh[s_idx] = y_dh;
    clear_if_excursion(
        s_idx,
        y_dv,
        y_dh,
        str_coupled,
        coupled_strs,
        n_coupled_strs,
        symp_has_gains,
        symp_gv,
        symp_gh,
        symp_mask,
    );
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_prefix_quad_damp<const K: usize>(
    st4: &mut [PrefixState],
    cf4: &[PrefixCoef],
    bd4: &[F32x4],
    ad4: &[F32x4],
    nvp: F32x4,
    h_proj: F32x4,
    s: usize,
    ts: usize,
    str_coupled: &mut [bool; TOTAL_STRINGS],
    coupled_strs: &mut [u8; TOTAL_STRINGS],
    n_coupled_strs: &mut usize,
    symp_has_gains: &mut [bool; TOTAL_STRINGS],
    symp_gv: &mut [f32; TOTAL_STRINGS],
    symp_gh: &mut [f32; TOTAL_STRINGS],
    symp_mask: &mut [f32; TOTAL_STRINGS],
    ports: &mut ContactPorts,
) -> (F32x4, F32x4) {
    let stride = 2 * K;
    let fdv4 = F32x4::from_slice(&ports.v_dv[s..s + 4]) * F32x4::from_slice(&symp_gv[s..s + 4]);
    let fdh4 = F32x4::from_slice(&ports.v_dh[s..s + 4]) * F32x4::from_slice(&symp_gh[s..s + 4]);
    let (st01, st23) = st4.split_at_mut(2 * stride);
    let (st0, st1) = st01.split_at_mut(stride);
    let (st2, st3) = st23.split_at_mut(stride);
    let (m0, br0, vdv0, vdh0) = step_one_prefix_vd_only::<K>(
        st0,
        &cf4[0..stride],
        &bd4[0..stride],
        fdv4.splat_lane::<0>(),
        fdh4.splat_lane::<0>(),
        nvp.splat_lane::<0>(),
        h_proj,
    );
    let (m1, br1, vdv1, vdh1) = step_one_prefix_vd_only::<K>(
        st1,
        &cf4[stride..2 * stride],
        &bd4[stride..2 * stride],
        fdv4.splat_lane::<1>(),
        fdh4.splat_lane::<1>(),
        nvp.splat_lane::<1>(),
        h_proj,
    );
    let br01 = pair_sum_2(br0, br1);
    let vdv01 = pair_sum_2(vdv0, vdv1);
    let vdh01 = pair_sum_2(vdh0, vdh1);
    let m01 = pair_alt_hi(m0, m1);
    let (m2, br2, vdv2, vdh2) = step_one_prefix_vd_only::<K>(
        st2,
        &cf4[2 * stride..3 * stride],
        &bd4[2 * stride..3 * stride],
        fdv4.splat_lane::<2>(),
        fdh4.splat_lane::<2>(),
        nvp.splat_lane::<2>(),
        h_proj,
    );
    let (m3, br3, vdv3, vdh3) = step_one_prefix_vd_only::<K>(
        st3,
        &cf4[3 * stride..4 * stride],
        &bd4[3 * stride..4 * stride],
        fdv4.splat_lane::<3>(),
        fdh4.splat_lane::<3>(),
        nvp.splat_lane::<3>(),
        h_proj,
    );
    let alpha4 = F32x4::from_slice(&ports.alpha_vel[s..s + 4]);
    let vdv4 = vdv01.slide2(pair_sum_2(vdv2, vdv3)) * alpha4;
    let vdh4 = vdh01.slide2(pair_sum_2(vdh2, vdh3)) * alpha4;
    vdv4.write_to_slice(&mut ports.v_dv[s..s + 4]);
    vdh4.write_to_slice(&mut ports.v_dh[s..s + 4]);
    let mask4 = F32x4::from_slice(&symp_mask[s..s + 4]);
    let mv = vdv4 * mask4;
    let mh = vdh4 * mask4;
    let v_sq = (mv * mv + mh * mh).reduce_sum();
    if v_sq > 2.5e-5 || ((ts & 31) == 31 && v_sq > 4.0e-6) {
        verify_quad_damp_y(
            K,
            s,
            st4,
            ad4,
            str_coupled,
            coupled_strs,
            n_coupled_strs,
            symp_has_gains,
            symp_gv,
            symp_gh,
            symp_mask,
            ports,
        );
    }
    (
        br01.slide2(pair_sum_2(br2, br3)),
        m01.slide2(pair_alt_lo(m2, m3)),
    )
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_single_prefix_damp<const K: usize>(
    s_idx: usize,
    ts: usize,
    st: &mut [PrefixState],
    cf: &[PrefixCoef],
    bd: &[F32x4],
    ad: &[F32x4],
    neg_vbr: F32x4,
    h_proj: F32x4,
    str_coupled: &mut [bool; TOTAL_STRINGS],
    coupled_strs: &mut [u8; TOTAL_STRINGS],
    n_coupled_strs: &mut usize,
    symp_has_gains: &mut [bool; TOTAL_STRINGS],
    symp_gv: &mut [f32; TOTAL_STRINGS],
    symp_gh: &mut [f32; TOTAL_STRINGS],
    symp_mask: &mut [f32; TOTAL_STRINGS],
    ports: &mut ContactPorts,
) -> (F32x4, F32x4) {
    if !symp_has_gains[s_idx] {
        return step_one_prefix::<K>(st, cf, neg_vbr, h_proj);
    }
    let fdv = F32x4::splat(symp_gv[s_idx] * ports.v_dv[s_idx]);
    let fdh = F32x4::splat(symp_gh[s_idx] * ports.v_dh[s_idx]);
    let (acc_m, acc_br, vdv, vdh) =
        step_one_prefix_vd_only::<K>(st, cf, bd, fdv, fdh, neg_vbr, h_proj);
    let alpha = ports.alpha_vel[s_idx];
    let v_dv = alpha * vdv.reduce_sum();
    let v_dh = alpha * vdh.reduce_sum();
    ports.v_dv[s_idx] = v_dv;
    ports.v_dh[s_idx] = v_dh;
    let v_sq = v_dv * v_dv + v_dh * v_dh;
    if v_sq > 2.5e-5 || ((ts & 31) == 31 && v_sq > 4.0e-6) {
        verify_single_damp_y(
            K,
            s_idx,
            st,
            ad,
            str_coupled,
            coupled_strs,
            n_coupled_strs,
            symp_has_gains,
            symp_gv,
            symp_gh,
            symp_mask,
            ports,
        );
    }
    (acc_m, acc_br)
}

/// Refreshes exact `y_dv` / `y_dh` once per subblock for linear-equilibrium closed-damper strings.
pub(crate) fn refresh_chunk_damp_y(
    bp: &BridgePorts,
    ports: &mut ContactPorts,
    symp_has_gains: &[bool; TOTAL_STRINGS],
) {
    for (s_idx, &has) in symp_has_gains.iter().enumerate() {
        if !has {
            continue;
        }
        let m = &bp.meta[s_idx];
        let (nv, nh) = (m.v_br.len(), m.h_br.len());
        ports.y_dv[s_idx] = eval_pol_y(
            &bp.state[m.beta_v_off..m.beta_v_off + nv],
            &bp.damp_ad[m.beta_v_off..m.beta_v_off + nv],
            nv,
        );
        ports.y_dh[s_idx] = eval_pol_y(
            &bp.state[m.beta_h_off..m.beta_h_off + nh],
            &bp.damp_ad[m.beta_h_off..m.beta_h_off + nh],
            nh,
        );
    }
}
