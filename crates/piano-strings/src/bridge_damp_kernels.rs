//! Contiguous 4-string linear-equilibrium closed-damper prefix stepping (`no_std`).

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_port::{BridgePorts, PrefixCoef, PrefixState};
#[allow(unused_imports)]
pub(crate) use crate::bridge_prefix_math::{
    pair_alt_hi, pair_alt_lo, pair_sum_2, reduce_sum_4, step_one_prefix, step_one_prefix_vd_only,
};
use crate::contact_ports::ContactPorts;

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
    y_lim: f32,
    str_coupled: &mut [bool; TOTAL_STRINGS],
    coupled_strs: &mut [u8; TOTAL_STRINGS],
    n_coupled_strs: &mut usize,
    symp_has_gains: &mut [bool; TOTAL_STRINGS],
    symp_gv: &mut [f32; TOTAL_STRINGS],
    symp_gh: &mut [f32; TOTAL_STRINGS],
    symp_mask: &mut [f32; TOTAL_STRINGS],
) {
    if y_dv.abs() > y_lim || y_dh.abs() > y_lim {
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
    symp_y_lim: &[f32; TOTAL_STRINGS],
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
            symp_y_lim[s_idx],
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
    y_lim: f32,
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
        y_lim,
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
    h_proj4: F32x4,
    s: usize,
    ts: usize,
    symp_y_lim: &[f32; TOTAL_STRINGS],
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
        h_proj4.splat_lane::<0>(),
    );
    let (m1, br1, vdv1, vdh1) = step_one_prefix_vd_only::<K>(
        st1,
        &cf4[stride..2 * stride],
        &bd4[stride..2 * stride],
        fdv4.splat_lane::<1>(),
        fdh4.splat_lane::<1>(),
        nvp.splat_lane::<1>(),
        h_proj4.splat_lane::<1>(),
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
        h_proj4.splat_lane::<2>(),
    );
    let (m3, br3, vdv3, vdh3) = step_one_prefix_vd_only::<K>(
        st3,
        &cf4[3 * stride..4 * stride],
        &bd4[3 * stride..4 * stride],
        fdv4.splat_lane::<3>(),
        fdh4.splat_lane::<3>(),
        nvp.splat_lane::<3>(),
        h_proj4.splat_lane::<3>(),
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
    if v_sq > 2.5e-5 || ((ts & 31) == 31 && v_sq > 1.0e-9) {
        verify_quad_damp_y(
            K,
            s,
            st4,
            ad4,
            symp_y_lim,
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
    y_lim: f32,
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
    if v_sq > 2.5e-5 || ((ts & 31) == 31 && v_sq > 1.0e-9) {
        verify_single_damp_y(
            K,
            s_idx,
            st,
            ad,
            y_lim,
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
