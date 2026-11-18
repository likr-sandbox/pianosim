//! Coupled-string WDF prefix stepping and injection kernels (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;

use crate::bridge_damp_kernels::reduce_sum_4;
use crate::bridge_port::{BridgePorts, PrefixState};
use crate::contact_ports::ContactPorts;
use crate::zoh_kernel::zoh_rotate;

#[inline(always)]
pub(crate) fn step_string_prefix_wdf_br(
    bp: &mut BridgePorts,
    beta_off: usize,
    nv: usize,
    nh: usize,
    c_sign_v: F32x4,
    c_sign_h: F32x4,
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
    let acc_c = acc_v.mul_add(c_sign_v, acc_h * c_sign_h);
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
    alpha_w: f32,
    f_d: f32,
    has_damper: bool,
) -> (F32x4, F32x4) {
    if alpha_w > 0.0 && f_h != 0.0 {
        step_pol_prefix_wdf_inj_inner::<true>(
            ports, bp, vr, beta_off, c_sign, neg_vbr, f_h, alpha_w, f_d, has_damper,
        )
    } else {
        step_pol_prefix_wdf_inj_inner::<false>(
            ports, bp, vr, beta_off, c_sign, neg_vbr, f_h, 0.0, f_d, has_damper,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_pol_prefix_wdf_inj_inner<const HAS_ALPHA: bool>(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    c_sign: F32x4,
    neg_vbr: F32x4,
    f_h: f32,
    alpha_w: f32,
    f_d: f32,
    has_damper: bool,
) -> (F32x4, F32x4) {
    let n = vr.end - vr.start;
    let st = &mut bp.state[beta_off..beta_off + n];
    let cf = &bp.coef[beta_off..beta_off + n];
    let bh = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let bh_diff = &ports.b_strike_diff.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr];
    let (fh_v, alpha_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(alpha_w), F32x4::splat(f_d));
    let (mut acc_v, mut acc_br) = (F32x4::zero(), F32x4::zero());
    for i in 0..n {
        let (c0, s0) = (cf[i], st[i]);
        let bh_eff = if HAS_ALPHA {
            alpha_v.mul_add(bh_diff[i], bh[i])
        } else {
            bh[i]
        };
        let u0 = if has_damper {
            fd_v.mul_add(bd[i], fh_v.mul_add(bh_eff, s0.u))
        } else {
            fh_v.mul_add(bh_eff, s0.u)
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
pub(crate) fn step_string_prefix_wdf_symp_damp_vh(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    v_br_r: Range<usize>,
    h_br_r: Range<usize>,
    beta_v_off: usize,
    beta_h_off: usize,
    c_sign_v: F32x4,
    c_sign_h: F32x4,
    neg_vbr: F32x4,
    fd_v: f32,
    fd_h: f32,
) -> (F32x4, F32x4, [f32; 4]) {
    if fd_v != 0.0 || fd_h != 0.0 {
        step_string_prefix_wdf_symp_damp_vh_inner::<true>(
            ports, bp, v_br_r, h_br_r, beta_v_off, beta_h_off, c_sign_v, c_sign_h, neg_vbr, fd_v,
            fd_h,
        )
    } else {
        step_string_prefix_wdf_symp_damp_vh_inner::<false>(
            ports, bp, v_br_r, h_br_r, beta_v_off, beta_h_off, c_sign_v, c_sign_h, neg_vbr, 0.0,
            0.0,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_string_prefix_wdf_symp_damp_vh_inner<const HAS_FD: bool>(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    v_br_r: Range<usize>,
    h_br_r: Range<usize>,
    beta_v_off: usize,
    beta_h_off: usize,
    c_sign_v: F32x4,
    c_sign_h: F32x4,
    neg_vbr: F32x4,
    fd_v: f32,
    fd_h: f32,
) -> (F32x4, F32x4, [f32; 4]) {
    let bd_all = ports.b_damp.as_f32x4_slice();
    let ad_all = ports.a_damp.as_f32x4_slice();
    let nv = v_br_r.end - v_br_r.start;
    let nh = h_br_r.end - h_br_r.start;
    let mut acc_br = F32x4::zero();

    let (mut acc_v, mut acc_yd_v, mut acc_vd_v) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    {
        let st = &mut bp.state[beta_v_off..beta_v_off + nv];
        let cf = &bp.coef[beta_v_off..beta_v_off + nv];
        let bd = &bd_all[v_br_r.clone()];
        let ad = &ad_all[v_br_r];
        let fdv = F32x4::splat(fd_v);
        for i in 0..nv {
            let (c0, s0, bdi) = (cf[i], st[i], bd[i]);
            let u0 = if HAS_FD { fdv.mul_add(bdi, s0.u) } else { s0.u };
            let (un, vn) = zoh_rotate(u0, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
            st[i] = PrefixState { u: un, v: vn };
            acc_v = acc_v + vn;
            acc_br = vn.mul_add(c0.beta, acc_br);
            acc_yd_v = vn.mul_add(ad[i], acc_yd_v);
            acc_vd_v = un.mul_add(bdi, acc_vd_v);
        }
    }

    let (mut acc_h, mut acc_yd_h, mut acc_vd_h) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    {
        let st = &mut bp.state[beta_h_off..beta_h_off + nh];
        let cf = &bp.coef[beta_h_off..beta_h_off + nh];
        let bd = &bd_all[h_br_r.clone()];
        let ad = &ad_all[h_br_r];
        let fdh = F32x4::splat(fd_h);
        for i in 0..nh {
            let (c0, s0, bdi) = (cf[i], st[i], bd[i]);
            let u0 = if HAS_FD { fdh.mul_add(bdi, s0.u) } else { s0.u };
            let (un, vn) = zoh_rotate(u0, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
            st[i] = PrefixState { u: un, v: vn };
            acc_h = acc_h + vn;
            acc_br = vn.mul_add(c0.beta, acc_br);
            acc_yd_h = vn.mul_add(ad[i], acc_yd_h);
            acc_vd_h = un.mul_add(bdi, acc_vd_h);
        }
    }

    let acc_c = acc_v.mul_add(c_sign_v, acc_h * c_sign_h);
    let port_sums = reduce_sum_4(acc_yd_v, acc_vd_v, acc_yd_h, acc_vd_h).to_array();
    (acc_c, acc_br, port_sums)
}
