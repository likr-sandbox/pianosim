//! 2x-unrolled packed WDF prefix ZOH kernels and 2-string paired SAV batching (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;

use crate::bridge_port::BridgePorts;
use crate::choir_map::StringMapEntry;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::ModalSoa;
use crate::phantom::MAX_PHANTOM_LONG_VECS;
use crate::sav_drift::{load_nl_state_n, load_vh_state, store_vh_u, SavBank};
use crate::sav_pair::{eval_no_phantom_pair, eval_phantom_pair};
use crate::zoh_kernel::zoh_rotate;

/// Advances WDF prefix `(u, v)` using packed `(d_br, s_br, c_br, beta_br)` tables with 2x
/// accumulator unrolling and defers horizontal vector reductions to the caller.
#[inline(always)]
pub(crate) fn step_pol_prefix_wdf(
    soa: &mut ModalSoa,
    bp: &BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
) -> (F32x4, F32x4) {
    let n = vr.end - vr.start;
    let br = beta_off..beta_off + n;
    let (u_all, v_all) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let (u, v) = (&mut u_all[vr.clone()], &mut v_all[vr]);
    let d = &bp.d_br.as_f32x4_slice()[br.clone()];
    let s = &bp.s_br.as_f32x4_slice()[br.clone()];
    let c = &bp.c_br.as_f32x4_slice()[br.clone()];
    let beta = &bp.beta_br.as_f32x4_slice()[br];

    let (mut acc_c0, mut acc_br0) = (F32x4::zero(), F32x4::zero());
    let (mut acc_c1, mut acc_br1) = (F32x4::zero(), F32x4::zero());
    let mut i = 0usize;
    while i + 1 < n {
        let (b0, b1) = (beta[i], beta[i + 1]);
        let (un0, vn0) = zoh_rotate(u[i], b0.mul_add(neg_vbr, v[i]), d[i], s[i]);
        let (un1, vn1) = zoh_rotate(u[i + 1], b1.mul_add(neg_vbr, v[i + 1]), d[i + 1], s[i + 1]);
        (u[i], v[i]) = (un0, vn0);
        (u[i + 1], v[i + 1]) = (un1, vn1);
        acc_c0 = vn0.mul_add(c[i], acc_c0);
        acc_br0 = vn0.mul_add(b0, acc_br0);
        acc_c1 = vn1.mul_add(c[i + 1], acc_c1);
        acc_br1 = vn1.mul_add(b1, acc_br1);
        i += 2;
    }
    if i < n {
        let bi = beta[i];
        let (un, vn) = zoh_rotate(u[i], bi.mul_add(neg_vbr, v[i]), d[i], s[i]);
        (u[i], v[i]) = (un, vn);
        acc_c0 = vn.mul_add(c[i], acc_c0);
        acc_br0 = vn.mul_add(bi, acc_br0);
    }
    (acc_c0 + acc_c1, acc_br0 + acc_br1)
}

/// Same as [`step_pol_prefix_wdf`] with simultaneous hammer/damper force injection into `u`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_inj(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    bp: &BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_h: f32,
    f_d: f32,
    has_damper: bool,
) -> (F32x4, F32x4) {
    let n = vr.end - vr.start;
    let br = beta_off..beta_off + n;
    let (u_all, v_all) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let (u, v) = (&mut u_all[vr.clone()], &mut v_all[vr.clone()]);
    let d = &bp.d_br.as_f32x4_slice()[br.clone()];
    let s = &bp.s_br.as_f32x4_slice()[br.clone()];
    let c = &bp.c_br.as_f32x4_slice()[br.clone()];
    let beta = &bp.beta_br.as_f32x4_slice()[br];
    let bh = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr];
    let (fh_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(f_d));

    let (mut acc_c0, mut acc_br0) = (F32x4::zero(), F32x4::zero());
    let (mut acc_c1, mut acc_br1) = (F32x4::zero(), F32x4::zero());
    let mut i = 0usize;
    while i + 1 < n {
        let u0 = if has_damper {
            fd_v.mul_add(bd[i], fh_v.mul_add(bh[i], u[i]))
        } else {
            fh_v.mul_add(bh[i], u[i])
        };
        let u1 = if has_damper {
            fd_v.mul_add(bd[i + 1], fh_v.mul_add(bh[i + 1], u[i + 1]))
        } else {
            fh_v.mul_add(bh[i + 1], u[i + 1])
        };
        let (b0, b1) = (beta[i], beta[i + 1]);
        let (un0, vn0) = zoh_rotate(u0, b0.mul_add(neg_vbr, v[i]), d[i], s[i]);
        let (un1, vn1) = zoh_rotate(u1, b1.mul_add(neg_vbr, v[i + 1]), d[i + 1], s[i + 1]);
        (u[i], v[i]) = (un0, vn0);
        (u[i + 1], v[i + 1]) = (un1, vn1);
        acc_c0 = vn0.mul_add(c[i], acc_c0);
        acc_br0 = vn0.mul_add(b0, acc_br0);
        acc_c1 = vn1.mul_add(c[i + 1], acc_c1);
        acc_br1 = vn1.mul_add(b1, acc_br1);
        i += 2;
    }
    if i < n {
        let u0 = if has_damper {
            fd_v.mul_add(bd[i], fh_v.mul_add(bh[i], u[i]))
        } else {
            fh_v.mul_add(bh[i], u[i])
        };
        let bi = beta[i];
        let (un, vn) = zoh_rotate(u0, bi.mul_add(neg_vbr, v[i]), d[i], s[i]);
        (u[i], v[i]) = (un, vn);
        acc_c0 = vn.mul_add(c[i], acc_c0);
        acc_br0 = vn.mul_add(bi, acc_br0);
    }
    (acc_c0 + acc_c1, acc_br0 + acc_br1)
}

/// Single-pass 2x-unrolled prefix WDF step accumulating `(acc_c, acc_br, yd, vd_raw)` in registers.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_symp_damp(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    bp: &BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_d: f32,
) -> (F32x4, F32x4, f32, f32) {
    if f_d != 0.0 {
        step_pol_prefix_wdf_symp_damp_inner::<true>(soa, ports, bp, vr, beta_off, neg_vbr, f_d)
    } else {
        step_pol_prefix_wdf_symp_damp_inner::<false>(soa, ports, bp, vr, beta_off, neg_vbr, 0.0)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_pol_prefix_wdf_symp_damp_inner<const HAS_FD: bool>(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    bp: &BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_d: f32,
) -> (F32x4, F32x4, f32, f32) {
    let n = vr.end - vr.start;
    let br = beta_off..beta_off + n;
    let (u_all, v_all) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let (u, v) = (&mut u_all[vr.clone()], &mut v_all[vr.clone()]);
    let d = &bp.d_br.as_f32x4_slice()[br.clone()];
    let s = &bp.s_br.as_f32x4_slice()[br.clone()];
    let c = &bp.c_br.as_f32x4_slice()[br.clone()];
    let beta = &bp.beta_br.as_f32x4_slice()[br];
    let bd = &ports.b_damp.as_f32x4_slice()[vr.clone()];
    let ad = &ports.a_damp.as_f32x4_slice()[vr];
    let fd_v = F32x4::splat(f_d);

    let (mut acc_c0, mut acc_br0, mut acc_yd0, mut acc_vd0) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    let (mut acc_c1, mut acc_br1, mut acc_yd1, mut acc_vd1) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    let mut i = 0usize;
    while i + 1 < n {
        let (bd0, bd1) = (bd[i], bd[i + 1]);
        let u0 = if HAS_FD {
            fd_v.mul_add(bd0, u[i])
        } else {
            u[i]
        };
        let u1 = if HAS_FD {
            fd_v.mul_add(bd1, u[i + 1])
        } else {
            u[i + 1]
        };
        let (b0, b1) = (beta[i], beta[i + 1]);
        let (un0, vn0) = zoh_rotate(u0, b0.mul_add(neg_vbr, v[i]), d[i], s[i]);
        let (un1, vn1) = zoh_rotate(u1, b1.mul_add(neg_vbr, v[i + 1]), d[i + 1], s[i + 1]);
        (u[i], v[i]) = (un0, vn0);
        (u[i + 1], v[i + 1]) = (un1, vn1);
        acc_c0 = vn0.mul_add(c[i], acc_c0);
        acc_br0 = vn0.mul_add(b0, acc_br0);
        acc_yd0 = vn0.mul_add(ad[i], acc_yd0);
        acc_vd0 = un0.mul_add(bd0, acc_vd0);
        acc_c1 = vn1.mul_add(c[i + 1], acc_c1);
        acc_br1 = vn1.mul_add(b1, acc_br1);
        acc_yd1 = vn1.mul_add(ad[i + 1], acc_yd1);
        acc_vd1 = un1.mul_add(bd1, acc_vd1);
        i += 2;
    }
    if i < n {
        let bdi = bd[i];
        let u0 = if HAS_FD {
            fd_v.mul_add(bdi, u[i])
        } else {
            u[i]
        };
        let bi = beta[i];
        let (un, vn) = zoh_rotate(u0, bi.mul_add(neg_vbr, v[i]), d[i], s[i]);
        (u[i], v[i]) = (un, vn);
        acc_c0 = vn.mul_add(c[i], acc_c0);
        acc_br0 = vn.mul_add(bi, acc_br0);
        acc_yd0 = vn.mul_add(ad[i], acc_yd0);
        acc_vd0 = un.mul_add(bdi, acc_vd0);
    }
    (
        acc_c0 + acc_c1,
        acc_br0 + acc_br1,
        (acc_yd0 + acc_yd1).reduce_sum(),
        (acc_vd0 + acc_vd1).reduce_sum(),
    )
}

/// Evaluates SAV drift + phantom tension for all active `sav_strings` in pairs bucketed by `nl`.
pub(crate) fn step_sav_batch_prefix(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    strings: &[StringMapEntry],
    sav_strings: &[u8],
    long_on: bool,
) {
    let mut pending: [Option<usize>; MAX_PHANTOM_LONG_VECS + 1] = [None; MAX_PHANTOM_LONG_VECS + 1];
    for &s_u8 in sav_strings {
        let s_idx = usize::from(s_u8);
        let (nv, nh, nl, m_ph) = sav.string_prefix_bounds(s_idx, long_on);
        if nv < 2 || nh < 2 {
            sav.step_string_sav(s_idx, &strings[s_idx], soa, long_on);
            continue;
        }
        let bucket = if m_ph == 0 {
            0
        } else {
            nl.min(MAX_PHANTOM_LONG_VECS)
        };
        if let Some(s0) = pending[bucket].take() {
            let (e0, e1) = (&strings[s0], &strings[s_idx]);
            if bucket == 0 {
                eval_sav_no_phantom_prefix(soa, sav, s0, e0, s_idx, e1);
            } else if bucket == 1 {
                eval_sav_phantom_prefix::<1>(soa, sav, s0, e0, s_idx, e1, 1);
            } else {
                eval_sav_phantom_prefix::<MAX_PHANTOM_LONG_VECS>(
                    soa, sav, s0, e0, s_idx, e1, bucket,
                );
            }
        } else {
            pending[bucket] = Some(s_idx);
        }
    }
    for s_idx in pending.into_iter().flatten() {
        sav.step_string_sav(s_idx, &strings[s_idx], soa, long_on);
    }
}

#[inline(always)]
fn eval_sav_no_phantom_prefix(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
) {
    let (vb0, hb0, vb1, hb1) = (
        e0.vert_vecs.start,
        e0.horiz_vecs.start,
        e1.vert_vecs.start,
        e1.horiz_vecs.start,
    );
    let (mut uv0, vv0) = load_vh_state(soa, vb0, 2);
    let (mut uh0, vh0) = load_vh_state(soa, hb0, 2);
    let (mut uv1, vv1) = load_vh_state(soa, vb1, 2);
    let (mut uh1, vh1) = load_vh_state(soa, hb1, 2);
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    eval_no_phantom_pair(
        sav.spec(s0),
        sav.spec(s1),
        sav.ts(),
        &mut r0,
        &mut r1,
        &mut uv0,
        &vv0,
        &mut uh0,
        &vh0,
        &mut uv1,
        &vv1,
        &mut uh1,
        &vh1,
    );
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_u(soa, vb0, 2, &uv0);
    store_vh_u(soa, hb0, 2, &uh0);
    store_vh_u(soa, vb1, 2, &uv1);
    store_vh_u(soa, hb1, 2, &uh1);
}

#[inline(always)]
fn eval_sav_phantom_prefix<const NL_MAX: usize>(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    nl: usize,
) {
    let n = if NL_MAX == 1 { 1 } else { nl.min(NL_MAX) };
    let (vb0, hb0, lb0) = (e0.vert_vecs.start, e0.horiz_vecs.start, e0.long_vecs.start);
    let (vb1, hb1, lb1) = (e1.vert_vecs.start, e1.horiz_vecs.start, e1.long_vecs.start);
    let (mut uv0, vv0) = load_vh_state(soa, vb0, 2);
    let (mut uh0, vh0) = load_vh_state(soa, hb0, 2);
    let (mut ul0, vl0) = load_nl_state_n::<NL_MAX>(soa, lb0, n);
    let (mut uv1, vv1) = load_vh_state(soa, vb1, 2);
    let (mut uh1, vh1) = load_vh_state(soa, hb1, 2);
    let (mut ul1, vl1) = load_nl_state_n::<NL_MAX>(soa, lb1, n);
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    eval_phantom_pair::<NL_MAX>(
        sav.spec(s0),
        sav.spec(s1),
        sav.ts(),
        n,
        &mut r0,
        &mut r1,
        &mut uv0,
        &vv0,
        &mut uh0,
        &vh0,
        &mut ul0,
        &vl0,
        &mut uv1,
        &vv1,
        &mut uh1,
        &vh1,
        &mut ul1,
        &vl1,
    );
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_u(soa, vb0, 2, &uv0);
    store_vh_u(soa, hb0, 2, &uh0);
    store_vh_u(soa, vb1, 2, &uv1);
    store_vh_u(soa, hb1, 2, &uh1);
    let u_s = soa.u.as_f32x4_slice_mut();
    if NL_MAX == 1 {
        u_s[lb0] = ul0[0];
        u_s[lb1] = ul1[0];
    } else {
        u_s[lb0..lb0 + n].copy_from_slice(&ul0[..n]);
        u_s[lb1..lb1 + n].copy_from_slice(&ul1[..n]);
    }
}
