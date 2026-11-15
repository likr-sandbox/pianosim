//! L1-packed WDF prefix ZOH kernels and 2-string paired SAV batching (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_port::{BridgePorts, FastPrefixMeta};
use crate::choir_map::StringMapEntry;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::ModalSoa;
use crate::phantom::MAX_PHANTOM_LONG_VECS;
use crate::sav_drift::{load_nl_state_n, SavBank};
use crate::sav_pair::{eval_no_phantom_pair, eval_phantom_pair};
use crate::zoh_kernel::zoh_rotate;

/// Contiguously steps all uncoupled strings' `[v_br | h_br]` L1-packed prefix vectors in one pass.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_uncoupled_prefixes_fast(
    u_br: &mut [F32x4],
    v_br: &mut [F32x4],
    d_br: &[F32x4],
    s_br: &[F32x4],
    c_br: &[F32x4],
    beta_br: &[F32x4],
    meta: &[FastPrefixMeta; TOTAL_STRINGS],
    str_coupled: &[bool; TOTAL_STRINGS],
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    may_be_nz: &mut [bool; TOTAL_STRINGS],
    ports_synced: &mut [bool; TOTAL_STRINGS],
) -> (f32, f32) {
    let len = u_br.len();
    assert!(
        v_br.len() == len
            && d_br.len() == len
            && s_br.len() == len
            && c_br.len() == len
            && beta_br.len() == len
    );
    let (mut u_rem, mut v_rem) = (u_br, v_br);
    let (mut d_rem, mut s_rem, mut c_rem, mut b_rem) = (d_br, s_br, c_br, beta_br);
    let (mut sum_l_vec, mut sum_r_vec) = (F32x4::zero(), F32x4::zero());

    for s_idx in 0..TOTAL_STRINGS {
        let fm = meta[s_idx];
        let n = fm.n_vh as usize;
        let (u, u_rest) = u_rem.split_at_mut(n);
        let (v, v_rest) = v_rem.split_at_mut(n);
        let (d, d_rest) = d_rem.split_at(n);
        let (s, s_rest) = s_rem.split_at(n);
        let (c, c_rest) = c_rem.split_at(n);
        let (beta, b_rest) = b_rem.split_at(n);
        (u_rem, v_rem, d_rem, s_rem, c_rem, b_rem) =
            (u_rest, v_rest, d_rest, s_rest, c_rest, b_rest);
        if str_coupled[s_idx] {
            continue;
        }
        let v_port = 0.5 * (a_waves[s_idx] + b_waves[s_idx]);
        let neg_vbr = F32x4::splat(-v_port);
        if may_be_nz[s_idx] || v_port != 0.0 {
            may_be_nz[s_idx] = true;
            ports_synced[s_idx] = false;
        }
        let (mut acc_c0, mut acc_br0) = (F32x4::zero(), F32x4::zero());
        let (mut acc_c1, mut acc_br1) = (F32x4::zero(), F32x4::zero());
        let mut i = 0usize;
        while i + 1 < n {
            let (b0, b1) = (beta[i], beta[i + 1]);
            let (un0, vn0) = zoh_rotate(u[i], b0.mul_add(neg_vbr, v[i]), d[i], s[i]);
            let (un1, vn1) =
                zoh_rotate(u[i + 1], b1.mul_add(neg_vbr, v[i + 1]), d[i + 1], s[i + 1]);
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
        let (acc_c, acc_br) = (acc_c0 + acc_c1, acc_br0 + acc_br1);
        a_waves[s_idx] = fm.a_wave_gain * acc_br.reduce_sum();
        sum_l_vec = acc_c.mul_add(F32x4::splat(fm.pan_l), sum_l_vec);
        sum_r_vec = acc_c.mul_add(F32x4::splat(fm.pan_r), sum_r_vec);
    }
    (sum_l_vec.reduce_sum(), sum_r_vec.reduce_sum())
}

/// Steps one string's merged `[v_br | h_br]` L1 prefix slice when damper is currently clear.
#[inline(always)]
pub(crate) fn step_string_prefix_wdf_br(
    bp: &mut BridgePorts,
    beta_off: usize,
    n_vh: usize,
    neg_vbr: F32x4,
) -> (F32x4, F32x4) {
    let br = beta_off..beta_off + n_vh;
    let (u_all, v_all) = (bp.u_br.as_f32x4_slice_mut(), bp.v_br.as_f32x4_slice_mut());
    let (u, v) = (&mut u_all[br.clone()], &mut v_all[br.clone()]);
    let d = &bp.d_br.as_f32x4_slice()[br.clone()];
    let s = &bp.s_br.as_f32x4_slice()[br.clone()];
    let c = &bp.c_br.as_f32x4_slice()[br.clone()];
    let beta = &bp.beta_br.as_f32x4_slice()[br];

    let (mut acc_c0, mut acc_br0) = (F32x4::zero(), F32x4::zero());
    let (mut acc_c1, mut acc_br1) = (F32x4::zero(), F32x4::zero());
    let mut i = 0usize;
    while i + 1 < n_vh {
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
    if i < n_vh {
        let bi = beta[i];
        let (un, vn) = zoh_rotate(u[i], bi.mul_add(neg_vbr, v[i]), d[i], s[i]);
        (u[i], v[i]) = (un, vn);
        acc_c0 = vn.mul_add(c[i], acc_c0);
        acc_br0 = vn.mul_add(bi, acc_br0);
    }
    (acc_c0 + acc_c1, acc_br0 + acc_br1)
}

/// Steps one polarization's L1 prefix WDF slice with simultaneous hammer/damper injection.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_inj(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_h: f32,
    f_d: f32,
    has_damper: bool,
) -> (F32x4, F32x4) {
    let n = vr.end - vr.start;
    let br = beta_off..beta_off + n;
    let (u_all, v_all) = (bp.u_br.as_f32x4_slice_mut(), bp.v_br.as_f32x4_slice_mut());
    let (u, v) = (&mut u_all[br.clone()], &mut v_all[br.clone()]);
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

/// Single-pass 2x-unrolled L1 prefix WDF step accumulating `(acc_c, acc_br, yd, vd_raw)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_symp_damp(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_d: f32,
) -> (F32x4, F32x4, f32, f32) {
    if f_d != 0.0 {
        step_pol_prefix_wdf_symp_damp_inner::<true>(ports, bp, vr, beta_off, neg_vbr, f_d)
    } else {
        step_pol_prefix_wdf_symp_damp_inner::<false>(ports, bp, vr, beta_off, neg_vbr, 0.0)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_pol_prefix_wdf_symp_damp_inner<const HAS_FD: bool>(
    ports: &ContactPorts,
    bp: &mut BridgePorts,
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_d: f32,
) -> (F32x4, F32x4, f32, f32) {
    let n = vr.end - vr.start;
    let br = beta_off..beta_off + n;
    let (u_all, v_all) = (bp.u_br.as_f32x4_slice_mut(), bp.v_br.as_f32x4_slice_mut());
    let (u, v) = (&mut u_all[br.clone()], &mut v_all[br.clone()]);
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

#[inline(always)]
fn load_vh_br(u: &[F32x4], v: &[F32x4], b: usize) -> ([F32x4; 2], [F32x4; 2]) {
    assert!(b + 1 < u.len() && b + 1 < v.len());
    ([u[b], u[b + 1]], [v[b], v[b + 1]])
}

#[inline(always)]
fn store_vh_u_br(u_s: &mut [F32x4], b: usize, u: &[F32x4; 2]) {
    assert!(b + 1 < u_s.len());
    u_s[b] = u[0];
    u_s[b + 1] = u[1];
}

/// Evaluates SAV drift + phantom tension for all active `sav_strings` in pairs bucketed by `nl`.
pub(crate) fn step_sav_batch_prefix(
    soa: &mut ModalSoa,
    bp: &mut BridgePorts,
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
            bp.sync_string_to_soa(soa, s_idx);
            sav.step_string_sav(s_idx, &strings[s_idx], soa, long_on);
            bp.sync_string_from_soa(soa, s_idx);
            continue;
        }
        let bucket = if m_ph == 0 {
            0
        } else {
            nl.min(MAX_PHANTOM_LONG_VECS)
        };
        if let Some(s0) = pending[bucket].take() {
            if bucket == 0 {
                eval_sav_no_phantom_prefix(bp, sav, s0, s_idx);
            } else if bucket == 1 {
                eval_sav_phantom_prefix::<1>(
                    soa,
                    bp,
                    sav,
                    s0,
                    &strings[s0],
                    s_idx,
                    &strings[s_idx],
                    1,
                );
            } else {
                eval_sav_phantom_prefix::<MAX_PHANTOM_LONG_VECS>(
                    soa,
                    bp,
                    sav,
                    s0,
                    &strings[s0],
                    s_idx,
                    &strings[s_idx],
                    bucket,
                );
            }
        } else {
            pending[bucket] = Some(s_idx);
        }
    }
    for s_idx in pending.into_iter().flatten() {
        bp.sync_string_to_soa(soa, s_idx);
        sav.step_string_sav(s_idx, &strings[s_idx], soa, long_on);
        bp.sync_string_from_soa(soa, s_idx);
    }
}

#[inline(always)]
fn eval_sav_no_phantom_prefix(bp: &mut BridgePorts, sav: &mut SavBank, s0: usize, s1: usize) {
    let (vb0, hb0) = (bp.meta[s0].beta_v_off, bp.meta[s0].beta_h_off);
    let (vb1, hb1) = (bp.meta[s1].beta_v_off, bp.meta[s1].beta_h_off);
    let (mut uv0, vv0, mut uh0, vh0, mut uv1, vv1, mut uh1, vh1) = {
        let (ub, vb) = (bp.u_br.as_f32x4_slice(), bp.v_br.as_f32x4_slice());
        let (uv0, vv0) = load_vh_br(ub, vb, vb0);
        let (uh0, vh0) = load_vh_br(ub, vb, hb0);
        let (uv1, vv1) = load_vh_br(ub, vb, vb1);
        let (uh1, vh1) = load_vh_br(ub, vb, hb1);
        (uv0, vv0, uh0, vh0, uv1, vv1, uh1, vh1)
    };
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
    let ub = bp.u_br.as_f32x4_slice_mut();
    store_vh_u_br(ub, vb0, &uv0);
    store_vh_u_br(ub, hb0, &uh0);
    store_vh_u_br(ub, vb1, &uv1);
    store_vh_u_br(ub, hb1, &uh1);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn eval_sav_phantom_prefix<const NL_MAX: usize>(
    soa: &mut ModalSoa,
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    nl: usize,
) {
    let n = if NL_MAX == 1 { 1 } else { nl.min(NL_MAX) };
    let (vb0, hb0, lb0) = (
        bp.meta[s0].beta_v_off,
        bp.meta[s0].beta_h_off,
        e0.long_vecs.start,
    );
    let (vb1, hb1, lb1) = (
        bp.meta[s1].beta_v_off,
        bp.meta[s1].beta_h_off,
        e1.long_vecs.start,
    );
    let (mut uv0, vv0, mut uh0, vh0, mut uv1, vv1, mut uh1, vh1) = {
        let (ub, vb) = (bp.u_br.as_f32x4_slice(), bp.v_br.as_f32x4_slice());
        let (uv0, vv0) = load_vh_br(ub, vb, vb0);
        let (uh0, vh0) = load_vh_br(ub, vb, hb0);
        let (uv1, vv1) = load_vh_br(ub, vb, vb1);
        let (uh1, vh1) = load_vh_br(ub, vb, hb1);
        (uv0, vv0, uh0, vh0, uv1, vv1, uh1, vh1)
    };
    let (mut ul0, vl0) = load_nl_state_n::<NL_MAX>(soa, lb0, n);
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
    let ub = bp.u_br.as_f32x4_slice_mut();
    store_vh_u_br(ub, vb0, &uv0);
    store_vh_u_br(ub, hb0, &uh0);
    store_vh_u_br(ub, vb1, &uv1);
    store_vh_u_br(ub, hb1, &uh1);
    let u_s = soa.u.as_f32x4_slice_mut();
    if NL_MAX == 1 {
        u_s[lb0] = ul0[0];
        u_s[lb1] = ul1[0];
    } else {
        u_s[lb0..lb0 + n].copy_from_slice(&ul0[..n]);
        u_s[lb1..lb1 + n].copy_from_slice(&ul1[..n]);
    }
}
