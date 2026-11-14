//! Register-resident `TILE = 8` SIMD prefix stepping for ringing strings with active SAV.

use core::ops::Range;

use piano_simd::F32x4;

use crate::choir_map::StringMapEntry;
use crate::modal_soa::ModalSoa;
use crate::phantom::MAX_PHANTOM_LONG_VECS;
use crate::sav_drift::{
    load_nl_state_n, load_vh_state, step_sav_no_phantom_spec, step_sav_phantom_spec_n,
    store_nl_state_n, SavBank, LAMBDA_REG_DEFAULT,
};
use crate::sav_spec::SavStringSpec;
use crate::zoh_kernel::{
    accum_string_tile_in, accum_tile_stereo, step_string_subblock, step_string_subblock_3,
    step_string_tile, zoh_rotate, TILE,
};

#[inline(always)]
pub(crate) fn store_vh_state(
    soa: &mut ModalSoa,
    b: usize,
    len: usize,
    u: &[F32x4; 2],
    v: &[F32x4; 2],
) {
    let (u_s, v_s) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    if len >= 2 {
        assert!(b + 1 < u_s.len() && b + 1 < v_s.len());
        u_s[b] = u[0];
        u_s[b + 1] = u[1];
        v_s[b] = v[0];
        v_s[b + 1] = v[1];
    } else if len == 1 {
        u_s[b] = u[0];
        v_s[b] = v[0];
    }
}

#[inline(always)]
pub(crate) fn step_2_slices(
    u: &mut [F32x4; 2],
    v: &mut [F32x4; 2],
    b: usize,
    d_s: &[F32x4],
    s_s: &[F32x4],
    c_s: &[F32x4],
) -> F32x4 {
    let (u0, v0) = zoh_rotate(u[0], v[0], d_s[b], s_s[b]);
    u[0] = u0;
    v[0] = v0;
    let (u1, v1) = zoh_rotate(u[1], v[1], d_s[b + 1], s_s[b + 1]);
    u[1] = u1;
    v[1] = v1;
    v0.mul_add(c_s[b], v1 * c_s[b + 1])
}

#[inline(always)]
fn step_flex_slices(
    u: &mut [F32x4; 2],
    v: &mut [F32x4; 2],
    b: usize,
    len: usize,
    d_s: &[F32x4],
    s_s: &[F32x4],
    c_s: &[F32x4],
) -> F32x4 {
    if len >= 2 {
        step_2_slices(u, v, b, d_s, s_s, c_s)
    } else if len == 1 {
        let (u0, v0) = zoh_rotate(u[0], v[0], d_s[b], s_s[b]);
        u[0] = u0;
        v[0] = v0;
        v0 * c_s[b]
    } else {
        F32x4::zero()
    }
}

#[inline(always)]
pub(crate) fn step_nl_slices<const NL_MAX: usize>(
    u: &mut [F32x4; NL_MAX],
    v: &mut [F32x4; NL_MAX],
    b: usize,
    nl: usize,
    d_s: &[F32x4],
    s_s: &[F32x4],
    c_s: &[F32x4],
) -> F32x4 {
    if NL_MAX == 1 {
        let (un, vn) = zoh_rotate(u[0], v[0], d_s[b], s_s[b]);
        u[0] = un;
        v[0] = vn;
        return vn * c_s[b];
    }
    let mut acc = F32x4::zero();
    for i in 0..nl.min(NL_MAX) {
        let idx = b + i;
        let (un, vn) = zoh_rotate(u[i], v[i], d_s[idx], s_s[idx]);
        u[i] = un;
        v[i] = vn;
        acc = vn.mul_add(c_s[idx], acc);
    }
    acc
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_tail_ranges<const FULL_TILE: bool>(
    soa: &mut ModalSoa,
    r0: Range<usize>,
    r1: Range<usize>,
    r2: Range<usize>,
    pan_l: f32,
    pan_r: f32,
    t_len: usize,
    tile_l: &mut [f32; TILE],
    tile_r: &mut [f32; TILE],
) {
    let mut acc = [F32x4::zero(); TILE];
    accum_string_tile_in::<FULL_TILE>(soa, r0, t_len, &mut acc);
    accum_string_tile_in::<FULL_TILE>(soa, r1, t_len, &mut acc);
    accum_string_tile_in::<FULL_TILE>(soa, r2, t_len, &mut acc);
    accum_tile_stereo(
        &acc,
        pan_l,
        pan_r,
        if FULL_TILE { TILE } else { t_len },
        tile_l,
        tile_r,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_tails(
    soa: &mut ModalSoa,
    r0: Range<usize>,
    r1: Range<usize>,
    r2: Range<usize>,
    entry: &StringMapEntry,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let steps = out_l.len().min(out_r.len());
    if steps > TILE {
        step_string_subblock_3(soa, [r0, r1, r2], entry.pan_l, entry.pan_r, out_l, out_r);
        return;
    }
    let (mut tile_l, mut tile_r) = ([0.0_f32; TILE], [0.0_f32; TILE]);
    if steps == TILE {
        step_tail_ranges::<true>(
            soa,
            r0,
            r1,
            r2,
            entry.pan_l,
            entry.pan_r,
            TILE,
            &mut tile_l,
            &mut tile_r,
        );
    } else {
        step_tail_ranges::<false>(
            soa,
            r0,
            r1,
            r2,
            entry.pan_l,
            entry.pan_r,
            steps,
            &mut tile_l,
            &mut tile_r,
        );
    }
    for ((lo, ro), (&tl, &tr)) in out_l[..steps]
        .iter_mut()
        .zip(out_r[..steps].iter_mut())
        .zip(tile_l[..steps].iter().zip(tile_r[..steps].iter()))
    {
        *lo += tl;
        *ro += tr;
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_no_phantom_flex_loop(
    soa: &ModalSoa,
    sp: &SavStringSpec,
    ts: f64,
    r_n: &mut f64,
    uv: &mut [F32x4; 2],
    vv: &mut [F32x4; 2],
    uh: &mut [F32x4; 2],
    vh: &mut [F32x4; 2],
    entry: &StringMapEntry,
    nv_vecs: usize,
    nh_vecs: usize,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let (vb, hb) = (entry.vert_vecs.start, entry.horiz_vecs.start);
    let (d_s, s_s, c_s) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    let steps = out_l.len().min(out_r.len());
    for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
        let av = step_flex_slices(uv, vv, vb, nv_vecs, d_s, s_s, c_s);
        let ah = step_flex_slices(uh, vh, hb, nh_vecs, d_s, s_s, c_s);
        step_sav_no_phantom_spec(sp, ts, r_n, uv, vv, uh, vh, LAMBDA_REG_DEFAULT);
        let term = (av + ah).reduce_sum();
        *lo += term * entry.pan_l;
        *ro += term * entry.pan_r;
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_no_phantom_sav_tile(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s_idx: usize,
    entry: &StringMapEntry,
    nv_vecs: usize,
    nh_vecs: usize,
    longitudinal_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let (vb, hb) = (entry.vert_vecs.start, entry.horiz_vecs.start);
    let (r1, r2) = if longitudinal_enabled {
        (hb + nh_vecs..entry.long_vecs.end, 0..0)
    } else {
        (hb + nh_vecs..entry.horiz_vecs.end, entry.long_vecs.clone())
    };
    step_tails(
        soa,
        vb + nv_vecs..entry.vert_vecs.end,
        r1,
        r2,
        entry,
        out_l,
        out_r,
    );

    let (mut uv, mut vv) = load_vh_state(soa, vb, nv_vecs);
    let (mut uh, mut vh) = load_vh_state(soa, hb, nh_vecs);
    let sp = sav.spec(s_idx);
    let ts = sav.ts();
    let mut r_n = sav.r_sav[s_idx];
    let steps = out_l.len().min(out_r.len());
    if nv_vecs >= 2 && nh_vecs >= 2 {
        let (d_s, s_s, c_s) = (
            soa.d_coeff.as_f32x4_slice(),
            soa.s_coeff.as_f32x4_slice(),
            soa.c_bridge.as_f32x4_slice(),
        );
        let min_len = d_s.len().min(s_s.len()).min(c_s.len());
        assert!(vb + 2 <= min_len && hb + 2 <= min_len);
        for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
            let av = step_2_slices(&mut uv, &mut vv, vb, d_s, s_s, c_s);
            let ah = step_2_slices(&mut uh, &mut vh, hb, d_s, s_s, c_s);
            step_sav_no_phantom_spec(
                sp,
                ts,
                &mut r_n,
                &mut uv,
                &vv,
                &mut uh,
                &vh,
                LAMBDA_REG_DEFAULT,
            );
            let term = (av + ah).reduce_sum();
            *lo += term * entry.pan_l;
            *ro += term * entry.pan_r;
        }
    } else {
        step_no_phantom_flex_loop(
            soa, sp, ts, &mut r_n, &mut uv, &mut vv, &mut uh, &mut vh, entry, nv_vecs, nh_vecs,
            out_l, out_r,
        );
    }
    sav.r_sav[s_idx] = r_n;
    store_vh_state(soa, vb, nv_vecs, &uv, &vv);
    store_vh_state(soa, hb, nh_vecs, &uh, &vh);
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_phantom_flex_loop<const NL_MAX: usize>(
    soa: &ModalSoa,
    sp: &SavStringSpec,
    ts: f64,
    r_n: &mut f64,
    uv: &mut [F32x4; 2],
    vv: &mut [F32x4; 2],
    uh: &mut [F32x4; 2],
    vh: &mut [F32x4; 2],
    ul: &mut [F32x4; NL_MAX],
    vl: &mut [F32x4; NL_MAX],
    entry: &StringMapEntry,
    nv_vecs: usize,
    nh_vecs: usize,
    nl_vecs: usize,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let (vb, hb, lb) = (
        entry.vert_vecs.start,
        entry.horiz_vecs.start,
        entry.long_vecs.start,
    );
    let (d_s, s_s, c_s) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    assert!(lb + nl_vecs <= d_s.len().min(s_s.len()).min(c_s.len()));
    let steps = out_l.len().min(out_r.len());
    for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
        let av = step_flex_slices(uv, vv, vb, nv_vecs, d_s, s_s, c_s);
        let ah = step_flex_slices(uh, vh, hb, nh_vecs, d_s, s_s, c_s);
        let al = step_nl_slices::<NL_MAX>(ul, vl, lb, nl_vecs, d_s, s_s, c_s);
        step_sav_phantom_spec_n::<NL_MAX>(
            sp,
            ts,
            nl_vecs,
            r_n,
            uv,
            vv,
            uh,
            vh,
            ul,
            vl,
            LAMBDA_REG_DEFAULT,
        );
        let term = (av + ah + al).reduce_sum();
        *lo += term * entry.pan_l;
        *ro += term * entry.pan_r;
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_phantom_sav_tile<const NL_MAX: usize>(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s_idx: usize,
    entry: &StringMapEntry,
    nv_vecs: usize,
    nh_vecs: usize,
    nl_vecs: usize,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let nl = if NL_MAX == 1 { 1 } else { nl_vecs.min(NL_MAX) };
    let (vb, hb, lb) = (
        entry.vert_vecs.start,
        entry.horiz_vecs.start,
        entry.long_vecs.start,
    );
    step_tails(
        soa,
        vb + nv_vecs..entry.vert_vecs.end,
        hb + nh_vecs..entry.horiz_vecs.end,
        lb + nl..entry.long_vecs.end,
        entry,
        out_l,
        out_r,
    );

    let (mut uv, mut vv) = load_vh_state(soa, vb, nv_vecs);
    let (mut uh, mut vh) = load_vh_state(soa, hb, nh_vecs);
    let (mut ul, mut vl) = load_nl_state_n::<NL_MAX>(soa, lb, nl);
    let sp = sav.spec(s_idx);
    let ts = sav.ts();
    let mut r_n = sav.r_sav[s_idx];
    let steps = out_l.len().min(out_r.len());
    if nv_vecs >= 2 && nh_vecs >= 2 {
        let (d_s, s_s, c_s) = (
            soa.d_coeff.as_f32x4_slice(),
            soa.s_coeff.as_f32x4_slice(),
            soa.c_bridge.as_f32x4_slice(),
        );
        let min_len = d_s.len().min(s_s.len()).min(c_s.len());
        assert!(vb + 2 <= min_len && hb + 2 <= min_len && lb + nl <= min_len);
        for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
            let av = step_2_slices(&mut uv, &mut vv, vb, d_s, s_s, c_s);
            let ah = step_2_slices(&mut uh, &mut vh, hb, d_s, s_s, c_s);
            let al = step_nl_slices::<NL_MAX>(&mut ul, &mut vl, lb, nl, d_s, s_s, c_s);
            step_sav_phantom_spec_n::<NL_MAX>(
                sp,
                ts,
                nl,
                &mut r_n,
                &mut uv,
                &vv,
                &mut uh,
                &vh,
                &mut ul,
                &vl,
                LAMBDA_REG_DEFAULT,
            );
            let term = (av + ah + al).reduce_sum();
            *lo += term * entry.pan_l;
            *ro += term * entry.pan_r;
        }
    } else {
        step_phantom_flex_loop::<NL_MAX>(
            soa, sp, ts, &mut r_n, &mut uv, &mut vv, &mut uh, &mut vh, &mut ul, &mut vl, entry,
            nv_vecs, nh_vecs, nl, out_l, out_r,
        );
    }
    sav.r_sav[s_idx] = r_n;
    store_vh_state(soa, vb, nv_vecs, &uv, &vv);
    store_vh_state(soa, hb, nh_vecs, &uh, &vh);
    store_nl_state_n::<NL_MAX>(soa, lb, nl, &ul, &vl);
}

/// Steps one ringing string (`may_be_nonzero && !hammer_active && !damper_engaged`) across
/// `out_l`/`out_r`, keeping prefix `F32x4` vectors resident in registers across the slice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_ringing_string_tile(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s_idx: usize,
    entry: &StringMapEntry,
    longitudinal_enabled: bool,
    sav_phantom_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    if !sav_phantom_enabled || sav.string_n_vh(s_idx) == 0 {
        if longitudinal_enabled {
            step_tails(soa, entry.all_vecs(), 0..0, 0..0, entry, out_l, out_r);
        } else {
            let steps = out_l.len().min(out_r.len());
            step_tails(
                soa,
                entry.vert_vecs.start..entry.horiz_vecs.end,
                0..0,
                0..0,
                entry,
                out_l,
                out_r,
            );
            if steps > TILE {
                step_string_subblock(soa, entry.long_vecs.clone(), 0.0, 0.0, out_l, out_r);
            } else {
                let _ = step_string_tile(soa, entry.long_vecs.clone(), steps);
            }
        }
        return;
    }

    let (nv_vecs, nh_vecs, nl_vecs, m_ph) = sav.string_prefix_bounds(s_idx, longitudinal_enabled);
    if m_ph == 0 {
        step_no_phantom_sav_tile(
            soa,
            sav,
            s_idx,
            entry,
            nv_vecs,
            nh_vecs,
            longitudinal_enabled,
            out_l,
            out_r,
        );
    } else if nl_vecs == 1 {
        step_phantom_sav_tile::<1>(soa, sav, s_idx, entry, nv_vecs, nh_vecs, 1, out_l, out_r);
    } else {
        step_phantom_sav_tile::<MAX_PHANTOM_LONG_VECS>(
            soa, sav, s_idx, entry, nv_vecs, nh_vecs, nl_vecs, out_l, out_r,
        );
    }
}
