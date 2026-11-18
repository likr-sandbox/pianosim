//! Register-resident `3x6` (`12` transverse x `24` longitudinal) SAV tile stepping
//! for bass keys `0..16` (`no_std`, isolated from `2x4` hot loops on `wasm32`).

use piano_simd::F32x4;

use crate::choir_map::StringMapEntry;
use crate::modal_soa::ModalSoa;
use crate::phantom::{MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_VECS};
use crate::sav_drift::{
    load_nl_state_n, load_vh_state_n, step_sav_no_phantom_spec_n, step_sav_phantom_spec_3x6,
    step_sav_phantom_spec_n, store_nl_state_n, SavBank, LAMBDA_REG_DEFAULT,
};
use crate::sav_spec::SavStringSpec;
use crate::sav_tile::{step_flex_slices, step_nl_slices, step_tails};
use crate::zoh_kernel::zoh_rotate;

#[inline(always)]
pub(crate) fn store_vh_state_n<const NT_MAX: usize>(
    soa: &mut ModalSoa,
    b: usize,
    len: usize,
    u: &[F32x4; NT_MAX],
    v: &[F32x4; NT_MAX],
) {
    let (u_s, v_s) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let n = len.min(NT_MAX);
    if n > 0 {
        assert!(b + n <= u_s.len() && b + n <= v_s.len());
        u_s[b..b + n].copy_from_slice(&u[..n]);
        v_s[b..b + n].copy_from_slice(&v[..n]);
    }
}

#[inline(always)]
fn step_3_slices(
    u: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    v: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    b: usize,
    d_s: &[F32x4],
    s_s: &[F32x4],
    c_s: &[F32x4],
) -> F32x4 {
    let mut acc = F32x4::zero();
    for i in 0..MAX_PHANTOM_TRANSVERSE_VECS {
        let idx = b + i;
        let (un, vn) = zoh_rotate(u[i], v[i], d_s[idx], s_s[idx]);
        u[i] = un;
        v[i] = vn;
        acc = vn.mul_add(c_s[idx], acc);
    }
    acc
}

#[inline(always)]
fn step_flex_3(
    u: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    v: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    b: usize,
    len: usize,
    d_s: &[F32x4],
    s_s: &[F32x4],
    c_s: &[F32x4],
) -> F32x4 {
    let n = len.min(MAX_PHANTOM_TRANSVERSE_VECS);
    let mut acc = F32x4::zero();
    for i in 0..n {
        let idx = b + i;
        let (un, vn) = zoh_rotate(u[i], v[i], d_s[idx], s_s[idx]);
        u[i] = un;
        v[i] = vn;
        acc = vn.mul_add(c_s[idx], acc);
    }
    acc
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_no_phantom_flex_loop(
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
        step_sav_no_phantom_spec_n::<2>(sp, ts, r_n, uv, vv, uh, vh, LAMBDA_REG_DEFAULT);
        let term = (av + ah).reduce_sum();
        *lo += term * entry.pan_l;
        *ro += term * entry.pan_r;
    }
}

#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_phantom_flex_loop<const NL_MAX: usize>(
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

#[cfg_attr(target_arch = "wasm32", inline(never))]
#[allow(clippy::too_many_arguments)]
fn step_no_phantom_sav_tile_3(
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

    let (mut uv, mut vv) = load_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, vb, nv_vecs);
    let (mut uh, mut vh) = load_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, hb, nh_vecs);
    let sp = sav.spec(s_idx);
    let ts = sav.ts();
    let mut r_n = sav.r_sav[s_idx];
    let steps = out_l.len().min(out_r.len());
    let (d_s, s_s, c_s) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    let min_len = d_s.len().min(s_s.len()).min(c_s.len());
    assert!(vb + nv_vecs <= min_len && hb + nh_vecs <= min_len);
    for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
        let av = if nv_vecs == MAX_PHANTOM_TRANSVERSE_VECS {
            step_3_slices(&mut uv, &mut vv, vb, d_s, s_s, c_s)
        } else {
            step_flex_3(&mut uv, &mut vv, vb, nv_vecs, d_s, s_s, c_s)
        };
        let ah = if nh_vecs == MAX_PHANTOM_TRANSVERSE_VECS {
            step_3_slices(&mut uh, &mut vh, hb, d_s, s_s, c_s)
        } else {
            step_flex_3(&mut uh, &mut vh, hb, nh_vecs, d_s, s_s, c_s)
        };
        step_sav_no_phantom_spec_n::<MAX_PHANTOM_TRANSVERSE_VECS>(
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
    sav.r_sav[s_idx] = r_n;
    store_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, vb, nv_vecs, &uv, &vv);
    store_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, hb, nh_vecs, &uh, &vh);
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
#[allow(clippy::too_many_arguments)]
fn step_phantom_sav_tile_3x6(
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
    let nl = nl_vecs.min(MAX_PHANTOM_LONG_VECS);
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

    let (mut uv, mut vv) = load_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, vb, nv_vecs);
    let (mut uh, mut vh) = load_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, hb, nh_vecs);
    let (mut ul, mut vl) = load_nl_state_n::<MAX_PHANTOM_LONG_VECS>(soa, lb, nl);
    let sp = sav.spec(s_idx);
    let ts = sav.ts();
    let mut r_n = sav.r_sav[s_idx];
    let steps = out_l.len().min(out_r.len());
    let (d_s, s_s, c_s) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    let min_len = d_s.len().min(s_s.len()).min(c_s.len());
    assert!(vb + nv_vecs <= min_len && hb + nh_vecs <= min_len && lb + nl <= min_len);
    for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
        let av = if nv_vecs == MAX_PHANTOM_TRANSVERSE_VECS {
            step_3_slices(&mut uv, &mut vv, vb, d_s, s_s, c_s)
        } else {
            step_flex_3(&mut uv, &mut vv, vb, nv_vecs, d_s, s_s, c_s)
        };
        let ah = if nh_vecs == MAX_PHANTOM_TRANSVERSE_VECS {
            step_3_slices(&mut uh, &mut vh, hb, d_s, s_s, c_s)
        } else {
            step_flex_3(&mut uh, &mut vh, hb, nh_vecs, d_s, s_s, c_s)
        };
        let al = step_nl_slices::<MAX_PHANTOM_LONG_VECS>(&mut ul, &mut vl, lb, nl, d_s, s_s, c_s);
        step_sav_phantom_spec_3x6(
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
    sav.r_sav[s_idx] = r_n;
    store_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, vb, nv_vecs, &uv, &vv);
    store_vh_state_n::<MAX_PHANTOM_TRANSVERSE_VECS>(soa, hb, nh_vecs, &uh, &vh);
    store_nl_state_n::<MAX_PHANTOM_LONG_VECS>(soa, lb, nl, &ul, &vl);
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_ringing_string_tile_3x6(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s_idx: usize,
    entry: &StringMapEntry,
    nv_vecs: usize,
    nh_vecs: usize,
    nl_vecs: usize,
    m_ph: usize,
    longitudinal_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    if m_ph == 0 {
        step_no_phantom_sav_tile_3(
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
    } else {
        step_phantom_sav_tile_3x6(
            soa, sav, s_idx, entry, nv_vecs, nh_vecs, nl_vecs, out_l, out_r,
        );
    }
}
