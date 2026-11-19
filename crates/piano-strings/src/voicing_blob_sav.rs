//! Binary packing helpers for `VoicingBlob` SAV specs and `f64`/`F32x4` slices (`no_std`).

use alloc::vec::Vec;
use piano_simd::F32x4;

use crate::phantom::{
    MAX_PHANTOM_LONG_MODES, MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_MODES,
    MAX_PHANTOM_TRANSVERSE_VECS,
};
use crate::sav_spec::{SavEnergySlopes, SavStringSpec};

pub(crate) const SAV_ENTRY_FLOATS: usize = 7
    + 4 * 2
    + (6 * MAX_PHANTOM_TRANSVERSE_VECS + 4 * MAX_PHANTOM_LONG_VECS) * 4
    + (2 * MAX_PHANTOM_TRANSVERSE_MODES + MAX_PHANTOM_LONG_MODES) * 2;

#[inline]
pub(crate) fn push_f64_bits(out: &mut Vec<f32>, val: f64) {
    let bits = val.to_bits();
    out.push(f32::from_bits(bits as u32));
    out.push(f32::from_bits((bits >> 32) as u32));
}

#[inline]
pub(crate) fn read_f64_bits(src: &[f32], cur: &mut usize) -> f64 {
    let lo = u64::from(src[*cur].to_bits());
    let hi = u64::from(src[*cur + 1].to_bits());
    *cur += 2;
    f64::from_bits(lo | (hi << 32))
}

#[inline]
fn push_vec4s(out: &mut Vec<f32>, slice: &[F32x4]) {
    for v in slice {
        out.extend_from_slice(&v.to_array());
    }
}

#[inline]
fn read_vec4s(src: &[f32], cur: &mut usize, dst: &mut [F32x4]) {
    for d in dst {
        *d = F32x4::from_slice(&src[*cur..*cur + 4]);
        *cur += 4;
    }
}

#[inline]
pub(crate) fn push_copy(out: &mut Vec<f32>, src: &[f32]) {
    out.extend_from_slice(src);
}

pub(crate) fn push_sav_entry(out: &mut Vec<f32>, sp: &SavStringSpec, sl: &SavEnergySlopes) {
    for &u in &[
        sp.n_vh, sp.m_ph, sp.nv_vecs, sp.nh_vecs, sp.nl_vecs, sl.nv_c, sl.nh_c,
    ] {
        out.push(f32::from_bits(u as u32));
    }
    for &f in &[sp.length_m, sp.quarter_l, sp.kappa_0, sp.kappa_ph] {
        push_f64_bits(out, f);
    }
    for arr in [
        &sp.sigma_v4,
        &sp.omega_sigma_v4,
        &sp.mask_v4,
        &sp.sigma_h4,
        &sp.omega_sigma_h4,
        &sp.mask_h4,
    ] {
        push_vec4s(out, arr);
    }
    for arr in [
        &sp.alpha_l4,
        &sp.half_ql_mask4,
        &sp.mask_l4,
        &sp.omega_alpha_l_norm4,
    ] {
        push_vec4s(out, arr);
    }
    for &f in &sl.sigma_v {
        push_f64_bits(out, f);
    }
    for &f in &sl.sigma_h {
        push_f64_bits(out, f);
    }
    for &f in &sl.alpha_l {
        push_f64_bits(out, f);
    }
}

pub(crate) fn read_sav_entry(
    src: &[f32],
    cur: &mut usize,
    sp: &mut SavStringSpec,
    sl: &mut SavEnergySlopes,
) {
    let ru = |c: &mut usize| -> usize {
        let v = src[*c].to_bits() as usize;
        *c += 1;
        v
    };
    (
        sp.n_vh, sp.m_ph, sp.nv_vecs, sp.nh_vecs, sp.nl_vecs, sl.nv_c, sl.nh_c,
    ) = (
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
    );
    sp.length_m = read_f64_bits(src, cur);
    sp.quarter_l = read_f64_bits(src, cur);
    sp.kappa_0 = read_f64_bits(src, cur);
    sp.kappa_ph = read_f64_bits(src, cur);
    for arr in [
        &mut sp.sigma_v4,
        &mut sp.omega_sigma_v4,
        &mut sp.mask_v4,
        &mut sp.sigma_h4,
        &mut sp.omega_sigma_h4,
        &mut sp.mask_h4,
    ] {
        read_vec4s(src, cur, arr);
    }
    for arr in [
        &mut sp.alpha_l4,
        &mut sp.half_ql_mask4,
        &mut sp.mask_l4,
        &mut sp.omega_alpha_l_norm4,
    ] {
        read_vec4s(src, cur, arr);
    }
    for d in &mut sl.sigma_v {
        *d = read_f64_bits(src, cur);
    }
    for d in &mut sl.sigma_h {
        *d = read_f64_bits(src, cur);
    }
    for d in &mut sl.alpha_l {
        *d = read_f64_bits(src, cur);
    }
}
