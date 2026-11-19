//! Coupled-form ZOH modal stepping kernel (`F32x4` SIMD, 8-sample L1-tiled).
//!
//! Loop order keeps samples inside L1-cached mode tiles (unrolled at compile time for
//! full `TILE = 8` tiles) while iterating across all 230 strings unconditionally every
//! sample (D1 invariant: zero voice-stealing or dormant-string skipping).

use core::ops::Range;

use piano_simd::F32x4;

use crate::choir_map::ChoirMap;
use crate::modal_soa::ModalSoa;
#[cfg(test)]
pub(crate) use crate::zoh_subblock::step_all_tiles;
pub(crate) use crate::zoh_subblock::{
    accum_soa_range_in, step_string_subblock, step_string_subblock_3,
};

/// Sample tile width kept resident across string/mode SIMD passes.
pub(crate) const TILE: usize = 8;

const SUBNORMAL_FLUSH_BITS: u32 = 1.0e-20_f32.to_bits();
const FINITE_EXP_BITS: u32 = 0x7f80_0000_u32;

/// Coupled-form exact incremental ZOH rotation on one 4-mode `F32x4` lane.
///
/// Advances `(u, v)` by pole `(1 - d) + i * s` without `2.0`-adjacent rounding:
/// `u_next = u - (d * u + s * v)` and `v_next = v + (s * u - d * v)`.
#[inline(always)]
pub(crate) fn zoh_rotate(u0: F32x4, v0: F32x4, d0: F32x4, s0: F32x4) -> (F32x4, F32x4) {
    let du = d0.mul_add(u0, s0 * v0);
    let dv = d0.nmul_add(v0, s0 * u0);
    (u0 - du, v0 + dv)
}

/// Injects scalar excitation impulse `force` into `u_slice` scaled by `b_slice`.
#[inline]
pub(crate) fn inject_excitation(u_slice: &mut [F32x4], b_slice: &[F32x4], force: f32) {
    let f_v = F32x4::splat(force);
    for (u, &b) in u_slice.iter_mut().zip(b_slice.iter()) {
        *u = b.mul_add(f_v, *u);
    }
}

/// Advances one contiguous string vector range by 1 sample (4x unrolled, branchless).
#[inline(always)]
pub(crate) fn step_string_modes(
    u: &mut [F32x4],
    v: &mut [F32x4],
    d: &[F32x4],
    s: &[F32x4],
    c: &[F32x4],
) -> f32 {
    let n = u.len();
    assert!(v.len() == n && d.len() == n && s.len() == n && c.len() == n);

    let zero = F32x4::zero();
    let (mut acc0, mut acc1, mut acc2, mut acc3) = (zero, zero, zero, zero);

    let mut u_chunks = u.chunks_exact_mut(4);
    let mut v_chunks = v.chunks_exact_mut(4);
    let mut d_chunks = d.chunks_exact(4);
    let mut s_chunks = s.chunks_exact(4);
    let mut c_chunks = c.chunks_exact(4);

    for ((((uc, vc), dc), sc), cc) in (&mut u_chunks)
        .zip(&mut v_chunks)
        .zip(&mut d_chunks)
        .zip(&mut s_chunks)
        .zip(&mut c_chunks)
    {
        for lane in 0..4 {
            let (un, vn) = zoh_rotate(uc[lane], vc[lane], dc[lane], sc[lane]);
            uc[lane] = un;
            vc[lane] = vn;
            match lane {
                0 => acc0 = vn.mul_add(cc[0], acc0),
                1 => acc1 = vn.mul_add(cc[1], acc1),
                2 => acc2 = vn.mul_add(cc[2], acc2),
                _ => acc3 = vn.mul_add(cc[3], acc3),
            }
        }
    }

    for ((((ur, vr), &d0), &s0), &c0) in u_chunks
        .into_remainder()
        .iter_mut()
        .zip(v_chunks.into_remainder().iter_mut())
        .zip(d_chunks.remainder().iter())
        .zip(s_chunks.remainder().iter())
        .zip(c_chunks.remainder().iter())
    {
        let (un, vn) = zoh_rotate(*ur, *vr, d0, s0);
        *ur = un;
        *vr = vn;
        acc0 = vn.mul_add(c0, acc0);
    }

    ((acc0 + acc1) + (acc2 + acc3)).reduce_sum()
}

/// Steps all 230 strings by 1 sample without allocations or branching.
pub(crate) fn step_all_strings(soa: &mut ModalSoa, map: &ChoirMap) -> (f32, f32) {
    let total_vecs = map.total_vecs();
    let mut u_rest = &mut soa.u.as_f32x4_slice_mut()[..total_vecs];
    let mut v_rest = &mut soa.v.as_f32x4_slice_mut()[..total_vecs];
    let mut d_rest = &soa.d_coeff.as_f32x4_slice()[..total_vecs];
    let mut s_rest = &soa.s_coeff.as_f32x4_slice()[..total_vecs];
    let mut c_rest = &soa.c_bridge.as_f32x4_slice()[..total_vecs];

    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for entry in map.strings() {
        let n = entry.vec_len();
        let (u_str, u_tail) = u_rest.split_at_mut(n);
        let (v_str, v_tail) = v_rest.split_at_mut(n);
        let (d_str, d_tail) = d_rest.split_at(n);
        let (s_str, s_tail) = s_rest.split_at(n);
        let (c_str, c_tail) = c_rest.split_at(n);
        (u_rest, v_rest, d_rest, s_rest, c_rest) = (u_tail, v_tail, d_tail, s_tail, c_tail);

        let f_term = step_string_modes(u_str, v_str, d_str, s_str, c_str);
        sum_l += f_term * entry.pan_l;
        sum_r += f_term * entry.pan_r;
    }
    (sum_l, sum_r)
}

#[inline(always)]
pub(crate) fn accum_string_tile_in<const FULL_TILE: bool>(
    soa: &mut ModalSoa,
    vr: Range<usize>,
    t_len: usize,
    acc: &mut [F32x4; TILE],
) {
    if vr.is_empty() {
        return;
    }
    let u = &mut soa.u.as_f32x4_slice_mut()[vr.clone()];
    let v = &mut soa.v.as_f32x4_slice_mut()[vr.clone()];
    let d = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c = &soa.c_bridge.as_f32x4_slice()[vr];
    let n = u.len();
    assert!(v.len() == n && d.len() == n && s.len() == n && c.len() == n);
    let steps = if FULL_TILE { TILE } else { t_len.min(TILE) };

    for ((((u_ref, v_ref), &d0), &s0), &c0) in u[..n]
        .iter_mut()
        .zip(v[..n].iter_mut())
        .zip(d[..n].iter())
        .zip(s[..n].iter())
        .zip(c[..n].iter())
    {
        let (mut u0, mut v0) = (*u_ref, *v_ref);
        for acc_slot in &mut acc[..steps] {
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            *acc_slot = v0.mul_add(c0, *acc_slot);
        }
        *u_ref = u0;
        *v_ref = v0;
    }
}

/// Register-tiled 8-sample coupled-form ZOH stepper for one string vector slice.
/// Loads each `(u, v, d, s, c)` vector once into SIMD registers across up to `TILE` samples.
#[inline(always)]
pub(crate) fn step_string_tile(
    soa: &mut ModalSoa,
    vr: Range<usize>,
    t_len: usize,
) -> [F32x4; TILE] {
    let mut acc = [F32x4::zero(); TILE];
    if t_len == TILE {
        accum_string_tile_in::<true>(soa, vr, TILE, &mut acc);
    } else {
        accum_string_tile_in::<false>(soa, vr, t_len, &mut acc);
    }
    acc
}

/// Accumulates `acc[..t_len]` reduced terminal outputs into L1 stack `tile_l`/`tile_r`.
#[inline(always)]
pub(crate) fn accum_tile_stereo(
    acc: &[F32x4; TILE],
    pan_l: f32,
    pan_r: f32,
    t_len: usize,
    tile_l: &mut [f32; TILE],
    tile_r: &mut [f32; TILE],
) {
    let steps = t_len.min(TILE);
    for ((acc_vec, tl), tr) in acc[..steps]
        .iter()
        .zip(&mut tile_l[..steps])
        .zip(&mut tile_r[..steps])
    {
        let f_term = acc_vec.reduce_sum();
        *tl += f_term * pan_l;
        *tr += f_term * pan_r;
    }
}

#[inline(always)]
fn flush_f32_slice(slice: &mut [f32]) -> u32 {
    const RANGE: u32 = FINITE_EXP_BITS - SUBNORMAL_FLUSH_BITS;
    #[inline(always)]
    fn flush_f32(x: &mut f32, nz: &mut u32) {
        let bits = x.to_bits();
        let abs_bits = bits & 0x7fff_ffff;
        let keep = abs_bits.wrapping_sub(SUBNORMAL_FLUSH_BITS) < RANGE;
        let kept = bits & 0_u32.wrapping_sub(u32::from(keep));
        *x = f32::from_bits(kept);
        *nz |= kept;
    }
    let (mut nz0, mut nz1, mut nz2, mut nz3) = (0_u32, 0_u32, 0_u32, 0_u32);
    let mut chunks = slice.chunks_exact_mut(4);
    for c in chunks.by_ref() {
        flush_f32(&mut c[0], &mut nz0);
        flush_f32(&mut c[1], &mut nz1);
        flush_f32(&mut c[2], &mut nz2);
        flush_f32(&mut c[3], &mut nz3);
    }
    for x in chunks.into_remainder() {
        flush_f32(x, &mut nz0);
    }
    nz0 | nz1 | nz2 | nz3
}

/// Branchlessly flushes `< 1e-20` and non-finite entries to zero within `vec_range`,
/// returning `true` if any remaining state slot is non-zero.
#[inline]
pub(crate) fn flush_subnormals_range(soa: &mut ModalSoa, vec_range: Range<usize>) -> bool {
    let slot_range = (vec_range.start * 4)..(vec_range.end * 4);
    let nz_u = flush_f32_slice(&mut soa.u.as_mut_slice()[slot_range.clone()]);
    let nz_v = flush_f32_slice(&mut soa.v.as_mut_slice()[slot_range]);
    ((nz_u | nz_v) & 0x7fff_ffff) != 0
}

/// Branchless flush of subnormals across the entire `ModalSoa` buffer.
#[cfg(test)]
pub(crate) fn flush_subnormals_soa(soa: &mut ModalSoa) {
    let total_vecs = soa.u.as_f32x4_slice().len();
    let _ = flush_subnormals_range(soa, 0..total_vecs);
}

/// Branchlessly flushes `< 1e-20` and non-finite entries across raw `(u, v)` slices.
#[inline]
pub(crate) fn flush_subnormals_slices(u: &mut [f32], v: &mut [f32]) -> bool {
    ((flush_f32_slice(u) | flush_f32_slice(v)) & 0x7fff_ffff) != 0
}

#[cfg(test)]
#[path = "zoh_kernel_tests.rs"]
mod tests;
