//! Subblock L1-resident ZOH stepping kernels (`CHUNK = 64`/`128`, `no_std`).

use core::ops::Range;

use piano_simd::F32x4;

use crate::modal_soa::ModalSoa;
use crate::zoh_kernel::zoh_rotate;
#[cfg(test)]
use crate::{
    choir_map::ChoirMap,
    zoh_kernel::{accum_tile_stereo, flush_subnormals_soa, step_string_tile, TILE},
};

/// Advances one linear string across an entire subblock in `CHUNK = 128` L1-resident passes
/// with 2-vector interleaved register rotation (`16` XMM registers, single-pass `ModalSoa` access).
#[inline]
pub(crate) fn step_string_subblock(
    soa: &mut ModalSoa,
    vr: Range<usize>,
    pan_l: f32,
    pan_r: f32,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    step_string_subblock_3(soa, [vr, 0..0, 0..0], pan_l, pan_r, out_l, out_r);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn step_ranges_chunk<const CHUNK: usize>(
    u: &mut [F32x4],
    v: &mut [F32x4],
    d: &[F32x4],
    s: &[F32x4],
    c: &[F32x4],
    ranges: [Range<usize>; 3],
    pan_l: f32,
    pan_r: f32,
    out_l: &mut [f32],
    out_r: &mut [f32],
    total: usize,
) {
    let mut base = 0usize;
    while base < total {
        let c_len = (total - base).min(CHUNK);
        let mut acc = [F32x4::zero(); CHUNK];
        let full4 = c_len & !3;
        for r in &ranges {
            let mut idx = r.start;
            let end = r.end;
            while idx + 1 < end {
                let (mut u0, mut v0) = (u[idx], v[idx]);
                let (d0, s0, c0) = (d[idx], s[idx], c[idx]);
                let (mut u1, mut v1) = (u[idx + 1], v[idx + 1]);
                let (d1, s1, c1) = (d[idx + 1], s[idx + 1], c[idx + 1]);
                for chunk in acc[..full4].chunks_exact_mut(4) {
                    let (mut a0, mut a1, mut a2, mut a3) = (chunk[0], chunk[1], chunk[2], chunk[3]);
                    (u0, v0) = zoh_rotate(u0, v0, d0, s0);
                    (u1, v1) = zoh_rotate(u1, v1, d1, s1);
                    a0 = v0.mul_add(c0, v1.mul_add(c1, a0));
                    (u0, v0) = zoh_rotate(u0, v0, d0, s0);
                    (u1, v1) = zoh_rotate(u1, v1, d1, s1);
                    a1 = v0.mul_add(c0, v1.mul_add(c1, a1));
                    (u0, v0) = zoh_rotate(u0, v0, d0, s0);
                    (u1, v1) = zoh_rotate(u1, v1, d1, s1);
                    a2 = v0.mul_add(c0, v1.mul_add(c1, a2));
                    (u0, v0) = zoh_rotate(u0, v0, d0, s0);
                    (u1, v1) = zoh_rotate(u1, v1, d1, s1);
                    a3 = v0.mul_add(c0, v1.mul_add(c1, a3));
                    (chunk[0], chunk[1], chunk[2], chunk[3]) = (a0, a1, a2, a3);
                }
                for slot in &mut acc[full4..c_len] {
                    (u0, v0) = zoh_rotate(u0, v0, d0, s0);
                    (u1, v1) = zoh_rotate(u1, v1, d1, s1);
                    *slot = v0.mul_add(c0, v1.mul_add(c1, *slot));
                }
                u[idx] = u0;
                v[idx] = v0;
                u[idx + 1] = u1;
                v[idx + 1] = v1;
                idx += 2;
            }
            if idx < end {
                let (mut u0, mut v0) = (u[idx], v[idx]);
                let (d0, s0, c0) = (d[idx], s[idx], c[idx]);
                for slot in &mut acc[..c_len] {
                    (u0, v0) = zoh_rotate(u0, v0, d0, s0);
                    *slot = v0.mul_add(c0, *slot);
                }
                u[idx] = u0;
                v[idx] = v0;
            }
        }
        for ((a, l), r) in acc[..c_len]
            .iter()
            .zip(&mut out_l[base..base + c_len])
            .zip(&mut out_r[base..base + c_len])
        {
            let f = a.reduce_sum();
            *l += f * pan_l;
            *r += f * pan_r;
        }
        base += c_len;
    }
}

/// Advances up to three modal ranges of one linear string across `out_l`/`out_r` in L1-resident
/// passes (`CHUNK = 64` or `128`), accumulating stereo once per sample across all ranges.
#[inline(never)]
pub(crate) fn step_string_subblock_3(
    soa: &mut ModalSoa,
    ranges: [Range<usize>; 3],
    pan_l: f32,
    pan_r: f32,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    if total == 0 || ranges.iter().all(Range::is_empty) {
        return;
    }
    let (u, v, d, s, c) = (
        soa.u.as_f32x4_slice_mut(),
        soa.v.as_f32x4_slice_mut(),
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    let min_len = u.len().min(v.len()).min(d.len()).min(s.len()).min(c.len());
    for r in &ranges {
        assert!(r.start <= r.end && r.end <= min_len);
    }

    if total <= 64 {
        step_ranges_chunk::<64>(u, v, d, s, c, ranges, pan_l, pan_r, out_l, out_r, total);
    } else {
        step_ranges_chunk::<128>(u, v, d, s, c, ranges, pan_l, pan_r, out_l, out_r, total);
    }
}

/// Accumulates ZOH-stepped `v_next * c` of `vr` into caller-provided `acc[..c_len]` (`<= 128`).
#[inline(always)]
pub(crate) fn accum_soa_range_in(soa: &mut ModalSoa, vr: Range<usize>, acc: &mut [F32x4]) {
    let c_len = acc.len();
    if c_len == 0 || vr.is_empty() {
        return;
    }
    let (u, v, d, s, c) = (
        &mut soa.u.as_f32x4_slice_mut()[vr.clone()],
        &mut soa.v.as_f32x4_slice_mut()[vr.clone()],
        &soa.d_coeff.as_f32x4_slice()[vr.clone()],
        &soa.s_coeff.as_f32x4_slice()[vr.clone()],
        &soa.c_bridge.as_f32x4_slice()[vr],
    );
    let n = u.len();
    assert!(v.len() == n && d.len() == n && s.len() == n && c.len() == n);
    let full4 = c_len & !3;
    let mut idx = 0usize;
    while idx + 2 < n {
        let (mut u0, mut v0, d0, s0, c0) = (u[idx], v[idx], d[idx], s[idx], c[idx]);
        let (mut u1, mut v1, d1, s1, c1) =
            (u[idx + 1], v[idx + 1], d[idx + 1], s[idx + 1], c[idx + 1]);
        let (mut u2, mut v2, d2, s2, c2) =
            (u[idx + 2], v[idx + 2], d[idx + 2], s[idx + 2], c[idx + 2]);
        for chunk in acc[..full4].chunks_exact_mut(4) {
            let (mut a0, mut a1, mut a2, mut a3) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            (u1, v1) = zoh_rotate(u1, v1, d1, s1);
            (u2, v2) = zoh_rotate(u2, v2, d2, s2);
            let m01_0 = v0.mul_add(c0, v1 * c1);
            a0 = v2.mul_add(c2, a0) + m01_0;
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            (u1, v1) = zoh_rotate(u1, v1, d1, s1);
            (u2, v2) = zoh_rotate(u2, v2, d2, s2);
            let m01_1 = v0.mul_add(c0, v1 * c1);
            a1 = v2.mul_add(c2, a1) + m01_1;
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            (u1, v1) = zoh_rotate(u1, v1, d1, s1);
            (u2, v2) = zoh_rotate(u2, v2, d2, s2);
            let m01_2 = v0.mul_add(c0, v1 * c1);
            a2 = v2.mul_add(c2, a2) + m01_2;
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            (u1, v1) = zoh_rotate(u1, v1, d1, s1);
            (u2, v2) = zoh_rotate(u2, v2, d2, s2);
            let m01_3 = v0.mul_add(c0, v1 * c1);
            a3 = v2.mul_add(c2, a3) + m01_3;
            (chunk[0], chunk[1], chunk[2], chunk[3]) = (a0, a1, a2, a3);
        }
        for slot in &mut acc[full4..c_len] {
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            (u1, v1) = zoh_rotate(u1, v1, d1, s1);
            (u2, v2) = zoh_rotate(u2, v2, d2, s2);
            let m01 = v0.mul_add(c0, v1 * c1);
            *slot = v2.mul_add(c2, *slot) + m01;
        }
        u[idx] = u0;
        v[idx] = v0;
        u[idx + 1] = u1;
        v[idx + 1] = v1;
        u[idx + 2] = u2;
        v[idx + 2] = v2;
        idx += 3;
    }
    while idx < n {
        let (mut u0, mut v0, d0, s0, c0) = (u[idx], v[idx], d[idx], s[idx], c[idx]);
        for chunk in acc[..full4].chunks_exact_mut(4) {
            let (mut a0, mut a1, mut a2, mut a3) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            a0 = v0.mul_add(c0, a0);
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            a1 = v0.mul_add(c0, a1);
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            a2 = v0.mul_add(c0, a2);
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            a3 = v0.mul_add(c0, a3);
            (chunk[0], chunk[1], chunk[2], chunk[3]) = (a0, a1, a2, a3);
        }
        for slot in &mut acc[full4..c_len] {
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            *slot = v0.mul_add(c0, *slot);
        }
        (u[idx], v[idx]) = (u0, v0);
        idx += 1;
    }
}

/// Steps all 230 strings across an uncoupled audio subblock in `TILE`-sample tiles.
#[cfg(test)]
pub(crate) fn step_all_tiles(
    soa: &mut ModalSoa,
    map: &ChoirMap,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    let mut base = 0usize;
    while base < total {
        let t_len = (total - base).min(TILE);
        let mut tile_l = [0.0_f32; TILE];
        let mut tile_r = [0.0_f32; TILE];
        for entry in map.strings() {
            let acc = step_string_tile(soa, entry.all_vecs(), t_len);
            accum_tile_stereo(
                &acc,
                entry.pan_l,
                entry.pan_r,
                t_len,
                &mut tile_l,
                &mut tile_r,
            );
        }
        out_l[base..base + t_len].copy_from_slice(&tile_l[..t_len]);
        out_r[base..base + t_len].copy_from_slice(&tile_r[..t_len]);
        base += t_len;
    }
    flush_subnormals_soa(soa);
}
