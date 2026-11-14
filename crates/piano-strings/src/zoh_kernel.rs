//! Coupled-form ZOH modal stepping kernel (`F32x4` SIMD, 8-sample L1-tiled).
//!
//! Loop order places samples inside L1-cached mode tiles (never more than `TILE`
//! samples per pass) while iterating across all 230 strings unconditionally every
//! sample (D1 invariant: zero voice-stealing or dormant-string skipping).

use core::ops::Range;

use piano_simd::F32x4;

use crate::choir_map::ChoirMap;
#[cfg(test)]
use crate::choir_map::{KeyMapEntry, StringMapEntry};
use crate::modal_soa::ModalSoa;

/// Sample tile width kept resident across string/mode SIMD passes.
pub(crate) const TILE: usize = 8;

/// Magnitude threshold below which state variables flush to exact zero.
pub(crate) const SUBNORMAL_FLOOR: f32 = 1.0e-20;

/// Coupled-form exact incremental ZOH rotation on one 4-mode `F32x4` lane.
///
/// Advances `(u, v)` by pole `(1 - d) + i * s` without `2.0`-adjacent rounding:
/// `u_next = u - (d * u + s * v)` and `v_next = v + (s * u - d * v)`.
#[inline(always)]
pub(crate) fn zoh_rotate(u: F32x4, v: F32x4, d: F32x4, s: F32x4) -> (F32x4, F32x4) {
    let u_next = u - d.mul_add(u, s * v);
    let v_next = v + s.mul_add(u, F32x4::zero() - (d * v));
    (u_next, v_next)
}

/// Injects scalar excitation impulse `force` into `u_slice` scaled by `b_slice`.
#[inline]
pub(crate) fn inject_excitation(u_slice: &mut [F32x4], b_slice: &[F32x4], force: f32) {
    let f_v = F32x4::splat(force);
    for (u, &b) in u_slice.iter_mut().zip(b_slice.iter()) {
        *u = f_v.mul_add(b, *u);
    }
}

/// Advances one contiguous vector range `[F32x4]` by 1 sample and returns `sum(v_next * c)`.
#[inline]
pub(crate) fn step_string_modes(
    u_slice: &mut [F32x4],
    v_slice: &mut [F32x4],
    d_slice: &[F32x4],
    s_slice: &[F32x4],
    c_slice: &[F32x4],
) -> f32 {
    let mut acc = F32x4::zero();
    for ((((u, v), &d), &s), &c) in u_slice
        .iter_mut()
        .zip(v_slice.iter_mut())
        .zip(d_slice.iter())
        .zip(s_slice.iter())
        .zip(c_slice.iter())
    {
        let (u_next, v_next) = zoh_rotate(*u, *v, d, s);
        *u = u_next;
        *v = v_next;
        acc = v_next.mul_add(c, acc);
    }
    acc.reduce_sum()
}

/// Steps all 230 strings by exactly 1 sample without allocations or branching.
pub(crate) fn step_all_strings(soa: &mut ModalSoa, map: &ChoirMap) -> (f32, f32) {
    let u_all = soa.u.as_f32x4_slice_mut();
    let v_all = soa.v.as_f32x4_slice_mut();
    let d_all = soa.d_coeff.as_f32x4_slice();
    let s_all = soa.s_coeff.as_f32x4_slice();
    let c_all = soa.c_bridge.as_f32x4_slice();

    let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
    for entry in map.strings() {
        let vr = entry.all_vecs();
        let term = step_string_modes(
            &mut u_all[vr.clone()],
            &mut v_all[vr.clone()],
            &d_all[vr.clone()],
            &s_all[vr.clone()],
            &c_all[vr],
        );
        sum_l += term * entry.pan_l;
        sum_r += term * entry.pan_r;
    }
    (sum_l, sum_r)
}

/// Steps one string's vector range across `out_l.len() <= TILE` samples in L1 cache.
#[inline]
pub(crate) fn step_string_tile(
    soa: &mut ModalSoa,
    vr: Range<usize>,
    pan_l: f32,
    pan_r: f32,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let t_len = out_l.len().min(out_r.len());
    debug_assert!(t_len <= TILE);
    if t_len == 0 {
        return;
    }

    let u_s = &mut soa.u.as_f32x4_slice_mut()[vr.clone()];
    let v_s = &mut soa.v.as_f32x4_slice_mut()[vr.clone()];
    let d_s = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s_s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c_s = &soa.c_bridge.as_f32x4_slice()[vr];

    let mut acc = [F32x4::zero(); TILE];
    for i in 0..u_s.len() {
        let (mut u, mut v) = (u_s[i], v_s[i]);
        let (d, s, c) = (d_s[i], s_s[i], c_s[i]);
        for acc_t in &mut acc[..t_len] {
            let (un, vn) = zoh_rotate(u, v, d, s);
            u = un;
            v = vn;
            *acc_t = vn.mul_add(c, *acc_t);
        }
        u_s[i] = u;
        v_s[i] = v;
    }

    for t in 0..t_len {
        let term = acc[t].reduce_sum();
        out_l[t] += term * pan_l;
        out_r[t] += term * pan_r;
    }
}

/// Flushes `< SUBNORMAL_FLOOR` state entries to zero within `vec_range`, returning
/// `true` if any remaining state sample is strictly non-zero.
#[inline]
pub(crate) fn flush_subnormals_range(soa: &mut ModalSoa, vec_range: Range<usize>) -> bool {
    let slot_range = (vec_range.start * 4)..(vec_range.end * 4);
    let mut any_nonzero = false;
    for x in &mut soa.u.as_mut_slice()[slot_range.clone()] {
        if x.abs() < SUBNORMAL_FLOOR {
            *x = 0.0;
        } else {
            any_nonzero = true;
        }
    }
    for x in &mut soa.v.as_mut_slice()[slot_range] {
        if x.abs() < SUBNORMAL_FLOOR {
            *x = 0.0;
        } else {
            any_nonzero = true;
        }
    }
    any_nonzero
}

/// Flushes subnormals across the entire `ModalSoa` buffer.
pub(crate) fn flush_subnormals_soa(soa: &mut ModalSoa) {
    let total_vecs = soa.u.as_f32x4_slice().len();
    let _ = flush_subnormals_range(soa, 0..total_vecs);
}

/// Steps all 230 strings across an uncoupled audio subblock in `TILE`-sample tiles.
pub(crate) fn step_all_tiles(
    soa: &mut ModalSoa,
    map: &ChoirMap,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    out_l[..total].fill(0.0);
    out_r[..total].fill(0.0);

    let mut base = 0usize;
    while base < total {
        let t_len = (total - base).min(TILE);
        let l_tile = &mut out_l[base..base + t_len];
        let r_tile = &mut out_r[base..base + t_len];
        for entry in map.strings() {
            step_string_tile(
                soa,
                entry.all_vecs(),
                entry.pan_l,
                entry.pan_r,
                l_tile,
                r_tile,
            );
        }
        base += t_len;
    }
    flush_subnormals_soa(soa);
}

#[cfg(test)]
pub(crate) fn step_key_subblock(
    soa: &mut ModalSoa,
    key: &KeyMapEntry,
    strings: &[StringMapEntry],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    out_l[..total].fill(0.0);
    out_r[..total].fill(0.0);

    let key_strings = &strings[key.string_range.clone()];
    let mut base = 0usize;
    while base < total {
        let t_len = (total - base).min(TILE);
        let l_tile = &mut out_l[base..base + t_len];
        let r_tile = &mut out_r[base..base + t_len];
        for entry in key_strings {
            step_string_tile(
                soa,
                entry.all_vecs(),
                entry.pan_l,
                entry.pan_r,
                l_tile,
                r_tile,
            );
        }
        base += t_len;
    }
    let _ = flush_subnormals_range(soa, key.vec_range.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StringBank;
    use piano_simd::{Complex, RealFft};
    use piano_spec::{build_scale_design, VoicingSpec};

    #[test]
    fn tiled_kernel_matches_single_sample_across_partial_tiles() {
        const FRAMES: usize = 13; // 1 full tile (8) + 1 partial tail (5)
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);
        let map = ChoirMap::build_with_specs(&specs, &voicing);
        let mut tiled_soa = ModalSoa::build(48_000.0, &voicing, &specs, &map);
        let mut ref_soa = tiled_soa.clone();

        for s_idx in [0usize, 120, 229] {
            let r = map.strings()[s_idx].all_vecs();
            inject_excitation(
                &mut tiled_soa.u.as_f32x4_slice_mut()[r.clone()],
                &tiled_soa.b_exc.as_f32x4_slice()[r.clone()],
                1.0,
            );
            inject_excitation(
                &mut ref_soa.u.as_f32x4_slice_mut()[r.clone()],
                &ref_soa.b_exc.as_f32x4_slice()[r],
                1.0,
            );
        }

        let (mut tl, mut tr) = (vec![0.0_f32; FRAMES], vec![0.0_f32; FRAMES]);
        step_all_tiles(&mut tiled_soa, &map, &mut tl, &mut tr);

        for f in 0..FRAMES {
            let (rl, rr) = step_all_strings(&mut ref_soa, &map);
            assert!((tl[f] - rl).abs() < 1e-6 && (tr[f] - rr).abs() < 1e-6);
        }
        flush_subnormals_soa(&mut ref_soa);
        for (a, b) in tiled_soa
            .u
            .as_mut_slice()
            .iter()
            .zip(ref_soa.u.as_mut_slice())
        {
            assert!((*a - *b).abs() < 1e-6);
        }
    }

    #[test]
    fn a4_fundamental_dominates_second_partial() {
        const N: usize = 32_768;
        let mut bank = StringBank::new(48_000.0, &VoicingSpec::default());
        bank.apply_excitation_sample(48, [1.0; 3], 1.0);
        let (mut out_l, mut out_r) = (vec![0.0_f32; N], vec![0.0_f32; N]);
        bank.render_key_subblock(48, &mut out_l, &mut out_r);
        let mono: Vec<f32> = out_l.iter().zip(&out_r).map(|(l, r)| l + r).collect();
        let mut rfft = RealFft::new(N);
        let mut spec = vec![Complex::ZERO; rfft.spectrum_len()];
        rfft.forward(&mono, &mut spec);

        let max_bin = (1..spec.len())
            .max_by(|&a, &b| spec[a].norm_sq().total_cmp(&spec[b].norm_sq()))
            .unwrap();
        let peak_hz = (max_bin as f64) * 48_000.0 / (N as f64);
        let band_max = |lo: usize, hi: usize| {
            spec[lo..=hi]
                .iter()
                .fold(0.0_f32, |m, c| m.max(c.norm_sq()))
        };
        let db_margin = 10.0 * (band_max(290, 310) / band_max(590, 615).max(1e-20)).log10();
        eprintln!("A4 peak_hz={peak_hz:.2} Hz, fund_vs_2nd={db_margin:.2} dB");
        assert!((435.0..=445.0).contains(&peak_hz) && db_margin > 3.9);
    }
}
