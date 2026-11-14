//! Continuous branchless SIMD coupled-form ZOH kernel (sample-outer / string-inner).
//!
//! Steps all 230 strings (~26,184 padded modes, ~6,546 `F32x4` vectors) every
//! sample with zero dormant-string skipping and zero heap allocations (D1).

use core::f64::consts::PI;

use piano_simd::{AlignedBuf, F32x4};

#[cfg(test)]
use crate::choir_map::KeyMapEntry;
use crate::choir_map::{ChoirMap, StringMapEntry};
use crate::modal_soa::{sin_cos_f64, ModalSoa};

const TILE: usize = 8;

const SUBNORMAL_FLUSH_BITS: u32 = 0x1e3c_e508; // 1.0e-20_f32.to_bits()
const FINITE_EXP_BITS: u32 = 0x7f80_0000;

/// Precomputed half-sine hammer excitation pulse specification for one key.
#[derive(Copy, Clone, Debug)]
pub struct KeyPulseSpec {
    pub pulse_len: u16,
    pub init_sin: f32,
    pub init_cos: f32,
    pub step_sin: f32,
    pub step_cos: f32,
    pub inv_sum: f32,
}

impl KeyPulseSpec {
    /// Builds a key-period-adapted half-sine pulse (`~0.20 * T0`, clamped to `[1, 2.2 ms]`)
    /// normalized so its sample sum equals `1.0`.
    #[must_use]
    pub fn for_key(f0_hz: f64, sample_rate: f64) -> Self {
        let max_pulse = ((0.0022 * sample_rate + 0.5) as usize).max(1);
        let raw_len = if f0_hz > 0.0 && f0_hz.is_finite() {
            (0.20 * sample_rate / f0_hz + 0.5) as usize
        } else {
            1
        };
        let pulse_len = raw_len.clamp(1, max_pulse);

        if pulse_len == 1 {
            return Self {
                pulse_len: 1,
                init_sin: 1.0,
                init_cos: 0.0,
                step_sin: 0.0,
                step_cos: 1.0,
                inv_sum: 1.0,
            };
        }

        let step = PI / (pulse_len as f64);
        let mut sum = 0.0_f64;
        for n in 0..pulse_len {
            sum += sin_cos_f64((n as f64 + 0.5) * step).0;
        }
        let (s0, c0) = sin_cos_f64(0.5 * step);
        let (ss, cs) = sin_cos_f64(step);
        Self {
            pulse_len: pulse_len as u16,
            init_sin: s0 as f32,
            init_cos: c0 as f32,
            step_sin: ss as f32,
            step_cos: cs as f32,
            inv_sum: (1.0 / sum) as f32,
        }
    }
}

/// Per-string zero-allocation half-sine pulse oscillator.
#[derive(Copy, Clone, Debug, Default)]
pub struct StringPulse {
    pub remaining: u16,
    sin_state: f32,
    cos_state: f32,
    step_sin: f32,
    step_cos: f32,
    gain: f32,
}

impl StringPulse {
    /// Starts a pulse on this string with scalar amplitude `amplitude`.
    #[inline]
    pub fn trigger(&mut self, spec: &KeyPulseSpec, amplitude: f32) {
        if !amplitude.is_finite() || amplitude == 0.0 {
            self.remaining = 0;
            return;
        }
        self.remaining = spec.pulse_len;
        self.sin_state = spec.init_sin;
        self.cos_state = spec.init_cos;
        self.step_sin = spec.step_sin;
        self.step_cos = spec.step_cos;
        self.gain = amplitude * spec.inv_sum;
    }

    /// Advances the pulse by one sample, returning the instantaneous force.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        if self.remaining == 0 {
            return 0.0;
        }
        let out = self.sin_state * self.gain;
        let next_s = self.sin_state * self.step_cos + self.cos_state * self.step_sin;
        let next_c = self.cos_state * self.step_cos - self.sin_state * self.step_sin;
        self.sin_state = next_s;
        self.cos_state = next_c;
        self.remaining -= 1;
        out
    }
}

/// Injects scalar excitation `force` across a string's `F32x4` vector slice:
/// `u[i] += b_exc[i] * force`.
#[inline]
pub fn inject_excitation(u_slice: &mut [F32x4], b_slice: &[F32x4], force: f32) {
    let f_vec = F32x4::splat(force);
    for (u_vec, &b_vec) in u_slice.iter_mut().zip(b_slice.iter()) {
        *u_vec = b_vec.mul_add(f_vec, *u_vec);
    }
}

#[inline(always)]
fn zoh_rotate(u0: F32x4, v0: F32x4, d0: F32x4, s0: F32x4) -> (F32x4, F32x4) {
    let du = d0.mul_add(u0, s0 * v0);
    let dv = s0 * u0 - d0 * v0;
    (u0 - du, v0 + dv)
}

/// Advances one contiguous string vector range by one sample using Exact Coupled-Form
/// Incremental ZOH (4x unrolled for instruction-level parallelism, zero branches).
#[inline(always)]
pub fn step_string_modes(
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
            let (u_next, v_next) = zoh_rotate(uc[lane], vc[lane], dc[lane], sc[lane]);
            uc[lane] = u_next;
            vc[lane] = v_next;
            match lane {
                0 => acc0 = v_next.mul_add(cc[0], acc0),
                1 => acc1 = v_next.mul_add(cc[1], acc1),
                2 => acc2 = v_next.mul_add(cc[2], acc2),
                _ => acc3 = v_next.mul_add(cc[3], acc3),
            }
        }
    }

    for ((((u0_ref, v0_ref), &d0), &s0), &c0) in u_chunks
        .into_remainder()
        .iter_mut()
        .zip(v_chunks.into_remainder().iter_mut())
        .zip(d_chunks.remainder().iter())
        .zip(s_chunks.remainder().iter())
        .zip(c_chunks.remainder().iter())
    {
        let (u_next, v_next) = zoh_rotate(*u0_ref, *v0_ref, d0, s0);
        *u0_ref = u_next;
        *v0_ref = v_next;
        acc0 = v_next.mul_add(c0, acc0);
    }

    ((acc0 + acc1) + (acc2 + acc3)).reduce_sum()
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_tile_inner<const FULL_TILE: bool, const HAS_EXC: bool>(
    u: &mut [F32x4],
    v: &mut [F32x4],
    d: &[F32x4],
    s: &[F32x4],
    c: &[F32x4],
    b: &[F32x4],
    f_tile: &[f32; TILE],
    t_len: usize,
) -> [F32x4; TILE] {
    let n = u.len();
    assert!(v.len() == n && d.len() == n && s.len() == n && c.len() == n);
    if HAS_EXC {
        assert!(b.len() >= n);
    }
    let steps = if FULL_TILE { TILE } else { t_len.min(TILE) };
    let mut acc = [F32x4::zero(); TILE];

    for (idx, ((((u_ref, v_ref), &d0), &s0), &c0)) in u[..n]
        .iter_mut()
        .zip(v[..n].iter_mut())
        .zip(d[..n].iter())
        .zip(s[..n].iter())
        .zip(c[..n].iter())
        .enumerate()
    {
        let (mut u0, mut v0) = (*u_ref, *v_ref);
        let b0 = if HAS_EXC { b[idx] } else { F32x4::zero() };
        for (acc_slot, &f_t) in acc[..steps].iter_mut().zip(&f_tile[..steps]) {
            if HAS_EXC {
                u0 = b0.mul_add(F32x4::splat(f_t), u0);
            }
            (u0, v0) = zoh_rotate(u0, v0, d0, s0);
            *acc_slot = v0.mul_add(c0, *acc_slot);
        }
        *u_ref = u0;
        *v_ref = v0;
    }
    acc
}

/// Register-tiled 8-sample coupled-form ZOH stepper for one string vector slice.
/// Loads each `(u, v, d, s, c[, b])` vector once into SIMD registers across up to `TILE` samples.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn step_string_tile(
    u: &mut [F32x4],
    v: &mut [F32x4],
    d: &[F32x4],
    s: &[F32x4],
    c: &[F32x4],
    b: &[F32x4],
    has_exc: bool,
    f_tile: &[f32; TILE],
    t_len: usize,
) -> [F32x4; TILE] {
    match (t_len == TILE, has_exc) {
        (true, false) => step_tile_inner::<true, false>(u, v, d, s, c, b, f_tile, TILE),
        (true, true) => step_tile_inner::<true, true>(u, v, d, s, c, b, f_tile, TILE),
        (false, false) => step_tile_inner::<false, false>(u, v, d, s, c, b, f_tile, t_len),
        (false, true) => step_tile_inner::<false, true>(u, v, d, s, c, b, f_tile, t_len),
    }
}

/// Steps all 230 strings for one sample unconditionally and returns stereo `(L, R)`.
#[inline]
pub fn step_all_strings(soa: &mut ModalSoa, map: &ChoirMap) -> (f32, f32) {
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
        u_rest = u_tail;
        v_rest = v_tail;
        d_rest = d_tail;
        s_rest = s_tail;
        c_rest = c_tail;

        let f_term = step_string_modes(u_str, v_str, d_str, s_str, c_str);
        sum_l += f_term * entry.pan_l;
        sum_r += f_term * entry.pan_r;
    }
    (sum_l, sum_r)
}

fn step_string_slice_tiles(
    soa: &mut ModalSoa,
    strings: &[StringMapEntry],
    pulses: &mut [StringPulse],
    vec_range: core::ops::Range<usize>,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let frames = out_l.len().min(out_r.len());
    if frames == 0 {
        return;
    }
    assert_eq!(strings.len(), pulses.len());
    let mut any_pulse = pulses.iter().any(|p| p.remaining > 0);
    let (v_start, v_end) = (vec_range.start, vec_range.end);

    let mut base = 0usize;
    while base < frames {
        let t_len = (frames - base).min(TILE);
        let mut tile_l = [0.0_f32; TILE];
        let mut tile_r = [0.0_f32; TILE];
        let mut still_active = false;

        {
            let mut u_rest = &mut soa.u.as_f32x4_slice_mut()[v_start..v_end];
            let mut v_rest = &mut soa.v.as_f32x4_slice_mut()[v_start..v_end];
            let mut d_rest = &soa.d_coeff.as_f32x4_slice()[v_start..v_end];
            let mut s_rest = &soa.s_coeff.as_f32x4_slice()[v_start..v_end];
            let mut c_rest = &soa.c_bridge.as_f32x4_slice()[v_start..v_end];
            let mut b_rest = &soa.b_exc.as_f32x4_slice()[v_start..v_end];

            for (entry, pulse) in strings.iter().zip(pulses.iter_mut()) {
                let n = entry.vec_len();
                let (u_str, u_tail) = u_rest.split_at_mut(n);
                let (v_str, v_tail) = v_rest.split_at_mut(n);
                let (d_str, d_tail) = d_rest.split_at(n);
                let (s_str, s_tail) = s_rest.split_at(n);
                let (c_str, c_tail) = c_rest.split_at(n);
                let (b_str, b_tail) = b_rest.split_at(n);
                (u_rest, v_rest, d_rest) = (u_tail, v_tail, d_tail);
                (s_rest, c_rest, b_rest) = (s_tail, c_tail, b_tail);

                let mut f_tile = [0.0_f32; TILE];
                let has_exc = if any_pulse && pulse.remaining > 0 {
                    let steps = t_len.min(pulse.remaining as usize);
                    for slot in &mut f_tile[..steps] {
                        *slot = pulse.next_sample();
                    }
                    still_active |= pulse.remaining > 0;
                    true
                } else {
                    false
                };

                let acc = step_string_tile(
                    u_str, v_str, d_str, s_str, c_str, b_str, has_exc, &f_tile, t_len,
                );
                for ((acc_vec, tl), tr) in acc[..t_len]
                    .iter()
                    .zip(&mut tile_l[..t_len])
                    .zip(&mut tile_r[..t_len])
                {
                    let f_term = acc_vec.reduce_sum();
                    *tl += f_term * entry.pan_l;
                    *tr += f_term * entry.pan_r;
                }
            }
        }

        any_pulse = still_active;
        out_l[base..base + t_len].copy_from_slice(&tile_l[..t_len]);
        out_r[base..base + t_len].copy_from_slice(&tile_r[..t_len]);
        base += t_len;
    }

    flush_subnormals_soa(soa);
}

/// Renders a subblock using 8-sample register tiling over all 230 strings,
/// advancing active string pulses and flushing subnormals once at block end.
pub fn render_subblock_core(
    soa: &mut ModalSoa,
    map: &ChoirMap,
    pulses: &mut [StringPulse],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    step_string_slice_tiles(
        soa,
        map.strings(),
        pulses,
        0..map.total_vecs(),
        out_l,
        out_r,
    );
}

/// Branchless flush of subnormals (`|x| < 1e-20`) and non-finite lanes in `u` and `v`.
#[inline]
pub fn flush_subnormals_soa(soa: &mut ModalSoa) {
    flush_subnormals_buf(&mut soa.u);
    flush_subnormals_buf(&mut soa.v);
}

#[inline]
fn flush_subnormals_buf(buf: &mut AlignedBuf) {
    const RANGE: u32 = FINITE_EXP_BITS - SUBNORMAL_FLUSH_BITS;
    for x in buf.as_mut_slice().iter_mut() {
        let bits = x.to_bits();
        let abs_bits = bits & 0x7fff_ffff;
        let keep = abs_bits.wrapping_sub(SUBNORMAL_FLUSH_BITS) < RANGE;
        let mask = 0_u32.wrapping_sub(u32::from(keep));
        *x = f32::from_bits(bits & mask);
    }
}

/// Single-key subblock stepper sharing `step_string_tile` on `soa` slices.
#[cfg(test)]
pub(crate) fn step_key_subblock(
    soa: &mut ModalSoa,
    key: &KeyMapEntry,
    strings: &[StringMapEntry],
    pulses: &mut [StringPulse],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let (sr, vr) = (key.string_range.clone(), key.vec_range.clone());
    step_string_slice_tiles(soa, &strings[sr.clone()], &mut pulses[sr], vr, out_l, out_r);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StringBank;
    use piano_simd::{Complex, RealFft};
    use piano_spec::{build_scale_design, key_tuned_f0_hz, VoicingSpec, TOTAL_STRINGS};

    #[test]
    fn tiled_kernel_matches_single_sample_across_partial_tiles_and_pulses() {
        const FRAMES: usize = 13; // 1 full tile (8) + 1 partial tail (5)
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);
        let map = ChoirMap::build_with_specs(&specs, &voicing);
        let mut tiled_soa = ModalSoa::build(48_000.0, &voicing, &specs, &map);
        let mut ref_soa = tiled_soa.clone();
        let mut tiled_pulses = [StringPulse::default(); TOTAL_STRINGS];
        let mut ref_pulses = [StringPulse::default(); TOTAL_STRINGS];
        for (k, str_idx) in [(0usize, 0usize), (87, 229)] {
            let r = map.keys()[k].vec_range.clone();
            tiled_soa.set_vec_range_hold(r.clone(), 1.0);
            ref_soa.set_vec_range_hold(r, 1.0);
            let spec = KeyPulseSpec::for_key(key_tuned_f0_hz(k, &specs, &voicing), 48_000.0);
            tiled_pulses[str_idx].trigger(&spec, 1.0);
            ref_pulses[str_idx].trigger(&spec, 1.0);
        }

        let (mut tl, mut tr) = (vec![0.0_f32; FRAMES], vec![0.0_f32; FRAMES]);
        render_subblock_core(&mut tiled_soa, &map, &mut tiled_pulses, &mut tl, &mut tr);

        for f in 0..FRAMES {
            for (entry, pulse) in map.strings().iter().zip(ref_pulses.iter_mut()) {
                if pulse.remaining > 0 {
                    let amp = pulse.next_sample();
                    let r = entry.all_vecs();
                    let u_s = &mut ref_soa.u.as_f32x4_slice_mut()[r.clone()];
                    inject_excitation(u_s, &ref_soa.b_exc.as_f32x4_slice()[r], amp);
                }
            }
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
        bank.set_key_hold(48, 1.0);
        bank.trigger_key_pulse(48, [1.0; 3], 1.0);
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
