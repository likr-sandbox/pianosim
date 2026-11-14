//! Continuous branchless SIMD coupled-form ZOH kernel (sample-outer / string-inner).
//!
//! Steps all 230 strings (~26,184 padded modes, ~6,546 `F32x4` vectors) every
//! sample with zero dormant-string skipping and zero heap allocations (D1).

use core::f64::consts::PI;

use piano_simd::{AlignedBuf, F32x4};

use crate::choir_map::ChoirMap;
#[cfg(test)]
use crate::choir_map::{KeyMapEntry, StringMapEntry};
use crate::modal_soa::{sin_cos_f64, ModalSoa};

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

/// Advances one contiguous string vector range by one sample using Exact Coupled-Form
/// Incremental ZOH (4x unrolled for instruction-level parallelism, zero branches):
///
/// ```text
/// u_next = u0 - (d * u0 + s * v)
/// v_next = v0 + (s * u0 - d * v0)
/// f_term = sum(v_next * c_bridge)
/// ```
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
    let mut acc0 = zero;
    let mut acc1 = zero;
    let mut acc2 = zero;
    let mut acc3 = zero;

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
            let u0 = uc[lane];
            let v0 = vc[lane];
            let d0 = dc[lane];
            let s0 = sc[lane];
            let u_next = u0 - d0.mul_add(u0, s0 * v0);
            let v_next = v0 + s0.mul_add(u0, zero - d0 * v0);
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

    let u_rem = u_chunks.into_remainder();
    let v_rem = v_chunks.into_remainder();
    let d_rem = d_chunks.remainder();
    let s_rem = s_chunks.remainder();
    let c_rem = c_chunks.remainder();

    for ((((u0_ref, v0_ref), &d0), &s0), &c0) in u_rem
        .iter_mut()
        .zip(v_rem.iter_mut())
        .zip(d_rem.iter())
        .zip(s_rem.iter())
        .zip(c_rem.iter())
    {
        let u0 = *u0_ref;
        let v0 = *v0_ref;
        let u_next = u0 - d0.mul_add(u0, s0 * v0);
        let v_next = v0 + s0.mul_add(u0, zero - d0 * v0);
        *u0_ref = u_next;
        *v0_ref = v_next;
        acc0 = v_next.mul_add(c0, acc0);
    }

    ((acc0 + acc1) + (acc2 + acc3)).reduce_sum()
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

    let mut sum_l = 0.0_f32;
    let mut sum_r = 0.0_f32;

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

/// Renders a subblock (sample-outer, string/mode-inner) over all 230 strings,
/// advancing any active string pulses and flushing subnormals once at block end.
pub fn render_subblock_core(
    soa: &mut ModalSoa,
    map: &ChoirMap,
    pulses: &mut [StringPulse],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let frames = out_l.len().min(out_r.len());
    let strings = map.strings();
    let total_vecs = map.total_vecs();
    let mut any_pulse = pulses.iter().any(|p| p.remaining > 0);

    for i in 0..frames {
        if any_pulse {
            let mut still_active = false;
            let u_vecs = &mut soa.u.as_f32x4_slice_mut()[..total_vecs];
            let b_vecs = &soa.b_exc.as_f32x4_slice()[..total_vecs];
            for (entry, pulse) in strings.iter().zip(pulses.iter_mut()) {
                if pulse.remaining > 0 {
                    let f_str = pulse.next_sample();
                    let range = entry.all_vecs();
                    inject_excitation(&mut u_vecs[range.clone()], &b_vecs[range], f_str);
                    still_active |= pulse.remaining > 0;
                }
            }
            any_pulse = still_active;
        }

        let (sum_l, sum_r) = step_all_strings(soa, map);
        out_l[i] = sum_l;
        out_r[i] = sum_r;
    }

    flush_subnormals_soa(soa);
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

/// Single-key subblock stepper sharing the exact `inject_excitation` and
/// `step_string_modes` kernel on `soa` slices (for fast unoptimized unit tests).
#[cfg(test)]
pub(crate) fn step_key_subblock(
    soa: &mut ModalSoa,
    key: &KeyMapEntry,
    strings: &[StringMapEntry],
    pulses: &mut [StringPulse],
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let frames = out_l.len().min(out_r.len());
    let key_strings = &strings[key.string_range.clone()];
    let key_pulses = &mut pulses[key.string_range.clone()];
    let mut any_pulse = key_pulses.iter().any(|p| p.remaining > 0);

    for i in 0..frames {
        if any_pulse {
            let mut still_active = false;
            let u_vecs = soa.u.as_f32x4_slice_mut();
            let b_vecs = soa.b_exc.as_f32x4_slice();
            for (entry, pulse) in key_strings.iter().zip(key_pulses.iter_mut()) {
                if pulse.remaining > 0 {
                    let f_str = pulse.next_sample();
                    let range = entry.all_vecs();
                    inject_excitation(&mut u_vecs[range.clone()], &b_vecs[range], f_str);
                    still_active |= pulse.remaining > 0;
                }
            }
            any_pulse = still_active;
        }

        let mut sum_l = 0.0_f32;
        let mut sum_r = 0.0_f32;
        let u_vecs = soa.u.as_f32x4_slice_mut();
        let v_vecs = soa.v.as_f32x4_slice_mut();
        let d_vecs = soa.d_coeff.as_f32x4_slice();
        let s_vecs = soa.s_coeff.as_f32x4_slice();
        let c_vecs = soa.c_bridge.as_f32x4_slice();

        for entry in key_strings {
            let r = entry.all_vecs();
            let f_term = step_string_modes(
                &mut u_vecs[r.clone()],
                &mut v_vecs[r.clone()],
                &d_vecs[r.clone()],
                &s_vecs[r.clone()],
                &c_vecs[r],
            );
            sum_l += f_term * entry.pan_l;
            sum_r += f_term * entry.pan_r;
        }
        out_l[i] = sum_l;
        out_r[i] = sum_r;
    }

    flush_subnormals_soa(soa);
}
