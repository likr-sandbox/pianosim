//! Short-string hammer-to-agraffe reflection delay line operating at `8x` action rate
//! (`384 kHz` @ `48 kHz`, `768 kHz` @ `96 kHz`) to produce physical multi-peak `F_h(t)`
//! strike-force ripples without double-counting host-rate `ModalSoa` low-frequency feedback.

extern crate alloc;

use alloc::boxed::Box;
use core::f64::consts::{LN_2, TAU};

use piano_spec::{
    equal_temperament_hz_with_a4, key_tuned_f0_hz, StringPhysicalParams, VoicingSpec,
    A4_FREQ_MIN_HZ, CHOIR_LAYOUTS, NUM_KEYS,
};

use crate::math::exp_f64;

const AGRAFFE_REFL_GAIN: f64 = 0.85;
const AGRAFFE_LP_CUTOFF_HZ: f64 = 10_500.0;
const AGRAFFE_LEAK_TAU_S: f64 = 20.0e-6;
const DIRECT_FRAC_MAX: f32 = 0.25;
const OVERSAMPLE_FACTOR: f64 = 8.0;
const WORST_CASE_PITCH_DROP_OCT: f64 = -2.0 / 12.0;

/// Relative strike-position ratio `beta(k)` along the speaking length from the agraffe.
#[must_use]
#[inline]
pub(crate) fn hammer_beta_for_key(key_idx: usize) -> f64 {
    let x = (key_idx.min(NUM_KEYS - 1) as f64) / ((NUM_KEYS - 1) as f64);
    0.138 - 0.043 * x
}

/// Pre-allocated variable-length per-key `8x` short-string agraffe reflection delay line.
#[derive(Clone, Debug)]
pub struct AgraffeDelayLine {
    buf: Box<[f32]>,
    write_idx: usize,
    cap: usize,
    delay_substeps: f32,
    wave_admittance: f32,
    direct_frac: f32,
    f_lp: f32,
    lp_alpha: f32,
    y_agg_m: f32,
    leak: f32,
}

impl AgraffeDelayLine {
    /// Pre-allocates the ring buffer for worst-case `tau_agg = beta / f0_min` at `8x * sample_rate`
    /// (covering `A4 = 392 Hz` + negative stretch/temperament offsets at `48 kHz` and `96 kHz`).
    #[must_use]
    pub fn new(
        key_idx: usize,
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
    ) -> Self {
        let k = key_idx.min(NUM_KEYS - 1);
        let sr_safe = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let f0_min = (equal_temperament_hz_with_a4(k, A4_FREQ_MIN_HZ)
            * exp_f64(WORST_CASE_PITCH_DROP_OCT * LN_2))
        .max(1.0);
        let sr_sub = OVERSAMPLE_FACTOR * sr_safe;
        let max_delay_sub = (((hammer_beta_for_key(k) / f0_min) * sr_sub) as usize) + 1;
        let cap = (max_delay_sub + 4).max(8);
        let buf = alloc::vec![0.0_f32; cap].into_boxed_slice();

        let mut line = Self {
            buf,
            write_idx: 0,
            cap,
            delay_substeps: 1.0,
            wave_admittance: 0.0,
            direct_frac: 0.0,
            f_lp: 0.0,
            lp_alpha: 0.0,
            y_agg_m: 0.0,
            leak: 1.0,
        };
        line.update_voicing(k, sr_safe, voicing, specs);
        line
    }

    /// Recomputes `tau_agg`, choir wave admittance, and high-pass residual filter coefficients.
    pub fn update_voicing(
        &mut self,
        key_idx: usize,
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
    ) {
        let k = key_idx.min(NUM_KEYS - 1);
        let sr_safe = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let f0 = key_tuned_f0_hz(k, specs, voicing).max(1.0);
        let tau_agg = hammer_beta_for_key(k) / f0;
        let max_delay = (self.cap.saturating_sub(2)).max(1) as f32;
        self.delay_substeps =
            ((tau_agg * OVERSAMPLE_FACTOR * sr_safe) as f32).clamp(1.0, max_delay);

        let n_str = f64::from(CHOIR_LAYOUTS[k].strings_per_key.max(1));
        let m_str = specs[k].linear_density_kg_m * specs[k].speaking_length_m;
        let z_choir = (n_str * 2.0 * f0 * m_str).max(1.0e-3);
        self.wave_admittance = (AGRAFFE_REFL_GAIN / (2.0 * z_choir)) as f32;
        self.direct_frac = (((k as f32) - 51.0) / 36.0).clamp(0.0, 1.0) * DIRECT_FRAC_MAX;

        let dt_sub = 1.0 / (OVERSAMPLE_FACTOR * sr_safe);
        self.lp_alpha = (1.0 - exp_f64(-TAU * AGRAFFE_LP_CUTOFF_HZ * dt_sub)) as f32;
        self.leak = exp_f64(-dt_sub / AGRAFFE_LEAK_TAU_S) as f32;
    }

    /// Overrides the fractional `8x` substep delay and wave admittance directly.
    pub fn set_delay_params(&mut self, delay_substeps: f32, wave_admittance: f32) {
        let max_delay = (self.cap.saturating_sub(2)).max(1) as f32;
        self.delay_substeps = if delay_substeps.is_finite() {
            delay_substeps.clamp(1.0, max_delay)
        } else {
            1.0
        };
        self.wave_admittance = if wave_admittance.is_finite() && wave_admittance >= 0.0 {
            wave_admittance
        } else {
            0.0
        };
    }

    /// Returns allocated delay-line ring buffer capacity in `8x` substeps.
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns current fractional round-trip delay in `8x` substeps.
    #[must_use]
    #[inline]
    pub const fn delay_substeps(&self) -> f32 {
        self.delay_substeps
    }

    /// Clears the ring buffer and filter state to zero (zero allocation).
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_idx = 0;
        self.f_lp = 0.0;
        self.y_agg_m = 0.0;
    }

    /// Evaluates the reflected agraffe residual `(y_agg_m, v_agg_mps)` before `p_end_m`.
    #[inline]
    pub fn step_substep_before(&mut self, dt_sub: f32, uc_choir_scale: f32) -> (f32, f32) {
        if self.cap < 2 || self.wave_admittance <= 0.0 {
            return (0.0, 0.0);
        }
        let max_delay = (self.cap - 2) as f32;
        let d = self.delay_substeps.clamp(1.0, max_delay);
        let d_int = d as usize;
        let frac = d - (d_int as f32);
        let idx0 = (self.write_idx + self.cap - d_int) % self.cap;
        let idx1 = (self.write_idx + self.cap - (d_int + 1)) % self.cap;
        let f_delayed = self.buf[idx0] + frac * (self.buf[idx1] - self.buf[idx0]);

        let scale = if uc_choir_scale.is_finite() && uc_choir_scale >= 0.0 {
            uc_choir_scale
        } else {
            1.0
        };
        let v_agg = -self.wave_admittance * scale * f_delayed;
        self.y_agg_m = (self.y_agg_m * self.leak + v_agg * dt_sub).clamp(-5.0e-4, 5.0e-4);
        (self.y_agg_m, v_agg)
    }

    /// Pushes the current `8x` substep contact force residual into the agraffe delay line.
    #[inline]
    pub fn push_force(&mut self, force_n: f32) {
        if self.cap == 0 {
            return;
        }
        let f = if force_n.is_finite() { force_n } else { 0.0 };
        self.f_lp += self.lp_alpha * (f - self.f_lp);
        let f_hf = (f - self.f_lp) + self.direct_frac * self.f_lp;
        self.buf[self.write_idx] = f_hf;
        self.write_idx += 1;
        if self.write_idx >= self.cap {
            self.write_idx = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use piano_spec::build_scale_design;

    use super::*;

    #[test]
    fn capacity_covers_a0_at_392hz_and_96khz_without_clamping() {
        let mut voicing = VoicingSpec {
            a4_freq_hz: A4_FREQ_MIN_HZ as f32,
            ..VoicingSpec::default()
        };
        voicing.note_offsets[0].pitch_cents = -100.0;
        let specs = build_scale_design(&voicing);

        for &sr in &[48_000.0_f64, 96_000.0] {
            let line = AgraffeDelayLine::new(0, sr, &voicing, &specs);
            let f0 = key_tuned_f0_hz(0, &specs, &voicing);
            let expected_sub = (hammer_beta_for_key(0) / f0) * 8.0 * sr;
            assert!(
                (line.delay_substeps() as f64 - expected_sub).abs() < 0.05,
                "sr={sr}: delay_substeps={} expected={expected_sub}, cap={}",
                line.delay_substeps(),
                line.capacity()
            );
            assert!(line.capacity() > (expected_sub as usize) + 2);
        }
    }

    #[test]
    fn fractional_delay_and_reset_work_across_ring_wrap() {
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);
        let mut line = AgraffeDelayLine::new(39, 48_000.0, &voicing, &specs);
        line.set_delay_params(4.5, 0.1);
        line.direct_frac = 1.0;

        let mut max_v = 0.0_f32;
        for step in 0..20 {
            let (_, v) = line.step_substep_before(1.0 / 384_000.0, 1.0);
            max_v = max_v.max(v.abs());
            line.push_force(if step == 0 { 10.0 } else { 0.0 });
        }
        assert!(max_v > 0.05, "expected reflected pulse, max_v={max_v}");

        line.reset();
        let (y_after, v_after) = line.step_substep_before(1.0 / 384_000.0, 1.0);
        assert_eq!((y_after, v_after), (0.0, 0.0));
    }
}
