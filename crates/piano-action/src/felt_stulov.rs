//! Nonlinear Stulov wool-felt contact model with two-timescale viscoelastic
//! relaxation (`tau_fast ~ 35 us` intra-strike, `tau_mem ~ 130 ms` compaction memory).

use piano_spec::{VoicingSpec, NUM_KEYS};

use crate::math::{exp_f64, pow_nonneg_f64, pow_usize_f32};
use crate::shank_timoshenko::ShankParams;

/// Fast intra-strike Stulov wool relaxation time constant (`35 us`).
pub const TAU_FAST_S: f64 = 35.0e-6;
/// Slow wool-compaction memory time constant (`130 ms`).
pub const TAU_MEM_S: f64 = 130.0e-3;
/// Nominal fast Stulov relaxation fraction `epsilon`.
pub const EPS_FAST_BASE: f32 = 0.28;

const MF_REF_SPEED_MPS: f64 = 1.826;
const TAU_COMPACT_S: f64 = 5.0e-3;
const MEM_ACCUM_GAIN: f32 = 10.0;
const MEM_STIFFEN_GAIN: f32 = 2.8;

/// Sample-rate-shared exponential relaxation factors for `FeltState`.
#[derive(Copy, Clone, Debug)]
pub struct FeltSharedCoeffs {
    pub a_fast_sub: f32,
    pub a_fast_host: f32,
    pub a_mem_host: f32,
}

impl FeltSharedCoeffs {
    /// Precomputes exact ZOH exponential decay factors at `sample_rate` and `8x`.
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        let dt_host = 1.0 / sample_rate;
        let dt_sub = dt_host / 8.0;
        let a_fast_sub = exp_f64(-dt_sub / TAU_FAST_S);
        let a_fast_host = exp_f64(-dt_host / TAU_FAST_S);
        let a_mem_host = exp_f64(-dt_host / TAU_MEM_S);
        Self {
            a_fast_sub: a_fast_sub as f32,
            a_fast_host: a_fast_host as f32,
            a_mem_host: a_mem_host as f32,
        }
    }
}

/// Per-key physical wool felt parameters derived from key index & [`VoicingSpec`].
#[derive(Copy, Clone, Debug)]
pub struct KeyFeltSpec {
    /// Power-law exponent `p in [2.2, 3.5]`.
    pub exponent_p: f32,
    /// Baseline physical felt stiffness `K_0` in SI units (`N / m^p`).
    pub k0_si: f32,
    /// Normalization displacement `delta_scale_m` (`m`) mapping active strikes into `[0, 1]`.
    pub delta_scale_m: f32,
    /// Dimensionless-coordinate force scale `K_0 * delta_scale_m^p` in Newtons.
    pub f_scale_n: f32,
    /// Nominal `mf` target contact duration (`s`) used to normalize compaction rate.
    pub nominal_contact_s: f32,
    /// Compaction accumulation step per unit normalized compression per `8x` substep.
    pub mem_sub_gain: f32,
}

fn compute_key_felt_spec(
    key_idx: usize,
    sample_rate: f64,
    voicing: &VoicingSpec,
    fixed_exponent_p: Option<f64>,
) -> KeyFeltSpec {
    let k = key_idx.min(NUM_KEYS - 1);
    let x = (k as f64) / ((NUM_KEYS - 1) as f64);
    let h = f64::from(voicing.effective_hammer_hardness(k));

    let p_base = 2.45 + 0.65 * x;
    let exponent_p =
        fixed_exponent_p.unwrap_or_else(|| (p_base + (h - 0.5) * 0.50).clamp(2.2, 3.5));

    // Smooth monotonic rigid-felt baseline tuned so coupled string contact durations hit
    // A0 ~ 3.2 ms, A4 mf ~ 1.5 ms, and C8 mf ~ 0.14 ms.
    let tau_c = 4.2e-3 * exp_f64(-2.85 * x - 0.70 * x * x * x);
    let c_hertz = 3.21 + 0.34 * (exponent_p - 2.2);
    let delta_mf = tau_c * MF_REF_SPEED_MPS / c_hertz;

    let shank = ShankParams::for_key(k, sample_rate);
    let m_h = f64::from(shank.hammer_mass_kg);
    let k0_nominal = (exponent_p + 1.0) * m_h * MF_REF_SPEED_MPS * MF_REF_SPEED_MPS
        / (2.0 * pow_nonneg_f64(delta_mf, exponent_p + 1.0));
    // Calibrate for Stulov hysteresis (~1 - 0.5 * eps) & voicing hardness.
    let voicing_stiff = exp_f64((h - 0.5) * 1.40);
    let k0_si = k0_nominal * 1.12 * voicing_stiff;

    // Normalize LUT domain so fff strikes sit cleanly inside u in [0, 1].
    let delta_mf_eff = if fixed_exponent_p.is_some() && (h - 0.5).abs() >= 1.0e-12 {
        delta_mf * pow_nonneg_f64(1.0 / voicing_stiff, 1.0 / (exponent_p + 1.0))
    } else {
        delta_mf
    };
    let delta_scale_m = (3.4 * delta_mf_eff).max(1.2e-4);
    let f_scale_n = k0_si * pow_nonneg_f64(delta_scale_m, exponent_p);
    let dt_sub = 1.0 / (8.0 * sample_rate);
    let mem_sub_gain = (f64::from(MEM_ACCUM_GAIN) * dt_sub / TAU_COMPACT_S) as f32;

    KeyFeltSpec {
        exponent_p: exponent_p as f32,
        k0_si: k0_si as f32,
        delta_scale_m: delta_scale_m as f32,
        f_scale_n: f_scale_n as f32,
        nominal_contact_s: tau_c as f32,
        mem_sub_gain,
    }
}

impl KeyFeltSpec {
    /// Builds physical felt parameters for `key_idx` under `voicing`.
    #[must_use]
    pub fn for_key(key_idx: usize, sample_rate: f64, voicing: &VoicingSpec) -> Self {
        compute_key_felt_spec(key_idx, sample_rate, voicing, None)
    }

    /// Updates `exponent_p`, stiffness, and scaling parameters in-place for `voicing`
    /// with zero heap allocations using the shared `p`-grid LUT.
    pub fn update_voicing_in_place(
        &mut self,
        key_idx: usize,
        sample_rate: f64,
        voicing: &VoicingSpec,
    ) {
        *self = compute_key_felt_spec(key_idx, sample_rate, voicing, None);
    }
}

/// Persistent two-timescale Stulov relaxation state (cleared ONLY in `reset`).
#[derive(Copy, Clone, Debug, Default)]
pub struct FeltState {
    /// Fast intra-strike IIR memory of `phi = ADAA(u)`.
    pub z_fast: f32,
    /// Slow wool-compaction memory in `[0.0, 1.0]` (`tau ~ 130 ms`).
    pub z_mem: f32,
}

impl FeltState {
    /// Clears both relaxation timescales (called only on `ActionBank::reset`).
    #[inline]
    pub fn reset(&mut self) {
        self.z_fast = 0.0;
        self.z_mem = 0.0;
    }

    /// Returns effective Newton force scale and hysteresis ratio modulated by `z_mem`.
    #[must_use]
    #[inline]
    pub fn effective_params(&self, spec: &KeyFeltSpec, memory_enabled: bool) -> (f32, f32) {
        if memory_enabled {
            let m = self.z_mem.clamp(0.0, 1.0);
            let f_scale_eff = spec.f_scale_n * (1.0 + MEM_STIFFEN_GAIN * m);
            let eps_eff = EPS_FAST_BASE * (1.0 - 0.20 * m);
            (f_scale_eff, eps_eff)
        } else {
            (spec.f_scale_n, EPS_FAST_BASE)
        }
    }

    /// Updates slow wool-compaction memory over one active `8x` substep.
    #[inline]
    pub fn accumulate_substep_compaction(
        &mut self,
        spec: &KeyFeltSpec,
        u_norm: f32,
        memory_enabled: bool,
    ) {
        if memory_enabled && u_norm > 0.0 {
            let inc = spec.mem_sub_gain * u_norm * (1.0 - self.z_mem);
            self.z_mem = (self.z_mem + inc).min(1.0);
        }
    }

    /// Decays slow memory (and fast state when idle) across `host_frames` samples.
    #[inline]
    pub fn decay_host_frames(&mut self, coeffs: &FeltSharedCoeffs, host_frames: usize) {
        if host_frames == 0 {
            return;
        }
        if self.z_mem > 0.0 {
            let factor = if host_frames == 1 {
                coeffs.a_mem_host
            } else {
                pow_usize_f32(coeffs.a_mem_host, host_frames)
            };
            self.z_mem *= factor;
            if self.z_mem < 1.0e-12 {
                self.z_mem = 0.0;
            }
        }
        if self.z_fast > 0.0 {
            let factor = if host_frames == 1 {
                coeffs.a_fast_host
            } else if host_frames >= 64 {
                0.0
            } else {
                pow_usize_f32(coeffs.a_fast_host, host_frames)
            };
            self.z_fast *= factor;
            if self.z_fast < 1.0e-12 {
                self.z_fast = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_felt_spec_scales_exponent_stiffness_and_voicing() {
        let nominal = VoicingSpec::default();
        let s0 = KeyFeltSpec::for_key(0, 48_000.0, &nominal);
        let s48 = KeyFeltSpec::for_key(48, 48_000.0, &nominal);
        let s87 = KeyFeltSpec::for_key(87, 48_000.0, &nominal);

        for s in [&s0, &s48, &s87] {
            assert!((2.2..=3.5).contains(&s.exponent_p));
            assert!(s.k0_si.is_finite() && s.k0_si > 0.0);
            assert!(s.f_scale_n.is_finite() && s.f_scale_n > 0.0);
        }
        assert!(s0.exponent_p < s48.exponent_p && s48.exponent_p < s87.exponent_p);
        assert!(s0.k0_si < s48.k0_si && s48.k0_si < s87.k0_si);
        assert!((s0.nominal_contact_s - 4.2e-3).abs() < 1e-4);
        assert!(
            s48.nominal_contact_s < s0.nominal_contact_s
                && s87.nominal_contact_s < s48.nominal_contact_s
        );
        assert!((s87.nominal_contact_s - 1.2e-4).abs() < 3e-5);

        let hard = VoicingSpec {
            hammer_hardness: 0.9,
            ..VoicingSpec::default()
        };
        let s48_hard = KeyFeltSpec::for_key(48, 48_000.0, &hard);
        assert!(s48_hard.exponent_p > s48.exponent_p);
        assert!(s48_hard.k0_si > s48.k0_si);
    }

    #[test]
    fn slow_memory_tau_matches_130_ms_decay() {
        let coeffs = FeltSharedCoeffs::new(48_000.0);
        let mut state = FeltState {
            z_fast: 0.5,
            z_mem: 1.0,
        };
        // 130 ms = 6240 samples @ 48 kHz -> exp(-1) ~= 0.367879
        state.decay_host_frames(&coeffs, 6_240);
        assert!(
            (state.z_mem - 0.367_879).abs() < 1e-4,
            "z_mem={}",
            state.z_mem
        );
        assert_eq!(state.z_fast, 0.0);
    }

    #[test]
    fn update_voicing_in_place_matches_default_and_updates_exponent_and_stiffness() {
        let nominal = VoicingSpec::default();
        for &k in &[0_usize, 39, 48, 87] {
            let baseline = KeyFeltSpec::for_key(k, 48_000.0, &nominal);
            let mut spec = baseline;

            // Default re-application matches baseline exactly.
            spec.update_voicing_in_place(k, 48_000.0, &nominal);
            assert_eq!(spec.exponent_p, baseline.exponent_p);
            assert!((spec.k0_si - baseline.k0_si).abs() <= baseline.k0_si * 1e-6);
            assert!((spec.delta_scale_m - baseline.delta_scale_m).abs() <= 1e-7);

            // Global + per-note hardness updates exponent_p, stiffness, and LUT scaling dynamically.
            let mut soft = nominal.clone();
            soft.hammer_hardness = 0.1;
            let mut hard = nominal.clone();
            hard.hammer_hardness = 0.9;
            hard.note_offsets[k].hammer_hardness_offset = 0.05;

            let mut s_soft = baseline;
            s_soft.update_voicing_in_place(k, 48_000.0, &soft);
            let mut s_hard = baseline;
            s_hard.update_voicing_in_place(k, 48_000.0, &hard);

            let fresh_soft = KeyFeltSpec::for_key(k, 48_000.0, &soft);
            let fresh_hard = KeyFeltSpec::for_key(k, 48_000.0, &hard);
            assert_eq!(s_soft.exponent_p, fresh_soft.exponent_p);
            assert_eq!(s_hard.exponent_p, fresh_hard.exponent_p);
            assert!(
                s_soft.exponent_p < baseline.exponent_p && baseline.exponent_p < s_hard.exponent_p
            );
            assert!(s_soft.k0_si < baseline.k0_si && baseline.k0_si < s_hard.k0_si);
            assert!(s_hard.delta_scale_m < s_soft.delta_scale_m);
            assert!(s_hard.f_scale_n > s_soft.f_scale_n);
        }
    }
}
