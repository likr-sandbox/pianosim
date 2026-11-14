//! Shared piano specifications, sample-accurate events, and scaling constants.

#![cfg_attr(not(test), no_std)]

pub mod dispersion;
pub mod events;
pub mod scale_design;
pub mod voicing;

pub use dispersion::{
    analyze_bank_mode_counts, count_longitudinal_partials, count_transverse_partials,
    effective_string_f0_hz, key_tuned_f0_hz, pad_to_simd4, partial_frequency_hz,
    polarization_offset_cents, railsback_stretch_cents, railsback_stretch_cents_table,
    scaled_railsback_stretch_cents, splitmix64, unison_detune_cents, BankModeCounts, Polarization,
    LONGITUDINAL_MAX_KEY_IDX, MAX_AUDIBLE_FREQ_HZ, MAX_LONGITUDINAL_PARTIALS_PER_STRING,
    MAX_TRANSVERSE_PARTIALS_PER_STRING,
};
pub use events::{u16_to_unit_f32, u7_to_u16, EventKind, PedalKind, SampleEvent};
pub use scale_design::{
    build_scale_design, equal_temperament_hz, key_choir, key_to_midi, midi_to_key,
    string_params_for_key, ChoirLayout, StringPhysicalParams, CHOIR_LAYOUTS, DAMPED_KEYS, MIDI_A0,
    MIDI_C8, NUM_KEYS, STEEL_DENSITY_KG_M3, STEEL_YOUNGS_MODULUS_PA, TOTAL_STRINGS,
};
pub use voicing::{NoteVoicingOffset, VoicingSpec};

/// Pure-Rust `no_std` `f64` mathematical primitives for initialization-time scaling.
#[allow(dead_code)]
pub(crate) mod math {
    use core::f64::consts::{FRAC_2_PI, LN_2, LOG2_E, SQRT_2};

    const PIO2_HI: f64 = 1.570_796_326_734_125_6;
    const PIO2_LO: f64 = 6.077_100_506_506_192e-11;
    const LN2_HI: f64 = 0.693_147_180_485_539_1;
    const LN2_LO: f64 = 7.440_614_790_965_583e-11;

    const SIN_COEFFS: [f64; 8] = [
        1.0,
        -1.0 / 6.0,
        1.0 / 120.0,
        -1.0 / 5_040.0,
        1.0 / 362_880.0,
        -1.0 / 39_916_800.0,
        1.0 / 6_227_020_800.0,
        -1.0 / 1_307_674_368_000.0,
    ];

    const COS_COEFFS: [f64; 8] = [
        1.0,
        -0.5,
        1.0 / 24.0,
        -1.0 / 720.0,
        1.0 / 40_320.0,
        -1.0 / 3_628_800.0,
        1.0 / 479_001_600.0,
        -1.0 / 87_178_291_200.0,
    ];

    const LN_ODD_COEFFS: [f64; 10] = [
        1.0,
        1.0 / 3.0,
        1.0 / 5.0,
        1.0 / 7.0,
        1.0 / 9.0,
        1.0 / 11.0,
        1.0 / 13.0,
        1.0 / 15.0,
        1.0 / 17.0,
        1.0 / 19.0,
    ];

    const EXP_COEFFS: [f64; 14] = [
        1.0,
        1.0,
        0.5,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5_040.0,
        1.0 / 40_320.0,
        1.0 / 362_880.0,
        1.0 / 3_628_800.0,
        1.0 / 39_916_800.0,
        1.0 / 479_001_600.0,
        1.0 / 6_227_020_800.0,
    ];

    #[inline]
    fn eval_horner(x: f64, coeffs: &[f64]) -> f64 {
        let mut acc = 0.0_f64;
        let mut i = coeffs.len();
        while i > 0 {
            i -= 1;
            acc = acc * x + coeffs[i];
        }
        acc
    }

    #[inline]
    fn round_f64(x: f64) -> i64 {
        if x >= 0.0 {
            (x + 0.5) as i64
        } else {
            (x - 0.5) as i64
        }
    }

    /// Computes `sqrt(x)` in pure Rust (`no_std`) to 53-bit `f64` precision
    /// using an IEEE-754 bit seed and 6 Newton-Raphson iterations.
    #[inline]
    pub(crate) fn sqrt_f64(x: f64) -> f64 {
        if x.is_nan() || x < 0.0 {
            return f64::NAN;
        }
        if x == 0.0 || x.is_infinite() {
            return x;
        }

        // Scale subnormals into normal range by 2^54 (post-scale 2^-27).
        let (scaled, post_scale) = if x < f64::MIN_POSITIVE {
            (x * 18_014_398_509_481_984.0, 1.0_f64 / 134_217_728.0)
        } else {
            (x, 1.0_f64)
        };

        let mut y = f64::from_bits((scaled.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
        for _ in 0..6 {
            y = 0.5 * (y + scaled / y);
        }
        y * post_scale
    }

    /// Computes `(sin(angle), cos(angle))` in `f64` without `std`.
    pub(crate) fn sin_cos_f64(angle: f64) -> (f64, f64) {
        if !angle.is_finite() {
            return (f64::NAN, f64::NAN);
        }
        let q = round_f64(angle * FRAC_2_PI);
        let qf = q as f64;
        let r = (angle - qf * PIO2_HI) - qf * PIO2_LO;
        let r2 = r * r;

        let s = r * eval_horner(r2, &SIN_COEFFS);
        let c = eval_horner(r2, &COS_COEFFS);

        match q.rem_euclid(4) {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }

    #[inline]
    pub(crate) fn sin_f64(angle: f64) -> f64 {
        sin_cos_f64(angle).0
    }

    #[inline]
    pub(crate) fn cos_f64(angle: f64) -> f64 {
        sin_cos_f64(angle).1
    }

    /// Computes `ln(x)` in `f64` using IEEE-754 mantissa decomposition.
    pub(crate) fn ln_f64(x: f64) -> f64 {
        if x < 0.0 || x.is_nan() {
            return f64::NAN;
        }
        if x == 0.0 {
            return f64::NEG_INFINITY;
        }
        if x.is_infinite() {
            return f64::INFINITY;
        }

        let mut val = x;
        let mut exp_bias: i32 = 0;
        if val < f64::MIN_POSITIVE {
            val *= 18_014_398_509_481_984.0;
            exp_bias = -54;
        }

        let bits = val.to_bits();
        let mut k = ((bits >> 52) as i32) - 1023 + exp_bias;
        let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
        if m > SQRT_2 {
            m *= 0.5;
            k += 1;
        }

        let u = (m - 1.0) / (m + 1.0);
        let series = eval_horner(u * u, &LN_ODD_COEFFS);
        f64::from(k) * LN_2 + 2.0 * u * series
    }

    /// Computes `exp(y)` in `f64` without `std`.
    pub(crate) fn exp_f64(y: f64) -> f64 {
        if y.is_nan() {
            return f64::NAN;
        }
        if y == 0.0 {
            return 1.0;
        }
        if y > 709.78 {
            return f64::INFINITY;
        }
        if y < -745.13 {
            return 0.0;
        }

        let k = round_f64(y * LOG2_E);
        let kf = k as f64;
        let r = (y - kf * LN2_HI) - kf * LN2_LO;
        let poly = eval_horner(r, &EXP_COEFFS);

        if k < -1022 {
            let shift = (k + 1077) as u64;
            let sub_scale = f64::from_bits(shift << 52);
            let tiny_scale = f64::from_bits((1023_u64 - 55) << 52);
            (poly * sub_scale) * tiny_scale
        } else {
            let pow2k = f64::from_bits(((k + 1023) as u64) << 52);
            poly * pow2k
        }
    }

    /// Computes `base.powf(exp)` in `f64` without `std`.
    pub(crate) fn powf_f64(base: f64, exp: f64) -> f64 {
        if exp == 0.0 {
            return 1.0;
        }
        if base.is_nan() || exp.is_nan() {
            return f64::NAN;
        }
        if base == 1.0 {
            return 1.0;
        }
        if base == 0.0 {
            return if exp > 0.0 { 0.0 } else { f64::INFINITY };
        }
        if base < 0.0 {
            return f64::NAN;
        }
        exp_f64(exp * ln_f64(base))
    }

    #[inline]
    pub(crate) fn exp2_f64(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            exp_f64(x * LN_2)
        }
    }

    #[inline]
    pub(crate) fn log2_f64(x: f64) -> f64 {
        ln_f64(x) * LOG2_E
    }
}

#[cfg(test)]
mod tests {
    use super::math::*;
    use super::*;

    #[test]
    fn nostd_f64_math_matches_std_to_machine_precision() {
        for &x in &[
            0.0_f64, 1e-315, 1e-300, 1e-12, 0.000_36, 0.5, 1.0, 2.0, 440.0, 1.2345e12,
        ] {
            let got = sqrt_f64(x);
            let expected = x.sqrt();
            let tol = 1e-15 * expected.max(1e-320);
            assert!(
                (got - expected).abs() <= tol,
                "sqrt_f64({x}) got={got} expected={expected}"
            );
        }
        assert!(sqrt_f64(-1.0).is_nan());
        assert!(sqrt_f64(f64::INFINITY).is_infinite());

        for i in -40..=40 {
            let angle = f64::from(i) * 0.417_234_567_89;
            assert!((sin_f64(angle) - angle.sin()).abs() < 1e-14);
            assert!((cos_f64(angle) - angle.cos()).abs() < 1e-14);
        }

        for &x in &[1e-12_f64, 0.00036, 0.5, 1.0, 2.0, 27.5, 440.0, 4186.0] {
            let l_got = ln_f64(x);
            let l_ref = x.ln();
            assert!((l_got - l_ref).abs() <= 1e-14 * l_ref.abs().max(1.0));
            let l2_got = log2_f64(x);
            let l2_ref = x.log2();
            assert!((l2_got - l2_ref).abs() <= 1e-14 * l2_ref.abs().max(1.0));
        }

        for &y in &[-30.0_f64, -2.5, -0.01, 0.0, 0.5, 1.0, 6.5, 20.0] {
            let e_got = exp_f64(y);
            let e_ref = y.exp();
            assert!((e_got - e_ref).abs() <= 1e-13 * e_ref.max(1.0));
        }

        for &base in &[0.0_f64, 0.044, 0.5, 1.0, 2.0, 13.7] {
            for &exp in &[-2.25_f64, 0.0, 0.5, 1.9, 3.25] {
                if base == 0.0 && exp <= 0.0 {
                    continue;
                }
                let got = powf_f64(base, exp);
                let expected = base.powf(exp);
                assert!((got - expected).abs() <= 1e-13 * expected.abs().max(1.0));
            }
        }
        assert_eq!(exp2_f64(0.0), 1.0);
    }

    #[test]
    fn full_piano_spec_pipeline_satisfies_concert_grand_invariants() {
        let mut total_strings = 0usize;
        let mut damped = 0usize;
        for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
            assert_eq!(layout.first_string_idx as usize, total_strings);
            total_strings += layout.strings_per_key as usize;
            if layout.has_damper {
                damped += 1;
            }
            assert_eq!(key_choir(k), Some(*layout));
        }
        assert_eq!(total_strings, TOTAL_STRINGS);
        assert_eq!(damped, DAMPED_KEYS);

        let nominal = VoicingSpec::default();
        let specs = build_scale_design(&nominal);
        assert!((0.00030..=0.00042).contains(&specs[0].inharmonicity_b));
        assert!((0.040..=0.050).contains(&specs[NUM_KEYS - 1].inharmonicity_b));
        assert_eq!(key_tuned_f0_hz(48, &specs, &nominal), 440.0);

        let counts = analyze_bank_mode_counts(&nominal);
        assert!(
            (25_000..=26_500).contains(&counts.total_padded),
            "{counts:?}"
        );

        // D4 per-note overrides shift targeted key without perturbing neighbors.
        let mut voiced = VoicingSpec::default();
        voiced.note_offsets[48].pitch_cents = 12.0;
        voiced.note_offsets[48].inharmonicity_scale = 0.5;
        let voiced_specs = build_scale_design(&voiced);
        assert!((voiced_specs[48].inharmonicity_b / specs[48].inharmonicity_b - 1.5).abs() < 1e-6);
        assert!((voiced_specs[49].inharmonicity_b - specs[49].inharmonicity_b).abs() < 1e-12);
        let a4_sharp = key_tuned_f0_hz(48, &voiced_specs, &voiced);
        assert!((a4_sharp - 440.0 * 2.0_f64.powf(12.0 / 1200.0)).abs() < 1e-11);

        // Every partial obeys the analytical dispersion law.
        for &k in &[0usize, 39, 48, 87] {
            let f0 = effective_string_f0_hz(k, 0, Polarization::Vertical, &specs, &nominal);
            let n_p = count_transverse_partials(f0, specs[k].inharmonicity_b, MAX_AUDIBLE_FREQ_HZ);
            assert!(n_p >= 1);
            for j in 1..=n_p {
                let f_j = partial_frequency_hz(f0, specs[k].inharmonicity_b, j);
                let ref_j =
                    (j as f64) * f0 * (1.0 + specs[k].inharmonicity_b * (j * j) as f64).sqrt();
                assert!((f_j - ref_j).abs() < 1e-10);
                assert!(f_j <= MAX_AUDIBLE_FREQ_HZ);
            }
        }
    }
}
