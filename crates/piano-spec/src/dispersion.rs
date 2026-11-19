//! Analytical string dispersion, Railsback stretch tuning, micro-detuning,
//! and audible mode-count analysis for the 88-key / 230-string piano bank.
//!
//! All partial frequencies strictly evaluate the exact physical dispersion law
//! `f_j = j * F0 * sqrt(1 + B * j^2)` without table approximations, RBF/SVM/NN
//! surrogates, or Taylor/Pade truncations.

use crate::dispersion_tuning::shift_f0_cents;
pub use crate::dispersion_tuning::{
    bridge_pin_angle_rad, bridge_pin_couplings, effective_string_f0_hz, key_tuned_f0_hz,
    polarization_offset_cents, railsback_stretch_cents, railsback_stretch_cents_table,
    scaled_railsback_stretch_cents, splitmix64, unison_detune_cents, BRIDGE_PIN_BASE_ANGLE_RAD,
};
use crate::math::sqrt_f64;
use crate::scale_design::{build_scale_design, CHOIR_LAYOUTS};
use crate::voicing::VoicingSpec;

/// Upper audible cutoff for modal bank sizing (Hz).
pub const MAX_AUDIBLE_FREQ_HZ: f64 = 20_000.0;

/// Hard per-string upper cap on transverse modes per polarization so SIMD
/// initialization buffers remain bounded even for extreme bass scale factors.
pub const MAX_TRANSVERSE_PARTIALS_PER_STRING: usize = 320;

/// Highest key index excluded from dedicated longitudinal mode synthesis.
///
/// Keys `0..52` (A0 through C5, covering 118 strings: 8 + 36 + 74) synthesize
/// explicit longitudinal phantom-partial modes up to [`MAX_AUDIBLE_FREQ_HZ`].
pub const LONGITUDINAL_MAX_KEY_IDX: usize = 52;

/// Per-string cap on longitudinal modes in keys `0..LONGITUDINAL_MAX_KEY_IDX`.
pub const MAX_LONGITUDINAL_PARTIALS_PER_STRING: usize = 48;

/// Transverse string polarization axis relative to the soundboard bridge.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Polarization {
    /// Perpendicular to the soundboard plane (strongly coupled fast decay).
    Vertical = 0,
    /// Parallel to the soundboard plane (weakly coupled sustaining aftersound).
    Horizontal = 1,
}

/// Aggregated modal count breakdown across all 88 keys and 230 strings.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BankModeCounts {
    /// Unpadded sum of audible transverse modes across 230 strings x 2 polarizations.
    pub transverse_unpadded: usize,
    /// Sum of transverse modes after per-string-polarization padding to 4 SIMD lanes.
    pub transverse_padded: usize,
    /// Unpadded sum of audible longitudinal modes across keys `0..52`.
    pub longitudinal_unpadded: usize,
    /// Sum of longitudinal modes after per-string padding to 4 SIMD lanes.
    pub longitudinal_padded: usize,
    /// Total unpadded active modes (`transverse_unpadded + longitudinal_unpadded`).
    pub total_unpadded: usize,
    /// Total SIMD4-padded modal slots (`transverse_padded + longitudinal_padded`).
    pub total_padded: usize,
}

/// Evaluates the exact analytical stiff-string dispersion equation for partial
/// `partial_index` (1-indexed: `1` = fundamental, `2` = second partial, ...):
///
/// `f_j = j * f0_hz * sqrt(1 + b * j^2)`.
///
/// Returns `0.0` if `partial_index == 0`, `f0_hz <= 0.0`, or inputs are non-finite.
#[must_use]
pub fn partial_frequency_hz(f0_hz: f64, b: f64, partial_index: usize) -> f64 {
    if partial_index == 0 || !f0_hz.is_finite() || f0_hz <= 0.0 {
        return 0.0;
    }
    let j = partial_index as f64;
    let b_safe = if b.is_finite() && b > 0.0 { b } else { 0.0 };
    j * f0_hz * sqrt_f64(1.0 + b_safe * j * j)
}

/// Counts transverse stiff-string partials with `f_j <= max_freq_hz`, capped
/// at [`MAX_TRANSVERSE_PARTIALS_PER_STRING`].
#[must_use]
pub fn count_transverse_partials(f0_hz: f64, b: f64, max_freq_hz: f64) -> usize {
    if !f0_hz.is_finite() || f0_hz <= 0.0 || !max_freq_hz.is_finite() || max_freq_hz < f0_hz {
        return 0;
    }
    (1..=MAX_TRANSVERSE_PARTIALS_PER_STRING)
        .take_while(|&j| partial_frequency_hz(f0_hz, b, j) <= max_freq_hz)
        .count()
}

/// Counts audible longitudinal phantom modes (`m * f_long_hz <= max_freq_hz`)
/// for bass/tenor keys `key_idx < LONGITUDINAL_MAX_KEY_IDX`.
#[must_use]
pub fn count_longitudinal_partials(key_idx: usize, f_long_hz: f64, max_freq_hz: f64) -> usize {
    if key_idx >= LONGITUDINAL_MAX_KEY_IDX
        || !f_long_hz.is_finite()
        || f_long_hz <= 0.0
        || !max_freq_hz.is_finite()
        || max_freq_hz < f_long_hz
    {
        return 0;
    }
    (1..=MAX_LONGITUDINAL_PARTIALS_PER_STRING)
        .take_while(|&m| (m as f64) * f_long_hz <= max_freq_hz)
        .count()
}

/// Rounds a per-voice mode count up to the next multiple of 4 for `f32x4` SIMD lanes.
#[must_use]
pub const fn pad_to_simd4(count: usize) -> usize {
    count.div_ceil(4) * 4
}

/// Computes exact unpadded and SIMD4-padded transverse and longitudinal mode
/// counts across all 88 keys and 230 strings for the given [`VoicingSpec`].
#[must_use]
pub fn analyze_bank_mode_counts(voicing: &VoicingSpec) -> BankModeCounts {
    let specs = build_scale_design(voicing);
    let stretch_table = railsback_stretch_cents_table(&specs);
    let stretch_scale = voicing.effective_stretch_amount() as f64;
    let (mut t_raw, mut t_pad, mut l_raw, mut l_pad) = (0usize, 0usize, 0usize, 0usize);

    for (key_idx, spec) in specs.iter().enumerate() {
        let choir = CHOIR_LAYOUTS[key_idx];
        let n_str = choir.strings_per_key as usize;
        let base_cents = stretch_table[key_idx] * stretch_scale
            + (voicing.effective_pitch_cents(key_idx) as f64);
        let temp_cents = voicing.temperament.cent_offset_for_key(key_idx);
        let note_cents = if temp_cents == 0.0 {
            base_cents
        } else {
            temp_cents + base_cents
        };

        let l_modes = count_longitudinal_partials(key_idx, spec.f_long_hz, MAX_AUDIBLE_FREQ_HZ);
        l_raw += l_modes * n_str;
        l_pad += pad_to_simd4(l_modes) * n_str;

        for s in 0..n_str {
            let detune = unison_detune_cents(key_idx, s, choir.strings_per_key, voicing);
            for pol in [Polarization::Vertical, Polarization::Horizontal] {
                let pol_cents = polarization_offset_cents(key_idx, s, pol, voicing);
                let f0 = shift_f0_cents(spec.f0_et_hz, note_cents + detune + pol_cents);
                let t_modes =
                    count_transverse_partials(f0, spec.inharmonicity_b, MAX_AUDIBLE_FREQ_HZ);
                t_raw += t_modes;
                t_pad += pad_to_simd4(t_modes);
            }
        }
    }

    BankModeCounts {
        transverse_unpadded: t_raw,
        transverse_padded: t_pad,
        longitudinal_unpadded: l_raw,
        longitudinal_padded: l_pad,
        total_unpadded: t_raw + l_raw,
        total_padded: t_pad + l_pad,
    }
}

#[cfg(test)]
#[path = "dispersion_tests.rs"]
mod tests;
