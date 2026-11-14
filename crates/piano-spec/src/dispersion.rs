//! Analytical string dispersion, Railsback stretch tuning, micro-detuning,
//! and audible mode-count analysis for the 88-key / 230-string piano bank.
//!
//! All partial frequencies strictly evaluate the exact physical dispersion law
//! `f_j = j * F0 * sqrt(1 + B * j^2)` without table approximations, RBF/SVM/NN
//! surrogates, or Taylor/Pade truncations.

use crate::math::{exp2_f64, log2_f64, sqrt_f64};
use crate::scale_design::{
    build_scale_design, StringPhysicalParams, A4_KEY_IDX, CHOIR_LAYOUTS, NUM_KEYS,
};
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

/// Differential Railsback stretch slope (in cents per semitone, always `>= 0`)
/// derived directly from key `key_idx`'s analytical inharmonicity `b`.
fn railsback_step_slope_cents(key_idx: usize, b: f64) -> f64 {
    let b_safe = if b.is_finite() && b > 0.0 { b } else { 0.0 };
    if b_safe == 0.0 {
        return 0.0;
    }
    // Below A4, aurally matching 2nd-to-6th bass partials stretches octaves
    // progressively wider; above A4, treble fundamentals match the lower
    // octave's stretched 2nd partial (~1.2 * B(k) effective shift rate).
    let coeff = if key_idx <= A4_KEY_IDX {
        let bass_depth = (A4_KEY_IDX - key_idx) as f64 / A4_KEY_IDX as f64;
        let j_eff = 2.0 + 4.4 * sqrt_f64(bass_depth) * sqrt_f64(sqrt_f64(bass_depth));
        j_eff * j_eff
    } else {
        1.2
    };
    50.0 * log2_f64(1.0 + coeff * b_safe)
}

/// Computes the unscaled concert-grand Railsback stretch curve (in cents)
/// across all 88 keys from their physical inharmonicity coefficients `B(k)`.
///
/// Anchored strictly at `A4` (`key_idx = 48`) = `0.0` cents, monotonically
/// increasing from bass (`~ -24 cents` at A0) to treble (`~ +31 cents` at C8).
#[must_use]
pub fn railsback_stretch_cents_table(specs: &[StringPhysicalParams; NUM_KEYS]) -> [f64; NUM_KEYS] {
    let mut table = [0.0_f64; NUM_KEYS];
    table[A4_KEY_IDX] = 0.0;

    let mut acc_down = 0.0_f64;
    let mut prev_slope = railsback_step_slope_cents(A4_KEY_IDX, specs[A4_KEY_IDX].inharmonicity_b);
    let mut k = A4_KEY_IDX;
    while k > 0 {
        let next_k = k - 1;
        let cur_slope = railsback_step_slope_cents(next_k, specs[next_k].inharmonicity_b);
        acc_down -= 0.5 * (prev_slope + cur_slope);
        table[next_k] = acc_down;
        prev_slope = cur_slope;
        k = next_k;
    }

    let mut acc_up = 0.0_f64;
    prev_slope = railsback_step_slope_cents(A4_KEY_IDX, specs[A4_KEY_IDX].inharmonicity_b);
    for (idx, spec) in specs.iter().enumerate().skip(A4_KEY_IDX + 1) {
        let cur_slope = railsback_step_slope_cents(idx, spec.inharmonicity_b);
        acc_up += 0.5 * (prev_slope + cur_slope);
        table[idx] = acc_up;
        prev_slope = cur_slope;
    }

    table
}

/// Returns the unscaled physical Railsback stretch offset (in cents) for `key_idx`
/// derived from `specs[..].inharmonicity_b`. Returns exact `0.0` for A4 (`48`).
#[must_use]
pub fn railsback_stretch_cents(key_idx: usize, specs: &[StringPhysicalParams; NUM_KEYS]) -> f64 {
    let clamped = key_idx.min(NUM_KEYS - 1);
    if clamped == A4_KEY_IDX {
        return 0.0;
    }
    railsback_stretch_cents_table(specs)[clamped]
}

/// Returns the Railsback stretch offset (in cents) scaled by `voicing.stretch_amount`.
#[must_use]
pub fn scaled_railsback_stretch_cents(
    key_idx: usize,
    specs: &[StringPhysicalParams; NUM_KEYS],
    voicing: &VoicingSpec,
) -> f64 {
    let stretch = voicing.effective_stretch_amount() as f64;
    if stretch == 0.0 {
        return 0.0;
    }
    let raw = railsback_stretch_cents(key_idx, specs);
    if raw == 0.0 {
        return 0.0;
    }
    raw * stretch
}

/// Deterministic SplitMix64 hash generator used for per-string micro-detuning
/// and polarization boundary offsets.
#[must_use]
pub const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
fn hash_to_unit(seed: u64) -> f64 {
    const INV_2_53: f64 = 1.0 / ((1u64 << 53) as f64);
    ((splitmix64(seed) >> 11) as f64) * INV_2_53
}

#[inline]
fn hash_to_signed_unit(seed: u64) -> f64 {
    2.0 * hash_to_unit(seed) - 1.0
}

/// Deterministic choir micro-detuning offset (in cents) for `string_in_choir`.
///
/// - Monochords (`strings_per_key <= 1`): strictly `0.0`.
/// - Bichords (`2`): symmetric spread around `0.0` with organic jitter.
/// - Trichords (`3`): characteristic `[-d, 0, +d]` triad (`d ~ 0.55..1.10 cents`)
///   plus fine organic SplitMix64 jitter, scaled by `voicing.unison_detune_scale`.
///
/// Returns exact `0.0` whenever `voicing.effective_unison_detune_scale() == 0.0`.
#[must_use]
pub fn unison_detune_cents(
    key_idx: usize,
    string_in_choir: usize,
    strings_per_key: u8,
    voicing: &VoicingSpec,
) -> f64 {
    let scale = voicing.effective_unison_detune_scale() as f64;
    if scale == 0.0
        || strings_per_key <= 1
        || key_idx >= NUM_KEYS
        || string_in_choir >= strings_per_key as usize
    {
        return 0.0;
    }

    let key_seed = (key_idx as u64).wrapping_mul(0x517c_c1b7_2722_0a95) ^ 0x1a2b_3c4d_5e6f_7788;
    let str_seed = key_seed
        ^ ((string_in_choir as u64)
            .wrapping_add(1)
            .wrapping_mul(0xd6e8_feb8_6659_fd93));

    // Nominal spread d in [0.55, 1.10) cents for rich slow choir beating.
    let d = 0.55 + 0.55 * hash_to_unit(key_seed);
    let base_offset = match (strings_per_key, string_in_choir) {
        (2, 0) => -0.55 * d,
        (2, 1) => 0.55 * d,
        (3, 0) => -d,
        (3, 1) => 0.0,
        (3, 2) => d,
        _ => 0.0,
    };
    let jitter = 0.14 * hash_to_signed_unit(str_seed);
    (base_offset + jitter) * scale
}

/// Deterministic frequency offset (in cents) arising from bridge boundary
/// anisotropy between vertical and horizontal string polarizations (~`0.16..0.28`
/// cents at nominal `unison_detune_scale = 1.0`).
///
/// Returns exact `0.0` for `Polarization::Vertical` or when
/// `voicing.effective_unison_detune_scale() == 0.0`.
#[must_use]
pub fn polarization_offset_cents(
    key_idx: usize,
    string_in_choir: usize,
    polarization: Polarization,
    voicing: &VoicingSpec,
) -> f64 {
    let scale = voicing.effective_unison_detune_scale() as f64;
    if scale == 0.0 || key_idx >= NUM_KEYS {
        return 0.0;
    }
    match polarization {
        Polarization::Vertical => 0.0,
        Polarization::Horizontal => {
            let seed = ((key_idx as u64) << 8) ^ (string_in_choir as u64) ^ 0x94d0_49bb_1331_11eb;
            let split_cents = 0.16 + 0.12 * hash_to_unit(seed);
            split_cents * scale
        }
    }
}

#[inline]
fn shift_f0_cents(f0_et_hz: f64, cents: f64) -> f64 {
    if cents == 0.0 {
        f0_et_hz
    } else {
        f0_et_hz * exp2_f64(cents / 1200.0)
    }
}

/// Computes the nominal tuned key fundamental (ET + Railsback + D4 pitch offset),
/// preserving exact `440.0 Hz` at `A4` (`key_idx = 48`) with default voicing.
#[must_use]
pub fn key_tuned_f0_hz(
    key_idx: usize,
    specs: &[StringPhysicalParams; NUM_KEYS],
    voicing: &VoicingSpec,
) -> f64 {
    let k = key_idx.min(NUM_KEYS - 1);
    let cents = scaled_railsback_stretch_cents(k, specs, voicing)
        + (voicing.effective_pitch_cents(k) as f64);
    shift_f0_cents(specs[k].f0_et_hz, cents)
}

/// Computes the exact effective fundamental frequency `F0` (Hz) for a single
/// polarized transverse string after ET, Railsback stretch, D4 note offset,
/// choir micro-detuning, and polarization anisotropy.
#[must_use]
pub fn effective_string_f0_hz(
    key_idx: usize,
    string_in_choir: usize,
    polarization: Polarization,
    specs: &[StringPhysicalParams; NUM_KEYS],
    voicing: &VoicingSpec,
) -> f64 {
    let k = key_idx.min(NUM_KEYS - 1);
    let choir = CHOIR_LAYOUTS[k];
    let cents = scaled_railsback_stretch_cents(k, specs, voicing)
        + (voicing.effective_pitch_cents(k) as f64)
        + unison_detune_cents(k, string_in_choir, choir.strings_per_key, voicing)
        + polarization_offset_cents(k, string_in_choir, polarization, voicing);
    shift_f0_cents(specs[k].f0_et_hz, cents)
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
        let note_cents = stretch_table[key_idx] * stretch_scale
            + (voicing.effective_pitch_cents(key_idx) as f64);

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
mod tests {
    use super::*;

    #[test]
    fn analytical_dispersion_law_matches_direct_formula() {
        let (f0, b) = (110.0, 0.00045);
        assert_eq!(partial_frequency_hz(f0, b, 0), 0.0);
        for j in 1..=32 {
            let expected = (j as f64) * f0 * (1.0 + b * (j as f64) * (j as f64)).sqrt();
            assert!((partial_frequency_hz(f0, b, j) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn railsback_stretch_is_zero_at_a4_and_monotone_across_88_keys() {
        let specs = build_scale_design(&VoicingSpec::default());
        let curve = railsback_stretch_cents_table(&specs);

        assert_eq!(curve[A4_KEY_IDX], 0.0);
        assert_eq!(railsback_stretch_cents(A4_KEY_IDX, &specs), 0.0);
        assert!((-32.0..=-20.0).contains(&curve[0]), "A0: {}", curve[0]);
        assert!(
            (25.0..=40.0).contains(&curve[NUM_KEYS - 1]),
            "C8: {}",
            curve[NUM_KEYS - 1]
        );
        for k in 1..NUM_KEYS {
            assert!(
                curve[k] > curve[k - 1],
                "k={k}: {} vs {}",
                curve[k - 1],
                curve[k]
            );
        }

        let no_stretch = VoicingSpec {
            stretch_amount: 0.0,
            ..VoicingSpec::default()
        };
        for k in 0..NUM_KEYS {
            assert_eq!(scaled_railsback_stretch_cents(k, &specs, &no_stretch), 0.0);
        }
    }

    #[test]
    fn unison_detune_and_polarization_respect_scale_and_determinism() {
        let zero_detune = VoicingSpec {
            unison_detune_scale: 0.0,
            ..VoicingSpec::default()
        };
        let nominal = VoicingSpec::default();
        let specs = build_scale_design(&nominal);

        for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
            for s in 0..(layout.strings_per_key as usize) {
                assert_eq!(
                    unison_detune_cents(k, s, layout.strings_per_key, &zero_detune),
                    0.0
                );
                assert_eq!(
                    polarization_offset_cents(k, s, Polarization::Vertical, &zero_detune),
                    0.0
                );
                assert_eq!(
                    polarization_offset_cents(k, s, Polarization::Horizontal, &zero_detune),
                    0.0
                );
            }
        }
        assert_eq!(
            effective_string_f0_hz(A4_KEY_IDX, 1, Polarization::Vertical, &specs, &zero_detune),
            440.0
        );
        assert_eq!(
            effective_string_f0_hz(
                A4_KEY_IDX,
                1,
                Polarization::Horizontal,
                &specs,
                &zero_detune
            ),
            440.0
        );

        assert_eq!(unison_detune_cents(0, 0, 1, &nominal), 0.0);
        let d0 = unison_detune_cents(A4_KEY_IDX, 0, 3, &nominal);
        let d1 = unison_detune_cents(A4_KEY_IDX, 1, 3, &nominal);
        let d2 = unison_detune_cents(A4_KEY_IDX, 2, 3, &nominal);
        assert!(d0 < -0.35 && d1.abs() < 0.20 && d2 > 0.35);
        assert_eq!(d0, unison_detune_cents(A4_KEY_IDX, 0, 3, &nominal));

        let pv = polarization_offset_cents(A4_KEY_IDX, 1, Polarization::Vertical, &nominal);
        let ph = polarization_offset_cents(A4_KEY_IDX, 1, Polarization::Horizontal, &nominal);
        assert_eq!(pv, 0.0);
        assert!((0.15..=0.30).contains(&(ph - pv)));
    }

    #[test]
    fn bank_mode_counts_hit_concert_grand_25k_to_26k5_budget() {
        let counts = analyze_bank_mode_counts(&VoicingSpec::default());
        assert!(
            (1_400..=1_900).contains(&counts.longitudinal_unpadded),
            "{counts:?}"
        );
        assert!(
            (25_000..=26_500).contains(&counts.total_padded),
            "{counts:?}"
        );
    }
}
