//! Railsback stretch tuning, SplitMix64 unison micro-detuning, bridge-pin V/H
//! coupling angles, and effective polarized string fundamental calculation.

use crate::dispersion::Polarization;
use crate::math::{exp2_f64, log2_f64, sin_cos_f64, sqrt_f64};
use crate::scale_design::{StringPhysicalParams, A4_KEY_IDX, CHOIR_LAYOUTS, NUM_KEYS};
use crate::voicing::VoicingSpec;

/// Key index of `C5` (MIDI 72), above which unison detuning tapers smoothly with frequency.
const C5_KEY_IDX: usize = 51;

/// Nominal vertical-to-horizontal bridge-pin inclination angle (`atan(0.18)` in radians).
pub const BRIDGE_PIN_BASE_ANGLE_RAD: f64 = 0.178_092_938_231_197_6;

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
/// - Trichords (`3`): characteristic `[-d, 0, +d]` triad (`d ~ 0.36..0.50 cents` up to `C5`,
///   smoothly tapering above `C5` down to `~0.42x` at `C8` so high-treble beating stays `< 0.9 Hz`)
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

    // Nominal spread d in [0.36, 0.50) cents up to C5, smoothly tapering by 2^(-0.42 * octaves)
    // above C5 down to ~0.42x at C8 (~0.20 cents) to keep high-treble beating < 0.9 Hz.
    let treble_taper = if key_idx > C5_KEY_IDX {
        let octaves_above_c5 = ((key_idx - C5_KEY_IDX) as f64) / 12.0;
        exp2_f64(-0.42 * octaves_above_c5)
    } else {
        1.0
    };
    let d = 0.36 + 0.14 * hash_to_unit(key_seed);
    let base_offset = match (strings_per_key, string_in_choir) {
        (2, 0) => -0.55 * d,
        (2, 1) => 0.55 * d,
        (3, 0) => -d,
        (3, 1) => 0.0,
        (3, 2) => d,
        _ => 0.0,
    };
    let jitter = 0.10 * hash_to_signed_unit(str_seed);
    let cond = f64::from(voicing.effective_condition());
    let cond_drift = if cond == 0.0 {
        0.0
    } else {
        1.8 * cond * hash_to_signed_unit(str_seed ^ 0x434f_4e44_554e_4953)
    };
    (base_offset + jitter + cond_drift) * treble_taper * scale
}

/// Deterministic structural bridge-pin V/H coupling angle `theta_pin(k, s)` in radians,
/// independent of `VoicingSpec::condition` and `unison_detune_scale`.
#[must_use]
pub fn bridge_pin_angle_rad(key_idx: usize, string_in_choir: usize, strings_per_key: u8) -> f64 {
    let k = key_idx.min(NUM_KEYS - 1);
    let n_str = strings_per_key.clamp(1, 3);
    let s = string_in_choir.min((n_str as usize) - 1);
    let spread = match (n_str, s) {
        (2, 0) => -0.032,
        (2, 1) => 0.032,
        (3, 0) => -0.045,
        (3, 1) => 0.004,
        (3, 2) => 0.045,
        _ => 0.0,
    };
    let seed = (k as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (s as u64)
            .wrapping_add(1)
            .wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ 0x5049_4e5f_414e_474c;
    let jitter = 0.018 * hash_to_signed_unit(seed);
    (BRIDGE_PIN_BASE_ANGLE_RAD + spread + jitter).clamp(0.10, 0.28)
}

/// Returns normalized `(c_vert, c_horiz) = (cos(theta_pin), sin(theta_pin))` bridge-pin
/// coupling projections for string `(key_idx, string_in_choir)`.
#[must_use]
pub fn bridge_pin_couplings(
    key_idx: usize,
    string_in_choir: usize,
    strings_per_key: u8,
) -> (f64, f64) {
    let theta = bridge_pin_angle_rad(key_idx, string_in_choir, strings_per_key);
    let (sin_t, cos_t) = sin_cos_f64(theta);
    (cos_t, sin_t)
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
pub(crate) fn shift_f0_cents(f0_et_hz: f64, cents: f64) -> f64 {
    if cents == 0.0 {
        f0_et_hz
    } else {
        f0_et_hz * exp2_f64(cents / 1200.0)
    }
}

/// Computes the nominal tuned key fundamental (ET + temperament + Railsback + D4 pitch offset),
/// preserving exact `440.0 Hz` at `A4` (`key_idx = 48`) with default voicing.
#[must_use]
pub fn key_tuned_f0_hz(
    key_idx: usize,
    specs: &[StringPhysicalParams; NUM_KEYS],
    voicing: &VoicingSpec,
) -> f64 {
    let k = key_idx.min(NUM_KEYS - 1);
    let base_cents = scaled_railsback_stretch_cents(k, specs, voicing)
        + (voicing.effective_pitch_cents(k) as f64);
    let temp_cents = voicing.temperament.cent_offset_for_key(k);
    let cents = if temp_cents == 0.0 {
        base_cents
    } else {
        temp_cents + base_cents
    };
    shift_f0_cents(specs[k].f0_et_hz, cents)
}

/// Computes the exact effective fundamental frequency `F0` (Hz) for a single
/// polarized transverse string after ET, temperament, Railsback stretch, D4 note offset,
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
    let base_cents = scaled_railsback_stretch_cents(k, specs, voicing)
        + (voicing.effective_pitch_cents(k) as f64)
        + unison_detune_cents(k, string_in_choir, choir.strings_per_key, voicing)
        + polarization_offset_cents(k, string_in_choir, polarization, voicing);
    let temp_cents = voicing.temperament.cent_offset_for_key(k);
    let cents = if temp_cents == 0.0 {
        base_cents
    } else {
        temp_cents + base_cents
    };
    shift_f0_cents(specs[k].f0_et_hz, cents)
}
