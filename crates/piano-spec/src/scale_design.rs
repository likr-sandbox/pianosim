//! Concert grand 88-key / 230-string physical scaling law (`A0`..=`C8`).
//!
//! Computes analytical speaking lengths, wire core radii, linear densities
//! (accounting for copper-wound bass strings on keys `0..26`), tensions via
//! Mersenne's law, bending stiffnesses, longitudinal fundamental frequencies,
//! and analytical inharmonicity coefficients `B(k)`.

use core::f64::consts::PI;

use crate::math::{exp2_f64, exp_f64, powf_f64, sqrt_f64};
use crate::voicing::VoicingSpec;

/// Standard 88-key piano keyboard count (`A0`..=`C8`).
pub const NUM_KEYS: usize = 88;
/// MIDI note number for `A0` (key index `0`).
pub const MIDI_A0: u8 = 21;
/// MIDI note number for `C8` (key index `87`).
pub const MIDI_C8: u8 = 108;
/// Key index of concert `A4` (MIDI 69).
pub const A4_KEY_IDX: usize = 48;
/// MIDI note number of concert `A4`.
pub const A4_MIDI: u8 = 69;
/// Concert pitch reference frequency for `A4` in Hz.
pub const A4_FREQ_HZ: f64 = 440.0;

/// Number of bass monochord keys (`0..8`, 1 string per key).
pub const MONOCHORD_KEYS: usize = 8;
/// Number of bass copper-wound bichord keys (`8..26`, 2 strings per key).
pub const BICHORD_KEYS: usize = 18;
/// Number of plain steel trichord keys (`26..88`, 3 strings per key).
pub const TRICHORD_KEYS: usize = 62;
/// Total physical strings across all 88 keys (`8 + 18*2 + 62*3 = 230`).
pub const TOTAL_STRINGS: usize = 230;
/// Number of damped keys (`0..70`, MIDI `21..=90`); top 18 keys are undamped.
pub const DAMPED_KEYS: usize = 70;

/// Young's modulus of music steel wire in Pascals (`2.0e11 Pa`).
pub const STEEL_YOUNGS_MODULUS_PA: f64 = 2.0e11;
/// Volumetric mass density of music steel wire in `kg/m^3`.
pub const STEEL_DENSITY_KG_M3: f64 = 7850.0;

const C8_SPEAKING_LENGTH_M: f64 = 0.044;
const LENGTH_FORESHORTEN_U0: f64 = 48.55;
const LENGTH_FORESHORTEN_Q: f64 = 1.9;
const BASS_B_FLOOR: f64 = 0.000_315;
const BASS_B_RISE: f64 = 0.000_060;
const TREBLE_B_C8_SCALE: f64 = 0.045_045;
const TREBLE_B_OCTAVE_DIV: f64 = 5.8;
const TREBLE_BLEND_MID_KEY: f64 = 51.0;
const TREBLE_BLEND_WIDTH: f64 = 5.2;
const BASS_CORE_RADIUS_A0_M: f64 = 0.000_44;
const BASS_TENSION_A0_N: f64 = 1280.0;

/// Choir geometry and damper presence for a single piano key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ChoirLayout {
    /// Number of unison strings struck for this key (`1`, `2`, or `3`).
    pub strings_per_key: u8,
    /// Starting index in the flat `0..230` string bank.
    pub first_string_idx: u16,
    /// Whether this key has an acoustic felt damper (`true` for keys `0..70`).
    pub has_damper: bool,
}

/// Complete `88`-key choir layout table partitioning `0..230` contiguously.
pub const CHOIR_LAYOUTS: [ChoirLayout; NUM_KEYS] = build_choir_layouts();

const fn build_choir_layouts() -> [ChoirLayout; NUM_KEYS] {
    let mut table = [ChoirLayout {
        strings_per_key: 0,
        first_string_idx: 0,
        has_damper: false,
    }; NUM_KEYS];
    let mut string_cursor: u16 = 0;
    let mut k = 0;
    while k < NUM_KEYS {
        let strings_per_key = if k < MONOCHORD_KEYS {
            1
        } else if k < MONOCHORD_KEYS + BICHORD_KEYS {
            2
        } else {
            3
        };
        table[k] = ChoirLayout {
            strings_per_key,
            first_string_idx: string_cursor,
            has_damper: k < DAMPED_KEYS,
        };
        string_cursor += strings_per_key as u16;
        k += 1;
    }
    table
}

/// Maps a keyboard index (`0..88`) to its MIDI note number (`21..=108`).
#[must_use]
pub const fn key_to_midi(key_idx: usize) -> Option<u8> {
    if key_idx < NUM_KEYS {
        Some(MIDI_A0 + key_idx as u8)
    } else {
        None
    }
}

/// Maps a MIDI note number (`21..=108`) to its 88-key index (`0..88`).
#[must_use]
pub const fn midi_to_key(midi_key: u8) -> Option<usize> {
    if midi_key >= MIDI_A0 && midi_key <= MIDI_C8 {
        Some((midi_key - MIDI_A0) as usize)
    } else {
        None
    }
}

/// Returns the [`ChoirLayout`] for `key_idx` (`0..88`), or `None` if out of range.
#[must_use]
pub const fn key_choir(key_idx: usize) -> Option<ChoirLayout> {
    if key_idx < NUM_KEYS {
        Some(CHOIR_LAYOUTS[key_idx])
    } else {
        None
    }
}

/// Computes the 12-tone equal temperament fundamental frequency in Hz (`A4 = 440.0`).
///
/// Returns `0.0` if `key_idx >= NUM_KEYS`.
#[must_use]
pub fn equal_temperament_hz(key_idx: usize) -> f64 {
    if key_idx >= NUM_KEYS {
        return 0.0;
    }
    if key_idx == A4_KEY_IDX {
        return A4_FREQ_HZ;
    }
    let semitones = (key_idx as i32) - (A4_KEY_IDX as i32);
    A4_FREQ_HZ * exp2_f64(f64::from(semitones) / 12.0)
}

/// Physical string parameters (`f64` precision) for one piano key choir.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StringPhysicalParams {
    /// Effective speaking string length `L(k)` in meters.
    pub speaking_length_m: f64,
    /// Steel core wire radius `r_core(k)` in meters.
    pub core_radius_m: f64,
    /// Total linear mass density `mu(k)` in `kg/m` (includes copper winding on `0..26`).
    pub linear_density_kg_m: f64,
    /// Static string tension `T_0(k)` in Newtons satisfying `F0 = (1/2L)*sqrt(T0/mu)`.
    pub tension_n: f64,
    /// Effective bending stiffness `E * S_core * K^2` in `N * m^2`.
    pub bending_stiffness_esk2: f64,
    /// Analytical inharmonicity coefficient `B(k) = pi^2 * ESK^2 / (T0 * L^2)`.
    pub inharmonicity_b: f64,
    /// Longitudinal wave fundamental frequency `(1/2L) * sqrt(E * S_core / mu)` in Hz.
    pub f_long_hz: f64,
    /// Reference 12-TET transverse fundamental frequency in Hz.
    pub f0_et_hz: f64,
}

#[inline]
fn base_speaking_length_m(key_idx: usize) -> f64 {
    let octaves_below_c8 = ((NUM_KEYS - 1) - key_idx) as f64 / 12.0;
    let u = exp2_f64(octaves_below_c8);
    let ratio = u / LENGTH_FORESHORTEN_U0;
    let denom = powf_f64(
        1.0 + powf_f64(ratio, LENGTH_FORESHORTEN_Q),
        1.0 / LENGTH_FORESHORTEN_Q,
    );
    C8_SPEAKING_LENGTH_M * u / denom
}

#[inline]
fn target_inharmonicity_b(key_idx: usize) -> f64 {
    let t = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let w = 1.0 / (1.0 + exp_f64(-((key_idx as f64) - TREBLE_BLEND_MID_KEY) / TREBLE_BLEND_WIDTH));
    let b_low = BASS_B_FLOOR + BASS_B_RISE * powf_f64(t, 2.2);
    let b_hi = TREBLE_B_C8_SCALE * exp2_f64(((key_idx as f64) - 87.0) / TREBLE_B_OCTAVE_DIV);
    let upper = if b_hi > b_low { b_hi } else { b_low };
    b_low * (1.0 - w) + upper * w
}

#[inline]
fn plain_steel_radius_and_tension(l_m: f64, f0_hz: f64, b_target: f64) -> (f64, f64, f64) {
    let l2 = l_m * l_m;
    let l4 = l2 * l2;
    let r_sq_num = 16.0 * STEEL_DENSITY_KG_M3 * l4 * f0_hz * f0_hz * b_target;
    let r_sq_den = PI * PI * STEEL_YOUNGS_MODULUS_PA;
    let r_core = sqrt_f64(r_sq_num / r_sq_den);
    let s_core = PI * r_core * r_core;
    let mu = STEEL_DENSITY_KG_M3 * s_core;
    let tension = 4.0 * mu * l2 * f0_hz * f0_hz;
    (r_core, mu, tension)
}

/// Computes the physical string parameters for `key_idx` under `voicing`.
///
/// Out-of-range indices clamp safely to `87` without panicking.
#[must_use]
pub fn string_params_for_key(key_idx: usize, voicing: &VoicingSpec) -> StringPhysicalParams {
    let k = if key_idx >= NUM_KEYS {
        NUM_KEYS - 1
    } else {
        key_idx
    };
    let f0_et_hz = equal_temperament_hz(k);
    let l_base = base_speaking_length_m(k);
    let b_base = target_inharmonicity_b(k);

    let plain_break = MONOCHORD_KEYS + BICHORD_KEYS;
    let (core_radius_m, linear_density_kg_m, tension_base, esk2_base) = if k >= plain_break {
        let (r_core, mu, t0) = plain_steel_radius_and_tension(l_base, f0_et_hz, b_base);
        let s_core = PI * r_core * r_core;
        let esk2 = 0.25 * STEEL_YOUNGS_MODULUS_PA * s_core * r_core * r_core;
        (r_core, mu, t0, esk2)
    } else {
        // Copper-wound bass strings: interpolate core radius and tension smoothly
        // into the plain-steel break while matching effective wound bending stiffness.
        let l_26 = base_speaking_length_m(plain_break);
        let f0_26 = equal_temperament_hz(plain_break);
        let b_26 = target_inharmonicity_b(plain_break);
        let (r_26, _, t_26) = plain_steel_radius_and_tension(l_26, f0_26, b_26);

        let s = (k as f64) / (plain_break as f64);
        let r_core = BASS_CORE_RADIUS_A0_M + (r_26 - BASS_CORE_RADIUS_A0_M) * powf_f64(s, 2.4);
        let t0 = BASS_TENSION_A0_N + (t_26 - BASS_TENSION_A0_N) * powf_f64(s, 1.4);
        let mu = t0 / (4.0 * l_base * l_base * f0_et_hz * f0_et_hz);
        let esk2 = b_base * t0 * l_base * l_base / (PI * PI);
        (r_core, mu, t0, esk2)
    };

    let length_scale = f64::from(voicing.effective_string_length_scale());
    let inharm_scale = f64::from(voicing.effective_inharmonicity_scale(k));

    let speaking_length_m = l_base * length_scale;
    let tension_n = tension_base * length_scale * length_scale;
    let bending_stiffness_esk2 = esk2_base * inharm_scale;
    let inharmonicity_b =
        (PI * PI * bending_stiffness_esk2) / (tension_n * speaking_length_m * speaking_length_m);
    let s_core = PI * core_radius_m * core_radius_m;
    let f_long_hz = (0.5 / speaking_length_m)
        * sqrt_f64(STEEL_YOUNGS_MODULUS_PA * s_core / linear_density_kg_m);

    StringPhysicalParams {
        speaking_length_m,
        core_radius_m,
        linear_density_kg_m,
        tension_n,
        bending_stiffness_esk2,
        inharmonicity_b,
        f_long_hz,
        f0_et_hz,
    }
}

/// Builds the full 88-key [`StringPhysicalParams`] table for `voicing`.
#[must_use]
pub fn build_scale_design(voicing: &VoicingSpec) -> [StringPhysicalParams; NUM_KEYS] {
    let placeholder = StringPhysicalParams {
        speaking_length_m: 0.0,
        core_radius_m: 0.0,
        linear_density_kg_m: 0.0,
        tension_n: 0.0,
        bending_stiffness_esk2: 0.0,
        inharmonicity_b: 0.0,
        f_long_hz: 0.0,
        f0_et_hz: 0.0,
    };
    let mut specs = [placeholder; NUM_KEYS];
    let mut k = 0;
    while k < NUM_KEYS {
        specs[k] = string_params_for_key(k, voicing);
        k += 1;
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choir_layout_partitions_230_strings_and_70_dampers() {
        let mut total_strings = 0usize;
        let mut damped = 0usize;
        let mut monochord = 0usize;
        let mut bichord = 0usize;
        let mut trichord = 0usize;

        for k in 0..NUM_KEYS {
            let choir = key_choir(k).expect("valid key");
            assert_eq!(usize::from(choir.first_string_idx), total_strings);
            total_strings += usize::from(choir.strings_per_key);
            if choir.has_damper {
                damped += 1;
            }
            match choir.strings_per_key {
                1 => monochord += 1,
                2 => bichord += 1,
                3 => trichord += 1,
                other => panic!("unexpected choir count {other}"),
            }
        }

        assert_eq!(monochord, 8);
        assert_eq!(bichord, 18);
        assert_eq!(trichord, 62);
        assert_eq!(total_strings, TOTAL_STRINGS);
        assert_eq!(8 + 18 * 2 + 62 * 3, TOTAL_STRINGS);
        assert_eq!(damped, DAMPED_KEYS);
        assert_eq!(NUM_KEYS - damped, 18);
        assert!(key_choir(NUM_KEYS).is_none());
    }

    #[test]
    fn midi_mapping_and_equal_temperament_precision() {
        assert_eq!(key_to_midi(0), Some(MIDI_A0));
        assert_eq!(key_to_midi(48), Some(69));
        assert_eq!(key_to_midi(87), Some(MIDI_C8));
        assert_eq!(key_to_midi(88), None);
        assert_eq!(midi_to_key(20), None);
        assert_eq!(midi_to_key(21), Some(0));
        assert_eq!(midi_to_key(69), Some(48));
        assert_eq!(midi_to_key(108), Some(87));
        assert_eq!(midi_to_key(109), None);

        assert_eq!(equal_temperament_hz(48), 440.0);
        assert!((equal_temperament_hz(0) - 27.5).abs() < 1e-12);
        assert!((equal_temperament_hz(87) - 4186.009_044_809_578).abs() < 1e-9);
    }

    #[test]
    fn physical_scaling_matches_analytical_laws_and_targets() {
        let specs = build_scale_design(&VoicingSpec::default());

        assert!((specs[0].speaking_length_m - 2.02).abs() < 0.03);
        assert!((specs[87].speaking_length_m - 0.044).abs() < 0.001);
        assert!(
            (0.000_30..=0.000_42).contains(&specs[0].inharmonicity_b),
            "B(0)={}",
            specs[0].inharmonicity_b
        );
        assert!(
            (0.040..=0.050).contains(&specs[87].inharmonicity_b),
            "B(87)={}",
            specs[87].inharmonicity_b
        );

        let bass_long_ratio = specs[0].f_long_hz / specs[0].f0_et_hz;
        assert!(
            (5.0..=15.0).contains(&bass_long_ratio),
            "bass f_long/F0 = {bass_long_ratio}"
        );

        for k in 0..NUM_KEYS {
            let p = specs[k];
            let f0_check =
                (0.5 / p.speaking_length_m) * (p.tension_n / p.linear_density_kg_m).sqrt();
            assert!((f0_check - p.f0_et_hz).abs() < 1e-10 * p.f0_et_hz);

            let b_check = (PI * PI * p.bending_stiffness_esk2)
                / (p.tension_n * p.speaking_length_m * p.speaking_length_m);
            assert!((b_check - p.inharmonicity_b).abs() < 1e-13);

            if k > 0 {
                assert!(p.speaking_length_m < specs[k - 1].speaking_length_m);
                assert!(p.inharmonicity_b > specs[k - 1].inharmonicity_b);
            }
        }
    }
}
