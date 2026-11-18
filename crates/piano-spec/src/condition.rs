//! Deterministic `SplitMix64` aging (`condition`) macro perturbations.
//!
//! Computes subtle, repeatable per-key aging offsets (`pitch_cents`,
//! `inharmonicity_scale`, `hammer_hardness_offset`, `level_db_offset`,
//! `damping_scale`) when `condition > 0.0`, and exact [`NoteVoicingOffset::ZERO`]
//! when `condition <= 0.0` or non-finite.

use crate::dispersion::splitmix64;
use crate::scale_design::NUM_KEYS;
use crate::voicing::NoteVoicingOffset;

/// Default aging condition macro (`0.0` = mint concert regulation).
pub const CONDITION_DEFAULT: f32 = 0.0;
/// Minimum clamped aging condition macro.
pub const CONDITION_MIN: f32 = 0.0;
/// Maximum clamped aging condition macro (`1.0` = heavily aged instrument).
pub const CONDITION_MAX: f32 = 1.0;

#[inline]
fn hash_signed_unit_f32(seed: u64) -> f32 {
    const INV_2_24: f32 = 1.0 / ((1u32 << 24) as f32);
    let unit = ((splitmix64(seed) >> 40) as f32) * INV_2_24;
    2.0 * unit - 1.0
}

#[inline]
pub(crate) fn clamp_condition(condition: f32) -> f32 {
    if condition.is_nan() {
        CONDITION_DEFAULT
    } else {
        condition.clamp(CONDITION_MIN, CONDITION_MAX)
    }
}

/// Computes deterministic per-key aging perturbations driven by `condition` (`[0.0, 1.0]`)
/// via `SplitMix64`. Returns exact [`NoteVoicingOffset::ZERO`] when `condition <= 0.0`
/// (or non-finite) or `key_idx >= 88`.
#[must_use]
pub fn condition_note_offset(condition: f32, key_idx: usize) -> NoteVoicingOffset {
    let c = clamp_condition(condition);
    if c == 0.0 || key_idx >= NUM_KEYS {
        return NoteVoicingOffset::ZERO;
    }

    let k64 = key_idx as u64;
    let pitch_hash =
        hash_signed_unit_f32(k64.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x434f_4e44_5049_5443);
    let inharm_hash =
        hash_signed_unit_f32(k64.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ 0x434f_4e44_494e_4841);
    let hammer_hash =
        hash_signed_unit_f32(k64.wrapping_mul(0xbf58_476d_1ce4_e5b9) ^ 0x434f_4e44_4841_4d4d);
    let level_hash =
        hash_signed_unit_f32(k64.wrapping_mul(0x517c_c1b7_2722_0a95) ^ 0x434f_4e44_4c45_564c);
    let damp_hash =
        hash_signed_unit_f32(k64.wrapping_mul(0x94d0_49bb_1331_11eb) ^ 0x434f_4e44_4441_4d50);

    // Mid-register hammer wear bias: greatest around C4..E5 where felt grooves most.
    let norm_mid = ((key_idx as f32) - 44.0) / 44.0;
    let mid_wear = (1.0 - norm_mid * norm_mid).max(0.0);

    NoteVoicingOffset {
        pitch_cents: c * (-1.2 + 6.0 * pitch_hash),
        inharmonicity_scale: c * (0.03 * inharm_hash),
        hammer_hardness_offset: c * (0.08 * mid_wear + 0.07 * hammer_hash),
        level_db_offset: c * (0.9 * level_hash),
        damping_scale: c * (0.04 + 0.08 * damp_hash),
    }
}
