//! 8-zone physical bridge projection (with 9th keybed radiator zone reserved).
//!
//! Maps all `TOTAL_STRINGS = 230` physical strings smoothly onto the 8 bridge
//! zones `(z0, w0, z1, w1)` obeying an exact Partition of Unity
//! (`w0 + w1 == 1.0`, `w0 >= 0.0`, `w1 >= 0.0`). Zone index `8` of
//! `NUM_RADIATOR_ZONES = 9` is strictly reserved for direct keybed thump in M6/M7
//! without artificial grouping or post-delay (FTO #5/#6).

use core::f64::consts::FRAC_PI_2;
use piano_spec::scale_design::{BICHORD_KEYS, MONOCHORD_KEYS};
use piano_spec::{CHOIR_LAYOUTS, NUM_KEYS, TOTAL_STRINGS};

use crate::math::sin_cos_f64;

/// Number of physical soundboard bridge coupling zones (`0..8`).
pub const NUM_BRIDGE_ZONES: usize = 8;

/// Total radiator input zones (`0..8` bridge + zone `8` reserved for keybed thump).
pub const NUM_RADIATOR_ZONES: usize = 9;

/// Reserved radiator zone index for Stage 4 keybed structural impulse (FTO #6).
pub const KEYBED_ZONE_IDX: usize = 8;

/// First bridge zone index coupled passively to `DuplexBank` (`2..8`).
pub const TREBLE_DUPLEX_ZONE_START: usize = 2;

const BASS_KEY_COUNT: usize = MONOCHORD_KEYS + BICHORD_KEYS;

/// Per-string bridge projection entry: `(zone0, weight0, zone1, weight1)`.
pub type ZoneWeight = (u8, f32, u8, f32);

/// Returns continuous bridge zone coordinate `u in [0.0, 7.0]` for `key_idx`.
///
/// Copper-wound bass strings (`0..26`) smoothly span the bass bridge (`0.0..2.0`);
/// plain-steel tenor/treble trichords (`26..88`) span the long bridge (`2.0..7.0`).
#[must_use]
#[inline]
pub fn key_zone_coord(key_idx: usize) -> f64 {
    let k = key_idx.min(NUM_KEYS - 1);
    if k < BASS_KEY_COUNT {
        2.0 * (k as f64) / (BASS_KEY_COUNT as f64)
    } else {
        let treble_span = (NUM_KEYS - 1 - BASS_KEY_COUNT) as f64;
        2.0 + 5.0 * ((k - BASS_KEY_COUNT) as f64) / treble_span
    }
}

/// Computes an exact IEEE-754 `f32` Partition-of-Unity pair `(w0, w1)` from
/// `raw_w1` in `[0.0, 1.0]` such that `w0 >= 0.0`, `w1 >= 0.0`, and `w0 + w1 == 1.0`.
#[must_use]
#[inline]
pub fn exact_partition_pair(raw_w1: f32) -> (f32, f32) {
    let clamped = if raw_w1.is_finite() {
        raw_w1.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let w0 = 1.0_f32 - clamped;
    let w1 = 1.0_f32 - w0;
    (w0, w1)
}

/// Builds the complete `230`-string 8-zone bridge projection map.
#[must_use]
pub fn build_zone_weights() -> [ZoneWeight; TOTAL_STRINGS] {
    let mut weights = [(0_u8, 1.0_f32, 1_u8, 0.0_f32); TOTAL_STRINGS];
    for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
        let u = key_zone_coord(k).clamp(0.0, (NUM_BRIDGE_ZONES - 1) as f64);
        let z0 = (u as usize).min(NUM_BRIDGE_ZONES - 2);
        let z1 = z0 + 1;
        let raw_w1 = (u - (z0 as f64)) as f32;
        let (w0, w1) = exact_partition_pair(raw_w1);
        let start = usize::from(layout.first_string_idx);
        let count = usize::from(layout.strings_per_key);
        for slot in &mut weights[start..start + count] {
            *slot = (z0 as u8, w0, z1 as u8, w1);
        }
    }
    weights
}

/// Accumulates per-string quantities (such as bridge force or broadband audio)
/// onto the 8 bridge zones via transpose projection `W^T x`.
#[must_use]
#[inline]
pub fn project_strings_to_zones(
    weights: &[ZoneWeight; TOTAL_STRINGS],
    string_vals: &[f32; TOTAL_STRINGS],
) -> [f32; NUM_BRIDGE_ZONES] {
    let mut zones = [0.0_f32; NUM_BRIDGE_ZONES];
    for (s_idx, &(z0, w0, z1, w1)) in weights.iter().enumerate() {
        let val = string_vals[s_idx];
        zones[usize::from(z0)] += w0 * val;
        zones[usize::from(z1)] += w1 * val;
    }
    zones
}

/// Gathers per-zone bridge velocities back onto each of the 230 strings via `W v_zone`.
#[inline]
pub fn gather_zones_to_strings(
    weights: &[ZoneWeight; TOTAL_STRINGS],
    zone_vals: &[f32; NUM_BRIDGE_ZONES],
    out: &mut [f32; TOTAL_STRINGS],
) {
    for (s_idx, &(z0, w0, z1, w1)) in weights.iter().enumerate() {
        out[s_idx] = w0 * zone_vals[usize::from(z0)] + w1 * zone_vals[usize::from(z1)];
    }
}

/// Constant-power left/right stereo pan `(pan_l, pan_r)` across bridge zones `0..8`.
#[must_use]
#[inline]
pub fn zone_stereo_pan(zone_idx: usize) -> (f32, f32) {
    let z = zone_idx.min(NUM_BRIDGE_ZONES - 1);
    let norm = (z as f64) / ((NUM_BRIDGE_ZONES - 1) as f64);
    let angle = (0.22 + 0.56 * norm) * FRAC_PI_2;
    let (sin_a, cos_a) = sin_cos_f64(angle);
    (cos_a as f32, sin_a as f32)
}
