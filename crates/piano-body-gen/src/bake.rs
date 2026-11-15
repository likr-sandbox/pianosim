//! Deterministic concert-grand Layer 1A body preset baker and diagnostic builder.

use crate::air_cavity::{populate_radiation_multipoles, synthesize_air_cavity_modes};
use crate::orthotropic_plate::{solve_soundboard_modes, PlateSolveStats};
use crate::pole_cancel::{extract_stage3_reduced_model, CALIBRATED_DIRECT_ZONE_GAIN};
use crate::ribs_rim::{
    synthesize_cast_iron_frame_modes, synthesize_rim_keybed_modes, CONCERT_ZONE_POS_M,
};
use crate::types::{zeta_to_t60, ModeSource};
use piano_spec::{BodyModeRecord, BodyPreset};

/// Complete Layer 1A solver breakdown and preset size diagnostics.
#[derive(Clone, Debug)]
pub struct BakeDiagnostics {
    /// Soundboard Rayleigh-Ritz and statistical modal overlap counts.
    pub plate: PlateSolveStats,
    /// Number of laminated beech rim and Zone 8 keybed coupling modes.
    pub rim_keybed_modes: usize,
    /// Number of weakly coupled cast-iron harp frame modes.
    pub frame_modes: usize,
    /// Number of 3D interior cabinet & under-board air cavity modes (`cavity_flag = 1`).
    pub cavity_modes: usize,
    /// Total structural-acoustic + cavity modes serialized into the preset.
    pub total_modes: usize,
    /// Lowest mode frequency in the baked preset [Hz].
    pub min_freq_hz: f64,
    /// Highest mode frequency in the baked preset [Hz].
    pub max_freq_hz: f64,
    /// Longest modal `T60` across the entire preset [s].
    pub max_t60_s: f64,
    /// Shortest modal `T60` across the entire preset [s].
    pub min_t60_s: f64,
    /// Size in bytes of `preset.encode()`.
    pub encoded_bytes: usize,
}

/// Runs the pure-Rust Layer 1A structural-acoustic concert-grand solver and returns
/// the calibrated [`BodyPreset`] along with solver diagnostics.
#[must_use]
pub fn bake_concert_grand_with_diagnostics() -> (BodyPreset, BakeDiagnostics) {
    let (mut modes, plate_stats) = solve_soundboard_modes();
    let rim_modes = synthesize_rim_keybed_modes();
    let frame_modes = synthesize_cast_iron_frame_modes();
    let cavity_modes = synthesize_air_cavity_modes();

    let rim_count = rim_modes.len();
    let frame_count = frame_modes.len();
    let cavity_count = cavity_modes.len();

    modes.extend(rim_modes);
    modes.extend(frame_modes);
    modes.extend(cavity_modes);

    modes.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
    populate_radiation_multipoles(&mut modes);

    let (y_inf, reduced_poles) = extract_stage3_reduced_model();

    let mut min_freq_hz = f64::MAX;
    let mut max_freq_hz = 0.0_f64;
    let mut max_t60_s = 0.0_f64;
    let mut min_t60_s = f64::MAX;

    let mut records = Vec::with_capacity(modes.len());
    for m in &modes {
        min_freq_hz = min_freq_hz.min(m.freq_hz);
        max_freq_hz = max_freq_hz.max(m.freq_hz);
        let t60 = zeta_to_t60(m.freq_hz, m.zeta);
        max_t60_s = max_t60_s.max(t60);
        min_t60_s = min_t60_s.min(t60);
        let _src: ModeSource = m.source;

        records.push(BodyModeRecord {
            freq_hz: m.freq_hz as f32,
            zeta: m.zeta as f32,
            zone_drive: std::array::from_fn(|z| m.zone_drive[z] as f32),
            rad_coeffs: std::array::from_fn(|r| m.rad_coeffs[r] as f32),
            cavity_flag: m.cavity_flag,
        });
    }

    let preset = BodyPreset {
        y_inf,
        direct_zone_gain: CALIBRATED_DIRECT_ZONE_GAIN,
        zone_pos_m: CONCERT_ZONE_POS_M,
        reduced_poles,
        modes: records,
    };

    let encoded_bytes = preset.encode().len();
    let diag = BakeDiagnostics {
        plate: plate_stats,
        rim_keybed_modes: rim_count,
        frame_modes: frame_count,
        cavity_modes: cavity_count,
        total_modes: preset.modes.len(),
        min_freq_hz,
        max_freq_hz,
        max_t60_s,
        min_t60_s,
        encoded_bytes,
    };
    (preset, diag)
}

/// Deterministic concert-grand Layer 1A structural-acoustic preset generator.
#[must_use]
pub fn bake_concert_grand_preset() -> BodyPreset {
    bake_concert_grand_with_diagnostics().0
}
