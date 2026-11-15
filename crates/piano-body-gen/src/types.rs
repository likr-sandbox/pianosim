//! Shared structural-acoustic mode types, unit conversions, and damping utilities.

use core::f64::consts::TAU;

use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};

/// Constant `3 * ln(10)` used to convert between `T60` decay time and modal `zeta`.
pub const THREE_LN_10: f64 = 6.907_755_278_982_137;

/// Minimum allowed structural-acoustic mode frequency in Hz.
pub const MIN_BODY_MODE_HZ: f64 = 30.0;

/// Upper extension frequency for concert-grand structural-acoustic modes in Hz.
pub const MAX_BODY_MODE_HZ: f64 = 16_500.0;

/// Upper bound on modal `T60` decay time in seconds (`<= 0.42 s` for clean `0.5 s` IRs).
pub const MAX_MODAL_T60_S: f64 = 0.42;

/// Physical subsystem that originated a solved structural-acoustic mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModeSource {
    /// Deterministic Rayleigh-Ritz eigenmode of the 2D orthotropic soundboard + ribs + bridges + rim.
    PlateRitz,
    /// Smooth high-frequency Rayleigh statistical overlap soundboard mode.
    PlateStatistical,
    /// Laminated beech rim and Zone 8 wooden keybed structural mode (`40..380 Hz`).
    RimKeybed,
    /// Weakly coupled cast-iron harp frame structural mode (`180..2,800 Hz`).
    CastIronFrame,
    /// 3D interior cabinet and under-board acoustic air cavity mode (`65..550 Hz`).
    AirCavity,
}

/// High-precision (`f64`) structural-acoustic mode produced by Layer 1A physical solvers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SolvedBodyMode {
    /// Natural modal frequency in Hz.
    pub freq_hz: f64,
    /// Viscous damping ratio `zeta > 0`.
    pub zeta: f64,
    /// Effective modal mass in kilograms (`> 0`).
    pub modal_mass_kg: f64,
    /// Modal admittance/drive weights across the 9 radiator zones (`0..7` bridge, `8` keybed).
    pub zone_drive: [f64; PRESET_RADIATOR_ZONES],
    /// Directional radiation residue `[monopole, dipole_x, dipole_y, dipole_z, lid_coupling]`.
    pub rad_coeffs: [f64; PRESET_RAD_COEFFS],
    /// Classification flag (`0` = structural plate/rim/frame, `1` = air cavity coupled).
    pub cavity_flag: u8,
    /// Originating physical subsystem.
    pub source: ModeSource,
}

/// Converts a 60 dB decay time `t60_s` at `freq_hz` into viscous damping ratio `zeta`.
#[must_use]
#[inline]
pub fn t60_to_zeta(freq_hz: f64, t60_s: f64) -> f64 {
    let f = freq_hz.max(1.0);
    let t60 = t60_s.max(1.0e-4);
    THREE_LN_10 / (TAU * f * t60)
}

/// Converts viscous damping ratio `zeta` at `freq_hz` into 60 dB decay time `T60` in seconds.
#[must_use]
#[inline]
pub fn zeta_to_t60(freq_hz: f64, zeta: f64) -> f64 {
    let denom = TAU * freq_hz.max(1.0) * zeta.max(1.0e-9);
    THREE_LN_10 / denom
}

/// Physical Sitka spruce modal `T60(f)` tapering from `~0.38 s` at low freq to `~0.04 s` at `16.5 kHz`.
#[must_use]
#[inline]
pub fn sitka_spruce_t60_s(freq_hz: f64) -> f64 {
    let f = freq_hz.clamp(20.0, 22_000.0);
    let t60 = 0.028 + 0.365 / (1.0 + (f / 460.0).powf(1.02));
    t60.clamp(0.032, 0.410)
}

/// Deterministic fractional hash in `[0.0, 1.0)` from integer indices (`SplitMix64`).
#[must_use]
#[inline]
pub fn det_unit_f64(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}
