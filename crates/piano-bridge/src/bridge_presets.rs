//! Precomputed physical soundboard/bridge modal constants and default string
//! port-resistance builders for [`crate::AdmittanceSsm`].

use piano_spec::{
    build_scale_design, PresetReducedPole, VoicingSpec, CHOIR_LAYOUTS, TOTAL_STRINGS,
};

use crate::math::sanitize_sample_rate;
use crate::zone_proj::NUM_BRIDGE_ZONES;

/// Number of reduced bidirectional soundboard/bridge modes (`~54..1,130 Hz`).
pub const NUM_REDUCED_MODES: usize = 10;

const MIN_Y_INF: f32 = 1.0e-9;
const MAX_Y_INF: f32 = 0.05;
const MIN_POLE_FREQ_HZ: f32 = 1.0;
const MAX_POLE_FREQ_HZ: f32 = 24_000.0;
const MIN_POLE_ZETA: f32 = 1.0e-6;
const MAX_POLE_ZETA: f32 = 5.0;
const MIN_MODAL_MASS_KG: f32 = 1.0e-4;
const MAX_MODAL_MASS_KG: f32 = 1.0e4;
const MAX_ABS_ZONE_GAIN: f32 = 10.0;

/// Publicly exported reduced bridge pole descriptor for M6 `piano-body-gen` pole cancellation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReducedBridgePole {
    /// Modal natural frequency in Hz.
    pub freq_hz: f32,
    /// Viscous damping ratio `zeta` (`0.0` in lossless test mode).
    pub zeta: f32,
    /// Modal quality factor `Q = 1 / (2 * zeta)`.
    pub q_factor: f32,
    /// Effective modal mass in kilograms.
    pub modal_mass_kg: f32,
    /// Dimensionless spatial mode gains at each of the 8 bridge zones.
    pub zone_gains: [f32; NUM_BRIDGE_ZONES],
}

/// Default high-frequency passive local resistive mobility `y_inf` `[m / (N*s)]`
/// across bridge zones `0..8`, preventing any `1.13 kHz` brick wall on treble/longitudinal signals
/// and matching acoustic literature bridge mobility (`~1e-4..1e-3 m/(N*s)`).
pub const DEFAULT_Y_INF: [f32; NUM_BRIDGE_ZONES] = [
    0.000_420, 0.000_360, 0.000_400, 0.000_350, 0.000_305, 0.000_510, 0.000_500, 0.000_520,
];

const MODE_FREQS_HZ: [f32; NUM_REDUCED_MODES] = [
    54.0, 89.0, 142.0, 212.0, 285.0, 405.0, 530.0, 730.0, 930.0, 1130.0,
];
const MODE_ZETAS: [f32; NUM_REDUCED_MODES] = [
    0.055, 0.042, 0.030, 0.020, 0.020, 0.0145, 0.024, 0.021, 0.020, 0.019,
];
const MODE_MASSES_KG: [f32; NUM_REDUCED_MODES] =
    [3.60, 2.50, 1.25, 0.68, 0.48, 0.385, 0.42, 0.58, 0.72, 0.88];

/// Dimensionless spatial mode gains at each of the 8 bridge zones for the 10 reduced modes.
pub const MODE_ZONE_GAINS: [[f32; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES] = [
    [0.12, 0.18, 0.30, 0.36, 0.34, 0.26, 0.18, 0.10],
    [0.16, 0.18, 0.18, 0.06, -0.16, -0.28, -0.26, -0.16],
    [0.05, 0.06, -0.26, -0.32, -0.28, -0.12, 0.18, 0.22],
    [0.04, -0.05, -0.34, -0.36, -0.20, 0.26, 0.20, -0.16],
    [-0.04, 0.05, 0.28, 0.40, 0.26, -0.20, -0.26, 0.18],
    [0.03, -0.04, 0.21, 0.45, -0.29, -0.38, 0.16, 0.24],
    [-0.05, 0.08, -0.20, 0.30, 0.30, -0.24, -0.22, 0.20],
    [-0.04, 0.07, -0.20, 0.24, -0.24, -0.24, 0.26, -0.22],
    [0.04, -0.06, 0.20, -0.22, 0.24, -0.22, -0.28, 0.26],
    [-0.04, 0.06, -0.20, 0.22, -0.24, 0.26, -0.26, 0.28],
];

/// Builds the 10 reduced soundboard/bridge poles for normal or lossless test mode.
#[must_use]
pub(crate) fn build_default_poles(lossless: bool) -> [ReducedBridgePole; NUM_REDUCED_MODES] {
    let placeholder = ReducedBridgePole {
        freq_hz: 0.0,
        zeta: 0.0,
        q_factor: 0.0,
        modal_mass_kg: 1.0,
        zone_gains: [0.0; NUM_BRIDGE_ZONES],
    };
    let mut poles = [placeholder; NUM_REDUCED_MODES];
    for m in 0..NUM_REDUCED_MODES {
        let freq_hz = MODE_FREQS_HZ[m];
        let zeta = if lossless { 0.0 } else { MODE_ZETAS[m] };
        let q_factor = if zeta > 0.0 {
            0.5 / zeta
        } else {
            f32::INFINITY
        };
        poles[m] = ReducedBridgePole {
            freq_hz,
            zeta,
            q_factor,
            modal_mass_kg: MODE_MASSES_KG[m],
            zone_gains: MODE_ZONE_GAINS[m],
        };
    }
    poles
}

/// Validates and clamps high-frequency bridge mobility `y_inf` to strictly positive passive bounds.
#[must_use]
pub(crate) fn sanitize_y_inf(y_inf: &[f32; NUM_BRIDGE_ZONES]) -> [f32; NUM_BRIDGE_ZONES] {
    let mut out = DEFAULT_Y_INF;
    for (z, &src) in y_inf.iter().enumerate() {
        out[z] = if src.is_finite() {
            src.clamp(MIN_Y_INF, MAX_Y_INF)
        } else {
            DEFAULT_Y_INF[z]
        };
    }
    out
}

/// Converts and validates [`PresetReducedPole`] entries into passive [`ReducedBridgePole`] values.
#[must_use]
pub(crate) fn sanitize_reduced_poles(
    reduced_poles: &[PresetReducedPole; NUM_REDUCED_MODES],
) -> [ReducedBridgePole; NUM_REDUCED_MODES] {
    let mut poles = build_default_poles(false);
    for (m, (dst, src)) in poles.iter_mut().zip(reduced_poles.iter()).enumerate() {
        let freq_hz = if src.freq_hz.is_finite() {
            src.freq_hz.clamp(MIN_POLE_FREQ_HZ, MAX_POLE_FREQ_HZ)
        } else {
            MODE_FREQS_HZ[m]
        };
        let zeta = if src.zeta.is_finite() {
            src.zeta.clamp(MIN_POLE_ZETA, MAX_POLE_ZETA)
        } else {
            MODE_ZETAS[m]
        };
        let modal_mass_kg = if src.modal_mass_kg.is_finite() {
            src.modal_mass_kg
                .clamp(MIN_MODAL_MASS_KG, MAX_MODAL_MASS_KG)
        } else {
            MODE_MASSES_KG[m]
        };
        let mut zone_gains = [0.0_f32; NUM_BRIDGE_ZONES];
        for (zg_dst, &zg_src) in zone_gains.iter_mut().zip(src.zone_gains.iter()) {
            *zg_dst = if zg_src.is_finite() {
                zg_src.clamp(-MAX_ABS_ZONE_GAIN, MAX_ABS_ZONE_GAIN)
            } else {
                0.0
            };
        }
        *dst = ReducedBridgePole {
            freq_hz,
            zeta,
            q_factor: 0.5 / zeta,
            modal_mass_kg,
            zone_gains,
        };
    }
    poles
}

/// Builds default physical per-string WDF port resistances `[kg/s]` for standalone initialization
/// before `StringBank::bridge_port_resistances()` is supplied.
#[must_use]
pub fn default_string_port_resistances(sample_rate: f32) -> [f32; TOTAL_STRINGS] {
    let safe_sr = f64::from(sanitize_sample_rate(sample_rate));
    let specs = build_scale_design(&VoicingSpec::default());
    let mut r0 = [1.0_f32; TOTAL_STRINGS];
    for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
        let p = specs[k];
        let m_str = p.linear_density_kg_m * p.speaking_length_m;
        let z0 = 2.0 * p.f0_et_hz * m_str;
        let scale = (11_000.0 / safe_sr).clamp(0.05, 0.6);
        let r_val = (z0 * scale).max(0.05) as f32;
        let start = usize::from(layout.first_string_idx);
        let count = usize::from(layout.strings_per_key);
        for slot in &mut r0[start..start + count] {
            *slot = r_val;
        }
    }
    r0
}
