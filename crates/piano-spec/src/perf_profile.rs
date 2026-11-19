//! Performance-quality trade-off profiles (`PerfPreset` / `PerfProfile`)
//! shared across all 4 physical synthesis stages and worker/Wasm boundaries.

use core::f64::consts::LN_10;

use crate::math::exp_f64;

/// Discrete performance-quality presets selectable from CLI or Web UI.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PerfPreset {
    /// Reference full-fidelity synthesis matching `Full` baseline behavior.
    #[default]
    Full = 0,
    /// Balanced real-time profile preserving acoustic character at ~2-3x lower cost.
    Balanced = 1,
    /// Aggressive real-time profile tuned for low-latency Web AudioWorklet targets.
    Realtime = 2,
}

impl PerfPreset {
    /// Converts a raw byte discriminant into a [`PerfPreset`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Full),
            1 => Some(Self::Balanced),
            2 => Some(Self::Realtime),
            _ => None,
        }
    }
}

/// Complete 19-parameter POD performance profile spanning Stages 1-4.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PerfProfile {
    // Stage 2: String bank
    /// Upper frequency cutoff (Hz) for horizontal string polarization modes.
    pub horiz_max_hz: f32,
    /// Tail mode pruning threshold (dB relative to fundamental excitation); `<= -120.0` disables.
    pub dead_mode_prune_db: f32,
    /// Post-Stage-4 modal amplitude horizon (dB) below which decaying low-order modes (`1..=8`)
    /// truncate to zero.
    pub low_order_energy_horizon_db: f32,
    /// Post-Stage-4 modal amplitude horizon (dB) below which decaying high-order modes (`> 8`)
    /// truncate to zero.
    pub energy_horizon_db: f32,
    /// Nonlinear SAV energy ratio threshold (dB) below which SAV enters smooth dormancy.
    pub sav_dormancy_db: f32,
    /// Exclusive upper key index (`0..=52`) for longitudinal / SAV phantom coupling.
    pub sav_max_key_idx: u8,
    /// Enables sparse resonant-pair phantom SAV kernels for bass keys `0..16`.
    pub bass_phantom_sparse: bool,
    /// Upper frequency cutoff (Hz) for passive sympathetic tail ringing.
    pub passive_res_max_hz: f32,

    // Stage 4: Radiator & HallFdn
    /// Synthesized structural soundboard impulse response duration (s).
    pub ir_duration_secs: f32,
    /// Sample tap index splitting 9-zone MIMO tails into shared stereo velvet diffuse tail.
    pub spatial_tail_split_tap: u16,
    /// Direct zero-latency FIR head tap count (`16`, `32`, or `64`).
    pub fir_head_taps: u8,
    /// Enables block-FFT evaluation of the FIR head instead of direct time-domain FIR.
    pub fir_block_fft: bool,
    /// Number of Hall FDN feedback delay lines (`4`, `8`, or `16`).
    pub hall_fdn_lines: u8,

    // Stage 1: Action
    /// Hammer-string contact substep oversampling factor (`2`, `4`, or `8`).
    pub action_oversample: u8,
    /// Uses analytical Bernstein-monomial polynomial ADAA + Newton K-method instead of LUTs.
    pub polynomial_adaa: bool,
    /// Enables Timoshenko flexible hammer shank bending dynamics.
    pub shank_flex_enabled: bool,
    /// Damper pad clearance control decimation factor (`1`, `2`, `4`, `8`, or `16`).
    pub damper_control_decim: u8,

    // Stage 3: Bridge
    /// Enables nonlinear Hertz bridge contact stiffness SAV coupling.
    pub hertz_nonlinear: bool,
    /// Number of active reduced bridge admittance poles (`4..=10`).
    pub active_bridge_poles: u8,
}

impl PerfProfile {
    /// Returns the full-fidelity reference profile matching `d2dab9c` defaults.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            horiz_max_hz: 20_000.0,
            dead_mode_prune_db: -120.0,
            low_order_energy_horizon_db: -120.0,
            energy_horizon_db: -120.0,
            sav_dormancy_db: -55.0,
            sav_max_key_idx: 52,
            bass_phantom_sparse: false,
            passive_res_max_hz: 20_000.0,
            ir_duration_secs: 0.50,
            spatial_tail_split_tap: u16::MAX,
            fir_head_taps: 64,
            fir_block_fft: false,
            hall_fdn_lines: 8,
            action_oversample: 8,
            polynomial_adaa: false,
            shank_flex_enabled: true,
            damper_control_decim: 1,
            hertz_nonlinear: true,
            active_bridge_poles: 10,
        }
    }

    /// Returns the balanced real-time profile.
    #[must_use]
    pub const fn balanced() -> Self {
        Self {
            horiz_max_hz: 3_500.0,
            dead_mode_prune_db: -50.0,
            low_order_energy_horizon_db: -112.0,
            energy_horizon_db: -96.0,
            sav_dormancy_db: -42.0,
            sav_max_key_idx: 39,
            bass_phantom_sparse: true,
            passive_res_max_hz: 4_000.0,
            ir_duration_secs: 0.25,
            spatial_tail_split_tap: 4_096,
            fir_head_taps: 64,
            fir_block_fft: false,
            hall_fdn_lines: 8,
            action_oversample: 4,
            polynomial_adaa: true,
            shank_flex_enabled: true,
            damper_control_decim: 8,
            hertz_nonlinear: false,
            active_bridge_poles: 8,
        }
    }

    /// Returns the aggressive low-latency real-time profile.
    #[must_use]
    pub const fn realtime() -> Self {
        Self {
            horiz_max_hz: 2_800.0,
            dead_mode_prune_db: -45.0,
            low_order_energy_horizon_db: -108.0,
            energy_horizon_db: -84.0,
            sav_dormancy_db: -38.0,
            sav_max_key_idx: 26,
            bass_phantom_sparse: true,
            passive_res_max_hz: 3_500.0,
            ir_duration_secs: 0.18,
            spatial_tail_split_tap: 2_048,
            fir_head_taps: 32,
            fir_block_fft: true,
            hall_fdn_lines: 4,
            action_oversample: 4,
            polynomial_adaa: true,
            shank_flex_enabled: false,
            damper_control_decim: 16,
            hertz_nonlinear: false,
            active_bridge_poles: 8,
        }
    }

    /// Constructs a [`PerfProfile`] from a [`PerfPreset`].
    #[must_use]
    pub const fn from_preset(preset: PerfPreset) -> Self {
        match preset {
            PerfPreset::Full => Self::full(),
            PerfPreset::Balanced => Self::balanced(),
            PerfPreset::Realtime => Self::realtime(),
        }
    }

    /// Returns a copy with all continuous bounds clamped (`NaN` -> `Full` default)
    /// and discrete options snapped to their valid supported sets.
    #[must_use]
    pub fn sanitized(&self) -> Self {
        let full = Self::full();
        Self {
            horiz_max_hz: sanitize_clamp(self.horiz_max_hz, full.horiz_max_hz, 500.0, 20_000.0),
            dead_mode_prune_db: sanitize_clamp(
                self.dead_mode_prune_db,
                full.dead_mode_prune_db,
                -120.0,
                -20.0,
            ),
            low_order_energy_horizon_db: sanitize_clamp(
                self.low_order_energy_horizon_db,
                full.low_order_energy_horizon_db,
                -120.0,
                -40.0,
            ),
            energy_horizon_db: sanitize_clamp(
                self.energy_horizon_db,
                full.energy_horizon_db,
                -120.0,
                -40.0,
            ),
            sav_dormancy_db: sanitize_clamp(
                self.sav_dormancy_db,
                full.sav_dormancy_db,
                -80.0,
                -20.0,
            ),
            sav_max_key_idx: self.sav_max_key_idx.min(52),
            bass_phantom_sparse: self.bass_phantom_sparse,
            passive_res_max_hz: sanitize_clamp(
                self.passive_res_max_hz,
                full.passive_res_max_hz,
                1_000.0,
                20_000.0,
            ),
            ir_duration_secs: sanitize_clamp(
                self.ir_duration_secs,
                full.ir_duration_secs,
                0.05,
                0.50,
            ),
            spatial_tail_split_tap: self.spatial_tail_split_tap.max(512),
            fir_head_taps: snap_fir_head_taps(self.fir_head_taps),
            fir_block_fft: self.fir_block_fft,
            hall_fdn_lines: snap_hall_fdn_lines(self.hall_fdn_lines),
            action_oversample: snap_action_oversample(self.action_oversample),
            polynomial_adaa: self.polynomial_adaa,
            shank_flex_enabled: self.shank_flex_enabled,
            damper_control_decim: snap_damper_control_decim(self.damper_control_decim),
            hertz_nonlinear: self.hertz_nonlinear,
            active_bridge_poles: self.active_bridge_poles.clamp(4, 10),
        }
    }

    /// Returns `true` when dead-mode tail pruning is enabled (`dead_mode_prune_db > -120.0`).
    #[must_use]
    pub fn is_dead_mode_pruning_enabled(&self) -> bool {
        self.dead_mode_prune_db > -120.0
    }

    /// Converts `low_order_energy_horizon_db` into Stage 2 modal `(u^2 + v^2)` horizon threshold
    /// for low-order modes (`1..=8`).
    ///
    /// Returns exact `1.0e-18` when `low_order_energy_horizon_db <= -120.0`.
    #[must_use]
    pub fn low_order_energy_horizon_sq(&self) -> f32 {
        energy_horizon_db_to_sq(self.low_order_energy_horizon_db)
    }

    /// Converts `energy_horizon_db` into Stage 2 modal `(u^2 + v^2)` horizon threshold.
    ///
    /// Returns exact `1.0e-18` when `energy_horizon_db <= -120.0`.
    #[must_use]
    pub fn energy_horizon_sq(&self) -> f32 {
        energy_horizon_db_to_sq(self.energy_horizon_db)
    }

    /// Converts `sav_dormancy_db` into the `E_nl / (2 * E_lin)` SAV dormancy ratio.
    ///
    /// Returns exact `1.581e-6` when `(sav_dormancy_db - -55.0).abs() < 1e-5`.
    #[must_use]
    pub fn sav_dormancy_ratio(&self) -> f64 {
        let db = sanitize_clamp(self.sav_dormancy_db, -55.0, -80.0, -20.0);
        let diff = db - -55.0;
        if diff > -1.0e-5 && diff < 1.0e-5 {
            1.581e-6
        } else {
            0.5 * exp_f64((f64::from(db) / 10.0) * LN_10)
        }
    }

    /// Returns `true` when Stage 2 modal counts match the full unpruned baseline.
    #[must_use]
    pub fn is_full_mode_counts(&self) -> bool {
        self.horiz_max_hz >= 20_000.0
            && !self.is_dead_mode_pruning_enabled()
            && !self.bass_phantom_sparse
    }
}

impl Default for PerfProfile {
    fn default() -> Self {
        Self::full()
    }
}

#[inline]
fn sanitize_clamp(v: f32, default: f32, min: f32, max: f32) -> f32 {
    if v.is_nan() {
        default
    } else {
        v.clamp(min, max)
    }
}

#[inline]
const fn snap_fir_head_taps(v: u8) -> u8 {
    if v <= 24 {
        16
    } else if v <= 48 {
        32
    } else {
        64
    }
}

#[inline]
const fn snap_hall_fdn_lines(v: u8) -> u8 {
    if v <= 6 {
        4
    } else if v <= 12 {
        8
    } else {
        16
    }
}

#[inline]
const fn snap_action_oversample(v: u8) -> u8 {
    if v <= 3 {
        2
    } else if v <= 6 {
        4
    } else {
        8
    }
}

#[inline]
fn energy_horizon_db_to_sq(raw_db: f32) -> f32 {
    let db = sanitize_clamp(raw_db, -120.0, -120.0, -40.0);
    if db <= -120.0 {
        1.0e-18
    } else {
        exp_f64(((f64::from(db) - 60.0) / 10.0) * LN_10) as f32
    }
}

#[inline]
const fn snap_damper_control_decim(v: u8) -> u8 {
    if v <= 1 {
        1
    } else if v <= 3 {
        2
    } else if v <= 6 {
        4
    } else if v <= 12 {
        8
    } else {
        16
    }
}

#[cfg(test)]
#[path = "perf_profile_tests.rs"]
mod tests;
