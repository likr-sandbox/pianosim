//! 1st-order Timoshenko flexible hammer shank (`8 * sample_rate`) with let-off
//! free flight under gravity, rest-rail / backcheck catch, and 12 ms safety timeout.

use core::f64::consts::TAU;

use piano_spec::{u16_to_unit_f32, NUM_KEYS};

use crate::math::pow_nonneg_f64;

/// Gravitational deceleration (`m/s^2`) acting on the hammer during free flight.
pub const GRAVITY_MPS2: f32 = 9.81;
/// Initial hammer position behind the let-off gap (`-20 um`).
pub const HAMMER_LET_OFF_Y0_M: f32 = -2.0e-5;
/// Backcheck catch threshold on downward rebound (`-60 um`).
pub const BACKCHECK_CATCH_Y_M: f32 = -6.0e-5;
/// Maximum hammer-active safety window (`12 ms`).
pub const HAMMER_TIMEOUT_S: f64 = 0.012;

pub(crate) const SILENT_KEY_MAX_U16: u16 = 64;
const SILENT_KEY_V_MIN_MPS: f32 = 0.004;
const SILENT_KEY_V_MAX_MPS: f32 = 0.015;
const PP_ANCHOR_U16: u16 = 516; // piano_spec::u7_to_u16(1)

const MASS_A0_KG: f64 = 0.0112;
const MASS_C8_KG: f64 = 0.0035;
const SHANK_FREQ_A0_HZ: f64 = 1_500.0;
const SHANK_FREQ_C8_HZ: f64 = 3_500.0;
const SHANK_ZETA: f64 = 0.025;

/// Precomputed per-key Timoshenko shank & trapezoidal integration parameters.
#[derive(Copy, Clone, Debug)]
pub struct ShankParams {
    pub hammer_mass_kg: f32,
    pub shank_freq_hz: f32,
    pub modal_mass_kg: f32,
    pub dt_sub: f32,
    half_dt: f32,
    g_dt: f32,
    half_g_dt2: f32,
    c0_f: f32,
    k0_ham: f32,
    c_vv: f32,
    c_vy: f32,
    c1_f: f32,
    k1_ham: f32,
    pub timeout_host_samples: u16,
}

impl ShankParams {
    /// Builds physical Timoshenko shank parameters for `key_idx` at `sample_rate`.
    #[must_use]
    pub fn for_key(key_idx: usize, sample_rate: f64) -> Self {
        let x = (key_idx.min(NUM_KEYS - 1) as f64) / ((NUM_KEYS - 1) as f64);
        let hammer_mass_kg = MASS_C8_KG + (MASS_A0_KG - MASS_C8_KG) * pow_nonneg_f64(1.0 - x, 1.35);
        let shank_freq_hz =
            SHANK_FREQ_A0_HZ + (SHANK_FREQ_C8_HZ - SHANK_FREQ_A0_HZ) * pow_nonneg_f64(x, 1.1);
        let modal_mass_kg = (1.4 + 1.0 * x) * hammer_mass_kg;

        let dt = 1.0 / (8.0 * sample_rate);
        let h = 0.5 * dt;
        let omega1 = TAU * shank_freq_hz;
        let gamma1 = 2.0 * SHANK_ZETA * omega1;
        let w2 = omega1 * omega1;
        let d1 = 1.0 + h * gamma1 + h * h * w2;

        let c_vv = (1.0 - h * gamma1 - h * h * w2) / d1;
        let c_vy = (2.0 * h * w2) / d1;
        let c1_f = dt / (modal_mass_kg * d1);
        let k1_ham = h * c1_f;

        let c0_f = dt / hammer_mass_kg;
        let k0_ham = h * c0_f;
        let g = f64::from(GRAVITY_MPS2);
        let g_dt = g * dt;
        let half_g_dt2 = 0.5 * g * dt * dt;
        let timeout_host_samples =
            ((HAMMER_TIMEOUT_S * sample_rate) + 0.5).clamp(8.0, 65_000.0) as u16;

        Self {
            hammer_mass_kg: hammer_mass_kg as f32,
            shank_freq_hz: shank_freq_hz as f32,
            modal_mass_kg: modal_mass_kg as f32,
            dt_sub: dt as f32,
            half_dt: h as f32,
            g_dt: g_dt as f32,
            half_g_dt2: half_g_dt2 as f32,
            c0_f: c0_f as f32,
            k0_ham: k0_ham as f32,
            c_vv: c_vv as f32,
            c_vy: c_vy as f32,
            c1_f: c1_f as f32,
            k1_ham: k1_ham as f32,
            timeout_host_samples,
        }
    }
}

/// Free-step prediction bundle passed to K-method solver.
#[derive(Copy, Clone, Debug)]
pub struct ShankFreePrediction {
    pub y_h_free: f32,
    pub k_ham: f32,
    y0_free: f32,
    v0_free: f32,
    eta1_free: f32,
    v1_free: f32,
}

/// Per-key mutable 1st-order Timoshenko hammer shank state.
#[derive(Copy, Clone, Debug)]
pub struct ShankState {
    pub y0: f32,
    pub v0: f32,
    pub eta1: f32,
    pub v1: f32,
    pub active: bool,
    pub has_contacted: bool,
    pub elapsed_host_samples: u16,
    pub flex_enabled: bool,
}

impl Default for ShankState {
    fn default() -> Self {
        Self {
            y0: HAMMER_LET_OFF_Y0_M,
            v0: 0.0,
            eta1: 0.0,
            v1: 0.0,
            active: false,
            has_contacted: false,
            elapsed_host_samples: 0,
            flex_enabled: true,
        }
    }
}

impl ShankState {
    /// Resets the shank to inactive resting state while preserving `flex_enabled`.
    pub fn reset(&mut self) {
        let flex = self.flex_enabled;
        *self = Self {
            flex_enabled: flex,
            ..Self::default()
        };
    }

    /// Launches the hammer from behind the let-off gap with initial speed `v_h0` (`m/s`).
    #[inline]
    pub fn launch(&mut self, v_h0: f32) {
        self.y0 = HAMMER_LET_OFF_Y0_M;
        self.v0 = if v_h0.is_finite() && v_h0 > 0.0 {
            v_h0
        } else {
            0.10
        };
        self.eta1 = 0.0;
        self.v1 = 0.0;
        self.active = true;
        self.has_contacted = false;
        self.elapsed_host_samples = 0;
    }

    /// Current hammer head vertical displacement `y0 + eta1` in meters.
    #[must_use]
    #[inline]
    pub fn y_h(&self) -> f32 {
        self.y0 + self.eta1
    }

    /// Current hammer head velocity `v0 + v1` in `m/s`.
    #[must_use]
    #[inline]
    pub fn v_h(&self) -> f32 {
        self.v0 + self.v1
    }

    /// Computes unforced (`F = 0`) trapezoidal endpoint prediction over one `8x` substep
    /// including gravitational deceleration `-GRAVITY_MPS2`.
    #[must_use]
    #[inline]
    pub fn predict_free(&self, params: &ShankParams) -> ShankFreePrediction {
        let y0_free = self.y0 + self.v0 * params.dt_sub - params.half_g_dt2;
        let v0_free = self.v0 - params.g_dt;
        if !self.flex_enabled {
            return ShankFreePrediction {
                y_h_free: y0_free,
                k_ham: params.k0_ham,
                y0_free,
                v0_free,
                eta1_free: 0.0,
                v1_free: 0.0,
            };
        }
        let v1_free = params.c_vv * self.v1 - params.c_vy * self.eta1;
        let eta1_free = self.eta1 + params.half_dt * (self.v1 + v1_free);
        ShankFreePrediction {
            y_h_free: y0_free + eta1_free,
            k_ham: params.k0_ham + params.k1_ham,
            y0_free,
            v0_free,
            eta1_free,
            v1_free,
        }
    }

    /// Advances `(y0, v0, eta1, v1)` under contact reaction `-force_n` (`force_n >= 0`).
    #[inline]
    pub fn apply_force(&mut self, params: &ShankParams, pred: ShankFreePrediction, force_n: f32) {
        if force_n > 0.0 {
            self.has_contacted = true;
        }
        self.v0 = pred.v0_free - params.c0_f * force_n;
        self.y0 = pred.y0_free - params.k0_ham * force_n;
        if self.flex_enabled {
            self.v1 = pred.v1_free - params.c1_f * force_n;
            self.eta1 = pred.eta1_free - params.k1_ham * force_n;
        } else {
            self.v1 = 0.0;
            self.eta1 = 0.0;
        }
    }

    /// Updates elapsed host sample count and applies backcheck catch, sub-escapement
    /// rest-rail catch, or 12 ms safety timeout. Returns `true` if hammer stays active.
    #[inline]
    pub fn finish_host_sample(&mut self, params: &ShankParams) -> bool {
        if !self.active {
            return false;
        }
        self.elapsed_host_samples = self.elapsed_host_samples.saturating_add(1);
        let caught_rebound =
            self.has_contacted && self.y_h() < BACKCHECK_CATCH_Y_M && self.v0 < 0.0;
        let caught_rest_rail =
            !self.has_contacted && self.v0 < 0.0 && self.y_h() <= HAMMER_LET_OFF_Y0_M;
        let timed_out = self.elapsed_host_samples >= params.timeout_host_samples;
        if caught_rest_rail {
            self.y0 = HAMMER_LET_OFF_Y0_M;
            self.v0 = 0.0;
        }
        if caught_rebound || caught_rest_rail || timed_out {
            self.active = false;
        }
        self.active
    }
}

#[inline]
fn main_velocity_curve(u: f32) -> f32 {
    let uf = f64::from(u.clamp(0.0, 1.0));
    (0.075 + 0.86 * uf + 3.665 * pow_nonneg_f64(uf, 2.15)) as f32
}

/// Maps 16-bit MIDI 2.0 velocity (`1..=65535`) strictly monotonically to hammer
/// strike speed:
/// - `1..=64`: sub-escapement silent-key range (`0.004..=0.015 m/s < v_esc`)
/// - `516` (`u7_to_u16(1)`): `~0.082 m/s` (`pp` audible strike, `> v_esc = 0.0198 m/s`)
/// - `33_025` (`u7_to_u16(64)`): `~1.35 m/s` (`mf`), `51_602` (`u7_to_u16(100)`): `~2.94 m/s`
/// - `65_535` (`u7_to_u16(127)`): `4.60 m/s` (`fff`).
#[must_use]
#[inline]
pub fn velocity_from_u16(velocity_u16: u16) -> f32 {
    if velocity_u16 == 0 {
        return 0.0;
    }
    if velocity_u16 <= SILENT_KEY_MAX_U16 {
        let t = f32::from(velocity_u16 - 1) / f32::from(SILENT_KEY_MAX_U16 - 1);
        return SILENT_KEY_V_MIN_MPS + (SILENT_KEY_V_MAX_MPS - SILENT_KEY_V_MIN_MPS) * t;
    }
    if velocity_u16 < PP_ANCHOR_U16 {
        let t = f32::from(velocity_u16 - SILENT_KEY_MAX_U16)
            / f32::from(PP_ANCHOR_U16 - SILENT_KEY_MAX_U16);
        let pp_v = main_velocity_curve((PP_ANCHOR_U16 as f32) / 65_535.0);
        return SILENT_KEY_V_MAX_MPS + (pp_v - SILENT_KEY_V_MAX_MPS) * t;
    }
    main_velocity_curve(u16_to_unit_f32(velocity_u16))
}

#[cfg(test)]
#[path = "shank_timoshenko_tests.rs"]
mod tests;
