//! 1st-order Timoshenko flexible hammer shank (`8 * sample_rate`) with let-off
//! free flight, backcheck catch deactivation, and 12 ms safety timeout.

use core::f64::consts::TAU;

use piano_spec::{u16_to_unit_f32, NUM_KEYS};

use crate::math::pow_nonneg_f64;

/// Initial hammer position behind the let-off gap (`-20 um`).
pub const HAMMER_LET_OFF_Y0_M: f32 = -2.0e-5;
/// Backcheck catch threshold on downward rebound (`-60 um`).
pub const BACKCHECK_CATCH_Y_M: f32 = -6.0e-5;
/// Maximum hammer-active safety window (`12 ms`).
pub const HAMMER_TIMEOUT_S: f64 = 0.012;

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
        let timeout_host_samples =
            ((HAMMER_TIMEOUT_S * sample_rate) + 0.5).clamp(8.0, 65_000.0) as u16;

        Self {
            hammer_mass_kg: hammer_mass_kg as f32,
            shank_freq_hz: shank_freq_hz as f32,
            modal_mass_kg: modal_mass_kg as f32,
            dt_sub: dt as f32,
            half_dt: h as f32,
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

    /// Computes unforced (`F = 0`) trapezoidal endpoint prediction over one `8x` substep.
    #[must_use]
    #[inline]
    pub fn predict_free(&self, params: &ShankParams) -> ShankFreePrediction {
        let y0_free = self.y0 + self.v0 * params.dt_sub;
        if !self.flex_enabled {
            return ShankFreePrediction {
                y_h_free: y0_free,
                k_ham: params.k0_ham,
                y0_free,
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
        self.v0 -= params.c0_f * force_n;
        self.y0 = pred.y0_free - params.k0_ham * force_n;
        if self.flex_enabled {
            self.v1 = pred.v1_free - params.c1_f * force_n;
            self.eta1 = pred.eta1_free - params.k1_ham * force_n;
        } else {
            self.v1 = 0.0;
            self.eta1 = 0.0;
        }
    }

    /// Updates elapsed host sample count and applies backcheck catch / 12 ms timeout.
    /// Returns `true` if hammer physics remains active for the next host sample.
    #[inline]
    pub fn finish_host_sample(&mut self, params: &ShankParams) -> bool {
        if !self.active {
            return false;
        }
        self.elapsed_host_samples = self.elapsed_host_samples.saturating_add(1);
        let caught = self.has_contacted && self.y_h() < BACKCHECK_CATCH_Y_M && self.v0 < 0.0;
        let timed_out = self.elapsed_host_samples >= params.timeout_host_samples;
        if caught || timed_out {
            self.active = false;
        }
        self.active
    }
}

/// Maps 16-bit MIDI 2.0 velocity (`1..=65535`) strictly monotonically to hammer
/// strike speed (`~0.10 m/s` at `pp`, `~1.83 m/s` at `51602`, `4.60 m/s` at `fff`).
#[must_use]
#[inline]
pub fn velocity_from_u16(velocity_u16: u16) -> f32 {
    let u = u16_to_unit_f32(velocity_u16);
    let u2 = u * u;
    let u5 = u2 * u2 * u;
    0.10 + 0.75 * u + 3.75 * u5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_mapping_is_strictly_monotonic_and_hits_targets() {
        let mut prev = 0.0_f32;
        for v in 1..=u16::MAX {
            let speed = velocity_from_u16(v);
            assert!(speed > prev, "non-monotonic at {v}: {speed} <= {prev}");
            prev = speed;
        }
        let v_pp = velocity_from_u16(1);
        let v_mf = velocity_from_u16(51_602);
        let v_fff = velocity_from_u16(u16::MAX);
        assert!((0.09..=0.11).contains(&v_pp), "pp={v_pp}");
        assert!((1.70..=1.95).contains(&v_mf), "mf={v_mf}");
        assert!((4.55..=4.65).contains(&v_fff), "fff={v_fff}");
    }

    #[test]
    fn shank_params_and_trapezoidal_passivity_hold() {
        let p0 = ShankParams::for_key(0, 48_000.0);
        let p48 = ShankParams::for_key(48, 48_000.0);
        let p87 = ShankParams::for_key(87, 48_000.0);
        assert!((p0.hammer_mass_kg - 0.0112).abs() < 1e-5);
        assert!((p87.hammer_mass_kg - 0.0035).abs() < 1e-5);
        assert!(p0.hammer_mass_kg > p48.hammer_mass_kg && p48.hammer_mass_kg > p87.hammer_mass_kg);
        assert!((1_500.0..=3_500.0).contains(&p0.shank_freq_hz));
        assert!((1_500.0..=3_500.0).contains(&p48.shank_freq_hz));
        assert!((1_500.0..=3_500.0).contains(&p87.shank_freq_hz));

        // Unforced ringing of flexible mode strictly dissipates energy.
        let mut s = ShankState {
            eta1: 1.0e-4,
            v1: 0.25,
            active: true,
            ..ShankState::default()
        };
        let omega1 = TAU * f64::from(p48.shank_freq_hz);
        let k1 = f64::from(p48.modal_mass_kg) * omega1 * omega1;
        let energy = |st: &ShankState| {
            0.5 * f64::from(p48.modal_mass_kg) * f64::from(st.v1) * f64::from(st.v1)
                + 0.5 * k1 * f64::from(st.eta1) * f64::from(st.eta1)
        };
        let mut e_prev = energy(&s);
        for _ in 0..400 {
            let pred = s.predict_free(&p48);
            s.apply_force(&p48, pred, 0.0);
            let e_cur = energy(&s);
            assert!(e_cur <= e_prev + 1e-15);
            e_prev = e_cur;
        }
        assert!(e_prev < 0.9 * (0.5 * f64::from(p48.modal_mass_kg) * 0.0625 + 0.5 * k1 * 1e-8));
    }

    #[test]
    fn backcheck_and_timeout_deactivate_hammer() {
        let params = ShankParams::for_key(48, 48_000.0);
        let mut s = ShankState::default();
        s.launch(1.8);
        let pred = s.predict_free(&params);
        s.apply_force(&params, pred, 100.0);
        assert!(s.has_contacted);
        s.y0 = -7.0e-5;
        s.v0 = -1.0;
        assert!(!s.finish_host_sample(&params));

        // Timeout without contact.
        s.launch(0.10);
        for _ in 0..params.timeout_host_samples - 1 {
            assert!(s.finish_host_sample(&params));
        }
        assert!(!s.finish_host_sample(&params));
    }
}
