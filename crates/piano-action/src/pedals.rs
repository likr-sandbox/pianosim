//! Pedal state machine (`Sustain`, `UnaCorda`, `Sostenuto`) and per-key
//! geometric damper-rail clearance smoother (`0..70` damped keys).

use piano_spec::{u16_to_unit_f32, PedalKind, CHOIR_LAYOUTS, DAMPED_KEYS, NUM_KEYS};

use crate::math::exp_f64;
pub use crate::pedals_clearance::sustain_clearance_m;
use crate::pedals_clearance::{alpha_for_decim, snap_control_decim};

/// Full open damper clearance (`+5.0 mm`) when key held, sustain=1.0, or undamped key.
pub const CLEARANCE_OPEN_M: f32 = 5.0e-3;
/// Nominal resting damper preload indentation (`-0.30 mm`) when released dry.
pub const CLEARANCE_PRELOAD_M: f32 = -3.0e-4;
/// Minimum geometric clearance (`+0.20 mm`) required to latch under Sostenuto.
pub const SOSTENUTO_OPEN_MIN_M: f32 = 2.0e-4;
/// Sostenuto engagement threshold in normalized `[0.0, 1.0]` pedal travel.
pub const SOSTENUTO_THRESHOLD: f32 = 0.5;

const LIFT_TAU_S: f64 = 0.008;
const DROP_TAU_SLOW_S: f64 = 0.028;
const DROP_TAU_FAST_S: f64 = 0.007;
pub(crate) const HALF_PEDAL_LOW: f32 = 0.35;
pub(crate) const HALF_PEDAL_HIGH: f32 = 0.65;
#[cfg(test)]
const HALF_PEDAL_MID_CLEARANCE_M: f32 = 3.2e-5;
pub(crate) const HALF_PEDAL_HI_CLEARANCE_M: f32 = 6.4e-5;
pub(crate) const SETTLE_EPS_M: f32 = 1.0e-7;
const PEEL_PULSE_DURATION_S: f64 = 2.0e-3;
pub(crate) const PEEL_MIN_PEAK_N: f32 = 5.0e-4;
pub(crate) const PEEL_MAX_PEAK_N: f32 = 2.0e-3;
pub(crate) const PEEL_ARM_CLEARANCE_M: f32 = -1.0e-5;

/// Continuous 3-pedal controller and per-key damper clearance smoother.
#[derive(Clone, Debug)]
pub struct PedalBank {
    pub(crate) sustain: f32,
    una_corda: f32,
    sostenuto_down: bool,
    pub(crate) key_down: [bool; NUM_KEYS],
    pub(crate) sostenuto_latched: [bool; DAMPED_KEYS],
    pub(crate) settling: [bool; DAMPED_KEYS],
    pub(crate) clearance_m: [f32; DAMPED_KEYS],
    pub(crate) release_preload_m: [f32; DAMPED_KEYS],
    pub(crate) alpha_down: [f32; DAMPED_KEYS],
    pub(crate) alpha_down_d: [f32; DAMPED_KEYS],
    pub(crate) alpha_lift: f32,
    pub(crate) alpha_lift_d: f32,
    alpha_down_slow: f32,
    alpha_down_fast: f32,
    pub(crate) control_decim: u8,
    pub(crate) peel_len: u16,
    pub(crate) peel_played: [bool; DAMPED_KEYS],
    pub(crate) peel_armed: [bool; DAMPED_KEYS],
    pub(crate) peel_min_clear_m: [f32; DAMPED_KEYS],
    pub(crate) peel_pos: [u16; DAMPED_KEYS],
    pub(crate) peel_peak_n: [f32; DAMPED_KEYS],
    pub(crate) last_peel_n: [f32; DAMPED_KEYS],
}

impl PedalBank {
    /// Initializes pedal bank and 1-pole rail coefficients for `sample_rate` (`1x` control rate).
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_decim(sample_rate, 1)
    }

    /// Initializes pedal bank with decimated control-rate step (`1`, `2`, or `4` samples).
    #[must_use]
    pub fn new_with_decim(sample_rate: f32, control_decim: u8) -> Self {
        let decim = snap_control_decim(control_decim);
        let sr = f64::from(sample_rate.max(8_000.0));
        let alpha_lift = (1.0 - exp_f64(-1.0 / (LIFT_TAU_S * sr))) as f32;
        let alpha_down_slow = (1.0 - exp_f64(-1.0 / (DROP_TAU_SLOW_S * sr))) as f32;
        let alpha_down_fast = (1.0 - exp_f64(-1.0 / (DROP_TAU_FAST_S * sr))) as f32;
        let default_alpha_down = 0.5 * (alpha_down_slow + alpha_down_fast);
        let alpha_lift_d = alpha_for_decim(alpha_lift, decim);
        let default_alpha_down_d = alpha_for_decim(default_alpha_down, decim);
        let peel_len = (((PEEL_PULSE_DURATION_S * sr) + 0.5).clamp(9.0, 1025.0) as u16) | 1;

        Self {
            sustain: 0.0,
            una_corda: 0.0,
            sostenuto_down: false,
            key_down: [false; NUM_KEYS],
            sostenuto_latched: [false; DAMPED_KEYS],
            settling: [false; DAMPED_KEYS],
            clearance_m: [CLEARANCE_PRELOAD_M; DAMPED_KEYS],
            release_preload_m: [CLEARANCE_PRELOAD_M; DAMPED_KEYS],
            alpha_down: [default_alpha_down; DAMPED_KEYS],
            alpha_down_d: [default_alpha_down_d; DAMPED_KEYS],
            alpha_lift,
            alpha_lift_d,
            alpha_down_slow,
            alpha_down_fast,
            control_decim: decim,
            peel_len,
            peel_played: [false; DAMPED_KEYS],
            peel_armed: [false; DAMPED_KEYS],
            peel_min_clear_m: [0.0; DAMPED_KEYS],
            peel_pos: [peel_len; DAMPED_KEYS],
            peel_peak_n: [0.0; DAMPED_KEYS],
            last_peel_n: [0.0; DAMPED_KEYS],
        }
    }

    /// Configured damper clearance control decimation factor (`1`, `2`, or `4`).
    #[must_use]
    #[inline]
    pub const fn control_decim(&self) -> u8 {
        self.control_decim
    }

    /// Updates damper clearance control decimation (`1`, `2`, or `4`) in-place with zero allocations.
    #[inline]
    pub fn set_control_decim(&mut self, control_decim: u8) {
        let decim = snap_control_decim(control_decim);
        if decim == self.control_decim {
            return;
        }
        self.control_decim = decim;
        self.alpha_lift_d = alpha_for_decim(self.alpha_lift, decim);
        for k in 0..DAMPED_KEYS {
            self.alpha_down_d[k] = alpha_for_decim(self.alpha_down[k], decim);
        }
    }

    /// Resets all pedals and returns all damped keys to resting `-0.30 mm` preload.
    pub fn reset(&mut self) {
        self.sustain = 0.0;
        self.una_corda = 0.0;
        self.sostenuto_down = false;
        self.key_down.fill(false);
        self.sostenuto_latched.fill(false);
        self.settling.fill(false);
        self.clearance_m.fill(CLEARANCE_PRELOAD_M);
        self.release_preload_m.fill(CLEARANCE_PRELOAD_M);
        let default_alpha_down = 0.5 * (self.alpha_down_slow + self.alpha_down_fast);
        self.alpha_down.fill(default_alpha_down);
        self.alpha_down_d
            .fill(alpha_for_decim(default_alpha_down, self.control_decim));
        self.peel_played.fill(false);
        self.peel_armed.fill(false);
        self.peel_min_clear_m.fill(0.0);
        self.peel_pos.fill(self.peel_len);
        self.peel_peak_n.fill(0.0);
        self.last_peel_n.fill(0.0);
    }

    #[inline]
    fn note_on_internal(&mut self, key_idx: usize, audible: bool) {
        if key_idx >= NUM_KEYS {
            return;
        }
        self.key_down[key_idx] = true;
        if key_idx < DAMPED_KEYS {
            self.clearance_m[key_idx] = CLEARANCE_OPEN_M;
            self.settling[key_idx] = false;
            if audible {
                self.peel_played[key_idx] = true;
            }
            self.peel_armed[key_idx] = false;
            self.peel_min_clear_m[key_idx] = 0.0;
            self.peel_pos[key_idx] = self.peel_len;
            self.last_peel_n[key_idx] = 0.0;
        }
    }

    /// Lifts the key's damper to full open clearance (`+5.0 mm`) on key strike.
    #[inline]
    pub fn note_on(&mut self, key_idx: usize) {
        self.note_on_internal(key_idx, true);
    }

    /// Lifts the key's damper on a sub-escapement silent key press without arming peel-off.
    #[inline]
    pub(crate) fn note_on_silent(&mut self, key_idx: usize) {
        self.note_on_internal(key_idx, false);
    }

    /// Releases the key, setting target preload and drop speed from `release_vel_u16`.
    #[inline]
    pub fn note_off(&mut self, key_idx: usize, release_vel_u16: u16) {
        if key_idx >= NUM_KEYS {
            return;
        }
        self.key_down[key_idx] = false;
        if key_idx < DAMPED_KEYS {
            let rel_u = u16_to_unit_f32(release_vel_u16);
            self.release_preload_m[key_idx] = CLEARANCE_PRELOAD_M * (0.80 + 0.20 * rel_u);
            let a1 = self.alpha_down_slow + (self.alpha_down_fast - self.alpha_down_slow) * rel_u;
            self.alpha_down[key_idx] = a1;
            self.alpha_down_d[key_idx] = alpha_for_decim(a1, self.control_decim);
            self.settling[key_idx] = true;
        }
    }

    /// Updates continuous pedal position (`0..=65535`).
    #[inline]
    pub fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        let unit = u16_to_unit_f32(value_u16).clamp(0.0, 1.0);
        match kind {
            PedalKind::Sustain => {
                if unit != self.sustain {
                    self.settling.fill(true);
                }
                self.sustain = unit;
                if unit > 0.0 {
                    // Pedal lift uses uniform rail preload as damper springs release.
                    let blend = (unit / HALF_PEDAL_LOW).min(1.0);
                    for preload in &mut self.release_preload_m {
                        *preload = *preload + blend * (CLEARANCE_PRELOAD_M - *preload);
                    }
                }
            }
            PedalKind::UnaCorda => {
                self.una_corda = unit;
            }
            PedalKind::Sostenuto => {
                let now_down = unit >= SOSTENUTO_THRESHOLD;
                if now_down && !self.sostenuto_down {
                    for k in 0..DAMPED_KEYS {
                        if self.clearance_m[k] > SOSTENUTO_OPEN_MIN_M {
                            self.sostenuto_latched[k] = true;
                            self.settling[k] = true;
                        }
                    }
                } else if !now_down && self.sostenuto_down {
                    self.sostenuto_latched.fill(false);
                    self.settling.fill(true);
                }
                self.sostenuto_down = now_down;
            }
        }
    }

    /// Current normalized sustain pedal position (`0.0..=1.0`).
    #[must_use]
    #[inline]
    pub const fn sustain(&self) -> f32 {
        self.sustain
    }

    /// Current normalized una corda pedal position (`0.0..=1.0`).
    #[must_use]
    #[inline]
    pub const fn una_corda(&self) -> f32 {
        self.una_corda
    }

    /// Returns whether key `key_idx` is currently latched by the sostenuto pedal.
    #[must_use]
    #[inline]
    pub fn is_sostenuto_latched(&self, key_idx: usize) -> bool {
        self.sostenuto_latched
            .get(key_idx)
            .copied()
            .unwrap_or(false)
    }

    /// Returns `Some(clearance_m)` when `key_idx`'s damper is completely settled open
    /// (`>= clear_threshold_m`) with no active smoothing or peel-off pulse.
    #[must_use]
    #[inline]
    pub fn is_damper_settled_open(&self, key_idx: usize, clear_threshold_m: f32) -> Option<f32> {
        if key_idx >= DAMPED_KEYS {
            return (CLEARANCE_OPEN_M >= clear_threshold_m).then_some(CLEARANCE_OPEN_M);
        }
        if !self.settling[key_idx]
            && self.peel_pos[key_idx] >= self.peel_len
            && self.last_peel_n[key_idx] == 0.0
            && self.clearance_m[key_idx] >= clear_threshold_m
        {
            Some(self.clearance_m[key_idx])
        } else {
            None
        }
    }

    /// Returns current smoothed damper clearance in meters (`+5.0 mm` for `70..88`).
    #[must_use]
    #[inline]
    pub fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        if key_idx < DAMPED_KEYS {
            self.clearance_m[key_idx]
        } else {
            CLEARANCE_OPEN_M
        }
    }

    /// Instantaneous damper peel-off adhesion force (`N`, `>= 0`) for `key_idx`
    /// produced by the most recent clearance step.
    #[must_use]
    #[inline]
    pub fn damper_peel_force_n(&self, key_idx: usize) -> f32 {
        self.last_peel_n.get(key_idx).copied().unwrap_or(0.0)
    }

    /// Returns per-string hammer strike weights `[f32; 3]` under current Una Corda.
    #[must_use]
    #[inline]
    pub fn strike_weights(&self, key_idx: usize) -> [f32; 3] {
        let Some(layout) = CHOIR_LAYOUTS.get(key_idx) else {
            return [0.0; 3];
        };
        let u = self.una_corda;
        match layout.strings_per_key {
            3 => [1.0 - u, 1.0, 1.0],
            2 => [1.0 - u, 1.0, 0.0],
            _ => [1.0 - 0.22 * u, 0.0, 0.0],
        }
    }

    /// Returns `(k_scale, eps_add)` modeling Una Corda virgin shoulder felt compliance
    /// softening and increased hysteresis for `key_idx`.
    #[must_use]
    #[inline]
    pub fn una_corda_felt_mod(&self, key_idx: usize) -> (f32, f32) {
        if self.una_corda <= 0.0 {
            return (1.0, 0.0);
        }
        let Some(layout) = CHOIR_LAYOUTS.get(key_idx) else {
            return (1.0, 0.0);
        };
        let u = self.una_corda.clamp(0.0, 1.0);
        let k_min = match layout.strings_per_key {
            3 => 0.28_f32,
            2 => 0.25_f32,
            _ => 0.52_f32,
        };
        (1.0 - (1.0 - k_min) * u, 0.12 * u)
    }
}

#[cfg(test)]
#[path = "pedals_tests.rs"]
mod tests;
