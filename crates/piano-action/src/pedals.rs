//! Pedal state machine (`Sustain`, `UnaCorda`, `Sostenuto`) and per-key
//! geometric damper-rail clearance smoother (`0..70` damped keys).

use piano_spec::{u16_to_unit_f32, PedalKind, CHOIR_LAYOUTS, DAMPED_KEYS, NUM_KEYS};

use crate::math::{exp_f64, sin_cos_f64};

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
const HALF_PEDAL_LOW: f32 = 0.35;
const HALF_PEDAL_HIGH: f32 = 0.65;
#[cfg(test)]
const HALF_PEDAL_MID_CLEARANCE_M: f32 = 3.2e-5;
const HALF_PEDAL_HI_CLEARANCE_M: f32 = 6.4e-5;
const SETTLE_EPS_M: f32 = 1.0e-7;
const PEEL_PULSE_DURATION_S: f64 = 2.0e-3;
const PEEL_MIN_PEAK_N: f32 = 5.0e-4;
const PEEL_MAX_PEAK_N: f32 = 2.0e-3;
const PEEL_ARM_CLEARANCE_M: f32 = -1.0e-5;

/// Continuous 3-pedal controller and per-key damper clearance smoother.
#[derive(Clone, Debug)]
pub struct PedalBank {
    sustain: f32,
    una_corda: f32,
    sostenuto_down: bool,
    key_down: [bool; NUM_KEYS],
    sostenuto_latched: [bool; DAMPED_KEYS],
    settling: [bool; DAMPED_KEYS],
    clearance_m: [f32; DAMPED_KEYS],
    release_preload_m: [f32; DAMPED_KEYS],
    alpha_down: [f32; DAMPED_KEYS],
    alpha_lift: f32,
    alpha_down_slow: f32,
    alpha_down_fast: f32,
    peel_len: u16,
    peel_played: [bool; DAMPED_KEYS],
    peel_armed: [bool; DAMPED_KEYS],
    peel_min_clear_m: [f32; DAMPED_KEYS],
    peel_pos: [u16; DAMPED_KEYS],
    peel_peak_n: [f32; DAMPED_KEYS],
    last_peel_n: [f32; DAMPED_KEYS],
}

impl PedalBank {
    /// Initializes pedal bank and 1-pole rail coefficients for `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let sr = f64::from(sample_rate.max(8_000.0));
        let alpha_lift = (1.0 - exp_f64(-1.0 / (LIFT_TAU_S * sr))) as f32;
        let alpha_down_slow = (1.0 - exp_f64(-1.0 / (DROP_TAU_SLOW_S * sr))) as f32;
        let alpha_down_fast = (1.0 - exp_f64(-1.0 / (DROP_TAU_FAST_S * sr))) as f32;
        let default_alpha_down = 0.5 * (alpha_down_slow + alpha_down_fast);
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
            alpha_lift,
            alpha_down_slow,
            alpha_down_fast,
            peel_len,
            peel_played: [false; DAMPED_KEYS],
            peel_armed: [false; DAMPED_KEYS],
            peel_min_clear_m: [0.0; DAMPED_KEYS],
            peel_pos: [peel_len; DAMPED_KEYS],
            peel_peak_n: [0.0; DAMPED_KEYS],
            last_peel_n: [0.0; DAMPED_KEYS],
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
            self.alpha_down[key_idx] =
                self.alpha_down_slow + (self.alpha_down_fast - self.alpha_down_slow) * rel_u;
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

    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and per-sample peel-off adhesion
    /// forces (`N`) into `out_peel_n[..n]`. Returns `(min_clear_m, any_peel_nonzero)`.
    #[inline]
    pub fn step_key_clearance_tile_with_peel(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
        out_peel_n: &mut [f32; 8],
    ) -> (f32, bool) {
        let n = t_len.min(8);
        if key_idx >= DAMPED_KEYS {
            out_clears[..n].fill(CLEARANCE_OPEN_M);
            out_peel_n[..n].fill(0.0);
            return (CLEARANCE_OPEN_M, false);
        }
        if n == 0 {
            return (self.clearance_m[key_idx], false);
        }
        let peel_active = self.peel_pos[key_idx] < self.peel_len;
        if !self.settling[key_idx] && !peel_active {
            let cur = self.clearance_m[key_idx];
            out_clears[..n].fill(cur);
            out_peel_n[..n].fill(0.0);
            if self.last_peel_n[key_idx] != 0.0 {
                self.last_peel_n[key_idx] = 0.0;
            }
            return (cur, false);
        }

        let target = if self.key_down[key_idx] || self.sostenuto_latched[key_idx] {
            CLEARANCE_OPEN_M
        } else {
            sustain_clearance_m(self.sustain, self.release_preload_m[key_idx])
        };
        let mut cur = self.clearance_m[key_idx];
        let alpha = if target >= cur {
            self.alpha_lift
        } else {
            self.alpha_down[key_idx]
        };

        let mut min_clear = f32::INFINITY;
        let mut settled = !self.settling[key_idx];
        let mut any_peel = false;
        let mut last_peel = 0.0_f32;
        for i in 0..n {
            let prev = cur;
            if settled || (target - cur).abs() < SETTLE_EPS_M {
                cur = target;
                settled = true;
            } else {
                cur += alpha * (target - cur);
                if (target - cur).abs() < SETTLE_EPS_M {
                    cur = target;
                    settled = true;
                }
            }
            out_clears[i] = cur;
            min_clear = min_clear.min(cur);

            if self.peel_played[key_idx] && cur <= PEEL_ARM_CLEARANCE_M {
                self.peel_armed[key_idx] = true;
                self.peel_min_clear_m[key_idx] = self.peel_min_clear_m[key_idx].min(cur);
            } else if self.peel_armed[key_idx] && prev < 0.0 && cur >= 0.0 {
                let x = (key_idx as f32) / ((DAMPED_KEYS - 1) as f32);
                let depth =
                    (-self.peel_min_clear_m[key_idx] / (-CLEARANCE_PRELOAD_M)).clamp(0.25, 1.0);
                let peak = PEEL_MIN_PEAK_N
                    + (PEEL_MAX_PEAK_N - PEEL_MIN_PEAK_N)
                        * (0.30 + 0.70 * (1.0 - x))
                        * (0.55 + 0.45 * depth);
                self.peel_peak_n[key_idx] = peak.clamp(PEEL_MIN_PEAK_N, PEEL_MAX_PEAK_N);
                self.peel_pos[key_idx] = 0;
                self.peel_armed[key_idx] = false;
                self.peel_min_clear_m[key_idx] = 0.0;
            }

            let f_peel = if self.peel_pos[key_idx] < self.peel_len {
                let u = (f64::from(self.peel_pos[key_idx]) + 0.5) / f64::from(self.peel_len);
                let (_, cos_tau) = sin_cos_f64(core::f64::consts::TAU * u);
                let c = 0.5 * (1.0 - cos_tau);
                let f = self.peel_peak_n[key_idx] * ((c * c) as f32);
                self.peel_pos[key_idx] += 1;
                any_peel |= f != 0.0;
                f
            } else {
                0.0
            };
            out_peel_n[i] = f_peel;
            last_peel = f_peel;
        }
        self.clearance_m[key_idx] = cur;
        self.last_peel_n[key_idx] = last_peel;
        if settled {
            self.settling[key_idx] = false;
        }
        (min_clear, any_peel)
    }

    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and returning the tile minimum.
    #[inline]
    pub fn step_key_clearance_tile(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
    ) -> f32 {
        let n = t_len.min(8);
        if key_idx >= DAMPED_KEYS {
            out_clears[..n].fill(CLEARANCE_OPEN_M);
            return CLEARANCE_OPEN_M;
        }
        if n == 0 {
            return self.clearance_m[key_idx];
        }
        if !self.settling[key_idx] && self.peel_pos[key_idx] >= self.peel_len {
            let cur = self.clearance_m[key_idx];
            out_clears[..n].fill(cur);
            if self.last_peel_n[key_idx] != 0.0 {
                self.last_peel_n[key_idx] = 0.0;
            }
            return cur;
        }
        let mut tmp_peel = [0.0_f32; 8];
        self.step_key_clearance_tile_with_peel(key_idx, t_len, out_clears, &mut tmp_peel)
            .0
    }

    /// Advances the 1-pole damper rail smoother for `key_idx` by one host sample.
    #[inline]
    pub fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        let mut tmp = [0.0_f32; 8];
        self.step_key_clearance_tile(key_idx, 1, &mut tmp)
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

/// Smooth strictly monotonic sustain-pedal lift curve mapping `s in [0.0, 1.0]`
/// to damper clearance in meters:
/// - `0.0 -> preload_m` (`-0.30 mm` nominal)
/// - `0.5 -> +32 um` (`+0.032 mm` half-pedal grazing zone)
/// - `1.0 -> +5.0 mm` (open gap)
#[must_use]
#[inline]
pub fn sustain_clearance_m(sustain: f32, preload_m: f32) -> f32 {
    let s = sustain.clamp(0.0, 1.0);
    if s <= HALF_PEDAL_LOW {
        let t = s / HALF_PEDAL_LOW;
        let smooth = 0.25 * t + 0.75 * (t * t * (3.0 - 2.0 * t));
        preload_m * (1.0 - smooth)
    } else if s <= HALF_PEDAL_HIGH {
        let t = (s - HALF_PEDAL_LOW) / (HALF_PEDAL_HIGH - HALF_PEDAL_LOW);
        t * HALF_PEDAL_HI_CLEARANCE_M
    } else {
        let span = 1.0 - HALF_PEDAL_HIGH;
        let ds = s - HALF_PEDAL_HIGH;
        let t = ds / span;
        let slope_lo = HALF_PEDAL_HI_CLEARANCE_M / (HALF_PEDAL_HIGH - HALF_PEDAL_LOW);
        let linear = HALF_PEDAL_HI_CLEARANCE_M + slope_lo * ds;
        linear + (CLEARANCE_OPEN_M - HALF_PEDAL_HI_CLEARANCE_M - slope_lo * span) * (t * t)
    }
}

#[cfg(test)]
#[path = "pedals_tests.rs"]
mod tests;
