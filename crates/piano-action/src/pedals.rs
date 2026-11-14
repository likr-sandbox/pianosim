//! Pedal state machine (`Sustain`, `UnaCorda`, `Sostenuto`) and per-key
//! geometric damper-rail clearance smoother (`0..70` damped keys).

use piano_spec::{u16_to_unit_f32, PedalKind, CHOIR_LAYOUTS, DAMPED_KEYS, NUM_KEYS};

use crate::math::exp_f64;

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

/// Continuous 3-pedal controller and per-key damper clearance smoother.
#[derive(Clone, Debug)]
pub struct PedalBank {
    sustain: f32,
    una_corda: f32,
    sostenuto_down: bool,
    key_down: [bool; NUM_KEYS],
    sostenuto_latched: [bool; DAMPED_KEYS],
    clearance_m: [f32; DAMPED_KEYS],
    release_preload_m: [f32; DAMPED_KEYS],
    alpha_down: [f32; DAMPED_KEYS],
    alpha_lift: f32,
    alpha_down_slow: f32,
    alpha_down_fast: f32,
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

        Self {
            sustain: 0.0,
            una_corda: 0.0,
            sostenuto_down: false,
            key_down: [false; NUM_KEYS],
            sostenuto_latched: [false; DAMPED_KEYS],
            clearance_m: [CLEARANCE_PRELOAD_M; DAMPED_KEYS],
            release_preload_m: [CLEARANCE_PRELOAD_M; DAMPED_KEYS],
            alpha_down: [default_alpha_down; DAMPED_KEYS],
            alpha_lift,
            alpha_down_slow,
            alpha_down_fast,
        }
    }

    /// Resets all pedals and returns all damped keys to resting `-0.30 mm` preload.
    pub fn reset(&mut self) {
        self.sustain = 0.0;
        self.una_corda = 0.0;
        self.sostenuto_down = false;
        self.key_down.fill(false);
        self.sostenuto_latched.fill(false);
        self.clearance_m.fill(CLEARANCE_PRELOAD_M);
        self.release_preload_m.fill(CLEARANCE_PRELOAD_M);
        let default_alpha_down = 0.5 * (self.alpha_down_slow + self.alpha_down_fast);
        self.alpha_down.fill(default_alpha_down);
    }

    /// Lifts the key's damper to full open clearance (`+5.0 mm`) on key strike.
    #[inline]
    pub fn note_on(&mut self, key_idx: usize) {
        if key_idx >= NUM_KEYS {
            return;
        }
        self.key_down[key_idx] = true;
        if key_idx < DAMPED_KEYS {
            self.clearance_m[key_idx] = CLEARANCE_OPEN_M;
        }
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
        }
    }

    /// Updates continuous pedal position (`0..=65535`).
    #[inline]
    pub fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        let unit = u16_to_unit_f32(value_u16).clamp(0.0, 1.0);
        match kind {
            PedalKind::Sustain => {
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
                        }
                    }
                } else if !now_down && self.sostenuto_down {
                    self.sostenuto_latched.fill(false);
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

    /// Advances the 1-pole damper rail smoother for `key_idx` by one host sample.
    #[inline]
    pub fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        if key_idx >= DAMPED_KEYS {
            return CLEARANCE_OPEN_M;
        }
        let target = if self.key_down[key_idx] || self.sostenuto_latched[key_idx] {
            CLEARANCE_OPEN_M
        } else {
            sustain_clearance_m(self.sustain, self.release_preload_m[key_idx])
        };
        let cur = self.clearance_m[key_idx];
        let alpha = if target >= cur {
            self.alpha_lift
        } else {
            self.alpha_down[key_idx]
        };
        let next = cur + alpha * (target - cur);
        self.clearance_m[key_idx] = next;
        next
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
mod tests {
    use super::*;

    #[test]
    fn sustain_curve_hits_landmarks_and_is_strictly_monotonic() {
        let c0 = sustain_clearance_m(0.0, CLEARANCE_PRELOAD_M);
        let chalf = sustain_clearance_m(0.5, CLEARANCE_PRELOAD_M);
        let c1 = sustain_clearance_m(1.0, CLEARANCE_PRELOAD_M);
        assert!((c0 - CLEARANCE_PRELOAD_M).abs() < 1e-9);
        assert!(
            (2.0e-5..=4.5e-5).contains(&chalf),
            "half-pedal clearance={chalf}"
        );
        assert!((chalf - HALF_PEDAL_MID_CLEARANCE_M).abs() < 1e-7);
        assert!((c1 - CLEARANCE_OPEN_M).abs() < 1e-7);

        let mut prev = c0;
        for i in 1..=200 {
            let s = (i as f32) / 200.0;
            let cur = sustain_clearance_m(s, CLEARANCE_PRELOAD_M);
            assert!(cur > prev, "not strictly monotonic at s={s}");
            prev = cur;
        }
    }

    #[test]
    fn sostenuto_latches_only_open_damped_keys_on_crossing() {
        let mut p = PedalBank::new(48_000.0);
        // Hold key 39 (open), release key 40 long ago (closed), undamped key 75.
        p.note_on(39);
        assert!(p.damper_clearance_m(39) > SOSTENUTO_OPEN_MIN_M);
        assert!(p.damper_clearance_m(40) < 0.0);

        p.set_pedal(PedalKind::Sostenuto, 40_000); // crosses >= 0.5
        assert!(p.is_sostenuto_latched(39));
        assert!(!p.is_sostenuto_latched(40));

        // Key struck AFTER sostenuto down must NOT latch.
        p.note_on(41);
        assert!(!p.is_sostenuto_latched(41));

        // Release keys and run smoother for 200 ms: key 39 stays open, key 41 closes.
        p.note_off(39, 50_000);
        p.note_off(41, 50_000);
        for _ in 0..9_600 {
            p.step_key_clearance(39);
            p.step_key_clearance(41);
        }
        assert!((p.damper_clearance_m(39) - CLEARANCE_OPEN_M).abs() < 1e-6);
        assert!(p.damper_clearance_m(41) < -2.0e-4);

        // Release sostenuto (< 0.5) and verify key 39 drops to preload.
        p.set_pedal(PedalKind::Sostenuto, 10_000);
        assert!(!p.is_sostenuto_latched(39));
        for _ in 0..9_600 {
            p.step_key_clearance(39);
        }
        assert!(p.damper_clearance_m(39) < -2.0e-4);
        assert_eq!(p.step_key_clearance(75), CLEARANCE_OPEN_M);
    }

    #[test]
    fn una_corda_weights_match_monochord_bichord_trichord() {
        let mut p = PedalBank::new(48_000.0);
        assert_eq!(p.strike_weights(0), [1.0, 0.0, 0.0]);
        assert_eq!(p.strike_weights(12), [1.0, 1.0, 0.0]);
        assert_eq!(p.strike_weights(48), [1.0, 1.0, 1.0]);

        p.set_pedal(PedalKind::UnaCorda, u16::MAX);
        assert!((p.strike_weights(0)[0] - 0.78).abs() < 1e-6);
        assert_eq!(p.strike_weights(12), [0.0, 1.0, 0.0]);
        assert_eq!(p.strike_weights(48), [0.0, 1.0, 1.0]);
    }
}
