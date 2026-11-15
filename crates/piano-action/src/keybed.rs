//! FTO #6 mechanical keybed bottoming impact model (`KeybedBank`).
//!
//! Produces the viscoelastic Stage 4 Zone 8 rail impact pulse (`N`) after physical
//! key-dip travel time (`>= 6.5 ms`), clearing the bass longitudinal precursor window
//! without using any separate decay envelope controller.

use piano_spec::{u16_to_unit_f32, u7_to_u16, NUM_KEYS};

use crate::math::{pow_nonneg_f64, sin_cos_f64};
use crate::shank_timoshenko::velocity_from_u16;

const KEYBED_PULSE_DURATION_S: f64 = 3.4e-3;
const KEYBED_MIN_DELAY_S: f64 = 6.5e-3;
const KEYBED_MAX_DELAY_S: f64 = 18.0e-3;
const KEYBED_TRAVEL_REF_M: f64 = 2.0e-3;
const KEY_HAMMER_LEVER_RATIO: f64 = 0.22;
const KEYBED_BASE_PEAK_N: f64 = 8.0;
const KEYBED_VEL_EXPONENT: f64 = 1.8;

#[derive(Clone, Copy, Debug, Default)]
struct KeybedSlot {
    delay_rem: u32,
    pulse_pos: u32,
    pulse_len: u32,
    peak_force_n: f32,
}

/// 88-key mechanical keybed impact pulse bank with O(1) idle bitmask dispatch.
#[derive(Clone, Debug)]
pub(crate) struct KeybedBank {
    sample_rate: f64,
    pulse_len: u32,
    active_mask: u128,
    keys: [KeybedSlot; NUM_KEYS],
}

impl KeybedBank {
    /// Initializes an idle keybed impact bank at `sample_rate`.
    #[must_use]
    pub(crate) fn new(sample_rate: f32) -> Self {
        let sr = f64::from(sample_rate.max(8_000.0));
        let pulse_len = ((KEYBED_PULSE_DURATION_S * sr) + 0.5).clamp(8.0, 4096.0) as u32;
        Self {
            sample_rate: sr,
            pulse_len,
            active_mask: 0,
            keys: [KeybedSlot::default(); NUM_KEYS],
        }
    }

    /// Clears all pending and active keybed impact pulses immediately.
    pub(crate) fn reset(&mut self) {
        self.active_mask = 0;
        self.keys = [KeybedSlot::default(); NUM_KEYS];
    }

    /// Schedules a keybed bottoming pulse for audible `NoteOn` strikes (`>= u7_to_u16(1)`).
    pub(crate) fn note_on(&mut self, key_idx: usize, velocity_u16: u16) {
        if key_idx >= NUM_KEYS || velocity_u16 < u7_to_u16(1) {
            return;
        }
        let v_key = (f64::from(velocity_from_u16(velocity_u16)) * KEY_HAMMER_LEVER_RATIO).max(1e-4);
        let t_delay = (KEYBED_TRAVEL_REF_M / v_key).clamp(KEYBED_MIN_DELAY_S, KEYBED_MAX_DELAY_S);
        let delay_rem = ((t_delay * self.sample_rate) + 0.999_999) as u32;

        let vel_norm = f64::from(u16_to_unit_f32(velocity_u16)).clamp(0.0, 1.0);
        let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
        let register_weight = 0.55 + 0.70 * x;
        let peak_force_n = (KEYBED_BASE_PEAK_N
            * pow_nonneg_f64(vel_norm, KEYBED_VEL_EXPONENT)
            * register_weight) as f32;

        if let Some(slot) = self.keys.get_mut(key_idx) {
            slot.delay_rem = delay_rem;
            slot.pulse_pos = 0;
            slot.pulse_len = self.pulse_len;
            slot.peak_force_n = peak_force_n;
            self.active_mask |= 1_u128 << (key_idx & 127);
        }
    }

    /// Advances active keybed pulses by one host sample and returns total Zone 8 force (`N`).
    #[inline]
    pub(crate) fn step_sample(&mut self) -> f32 {
        let mut pending = self.active_mask;
        if pending == 0 {
            return 0.0;
        }
        let mut total_force_n = 0.0_f32;
        while pending != 0 {
            let k = pending.trailing_zeros() as usize;
            pending &= pending - 1;
            let bit = 1_u128 << (k & 127);
            let Some(slot) = self.keys.get_mut(k) else {
                self.active_mask &= !bit;
                continue;
            };
            if slot.delay_rem > 0 {
                slot.delay_rem -= 1;
                continue;
            }
            if slot.pulse_pos < slot.pulse_len {
                let u = (f64::from(slot.pulse_pos) + 0.5) / f64::from(slot.pulse_len);
                let (_, cos_tau) = sin_cos_f64(core::f64::consts::TAU * u);
                let c = 0.5 * (1.0 - cos_tau);
                total_force_n += slot.peak_force_n * ((c * c) as f32);
                slot.pulse_pos += 1;
                if slot.pulse_pos >= slot.pulse_len {
                    self.active_mask &= !bit;
                }
            } else {
                self.active_mask &= !bit;
            }
        }
        total_force_n
    }

    #[cfg(test)]
    pub(crate) fn is_active(&self) -> bool {
        self.active_mask != 0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::f64::consts::TAU;

    use piano_spec::{u7_to_u16, NUM_KEYS};

    use super::KeybedBank;

    #[test]
    fn keybed_is_idle_initially_ignores_silent_vel_and_resets_cleanly() {
        let mut bank = KeybedBank::new(48_000.0);
        assert!(!bank.is_active());
        assert_eq!(bank.step_sample(), 0.0);

        bank.note_on(39, 0);
        bank.note_on(39, u7_to_u16(1) - 1);
        bank.note_on(NUM_KEYS, u7_to_u16(100));
        assert!(!bank.is_active());
        assert_eq!(bank.step_sample(), 0.0);

        bank.note_on(39, u7_to_u16(100));
        assert!(bank.is_active());
        bank.reset();
        assert!(!bank.is_active());
        assert_eq!(bank.step_sample(), 0.0);
    }

    #[test]
    fn keybed_delay_clears_precursor_is_bandlimited_and_monotonic_across_vel() {
        let sr = 48_000.0_f32;
        let min_delay_samples = (6.5e-3 * f64::from(sr)) as usize;
        let n_frames = 1_536_usize; // 32 ms

        let mut prev_peak = 0.0_f32;
        for &vel in &[
            u7_to_u16(1),
            u7_to_u16(24),
            u7_to_u16(64),
            u7_to_u16(100),
            u16::MAX,
        ] {
            let mut bank = KeybedBank::new(sr);
            bank.note_on(12, vel);

            let mut trace = vec![0.0_f32; n_frames];
            let mut first_nz = None;
            let mut peak = 0.0_f32;
            for (i, sample) in trace.iter_mut().enumerate() {
                let f = bank.step_sample();
                assert!(f >= 0.0 && f.is_finite());
                *sample = f;
                if f > 0.0 && first_nz.is_none() {
                    first_nz = Some(i);
                }
                peak = peak.max(f);
            }
            assert!(!bank.is_active());

            let onset = first_nz.expect("audible note must trigger keybed pulse");
            assert!(
                onset >= min_delay_samples && onset <= (18.5e-3 * f64::from(sr)) as usize,
                "vel={vel}: onset={onset} samples out of [6.5ms, 18.5ms]"
            );
            assert!(
                peak > prev_peak,
                "vel={vel}: peak={peak} did not strictly increase over {prev_peak}"
            );
            prev_peak = peak;

            // Verify C^2 raised-cosine pulse sits < 900 Hz with negligible > 18 kHz energy.
            let mut e_total = 0.0_f64;
            let mut e_low = 0.0_f64;
            let mut e_high = 0.0_f64;
            for k in 0..=n_frames / 2 {
                let freq = (k as f64) * f64::from(sr) / (n_frames as f64);
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (n, &x) in trace.iter().enumerate() {
                    let ph = TAU * (k as f64) * (n as f64) / (n_frames as f64);
                    re += f64::from(x) * ph.cos();
                    im -= f64::from(x) * ph.sin();
                }
                let pwr = (re * re + im * im)
                    * if k == 0 || k * 2 == n_frames {
                        1.0
                    } else {
                        2.0
                    };
                e_total += pwr;
                if freq <= 900.0 {
                    e_low += pwr;
                }
                if freq >= 18_000.0 {
                    e_high += pwr;
                }
            }
            assert!(
                e_low / e_total > 0.99,
                "vel={vel}: low-band ratio={}",
                e_low / e_total
            );
            assert!(
                e_high / e_total < 1e-8,
                "vel={vel}: ultrasonic ratio={}",
                e_high / e_total
            );
        }

        // Verify higher register key produces slightly stronger keybed impulse at fixed velocity.
        let peak_of = |key| {
            let mut b = KeybedBank::new(sr);
            b.note_on(key, u7_to_u16(96));
            (0..n_frames).fold(0.0_f32, |m, _| m.max(b.step_sample()))
        };
        assert!(peak_of(75) > peak_of(5));
    }
}
