//! Stage 1 physical piano action: pedals, 1st-order Timoshenko hammer shank,
//! two-timescale Stulov wool felt, Borin K-method + 1st-order ADAA contact solver,
//! `8x` oversampled strike loop, and FTO #3 unilateral hysteretic damper pad.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use piano_simd::AdaaPowerLut;
use piano_spec::{PedalKind, VoicingSpec, NUM_KEYS};

pub mod damper_pad;
pub mod felt_stulov;
pub mod k_method;
pub mod keybed;
mod math;
pub mod oversample;
pub mod pedals;
pub mod shank_timoshenko;

#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_damper;
#[cfg(test)]
mod tests_hammer;

use damper_pad::DamperPadBank;
use felt_stulov::{FeltSharedCoeffs, FeltState, KeyFeltSpec};
use k_method::{BorinKLut, BORIN_LUT_SIZE, BORIN_P_MAX};
use keybed::KeybedBank;
use oversample::{DecimationFir16, HammerOversampleState};
use pedals::PedalBank;
use shank_timoshenko::{velocity_from_u16, ShankParams};

const ADAA_LUT_SIZE: usize = 1024;
const ADAA_LUT_U_MAX: f32 = 1.0;

#[derive(Clone, Debug)]
struct KeyActionSlot {
    shank_params: ShankParams,
    felt_spec: KeyFeltSpec,
    felt_state: FeltState,
    adaa_lut: AdaaPowerLut,
    borin_lut: BorinKLut,
    hammer: HammerOversampleState,
}

impl KeyActionSlot {
    fn new(key_idx: usize, sample_rate: f64, voicing: &VoicingSpec) -> Self {
        let shank_params = ShankParams::for_key(key_idx, sample_rate);
        let felt_spec = KeyFeltSpec::for_key(key_idx, sample_rate, voicing);
        let adaa_lut = AdaaPowerLut::new(felt_spec.exponent_p, ADAA_LUT_U_MAX, ADAA_LUT_SIZE);
        let borin_lut = BorinKLut::new(felt_spec.exponent_p, BORIN_P_MAX, BORIN_LUT_SIZE);

        Self {
            shank_params,
            felt_spec,
            felt_state: FeltState::default(),
            adaa_lut,
            borin_lut,
            hammer: HammerOversampleState::default(),
        }
    }
}

/// Complete Stage 1 physical piano action bank for all 88 keys and 230 strings.
///
/// All LUTs and per-key buffers are allocated once inside [`ActionBank::new`].
/// Every sample-rate and event method executes with zero heap allocations, zero
/// locks, zero syscalls, and zero panics.
#[derive(Clone, Debug)]
pub struct ActionBank {
    sample_rate: f32,
    memory_enabled: bool,
    felt_coeffs: FeltSharedCoeffs,
    decimation_fir: DecimationFir16,
    pedals: PedalBank,
    damper_pad: DamperPadBank,
    keybed: KeybedBank,
    keys: Box<[KeyActionSlot]>,
}

impl ActionBank {
    /// Constructs and pre-tabulates the Stage 1 action bank for `sample_rate` and `voicing`.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let sr_f64 = f64::from(sr);

        let mut keys = Vec::with_capacity(NUM_KEYS);
        for k in 0..NUM_KEYS {
            keys.push(KeyActionSlot::new(k, sr_f64, voicing));
        }

        Self {
            sample_rate: sr,
            memory_enabled: true,
            felt_coeffs: FeltSharedCoeffs::new(sr_f64),
            decimation_fir: DecimationFir16::new(),
            pedals: PedalBank::new(sr),
            damper_pad: DamperPadBank::new(sr, voicing),
            keybed: KeybedBank::new(sr),
            keys: keys.into_boxed_slice(),
        }
    }

    /// Configured host audio sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Resets hammers, dampers, pedals, keybed, and Stulov compaction memory to initial rest.
    pub fn reset(&mut self) {
        self.pedals.reset();
        self.damper_pad.reset();
        self.keybed.reset();
        for slot in self.keys.iter_mut() {
            slot.hammer.reset();
            slot.felt_state.reset();
        }
    }

    /// Triggers key strike (`velocity_u16 > 0`), or release (`velocity_u16 == 0`).
    ///
    /// Lifts the key's damper to `+5.0 mm` and launches the hammer behind `-20 um` let-off.
    /// Slow Stulov wool compaction memory `z_mem` persists across strikes.
    #[inline]
    pub fn note_on(&mut self, key_idx: usize, velocity_u16: u16) {
        if key_idx >= NUM_KEYS {
            return;
        }
        if velocity_u16 == 0 {
            self.note_off(key_idx, 0);
            return;
        }
        self.pedals.note_on(key_idx);
        self.keybed.note_on(key_idx, velocity_u16);
        let slot = &mut self.keys[key_idx];
        if !self.memory_enabled {
            slot.felt_state.reset();
        }
        slot.hammer.launch(velocity_from_u16(velocity_u16));
    }

    /// Releases `key_idx`, initiating damper drop at a rate scaled by `release_vel_u16`.
    #[inline]
    pub fn note_off(&mut self, key_idx: usize, release_vel_u16: u16) {
        if key_idx >= NUM_KEYS {
            return;
        }
        self.pedals.note_off(key_idx, release_vel_u16);
    }

    /// Updates continuous pedal position (`Sustain`, `UnaCorda`, `Sostenuto`).
    #[inline]
    pub fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        self.pedals.set_pedal(kind, value_u16);
    }

    /// Decays Stulov felt memory across `host_frames` idle samples for inactive hammers.
    #[inline]
    pub fn decay_idle_memory(&mut self, host_frames: usize) {
        if host_frames == 0 {
            return;
        }
        for slot in self.keys.iter_mut() {
            if !slot.hammer.is_active() {
                slot.felt_state
                    .decay_host_frames(&self.felt_coeffs, host_frames);
            }
        }
    }

    /// Enables or disables slow Stulov compaction memory across repeated strikes.
    #[inline]
    pub fn set_memory_enabled(&mut self, enabled: bool) {
        self.memory_enabled = enabled;
        if !enabled {
            for slot in self.keys.iter_mut() {
                slot.felt_state.z_mem = 0.0;
            }
        }
    }

    /// Returns `true` if `key_idx`'s hammer or polyphase FIR tail is active.
    #[must_use]
    #[inline]
    pub fn is_hammer_active(&self, key_idx: usize) -> bool {
        self.keys
            .get(key_idx)
            .is_some_and(|slot| slot.hammer.is_active())
    }

    /// Current smoothed damper clearance (`m`) for `key_idx`.
    #[must_use]
    #[inline]
    pub fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        self.pedals.damper_clearance_m(key_idx)
    }

    /// Current continuous sustain pedal position (`0.0..=1.0`).
    #[must_use]
    #[inline]
    pub fn sustain(&self) -> f32 {
        self.pedals.sustain()
    }

    /// Una Corda choir strike weights `[w0, w1, w2]` for `key_idx`.
    #[must_use]
    #[inline]
    pub fn strike_weights(&self, key_idx: usize) -> [f32; 3] {
        self.pedals.strike_weights(key_idx)
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
        self.pedals
            .step_key_clearance_tile(key_idx, t_len, out_clears)
    }

    /// Advances the 1-pole damper rail smoother by one host sample and returns `y_clear_m`.
    #[inline]
    pub fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        self.pedals.step_key_clearance(key_idx)
    }

    /// Steps one host sample of the `8x` oversampled hammer-string strike interaction,
    /// returning the decimated upward force (`N >= 0`) applied to the string choir.
    #[inline]
    pub fn step_hammer_sample(
        &mut self,
        key_idx: usize,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32 {
        let uc_mod = self.pedals.una_corda_felt_mod(key_idx);
        let Some(slot) = self.keys.get_mut(key_idx) else {
            return 0.0;
        };
        slot.hammer.step_host_sample(
            &slot.shank_params,
            &slot.felt_spec,
            &self.felt_coeffs,
            &mut slot.felt_state,
            &slot.adaa_lut,
            &slot.borin_lut,
            &self.decimation_fir,
            self.memory_enabled,
            uc_mod,
            y_str_m,
            v_str_mps,
            r_port_strike,
        )
    }

    /// Steps one host sample of the unilateral hysteretic damper pad for `(key_idx, slot_idx)`
    /// (`slot_idx = 2 * s_idx` vertical, `2 * s_idx + 1` horizontal),
    /// returning the passive damping force `F_damp` (`N`) applied **ON the string**.
    #[inline]
    pub fn step_damper_sample(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        self.damper_pad.step_damper_sample(
            key_idx,
            slot_idx,
            y_damp_m,
            v_damp_mps,
            r_port_damp,
            y_clear_m,
        )
    }

    /// Advances active keybed rail impacts by one host sample, returning the total
    /// Stage 4 Zone 8 mechanical keybed bottoming force (`N`).
    #[inline]
    pub fn step_keybed_force_sample(&mut self) -> f32 {
        self.keybed.step_sample()
    }

    /// Updates felt stiffness/scaling and damper pad impedance smoothly in-place
    /// for `voicing` with zero heap allocations, zero locks, and zero panics.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        let sr = f64::from(self.sample_rate);
        for (key_idx, slot) in self.keys.iter_mut().enumerate() {
            slot.felt_spec.update_voicing_in_place(key_idx, sr, voicing);
        }
        self.damper_pad.update_voicing_in_place(voicing);
    }

    #[cfg(test)]
    pub(crate) fn set_shank_flex_enabled(&mut self, key_idx: usize, enabled: bool) {
        if let Some(slot) = self.keys.get_mut(key_idx) {
            slot.hammer.shank.flex_enabled = enabled;
        }
    }

    #[cfg(test)]
    pub(crate) fn felt_memory(&self, key_idx: usize) -> f32 {
        self.keys
            .get(key_idx)
            .map_or(0.0, |slot| slot.felt_state.z_mem)
    }
}

#[cfg(test)]
mod escapement_tests {
    use alloc::vec;

    use piano_spec::{u7_to_u16, PedalKind, VoicingSpec};

    use super::ActionBank;
    use crate::tests_common::{spectral_centroid_and_band, TestModalString};

    #[test]
    fn silent_key_lifts_damper_without_string_contact_and_3_pedals_hold() {
        let voicing = VoicingSpec::default();
        let mut bank = ActionBank::new(48_000.0, &voicing);
        let c4 = 39_usize;
        let e4 = 43_usize;

        for vel in [1_u16, 16, 32, 64] {
            bank.reset();
            bank.note_on(c4, vel);
            let mut peak_force = 0.0_f32;
            for _ in 0..2_400 {
                let f = bank.step_hammer_sample(c4, 0.0, 0.0, 0.01);
                peak_force = peak_force.max(f);
                let c = bank.step_key_clearance(c4);
                assert!(c >= 4.5e-3, "vel={vel} clearance dropped: {c}");
            }
            assert_eq!(peak_force, 0.0, "silent vel={vel} struck string");
            assert!(!bank.is_hammer_active(c4));
            assert!(bank.damper_clearance_m(c4) >= 4.5e-3);
        }

        // Audible MIDI velocity 1 (`u7_to_u16(1)`) DOES strike the string.
        bank.reset();
        bank.note_on(c4, u7_to_u16(1));
        let mut pp_peak = 0.0_f32;
        for _ in 0..480 {
            pp_peak = pp_peak.max(bank.step_hammer_sample(c4, 0.0, 0.0, 0.01));
        }
        assert!(pp_peak > 0.0, "pp strike produced zero force");

        // 3-pedal invariants: Una Corda, Sostenuto, and Sustain.
        bank.reset();
        bank.set_pedal(PedalKind::UnaCorda, u16::MAX);
        assert_eq!(bank.strike_weights(c4), [0.0, 1.0, 1.0]);

        bank.note_on(c4, 1); // Silent C4 held open -> latched by Sostenuto.
        bank.set_pedal(PedalKind::Sostenuto, u16::MAX);
        bank.note_off(c4, 32_768);
        bank.note_on(e4, 40_000);
        bank.note_off(e4, 32_768);
        let mut tile = [0.0_f32; 8];
        for _ in 0..1_200 {
            bank.step_key_clearance_tile(c4, 8, &mut tile);
            bank.step_key_clearance_tile(e4, 8, &mut tile);
        }
        assert!(bank.damper_clearance_m(c4) >= 4.5e-3);
        assert!(bank.damper_clearance_m(e4) < 0.0);

        bank.set_pedal(PedalKind::Sostenuto, 0);
        bank.set_pedal(PedalKind::Sustain, u16::MAX);
        for _ in 0..1_200 {
            bank.step_key_clearance_tile(e4, 8, &mut tile);
        }
        assert!(bank.damper_clearance_m(e4) >= 4.5e-3);
    }

    #[test]
    fn una_corda_felt_softening_extends_contact_and_lowers_centroid() {
        let voicing = VoicingSpec::default();
        let measure_pulse =
            |key_idx: usize, vel_u16: u16, una_corda: bool| -> (f32, f32, Vec<f32>) {
                let mut bank = ActionBank::new(48_000.0, &voicing);
                if una_corda {
                    bank.set_pedal(PedalKind::UnaCorda, u16::MAX);
                    assert!(bank.pedals.una_corda_felt_mod(key_idx).0 < 0.30);
                } else {
                    assert_eq!(bank.pedals.una_corda_felt_mod(key_idx), (1.0, 0.0));
                }
                let mut s = TestModalString::new(key_idx, 48_000.0, &voicing);
                let port_scale = if una_corda { 1.5_f32 } else { 1.0 };
                bank.note_on(key_idx, vel_u16);
                let mut pulse = vec![0.0_f32; 512];
                for out in &mut pulse {
                    let (y, v) = s.strike_port();
                    let f = bank.step_hammer_sample(key_idx, y, v, s.r_port_strike * port_scale);
                    *out = f;
                    s.step_forces(f * port_scale, 0.0);
                }
                let peak = pulse.iter().copied().fold(0.0_f32, f32::max);
                let dur_ms =
                    (pulse.iter().filter(|&&f| f > 0.05 * peak).count() as f32) * 1000.0 / 48_000.0;
                let (centroid, _) = spectral_centroid_and_band(&pulse, 48_000.0, 100.0, 20_000.0);
                (dur_ms, centroid, pulse)
            };

        for &key_idx in &[39_usize, 48] {
            for &vel_u16 in &[48_000_u16, 60_000] {
                let (dur_off_a, cent_off_a, pulse_a) = measure_pulse(key_idx, vel_u16, false);
                let (dur_off_b, _, pulse_b) = measure_pulse(key_idx, vel_u16, false);
                assert_eq!(pulse_a, pulse_b);

                let (dur_uc, cent_uc, _) = measure_pulse(key_idx, vel_u16, true);
                assert!(
                    dur_uc > dur_off_a,
                    "k={key_idx} v={vel_u16}: dur_uc={dur_uc} <= dur_off={dur_off_a}"
                );
                assert!(
                    cent_uc < cent_off_a,
                    "k={key_idx} v={vel_u16}: cent_uc={cent_uc} >= cent_off={cent_off_a}"
                );
                let _ = dur_off_b;
            }
        }
    }

    #[test]
    fn keybed_and_voicing_update_work_in_place_on_action_bank() {
        let nominal = VoicingSpec::default();
        let mut bank = ActionBank::new(48_000.0, &nominal);
        assert_eq!(bank.step_keybed_force_sample(), 0.0);
        bank.note_on(48, u7_to_u16(1) - 1);
        assert_eq!(bank.step_keybed_force_sample(), 0.0);

        bank.note_on(48, u7_to_u16(96));
        for _ in 0..300 {
            assert_eq!(bank.step_keybed_force_sample(), 0.0);
        }
        let peak_kb = (0..300).fold(0.0_f32, |m, _| m.max(bank.step_keybed_force_sample()));
        assert!(peak_kb > 0.0);
        bank.note_on(48, u7_to_u16(96));
        bank.reset();
        assert_eq!(bank.step_keybed_force_sample(), 0.0);

        let orig_p = bank.keys[48].felt_spec.exponent_p;
        let strike_stats = |b: &mut ActionBank| -> (f32, f32) {
            b.reset();
            let mut s = TestModalString::new(48, 48_000.0, &nominal);
            b.note_on(48, 51_602);
            let mut pulse = vec![0.0_f32; 256];
            for out in &mut pulse {
                let (y, v) = s.strike_port();
                let f = b.step_hammer_sample(48, y, v, s.r_port_strike);
                *out = f;
                s.step_forces(f, 0.0);
            }
            let pk = pulse.iter().copied().fold(0.0_f32, f32::max);
            let dur = pulse.iter().filter(|&&f| f > 0.05 * pk).count() as f32;
            let (cent, _) = spectral_centroid_and_band(&pulse, 48_000.0, 100.0, 20_000.0);
            (dur, cent)
        };

        let (dur_nom, cent_nom) = strike_stats(&mut bank);
        let vspec = |h: f32| VoicingSpec {
            hammer_hardness: h,
            ..VoicingSpec::default()
        };
        bank.update_voicing_in_place(&vspec(0.1));
        assert_eq!(bank.keys[48].felt_spec.exponent_p, orig_p);
        let (dur_soft, cent_soft) = strike_stats(&mut bank);
        bank.update_voicing_in_place(&vspec(0.9));
        assert_eq!(bank.keys[48].felt_spec.exponent_p, orig_p);
        let (dur_hard, cent_hard) = strike_stats(&mut bank);
        assert!(dur_hard < dur_soft && cent_hard > cent_soft);

        bank.update_voicing_in_place(&nominal);
        let (dur_restored, cent_restored) = strike_stats(&mut bank);
        assert!((dur_restored - dur_nom).abs() <= 1.0 && (cent_restored - cent_nom).abs() < 1e-2);
    }
}
