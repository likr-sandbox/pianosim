//! Stage 1 physical piano action: pedals, 1st-order Timoshenko hammer shank,
//! two-timescale Stulov wool felt, Borin K-method + 1st-order ADAA contact solver,
//! `8x` oversampled strike loop, and FTO #3 unilateral hysteretic damper pad.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use piano_spec::{
    build_scale_design, PedalKind, StringPhysicalParams, VoicingSpec, CHOIR_LAYOUTS, NUM_KEYS,
};

pub mod agraffe_delay;
pub mod damper_pad;
pub mod felt_stulov;
pub mod k_method;
pub mod keybed;
mod math;
pub mod oversample;
pub mod p_grid;
pub mod pedals;
pub mod shank_timoshenko;

#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_damper;
#[cfg(test)]
mod tests_escapement;
#[cfg(test)]
mod tests_hammer;

use agraffe_delay::AgraffeDelayLine;
use damper_pad::DamperPadBank;
use felt_stulov::{FeltSharedCoeffs, FeltState, KeyFeltSpec};
use keybed::KeybedBank;
use oversample::{DecimationFir16, HammerOversampleState};
use p_grid::PGridLuts;
use pedals::PedalBank;
use shank_timoshenko::{velocity_from_u16, ShankParams, SILENT_KEY_MAX_U16};

#[derive(Clone, Debug)]
struct KeyActionSlot {
    shank_params: ShankParams,
    felt_spec: KeyFeltSpec,
    felt_state: FeltState,
    hammer: HammerOversampleState,
    agraffe: AgraffeDelayLine,
}

impl KeyActionSlot {
    fn new(
        key_idx: usize,
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
    ) -> Self {
        let shank_params = ShankParams::for_key(key_idx, sample_rate);
        let felt_spec = KeyFeltSpec::for_key(key_idx, sample_rate, voicing);
        let agraffe = AgraffeDelayLine::new(key_idx, sample_rate, voicing, specs);

        Self {
            shank_params,
            felt_spec,
            felt_state: FeltState::default(),
            hammer: HammerOversampleState::default(),
            agraffe,
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
    p_grid: PGridLuts,
    decimation_fir: DecimationFir16,
    pedals: PedalBank,
    damper_pad: DamperPadBank,
    keybed: KeybedBank,
    keys: Box<[KeyActionSlot]>,
    hammer_voicing: VoicingSpec,
    key_hardness_applied: [f32; NUM_KEYS],
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

        let specs = build_scale_design(voicing);
        let mut keys = Vec::with_capacity(NUM_KEYS);
        for k in 0..NUM_KEYS {
            keys.push(KeyActionSlot::new(k, sr_f64, voicing, &specs));
        }

        Self {
            sample_rate: sr,
            memory_enabled: true,
            felt_coeffs: FeltSharedCoeffs::new(sr_f64),
            p_grid: PGridLuts::new(),
            decimation_fir: DecimationFir16::new(),
            pedals: PedalBank::new(sr),
            damper_pad: DamperPadBank::new(sr, voicing),
            keybed: KeybedBank::new(sr),
            keys: keys.into_boxed_slice(),
            hammer_voicing: voicing.sanitized(),
            key_hardness_applied: [voicing.effective_global_hammer_hardness(); NUM_KEYS],
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
            slot.agraffe.reset();
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
        if velocity_u16 > SILENT_KEY_MAX_U16 {
            self.pedals.note_on(key_idx);
        } else {
            self.pedals.note_on_silent(key_idx);
        }
        self.damper_pad.lift_key_open(key_idx);
        self.keybed.note_on(key_idx, velocity_u16);
        let slot = &mut self.keys[key_idx];
        let target_h = self.hammer_voicing.hammer_hardness;
        if (self.key_hardness_applied[key_idx] - target_h).abs() > 1.0e-6 {
            slot.felt_spec.update_voicing_in_place(
                key_idx,
                f64::from(self.sample_rate),
                &self.hammer_voicing,
            );
            self.key_hardness_applied[key_idx] = target_h;
        }
        if !self.memory_enabled {
            slot.felt_state.reset();
        }
        slot.agraffe.reset();
        slot.hammer.launch(velocity_from_u16(velocity_u16));
    }

    /// Releases `key_idx`, initiating damper drop and key-return Zone 8 shock scaled by `release_vel_u16`.
    #[inline]
    pub fn note_off(&mut self, key_idx: usize, release_vel_u16: u16) {
        if key_idx >= NUM_KEYS {
            return;
        }
        self.pedals.note_off(key_idx, release_vel_u16);
        self.keybed.note_off(key_idx, release_vel_u16);
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

    /// Instantaneous damper peel-off adhesion force (`N`, `>= 0`) for `key_idx`.
    #[must_use]
    #[inline]
    pub fn damper_peel_force_n(&self, key_idx: usize) -> f32 {
        self.pedals.damper_peel_force_n(key_idx)
    }

    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and peel-off adhesion forces (`N`)
    /// into `out_peel_n[..n]`. Returns `(min_clear_m, any_peel_nonzero)`.
    #[inline]
    pub fn step_key_clearance_tile_with_peel(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
        out_peel_n: &mut [f32; 8],
    ) -> (f32, bool) {
        self.pedals
            .step_key_clearance_tile_with_peel(key_idx, t_len, out_clears, out_peel_n)
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
        let w = self.pedals.strike_weights(key_idx);
        let Some(slot) = self.keys.get_mut(key_idx) else {
            return 0.0;
        };
        let w_sum = (w[0] + w[1] + w[2]).max(1.0);
        let n_str = f32::from(CHOIR_LAYOUTS[key_idx].strings_per_key.max(1));
        let uc_choir_scale = n_str / w_sum;
        let (p_lo, p_hi, p_t) = self.p_grid.lookup(slot.felt_spec.exponent_p);
        let (adaa_lo, adaa_hi, borin_lo, borin_hi) = self.p_grid.luts_for(p_lo, p_hi);
        slot.hammer.step_host_sample_with_agraffe(
            &slot.shank_params,
            &slot.felt_spec,
            &self.felt_coeffs,
            &mut slot.felt_state,
            adaa_lo,
            adaa_hi,
            borin_lo,
            borin_hi,
            p_t,
            &self.decimation_fir,
            self.memory_enabled,
            uc_mod,
            y_str_m,
            v_str_mps,
            r_port_strike,
            Some(&mut slot.agraffe),
            uc_choir_scale,
        )
    }

    /// Overrides `key_idx`'s `8x` agraffe round-trip delay and wave admittance.
    #[inline]
    pub fn set_agraffe_delay_samples(
        &mut self,
        key_idx: usize,
        delay_substeps: f32,
        wave_admittance: f32,
    ) {
        if let Some(slot) = self.keys.get_mut(key_idx) {
            slot.agraffe
                .set_delay_params(delay_substeps, wave_admittance);
        }
    }

    /// Returns the most recent peak normalized felt compression (`[0.0, 1.0]`) for `key_idx`.
    #[must_use]
    #[inline]
    pub fn hammer_compression_norm(&self, key_idx: usize) -> f32 {
        self.keys
            .get(key_idx)
            .map_or(0.0, |slot| slot.hammer.last_u_norm)
    }

    /// Returns `(last_host_force_n, prev_sub_forces)` from the most recent strike step of `key_idx`.
    #[must_use]
    #[inline]
    pub fn last_hammer_forces(&self, key_idx: usize) -> (f32, [f32; 8]) {
        self.keys.get(key_idx).map_or((0.0, [0.0; 8]), |slot| {
            (slot.hammer.last_host_force_n, slot.hammer.prev_sub_forces)
        })
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

    /// Evaluates one host sample for both V (`2 * str_idx`) and H (`2 * str_idx + 1`)
    /// damper slots of `(key_idx, str_idx)`, sharing geometric clearance computations.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_damper_pair(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        y_dv_m: f32,
        v_dv_mps: f32,
        r_dv: f32,
        y_dh_m: f32,
        v_dh_mps: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> (f32, f32) {
        self.damper_pad.step_damper_pair(
            key_idx, str_idx, y_dv_m, v_dv_mps, r_dv, y_dh_m, v_dh_mps, r_dh, y_clear_m,
        )
    }

    /// Returns exact closed-damper linear-equilibrium V/H velocity gains when felt memory has settled.
    #[inline]
    pub fn symp_linear_gains_if_settled(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> Option<(f32, f32)> {
        self.damper_pad
            .symp_linear_gains_if_settled(key_idx, str_idx, r_dv, r_dh, y_clear_m)
    }

    /// Advances quiescent damper felt relaxation over `sub` samples and returns linear V/H gains.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn symp_linear_gains_for_chunk(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        yc0: f32,
        yc_end: f32,
        sub: usize,
    ) -> Option<(f32, f32)> {
        self.damper_pad
            .symp_linear_gains_for_chunk(key_idx, str_idx, r_dv, r_dh, yc0, yc_end, sub)
    }

    /// Relaxes damper felt hysteresis states `(z_pos, z_neg)` across `sub` open-gap samples.
    #[inline]
    pub fn relax_open_gap_chunk(&mut self, str_idx: usize, sub: usize) {
        self.damper_pad.relax_open_gap_chunk(str_idx, sub);
    }

    /// Advances active keybed rail impacts by one host sample, returning the total
    /// Stage 4 Zone 8 mechanical keybed bottoming force (`N`).
    #[inline]
    pub fn step_keybed_force_sample(&mut self) -> f32 {
        self.keybed.step_sample()
    }

    /// Updates only global hammer hardness cleanly (`< 1 us`), updating active hammers
    /// immediately and lazily syncing resting hammers on next `note_on` without recomputing dampers.
    pub fn update_hammer_hardness_in_place(&mut self, voicing: &VoicingSpec) {
        let sr = f64::from(self.sample_rate);
        let h = voicing.effective_global_hammer_hardness();
        self.hammer_voicing.hammer_hardness = h;
        for (key_idx, slot) in self.keys.iter_mut().enumerate() {
            if slot.hammer.is_active() {
                slot.felt_spec
                    .update_voicing_in_place(key_idx, sr, &self.hammer_voicing);
                self.key_hardness_applied[key_idx] = h;
            }
        }
    }

    /// Updates felt stiffness/scaling and damper pad impedance smoothly in-place
    /// for `voicing` with zero heap allocations, zero locks, and zero panics.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        let sr = f64::from(self.sample_rate);
        self.hammer_voicing = voicing.sanitized();
        let specs = build_scale_design(&self.hammer_voicing);
        self.key_hardness_applied
            .fill(self.hammer_voicing.hammer_hardness);
        for (key_idx, slot) in self.keys.iter_mut().enumerate() {
            slot.felt_spec
                .update_voicing_in_place(key_idx, sr, &self.hammer_voicing);
            slot.agraffe
                .update_voicing(key_idx, sr, &self.hammer_voicing, &specs);
        }
        self.damper_pad
            .update_voicing_in_place(&self.hammer_voicing);
    }
}
