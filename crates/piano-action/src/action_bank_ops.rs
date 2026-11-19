//! [`ActionBank`] lifecycle, `PerfProfile` configuration, MIDI event handling,
//! and in-place voicing/hardness updates.

use alloc::vec::Vec;

use piano_simd::PolyPowerApprox;
use piano_spec::{
    build_scale_design, PedalKind, PerfProfile, StringPhysicalParams, VoicingSpec, NUM_KEYS,
};

use crate::agraffe_delay::AgraffeDelayLine;
use crate::damper_pad::DamperPadBank;
use crate::felt_stulov::{FeltSharedCoeffs, FeltState, KeyFeltSpec};
use crate::keybed::KeybedBank;
use crate::oversample::{DecimationFir, HammerOversampleState};
use crate::p_grid::PGridLuts;
use crate::pedals::PedalBank;
use crate::shank_timoshenko::{velocity_from_u16, ShankParams, SILENT_KEY_MAX_U16};
use crate::ActionBank;

#[derive(Clone, Debug)]
pub(crate) struct KeyActionSlot {
    pub(crate) shank_params: ShankParams,
    pub(crate) felt_spec: KeyFeltSpec,
    pub(crate) poly_adaa: PolyPowerApprox,
    pub(crate) felt_state: FeltState,
    pub(crate) hammer: HammerOversampleState,
    pub(crate) agraffe: AgraffeDelayLine,
}

impl KeyActionSlot {
    pub(crate) fn new_with_profile(
        key_idx: usize,
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        perf: &PerfProfile,
    ) -> Self {
        let shank_params =
            ShankParams::for_key_oversampled(key_idx, sample_rate, perf.action_oversample);
        let felt_spec =
            KeyFeltSpec::for_key_oversampled(key_idx, sample_rate, voicing, perf.action_oversample);
        let poly_adaa = PolyPowerApprox::new(felt_spec.exponent_p);
        let agraffe = AgraffeDelayLine::new_oversampled(
            key_idx,
            sample_rate,
            voicing,
            specs,
            perf.action_oversample,
        );
        let mut hammer = HammerOversampleState::default();
        hammer.shank.flex_enabled = perf.shank_flex_enabled;

        Self {
            shank_params,
            felt_spec,
            poly_adaa,
            felt_state: FeltState::default(),
            hammer,
            agraffe,
        }
    }
}

impl ActionBank {
    /// Constructs and pre-tabulates the Stage 1 action bank for `sample_rate_hz`, `voicing`,
    /// and performance profile `perf`.
    ///
    /// When `perf.polynomial_adaa` is `true`, skips allocating the `384 KB` `PGridLuts` table.
    #[must_use]
    pub fn new_with_profile(
        sample_rate_hz: f32,
        voicing: &VoicingSpec,
        perf: &PerfProfile,
    ) -> Self {
        let perf = perf.sanitized();
        let sr = if sample_rate_hz.is_finite() {
            sample_rate_hz.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let sr_f64 = f64::from(sr);

        let specs = build_scale_design(voicing);
        let mut keys = Vec::with_capacity(NUM_KEYS);
        for k in 0..NUM_KEYS {
            keys.push(KeyActionSlot::new_with_profile(
                k, sr_f64, voicing, &specs, &perf,
            ));
        }

        let p_grid = if perf.polynomial_adaa {
            None
        } else {
            Some(PGridLuts::new())
        };

        Self {
            sample_rate: sr,
            memory_enabled: true,
            perf,
            felt_coeffs: FeltSharedCoeffs::new_oversampled(sr_f64, perf.action_oversample),
            p_grid,
            decimation_fir: DecimationFir::for_oversample(perf.action_oversample),
            pedals: PedalBank::new_with_decim(sr, perf.damper_control_decim),
            damper_pad: DamperPadBank::new_with_decim(sr, voicing, perf.damper_control_decim),
            keybed: KeybedBank::new(sr),
            keys: keys.into_boxed_slice(),
            hammer_voicing: voicing.sanitized(),
            scale_specs: specs,
            key_hardness_applied: [voicing.effective_global_hammer_hardness(); NUM_KEYS],
        }
    }

    /// Returns the active sanitized Stage 1 performance profile.
    #[must_use]
    #[inline]
    pub const fn perf_profile(&self) -> PerfProfile {
        self.perf
    }

    /// Applies zero-allocation runtime Stage 1 `PerfProfile` knobs (`shank_flex_enabled`,
    /// `damper_control_decim`, and `polynomial_adaa` when `PGridLuts` was pre-allocated).
    ///
    /// If initialized with `polynomial_adaa == true` (`p_grid: None`), keeps `polynomial_adaa = true`
    /// to preserve the zero-allocation guarantee.
    pub fn set_perf_profile_runtime(&mut self, perf: &PerfProfile) {
        let p = perf.sanitized();
        self.perf.shank_flex_enabled = p.shank_flex_enabled;
        for slot in self.keys.iter_mut() {
            slot.hammer.shank.flex_enabled = p.shank_flex_enabled;
        }
        self.perf.damper_control_decim = p.damper_control_decim;
        self.pedals.set_control_decim(p.damper_control_decim);
        self.damper_pad.set_control_decim(p.damper_control_decim);
        self.perf.polynomial_adaa = if self.p_grid.is_some() {
            p.polynomial_adaa
        } else {
            true
        };
    }

    /// Applies all Stage 1 `PerfProfile` settings, delegating to [`Self::reconfigure_profile`].
    pub fn set_perf_profile(&mut self, perf: &PerfProfile) {
        self.reconfigure_profile(perf);
    }

    /// Reconfigures all Stage 1 `PerfProfile` knobs including `action_oversample` and
    /// lazy `PGridLuts` allocation if switching from `polynomial_adaa`-only initialization.
    pub fn reconfigure_profile(&mut self, perf: &PerfProfile) {
        let p = perf.sanitized();
        if !p.polynomial_adaa && self.p_grid.is_none() {
            self.p_grid = Some(PGridLuts::new());
        }
        self.set_perf_profile_runtime(&p);

        if p.action_oversample != self.perf.action_oversample {
            let os = p.action_oversample;
            let sr_f64 = f64::from(self.sample_rate);
            self.felt_coeffs = FeltSharedCoeffs::new_oversampled(sr_f64, os);
            self.decimation_fir = DecimationFir::for_oversample(os);
            for (k, slot) in self.keys.iter_mut().enumerate() {
                slot.shank_params = ShankParams::for_key_oversampled(k, sr_f64, os);
                slot.felt_spec.update_voicing_in_place_oversampled(
                    k,
                    sr_f64,
                    &self.hammer_voicing,
                    os,
                );
                slot.poly_adaa = PolyPowerApprox::new(slot.felt_spec.exponent_p);
                slot.agraffe
                    .set_oversample(os, k, sr_f64, &self.hammer_voicing, &self.scale_specs);
                if usize::from(os) < slot.hammer.prev_sub_forces.len() {
                    slot.hammer.prev_sub_forces[usize::from(os)..].fill(0.0);
                }
            }
            self.perf.action_oversample = os;
        }
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
        let os = self.perf.action_oversample;
        let slot = &mut self.keys[key_idx];
        let target_h = self.hammer_voicing.hammer_hardness;
        if (self.key_hardness_applied[key_idx] - target_h).abs() > 1.0e-6 {
            slot.felt_spec.update_voicing_in_place_oversampled(
                key_idx,
                f64::from(self.sample_rate),
                &self.hammer_voicing,
                os,
            );
            slot.poly_adaa = PolyPowerApprox::new(slot.felt_spec.exponent_p);
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

    /// Updates only global hammer hardness cleanly (`< 1 us`), updating active hammers
    /// immediately and lazily syncing resting hammers on next `note_on` without recomputing dampers.
    pub fn update_hammer_hardness_in_place(&mut self, voicing: &VoicingSpec) {
        let sr = f64::from(self.sample_rate);
        let os = self.perf.action_oversample;
        let h = voicing.effective_global_hammer_hardness();
        self.hammer_voicing.hammer_hardness = h;
        for (key_idx, slot) in self.keys.iter_mut().enumerate() {
            if slot.hammer.is_active() {
                slot.felt_spec.update_voicing_in_place_oversampled(
                    key_idx,
                    sr,
                    &self.hammer_voicing,
                    os,
                );
                slot.poly_adaa = PolyPowerApprox::new(slot.felt_spec.exponent_p);
                self.key_hardness_applied[key_idx] = h;
            }
        }
    }

    /// Preloads sanitized voicing and precomputed 88-key physical parameters prior to
    /// incremental [`Self::update_voicing_key`] commits, avoiding any duplicate `build_scale_design`.
    pub fn set_voicing_scale_specs(
        &mut self,
        spec: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
    ) {
        self.hammer_voicing = spec.sanitized();
        self.scale_specs = *specs;
    }

    /// Updates felt stiffness/scaling, agraffe delay, and damper pad impedance in-place for a
    /// single key `k` from precomputed `phys` without calling `build_scale_design`.
    pub fn update_voicing_key(
        &mut self,
        k: usize,
        spec: &VoicingSpec,
        phys: &StringPhysicalParams,
    ) {
        if k >= NUM_KEYS {
            return;
        }
        let sr = f64::from(self.sample_rate);
        let os = self.perf.action_oversample;
        self.hammer_voicing = spec.sanitized();
        self.scale_specs[k] = *phys;
        self.key_hardness_applied[k] = self.hammer_voicing.hammer_hardness;
        if let Some(slot) = self.keys.get_mut(k) {
            slot.felt_spec
                .update_voicing_in_place_oversampled(k, sr, &self.hammer_voicing, os);
            slot.poly_adaa = PolyPowerApprox::new(slot.felt_spec.exponent_p);
            slot.agraffe
                .update_voicing(k, sr, &self.hammer_voicing, &self.scale_specs);
        }
        self.damper_pad.update_voicing_key(k, phys);
    }

    /// Updates felt stiffness/scaling and damper pad impedance smoothly in-place
    /// for `voicing` with zero heap allocations, zero locks, and zero panics.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        let sr = f64::from(self.sample_rate);
        let os = self.perf.action_oversample;
        self.hammer_voicing = voicing.sanitized();
        self.scale_specs = build_scale_design(&self.hammer_voicing);
        self.key_hardness_applied
            .fill(self.hammer_voicing.hammer_hardness);
        for (key_idx, slot) in self.keys.iter_mut().enumerate() {
            slot.felt_spec.update_voicing_in_place_oversampled(
                key_idx,
                sr,
                &self.hammer_voicing,
                os,
            );
            slot.poly_adaa = PolyPowerApprox::new(slot.felt_spec.exponent_p);
            slot.agraffe
                .update_voicing(key_idx, sr, &self.hammer_voicing, &self.scale_specs);
        }
        self.damper_pad
            .update_voicing_in_place_from_specs(&self.scale_specs);
    }
}
