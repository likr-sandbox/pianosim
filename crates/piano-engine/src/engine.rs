//! Real-time 88-key physical piano engine facade.

use piano_spec::{
    LidState, ListenerPerspective, PerfProfile, SampleEvent, Temperament, VoicingSpec,
};

use crate::engine_core::{AudioTap, EngineCore};
use crate::event_queue::{is_sorted_by_offset, EventQueue};
use crate::param_smooth::{
    IntoLidState, IntoListenerPerspective, SpaceCrossfadeGuard, VoicingSmoother, DIRTY_TUNING,
};
use crate::telemetry::{compute_block_telemetry, EngineTelemetry};
use crate::voicing_commit::VoicingCommitState;

/// Zero-allocation 88-key polyphonic piano engine coupling Stage 1 action,
/// Stage 2 strings, Stage 3 bridge, and Stage 4 spatial acoustic radiator.
#[derive(Clone, Debug)]
pub struct PianoEngine {
    pub(crate) core: EngineCore,
    scratch_events: EventQueue,
    telemetry: EngineTelemetry,
    pub(crate) voicing_smoother: VoicingSmoother,
    pub(crate) voicing_commit: VoicingCommitState,
    pub(crate) space_guard: SpaceCrossfadeGuard,
}

impl PianoEngine {
    /// Allocates and precomputes all 88-key action, string, bridge, and radiator tables with default voicing.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_profile(sample_rate, VoicingSpec::default(), PerfProfile::full())
    }

    /// Allocates and precomputes all 88-key action, string, bridge, and radiator tables for `voicing`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::new_with_profile(sample_rate, voicing.clone(), PerfProfile::full())
    }

    /// Allocates and precomputes all 88-key action, string, bridge, and radiator tables
    /// for `voicing` and performance profile `profile`.
    #[must_use]
    pub fn new_with_profile(
        sample_rate_hz: f32,
        voicing: VoicingSpec,
        profile: PerfProfile,
    ) -> Self {
        let core = EngineCore::new_with_profile(sample_rate_hz, &voicing, profile);
        let voicing_smoother = VoicingSmoother::new_with_voicing(&voicing);
        let voicing_commit = VoicingCommitState::new(core.bank.sample_rate());
        let space_guard = SpaceCrossfadeGuard::new(ListenerPerspective::Player, LidState::Open);
        Self {
            core,
            scratch_events: EventQueue::new(),
            telemetry: EngineTelemetry::default(),
            voicing_smoother,
            voicing_commit,
            space_guard,
        }
    }

    /// Returns the configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.core.bank.sample_rate()
    }

    /// Returns total SIMD4-padded modal slots driven on every sample (`~26,184`).
    #[must_use]
    #[inline]
    pub const fn total_modes(&self) -> usize {
        self.core.bank.total_modes()
    }

    /// Returns the telemetry captured during the most recent `render_block` call.
    #[must_use]
    #[inline]
    pub const fn telemetry(&self) -> EngineTelemetry {
        self.telemetry
    }

    /// Returns the count of currently active ringing keys.
    #[must_use]
    pub fn active_voices(&self) -> u32 {
        self.core.active_voice_count()
    }

    /// Resets all 88 keys, pedals, hammers, dampers, and modal buffers without reallocating.
    pub fn reset(&mut self) {
        self.core.all_notes_off();
        self.space_guard.on_silence_reset();
        self.space_guard.clear_standby_delay_dirty();
        self.telemetry = EngineTelemetry::default();
    }

    /// Sets target voicing parameters (`A4` unchanged) with 20 ms exponential smoothing.
    pub fn set_voicing(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
    ) {
        self.voicing_smoother
            .set_target(hammer_hardness, unison_detune, stretch, length_scale);
    }

    /// Sets target voicing parameters including `A4` concert pitch (`[392.0, 466.2]` Hz) with 20 ms smoothing.
    pub fn set_voicing_with_a4(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        a4_freq_hz: f32,
    ) {
        self.voicing_smoother.set_target_with_a4(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            a4_freq_hz,
        );
    }

    /// Sets all global voicing parameters including `concert_pitch_hz` (`[392.0, 466.2]` Hz),
    /// classical `temperament`, and deterministic aging `condition` (`[0.0, 1.0]`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_voicing_full(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        concert_pitch_hz: f32,
        temperament: Temperament,
        condition: f32,
    ) {
        self.voicing_smoother.set_target_full(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            concert_pitch_hz,
            temperament,
            condition,
        );
    }

    /// Sets target `A4` concert pitch (`[392.0, 466.2]` Hz) with 20 ms exponential smoothing.
    pub fn set_a4_freq_hz(&mut self, a4_freq_hz: f32) {
        self.voicing_smoother.set_a4_target(a4_freq_hz);
    }

    /// Alias for [`Self::set_a4_freq_hz`] (`[392.0, 466.2]` Hz).
    pub fn set_concert_pitch_hz(&mut self, concert_pitch_hz: f32) {
        self.voicing_smoother
            .set_concert_pitch_target(concert_pitch_hz);
    }

    /// Sets the active classical 12-tone [`Temperament`] for Stage 2/3 tuning.
    pub fn set_temperament(&mut self, temperament: Temperament) {
        self.voicing_smoother.set_temperament(temperament);
    }

    /// Sets target deterministic aging `condition` (`[0.0, 1.0]`) with 20 ms exponential smoothing.
    pub fn set_condition(&mut self, condition: f32) {
        self.voicing_smoother.set_condition_target(condition);
    }

    /// Updates only hammer hardness smoothly (`< 50 us`), leaving string tuning untouched.
    pub fn set_hammer_hardness(&mut self, hammer_hardness: f32) {
        self.voicing_smoother.set_target(
            hammer_hardness,
            self.voicing_smoother.current_unison_detune(),
            self.voicing_smoother.current_stretch(),
            self.voicing_smoother.current_length_scale(),
        );
    }

    /// Installs an off-thread precomputed Stage 2/3 tuning table blob with zero allocations.
    pub fn apply_precomputed_voicing(&mut self, spec: VoicingSpec, blob: &[f32]) -> bool {
        if !self.begin_voicing_commit(spec, blob) {
            return false;
        }
        while self.is_voicing_applying() {
            self.step_voicing_commit();
        }
        true
    }

    /// Installs an off-thread precomputed Stage 4 FIR/NUPOLS blob and starts crossfade/switch.
    pub fn apply_precomputed_space(
        &mut self,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
        blob: &[f32],
    ) -> bool {
        self.apply_precomputed_body_space(self.core.body_archetype, perspective, lid, blob)
    }

    /// Streams a chunk `[blob_off..blob_off + chunk.len())` into the standby Stage 4 radiator.
    pub fn stage_precomputed_space_chunk(
        &mut self,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
        blob_off: usize,
        chunk: &[f32],
        total_len: usize,
    ) -> bool {
        self.stage_precomputed_body_space_chunk(
            self.core.body_archetype,
            perspective,
            lid,
            blob_off,
            chunk,
            total_len,
        )
    }

    /// Activates the pre-staged standby radiator bank via instant swap (silent) or 8 ms crossfade.
    pub fn commit_precomputed_space(
        &mut self,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
    ) -> bool {
        self.commit_precomputed_body_space(self.core.body_archetype, perspective, lid)
    }

    /// Sets listener perspective and lid opening state with 8 ms raised-cosine crossfade.
    pub fn set_space(&mut self, perspective: impl IntoListenerPerspective, lid: impl IntoLidState) {
        let _ = self.set_body_space(self.core.body_archetype, perspective, lid);
    }

    /// Returns reference to voicing smoother.
    #[must_use]
    #[inline]
    pub const fn voicing_smoother(&self) -> &VoicingSmoother {
        &self.voicing_smoother
    }

    /// Returns reference to spatial crossfade guard.
    #[must_use]
    #[inline]
    pub const fn space_guard(&self) -> &SpaceCrossfadeGuard {
        &self.space_guard
    }

    /// Diagnostic toggle for Stulov wool hysteresis memory.
    pub fn set_felt_memory_enabled(&mut self, enabled: bool) {
        self.core.action.set_memory_enabled(enabled);
    }

    /// Diagnostic toggle for longitudinal precursor and SAV coupling.
    pub fn set_nonlinear_coupling(
        &mut self,
        longitudinal_enabled: bool,
        sav_phantom_enabled: bool,
    ) {
        self.core
            .bank
            .set_nonlinear_coupling(longitudinal_enabled, sav_phantom_enabled);
    }

    #[cfg(test)]
    pub(crate) fn total_sav_energy_joules(&self) -> f64 {
        self.core.bank.total_sav_energy_joules()
            + self
                .core
                .bank
                .bridge_link_energy_joules(self.core.bridge.port_resistances())
            + self.core.bridge.total_energy_joules()
    }

    #[cfg(test)]
    pub(crate) fn string_linear_energy_joules(&self, s_idx: usize) -> f64 {
        self.core.bank.string_linear_energy_joules(s_idx)
    }

    #[cfg(test)]
    pub(crate) fn key_sav_energy_joules(&self, key_idx: usize) -> f64 {
        self.core.bank.key_sav_energy_joules(key_idx)
    }

    #[cfg(test)]
    pub(crate) fn key_nonlinear_energy_joules(&self, key_idx: usize) -> f64 {
        self.core.bank.key_nonlinear_energy_joules(key_idx)
    }

    #[cfg(test)]
    pub(crate) fn is_any_hammer_active(&self) -> bool {
        (0..piano_spec::NUM_KEYS).any(|k| self.core.action.is_hammer_active(k))
    }

    #[cfg(test)]
    pub(crate) fn set_duplex_gain(&mut self, gain: f32) {
        self.core.bridge.set_duplex_gain(gain);
    }

    #[cfg(test)]
    pub(crate) fn is_bridge_quiescent(&self) -> bool {
        self.core.bridge.is_quiescent()
    }

    #[cfg(test)]
    pub(crate) fn excite_bridge_subnormal_for_test(&mut self) {
        let mut a = [0.0_f32; piano_spec::TOTAL_STRINGS];
        a[140] = 1.0e-14;
        let mut b = [0.0_f32; piano_spec::TOTAL_STRINGS];
        let mut z = [0.0_f32; piano_bridge::NUM_BRIDGE_ZONES];
        let _ = self.core.bridge.step_bridge_sample(
            &a,
            &[0.0; piano_bridge::NUM_BRIDGE_ZONES],
            &mut b,
            &mut z,
        );
    }

    /// Processes sample-accurate `events` and renders Stage 4 acoustic sound into `out_l`/`out_r`
    /// with zero heap allocations.
    pub fn render_block(&mut self, events: &[SampleEvent], out_l: &mut [f32], out_r: &mut [f32]) {
        self.render_block_with_tap(events, out_l, out_r, AudioTap::AcousticRadiator);
    }

    /// Processes sample-accurate `events` and renders Stage 3 direct bridge velocity into `out_l`/`out_r`
    /// with zero heap allocations for stage diagnostics and legacy tests.
    pub fn render_block_bridge(
        &mut self,
        events: &[SampleEvent],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        self.render_block_with_tap(events, out_l, out_r, AudioTap::BridgeDirect);
    }

    fn render_block_with_tap(
        &mut self,
        events: &[SampleEvent],
        out_l: &mut [f32],
        out_r: &mut [f32],
        tap: AudioTap,
    ) {
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }

        let out_l = &mut out_l[..frames];
        let out_r = &mut out_r[..frames];

        let _ = self.step_voicing_commit();

        if let Some((spec, dirty)) = self.voicing_smoother.step_block_with_dirty_ex(
            frames,
            self.sample_rate(),
            !self.voicing_commit.active,
        ) {
            if (dirty & DIRTY_TUNING) != 0 && !self.voicing_commit.active {
                self.core.current_voicing.hammer_hardness = spec.hammer_hardness;
                self.core.current_voicing.unison_detune_scale = spec.unison_detune_scale;
                self.core.current_voicing.stretch_amount = spec.stretch_amount;
                self.core.current_voicing.string_length_scale = spec.string_length_scale;
                self.core.current_voicing.a4_freq_hz = spec.a4_freq_hz;
                self.core.current_voicing.temperament = spec.temperament;
                self.core.current_voicing.condition = spec.condition;
                let applied_spec = self.core.current_voicing.clone();
                self.core.action.update_voicing_in_place(&applied_spec);
                self.core.bank.update_voicing_in_place(&applied_spec);
                self.core.bridge.set_voicing(&applied_spec);
                self.core
                    .bridge
                    .set_port_resistances(self.core.bank.bridge_port_resistances());
            } else {
                self.core.current_voicing.hammer_hardness = spec.hammer_hardness;
                self.core
                    .action
                    .update_hammer_hardness_in_place(&self.core.current_voicing);
            }
        }

        if is_sorted_by_offset(events) {
            self.core
                .render_event_slice(events, out_l, out_r, &mut self.space_guard, tap);
        } else {
            self.scratch_events.copy_sorted_from(events);
            self.core.render_event_slice(
                self.scratch_events.as_slice(),
                out_l,
                out_r,
                &mut self.space_guard,
                tap,
            );
        }

        self.telemetry = compute_block_telemetry(self.core.active_voice_count(), out_l, out_r);

        let peak = out_l
            .iter()
            .chain(out_r.iter())
            .fold(0.0_f32, |m, &x| m.max(x.abs()));
        self.space_guard.update_envelope(peak);
    }
}
