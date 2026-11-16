//! Real-time 88-key physical piano engine facade.

#[cfg(test)]
use piano_spec::NUM_KEYS;
use piano_spec::{LidState, ListenerPerspective, SampleEvent, VoicingSpec};

use crate::engine_core::{AudioTap, EngineCore};
use crate::event_queue::{is_sorted_by_offset, EventQueue};
use crate::param_smooth::{
    IntoLidState, IntoListenerPerspective, SpaceCrossfadeGuard, SpaceTransitionAction,
    VoicingSmoother, DIRTY_TUNING,
};
use crate::telemetry::{compute_block_telemetry, EngineTelemetry};

/// Zero-allocation 88-key polyphonic piano engine coupling Stage 1 action,
/// Stage 2 strings, Stage 3 bridge, and Stage 4 spatial acoustic radiator.
#[derive(Clone, Debug)]
pub struct PianoEngine {
    core: EngineCore,
    scratch_events: EventQueue,
    telemetry: EngineTelemetry,
    voicing_smoother: VoicingSmoother,
    space_guard: SpaceCrossfadeGuard,
}

impl PianoEngine {
    /// Allocates and precomputes all 88-key action, string, bridge, and radiator tables with default voicing.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_voicing(sample_rate, &VoicingSpec::default())
    }

    /// Allocates and precomputes all 88-key action, string, bridge, and radiator tables for `voicing`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let core = EngineCore::new(sample_rate, voicing);
        let voicing_smoother = VoicingSmoother::new_with_voicing(voicing);
        let space_guard = SpaceCrossfadeGuard::new(ListenerPerspective::Player, LidState::Open);
        Self {
            core,
            scratch_events: EventQueue::new(),
            telemetry: EngineTelemetry::default(),
            voicing_smoother,
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
        self.telemetry = EngineTelemetry::default();
    }

    /// Sets target voicing parameters with 20 ms exponential smoothing.
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
        let v = spec.sanitized();
        if !self.core.bank.install_precomputed_voicing(&v, blob) {
            return false;
        }
        self.core
            .bridge
            .set_port_resistances(self.core.bank.bridge_port_resistances());
        self.core.action.update_voicing_in_place(&v);
        self.voicing_smoother.set_tuning_applied(
            v.unison_detune_scale,
            v.stretch_amount,
            v.string_length_scale,
        );
        true
    }

    /// Installs an off-thread precomputed Stage 4 FIR/NUPOLS blob and starts crossfade/switch.
    pub fn apply_precomputed_space(
        &mut self,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
        blob: &[f32],
    ) -> bool {
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        let action = self
            .space_guard
            .request_space(persp, lid_state, self.sample_rate());
        match action {
            SpaceTransitionAction::None => true,
            SpaceTransitionAction::InstantSwitch {
                target_idx,
                perspective,
                lid,
            } => self.core.radiators[target_idx].install_precomputed_space(perspective, lid, blob),
            SpaceTransitionAction::StartCrossfade {
                fade_in_idx,
                perspective,
                lid,
                ..
            } => self.core.radiators[fade_in_idx].install_precomputed_space(perspective, lid, blob),
        }
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
        if self.space_guard.is_crossfading() {
            return false;
        }
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        let standby = self.space_guard.standby_radiator_idx();
        if blob_off == 0 {
            self.core.radiators[standby].reset();
        }
        self.core.radiators[standby]
            .install_precomputed_space_chunk(persp, lid_state, blob_off, chunk, total_len)
    }

    /// Activates the pre-staged standby radiator bank via instant swap (silent) or 8 ms crossfade.
    pub fn commit_precomputed_space(
        &mut self,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
    ) -> bool {
        if self.space_guard.is_crossfading() {
            return false;
        }
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        let standby = self.space_guard.standby_radiator_idx();
        let rad = &self.core.radiators[standby];
        if rad.perspective() != persp || rad.lid_state() != lid_state {
            return false;
        }
        let _ = self
            .space_guard
            .commit_standby(persp, lid_state, self.sample_rate());
        true
    }

    /// Sets listener perspective and lid opening state with 8 ms raised-cosine crossfade.
    pub fn set_space(&mut self, perspective: impl IntoListenerPerspective, lid: impl IntoLidState) {
        let Some(persp) = perspective.into_perspective() else {
            return;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return;
        };
        let action = self
            .space_guard
            .request_space(persp, lid_state, self.sample_rate());
        match action {
            SpaceTransitionAction::None => {}
            SpaceTransitionAction::InstantSwitch {
                target_idx,
                perspective,
                lid,
            } => {
                self.core.radiators[target_idx].reconfigure_space(perspective, lid);
            }
            SpaceTransitionAction::StartCrossfade {
                fade_in_idx,
                perspective,
                lid,
                ..
            } => {
                self.core.radiators[fade_in_idx].reconfigure_space(perspective, lid);
            }
        }
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
        (0..NUM_KEYS).any(|k| self.core.action.is_hammer_active(k))
    }

    #[cfg(test)]
    pub(crate) fn set_duplex_gain(&mut self, gain: f32) {
        self.core.bridge.set_duplex_gain(gain);
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

        if let Some((spec, dirty)) = self
            .voicing_smoother
            .step_block_with_dirty(frames, self.sample_rate())
        {
            if (dirty & DIRTY_TUNING) != 0 {
                self.core.action.update_voicing_in_place(&spec);
                self.core.bank.update_voicing_in_place(&spec);
                self.core
                    .bridge
                    .set_port_resistances(self.core.bank.bridge_port_resistances());
            } else {
                self.core.action.update_hammer_hardness_in_place(&spec);
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

        self.core.retire_silent_keys();
        self.telemetry = compute_block_telemetry(self.core.active_voice_count(), out_l, out_r);

        let peak = out_l
            .iter()
            .chain(out_r.iter())
            .fold(0.0_f32, |m, &x| m.max(x.abs()));
        self.space_guard.update_envelope(peak);
    }
}
