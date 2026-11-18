//! Multi-body preset synchronization, reverb control, and spatial transition helpers.

use piano_radiator::{embedded_body_bytes, BodyArchetype};
use piano_spec::{BodyPreset, PresetReducedPole, PRESET_BRIDGE_ZONES, PRESET_REDUCED_POLES};

use crate::engine::PianoEngine;
use crate::param_smooth::{IntoLidState, IntoListenerPerspective, SpaceTransitionAction};

/// Default musical hall reverb decay time in seconds for [`PianoEngine::new`].
pub const DEFAULT_REVERB_DECAY_S: f32 = 1.6;
/// Default musical hall reverb wet mix `[0.0..=1.0]` for [`PianoEngine::new`].
pub const DEFAULT_REVERB_MIX: f32 = 0.18;

const SPACE_BLOB_MAGIC: u32 = 0x5042_5350;

/// Trait for converting strongly typed or numeric body discriminants into [`BodyArchetype`].
pub trait IntoBodyArchetype {
    fn into_body_archetype(self) -> Option<BodyArchetype>;
}

impl IntoBodyArchetype for BodyArchetype {
    #[inline]
    fn into_body_archetype(self) -> Option<BodyArchetype> {
        Some(self)
    }
}

impl IntoBodyArchetype for u8 {
    #[inline]
    fn into_body_archetype(self) -> Option<BodyArchetype> {
        BodyArchetype::from_u8(self)
    }
}

impl IntoBodyArchetype for u32 {
    #[inline]
    fn into_body_archetype(self) -> Option<BodyArchetype> {
        u8::try_from(self).ok().and_then(BodyArchetype::from_u8)
    }
}

impl IntoBodyArchetype for i32 {
    #[inline]
    fn into_body_archetype(self) -> Option<BodyArchetype> {
        u8::try_from(self).ok().and_then(BodyArchetype::from_u8)
    }
}

#[inline]
const fn body_index(archetype: BodyArchetype) -> usize {
    match archetype {
        BodyArchetype::ConcertGrand => 0,
        BodyArchetype::UprightStudio => 1,
        BodyArchetype::VintageSaloon => 2,
    }
}

/// Pre-decoded Stage 3 `(y_inf, reduced_poles)` cache for all 3 baked body archetypes
/// so real-time audio-thread body switching incurs zero heap allocations.
#[derive(Clone, Debug)]
pub(crate) struct BodyModelCache {
    models: [(
        [f32; PRESET_BRIDGE_ZONES],
        [PresetReducedPole; PRESET_REDUCED_POLES],
    ); 3],
}

impl BodyModelCache {
    #[must_use]
    pub fn new() -> Self {
        let decode_model = |arch: BodyArchetype| {
            let preset = BodyPreset::decode(embedded_body_bytes(arch))
                .expect("embedded .pbdy asset must decode");
            (preset.y_inf, preset.reduced_poles)
        };
        Self {
            models: [
                decode_model(BodyArchetype::ConcertGrand),
                decode_model(BodyArchetype::UprightStudio),
                decode_model(BodyArchetype::VintageSaloon),
            ],
        }
    }

    #[inline]
    #[must_use]
    pub fn reduced_model(
        &self,
        archetype: BodyArchetype,
    ) -> &(
        [f32; PRESET_BRIDGE_ZONES],
        [PresetReducedPole; PRESET_REDUCED_POLES],
    ) {
        &self.models[body_index(archetype)]
    }
}

impl PianoEngine {
    /// Sets hall reverb decay `[0.0..=3.2 s]` and wet mix `[0.0..=1.0]` across both Stage 4
    /// radiator banks with zero allocations.
    pub fn set_reverb(&mut self, decay_s: f32, mix: f32) {
        self.core.set_reverb(decay_s, mix);
    }

    /// Returns the active `(reverb_decay_s, reverb_mix)` parameters.
    #[must_use]
    #[inline]
    pub const fn reverb_params(&self) -> (f32, f32) {
        self.core.reverb_params()
    }

    /// Returns the active physical piano [`BodyArchetype`].
    #[must_use]
    #[inline]
    pub const fn body_archetype(&self) -> BodyArchetype {
        self.core.body_archetype
    }

    /// Switches the active piano body archetype while keeping the current perspective and lid,
    /// synchronizing Stage 3 bridge admittance and Stage 4 radiator IRs with 8 ms crossfade.
    pub fn set_body_archetype(&mut self, archetype: impl IntoBodyArchetype) {
        let persp = self.space_guard.current_perspective();
        let lid = self.space_guard.current_lid();
        let _ = self.set_body_space(archetype, persp, lid);
    }

    /// Synchronously switches `(body_archetype, perspective, lid)` across Stage 3 and Stage 4.
    pub fn set_body_space(
        &mut self,
        archetype: impl IntoBodyArchetype,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
    ) -> bool {
        let Some(body) = archetype.into_body_archetype() else {
            return false;
        };
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        let action =
            self.space_guard
                .request_body_space(body, persp, lid_state, self.sample_rate());
        match action {
            SpaceTransitionAction::None => {}
            SpaceTransitionAction::InstantSwitch {
                target_idx,
                perspective,
                lid,
            } => {
                self.reconfigure_radiator_for_body(target_idx, body, perspective, lid);
                self.core.sync_to_body_if_changed(body);
            }
            SpaceTransitionAction::StartCrossfade {
                fade_in_idx,
                perspective,
                lid,
                ..
            } => {
                self.reconfigure_radiator_for_body(fade_in_idx, body, perspective, lid);
                self.core.sync_to_body_if_changed(body);
            }
        }
        true
    }

    fn reconfigure_radiator_for_body(
        &mut self,
        idx: usize,
        body: BodyArchetype,
        perspective: piano_spec::ListenerPerspective,
        lid: piano_spec::LidState,
    ) {
        if self.core.radiator_bodies[idx] != Some(body) {
            let _ = self.core.radiators[idx].reconfigure_from_preset_bytes(
                embedded_body_bytes(body),
                perspective,
                lid,
            );
            self.core.radiator_bodies[idx] = Some(body);
        } else {
            self.core.radiators[idx].reconfigure_space(perspective, lid);
        }
    }

    /// Installs an off-thread precomputed Stage 4 FIR/NUPOLS blob for `(body, perspective, lid)`,
    /// synchronizing Stage 3 bridge admittance without heap allocations.
    pub fn apply_precomputed_body_space(
        &mut self,
        archetype: impl IntoBodyArchetype,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
        blob: &[f32],
    ) -> bool {
        let Some(body) = archetype.into_body_archetype() else {
            return false;
        };
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        if !self.space_guard.is_crossfading()
            && body == self.space_guard.current_body()
            && persp == self.space_guard.current_perspective()
            && lid_state == self.space_guard.current_lid()
        {
            let expected =
                piano_radiator::space_blob_len_for_sample_rate(f64::from(self.sample_rate()));
            return blob.len() == expected && blob[0].to_bits() == SPACE_BLOB_MAGIC;
        }
        let target_idx = self.space_guard.target_radiator_idx();
        if !self.core.radiators[target_idx].install_precomputed_space(persp, lid_state, blob) {
            return false;
        }
        if self.core.radiator_bodies[target_idx] != Some(body) {
            self.core.radiator_bodies[target_idx] = None;
        }
        let _ = self
            .space_guard
            .request_body_space(body, persp, lid_state, self.sample_rate());
        self.core.sync_to_body_if_changed(body);
        true
    }

    /// Alias for [`Self::apply_precomputed_body_space`] committing a precomputed body/space blob.
    pub fn commit_body_space_blob(
        &mut self,
        archetype: impl IntoBodyArchetype,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
        blob: &[f32],
    ) -> bool {
        self.apply_precomputed_body_space(archetype, perspective, lid, blob)
    }

    /// Streams a precomputed space blob chunk for `(body, perspective, lid)` into standby.
    pub fn stage_precomputed_body_space_chunk(
        &mut self,
        archetype: impl IntoBodyArchetype,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
        blob_off: usize,
        chunk: &[f32],
        total_len: usize,
    ) -> bool {
        if self.space_guard.is_crossfading() {
            return false;
        }
        let Some(body) = archetype.into_body_archetype() else {
            return false;
        };
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        let standby = self.space_guard.standby_radiator_idx();
        if blob_off == 0 {
            self.core.radiators[standby].reset();
            self.core.staged_standby_body = None;
            if self.core.radiator_bodies[standby] != Some(body) {
                self.core.radiator_bodies[standby] = None;
            }
        }
        let ok = self.core.radiators[standby]
            .install_precomputed_space_chunk(persp, lid_state, blob_off, chunk, total_len);
        if !ok {
            self.core.staged_standby_body = None;
            return false;
        }
        if blob_off.saturating_add(chunk.len()) == total_len {
            self.core.staged_standby_body = Some(body);
        }
        true
    }

    /// Activates the pre-staged standby radiator bank for `(body, perspective, lid)`
    /// via instant swap (silent) or 8 ms crossfade, syncing Stage 3 bridge admittance zero-alloc.
    pub fn commit_precomputed_body_space(
        &mut self,
        archetype: impl IntoBodyArchetype,
        perspective: impl IntoListenerPerspective,
        lid: impl IntoLidState,
    ) -> bool {
        if self.space_guard.is_crossfading() {
            return false;
        }
        let Some(body) = archetype.into_body_archetype() else {
            return false;
        };
        let Some(persp) = perspective.into_perspective() else {
            return false;
        };
        let Some(lid_state) = lid.into_lid_state() else {
            return false;
        };
        if self.core.staged_standby_body != Some(body) {
            return false;
        }
        let standby = self.space_guard.standby_radiator_idx();
        let rad = &self.core.radiators[standby];
        if rad.perspective() != persp || rad.lid_state() != lid_state {
            return false;
        }
        let _ = self
            .space_guard
            .commit_body_standby(body, persp, lid_state, self.sample_rate());
        self.core.sync_to_body_if_changed(body);
        self.core.staged_standby_body = None;
        true
    }
}
