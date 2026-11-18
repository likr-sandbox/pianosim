//! Off-thread Prep Web Worker context (`SpacePrepContext` + `VoicingPrepContext`)
//! avoiding real-time audio buffer allocations.

use piano_engine::{embedded_body_bytes, BodyArchetype, SpacePrepContext, VoicingPrepContext};
use piano_spec::{LidState, ListenerPerspective, Temperament, VoicingSpec};

#[derive(Copy, Clone, PartialEq, Eq)]
enum LastKind {
    None,
    Space,
    Voicing,
}

/// Lightweight off-thread precomputation state instantiated inside the Prep Web Worker.
pub struct PrepContext {
    space: SpacePrepContext,
    voicing: VoicingPrepContext,
    body_archetype: BodyArchetype,
    last: LastKind,
}

impl PrepContext {
    /// Initializes spatial + tuning precomputation tables for `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let sr = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        Self {
            space: SpacePrepContext::new_concert_grand(f64::from(sr)),
            voicing: VoicingPrepContext::new(sr),
            body_archetype: BodyArchetype::ConcertGrand,
            last: LastKind::None,
        }
    }

    /// Returns the currently loaded [`BodyArchetype`] in this prep context.
    #[must_use]
    #[inline]
    pub const fn body_archetype(&self) -> BodyArchetype {
        self.body_archetype
    }

    /// Precomputes Stage 4 spatial FIR + NUPOLS spectra for the active body and returns `f32` element count.
    pub fn prepare_space(&mut self, perspective: ListenerPerspective, lid: LidState) -> usize {
        let len = self.space.prepare_space(perspective, lid).len();
        self.last = LastKind::Space;
        len
    }

    /// Updates the loaded body archetype if needed and precomputes Stage 4 spatial FIR + NUPOLS spectra.
    pub fn prepare_space_for_body(
        &mut self,
        body: BodyArchetype,
        perspective: ListenerPerspective,
        lid: LidState,
    ) -> usize {
        if self.body_archetype != body {
            if self
                .space
                .set_body_preset_bytes(embedded_body_bytes(body))
                .is_err()
            {
                return 0;
            }
            self.body_archetype = body;
        }
        self.prepare_space(perspective, lid)
    }

    /// Precomputes Stage 2/3 tuning tables and returns `f32` element count (`Equal`, `condition = 0.0`).
    pub fn prepare_voicing(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        a4_freq_hz: f32,
    ) -> usize {
        self.prepare_voicing_full(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            a4_freq_hz,
            Temperament::Equal,
            0.0,
        )
    }

    /// Precomputes Stage 2/3 tuning tables including concert pitch, temperament, and aging `condition`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_voicing_full(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        concert_pitch_hz: f32,
        temperament: Temperament,
        condition: f32,
    ) -> usize {
        let spec = VoicingSpec::from_globals_full(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            concert_pitch_hz,
            temperament,
            condition,
        );
        let len = self.voicing.prepare_voicing(&spec).len();
        self.last = LastKind::Voicing;
        len
    }

    /// Pointer to the most recently prepared blob's `f32` buffer in Wasm linear memory.
    #[must_use]
    pub fn out_ptr(&self) -> *const f32 {
        match self.last {
            LastKind::Space => self.space.out_blob().as_ptr(),
            LastKind::Voicing => self.voicing.out_blob().as_ptr(),
            LastKind::None => core::ptr::null(),
        }
    }
}
