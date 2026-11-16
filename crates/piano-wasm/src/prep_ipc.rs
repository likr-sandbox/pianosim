//! Off-thread Prep Web Worker context (`SpacePrepContext` + `VoicingPrepContext`)
//! avoiding real-time audio buffer allocations.

use piano_engine::{SpacePrepContext, VoicingPrepContext};
use piano_spec::{LidState, ListenerPerspective, VoicingSpec};

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
            last: LastKind::None,
        }
    }

    /// Precomputes Stage 4 spatial FIR + NUPOLS spectra and returns `f32` element count.
    pub fn prepare_space(&mut self, perspective: ListenerPerspective, lid: LidState) -> usize {
        let len = self.space.prepare_space(perspective, lid).len();
        self.last = LastKind::Space;
        len
    }

    /// Precomputes Stage 2/3 tuning tables and returns `f32` element count.
    pub fn prepare_voicing(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
    ) -> usize {
        let spec = VoicingSpec::from_globals(hammer_hardness, unison_detune, stretch, length_scale);
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
