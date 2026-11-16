//! Linear-memory zero-allocation IPC context for `AudioWorkletGlobalScope`.

use piano_engine::{space_blob_len_for_sample_rate, voicing_blob_len, EventQueue, PianoEngine};
use piano_spec::{EventKind, LidState, ListenerPerspective, PedalKind, SampleEvent, VoicingSpec};

/// Maximum audio frames rendered per `piano_render` call.
pub const MAX_BLOCK: usize = 512;
/// Number of `f32` scalars in the shared telemetry buffer.
pub const TELEMETRY_LEN: usize = 8;

#[repr(C, align(16))]
struct AlignedBlock([f32; MAX_BLOCK]);

#[repr(C, align(16))]
struct AlignedTelemetry([f32; TELEMETRY_LEN]);

/// Self-contained Wasm audio-thread context allocated once at `piano_init`.
pub struct WasmContext {
    engine: PianoEngine,
    events: EventQueue,
    out_l: AlignedBlock,
    out_r: AlignedBlock,
    telemetry: AlignedTelemetry,
    staging: Vec<f32>,
}

impl WasmContext {
    /// Initializes the piano engine and pre-allocates fixed audio/telemetry/staging buffers.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let engine = PianoEngine::new(sample_rate);
        let sr_f64 = f64::from(engine.sample_rate());
        let cap =
            space_blob_len_for_sample_rate(sr_f64).max(voicing_blob_len(engine.sample_rate()));
        Self {
            engine,
            events: EventQueue::new(),
            out_l: AlignedBlock([0.0; MAX_BLOCK]),
            out_r: AlignedBlock([0.0; MAX_BLOCK]),
            telemetry: AlignedTelemetry([0.0; TELEMETRY_LEN]),
            staging: vec![0.0; cap],
        }
    }

    /// Enqueues a sample-accurate note-on event without allocating.
    #[inline]
    pub fn note_on(&mut self, midi_key: u8, velocity_u16: u16, sample_offset: u32) {
        let _ = self.events.push(SampleEvent {
            sample_offset,
            kind: EventKind::NoteOn {
                midi_key,
                velocity_u16,
            },
        });
    }

    /// Enqueues a sample-accurate note-off event without allocating.
    #[inline]
    pub fn note_off(&mut self, midi_key: u8, release_vel_u16: u16, sample_offset: u32) {
        let _ = self.events.push(SampleEvent {
            sample_offset,
            kind: EventKind::NoteOff {
                midi_key,
                release_vel_u16,
            },
        });
    }

    /// Enqueues a sample-accurate pedal event (`0`=Sustain, `1`=UnaCorda, `2`=Sostenuto).
    #[inline]
    pub fn pedal(&mut self, pedal_kind: u8, value_u16: u16, sample_offset: u32) {
        if let Some(kind) = PedalKind::from_u8(pedal_kind) {
            let _ = self.events.push(SampleEvent {
                sample_offset,
                kind: EventKind::Pedal { kind, value_u16 },
            });
        }
    }

    /// Enqueues an all-notes-off panic event without allocating.
    #[inline]
    pub fn all_notes_off(&mut self) {
        let ev = SampleEvent {
            sample_offset: 0,
            kind: EventKind::AllNotesOff,
        };
        if !self.events.push(ev) {
            self.events.clear();
            let _ = self.events.push(ev);
        }
    }

    /// Updates voicing targets for smooth real-time parameter interpolation.
    #[inline]
    pub fn set_voicing(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
    ) {
        self.engine
            .set_voicing(hammer_hardness, unison_detune, stretch, length_scale);
    }

    /// Reconfigures spatial listening perspective and lid opening state.
    #[inline]
    pub fn set_space(&mut self, perspective: ListenerPerspective, lid: LidState) {
        self.engine.set_space(perspective, lid);
    }

    /// Updates only hammer hardness smoothly (`< 50 us`), avoiding string table rebuild.
    #[inline]
    pub fn set_hammer_hardness(&mut self, hammer_hardness: f32) {
        self.engine.set_hammer_hardness(hammer_hardness);
    }

    /// Returns mutable pointer to pre-allocated staging buffer if `len_f32` fits.
    #[inline]
    pub fn staging_mut_ptr(&mut self, len_f32: usize) -> *mut f32 {
        if len_f32 == 0 || len_f32 > self.staging.len() {
            return core::ptr::null_mut();
        }
        self.staging.as_mut_ptr()
    }

    /// Applies an off-thread precomputed spatial FIR + NUPOLS blob from staging memory.
    pub fn apply_space_blob(
        &mut self,
        perspective: ListenerPerspective,
        lid: LidState,
        len_f32: usize,
    ) -> bool {
        if len_f32 == 0 || len_f32 > self.staging.len() {
            return false;
        }
        let blob = &self.staging[..len_f32];
        self.engine.apply_precomputed_space(perspective, lid, blob)
    }

    /// Applies an off-thread precomputed Stage 2/3 tuning table blob from staging memory.
    pub fn apply_voicing_blob(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        len_f32: usize,
    ) -> bool {
        if len_f32 == 0 || len_f32 > self.staging.len() {
            return false;
        }
        let spec = VoicingSpec::from_globals(hammer_hardness, unison_detune, stretch, length_scale);
        self.engine
            .apply_precomputed_voicing(spec, &self.staging[..len_f32])
    }

    /// Renders up to `MAX_BLOCK` frames into the internal L/R buffers, updates
    /// telemetry, and clears queued events. Zero heap allocations.
    pub fn render(&mut self, frames: usize) {
        let n = frames.min(MAX_BLOCK);
        self.engine.render_block(
            self.events.as_slice(),
            &mut self.out_l.0[..n],
            &mut self.out_r.0[..n],
        );
        self.telemetry.0 = self.engine.telemetry().to_array();
        self.events.clear();
    }

    /// Returns a pointer to the left channel output buffer (`MAX_BLOCK` floats).
    #[must_use]
    #[inline]
    pub fn out_l_ptr(&self) -> *const f32 {
        self.out_l.0.as_ptr()
    }

    /// Returns a pointer to the right channel output buffer (`MAX_BLOCK` floats).
    #[must_use]
    #[inline]
    pub fn out_r_ptr(&self) -> *const f32 {
        self.out_r.0.as_ptr()
    }

    /// Returns a pointer to the 8-float telemetry array.
    #[must_use]
    #[inline]
    pub fn telemetry_ptr(&self) -> *const f32 {
        self.telemetry.0.as_ptr()
    }
}
