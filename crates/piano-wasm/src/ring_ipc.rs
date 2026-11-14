//! Linear-memory zero-allocation IPC context for `AudioWorkletGlobalScope`.

use piano_engine::{EventQueue, PianoEngine};
use piano_spec::{EventKind, PedalKind, SampleEvent};

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
}

impl WasmContext {
    /// Initializes the piano engine and pre-allocates fixed audio/telemetry buffers.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            engine: PianoEngine::new(sample_rate),
            events: EventQueue::new(),
            out_l: AlignedBlock([0.0; MAX_BLOCK]),
            out_r: AlignedBlock([0.0; MAX_BLOCK]),
            telemetry: AlignedTelemetry([0.0; TELEMETRY_LEN]),
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
