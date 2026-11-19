//! Linear-memory zero-allocation IPC context for `AudioWorkletGlobalScope`.

use piano_engine::{
    space_blob_len_for_sample_rate, voicing_blob_len, BodyArchetype, EventQueue, PerfProfile,
    PianoEngine,
};
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
    pub(crate) engine: PianoEngine,
    events: EventQueue,
    out_l: AlignedBlock,
    out_r: AlignedBlock,
    telemetry: AlignedTelemetry,
    pub(crate) staging: Vec<f32>,
}

impl WasmContext {
    /// Initializes the piano engine (`Full` profile) and pre-allocates fixed audio/telemetry/staging buffers.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_profile(sample_rate, PerfProfile::full())
    }

    /// Initializes the piano engine with `profile` while sizing `staging` to max `Full` capacity.
    #[must_use]
    pub fn new_with_profile(sample_rate: f32, profile: PerfProfile) -> Self {
        let engine = PianoEngine::new_with_profile(sample_rate, VoicingSpec::default(), profile);
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

    /// Zero-allocation audio-thread update of runtime-tunable `PerfProfile` knobs.
    #[inline]
    pub fn set_perf_profile_runtime(&mut self, profile: PerfProfile) -> bool {
        self.engine.set_perf_profile_runtime(profile)
    }

    /// Returns the active [`PerfProfile`] of the underlying engine.
    #[must_use]
    #[inline]
    pub const fn perf_profile(&self) -> PerfProfile {
        self.engine.perf_profile()
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

    /// Discards queued future events and enqueues an immediate all-notes-off panic event.
    #[inline]
    pub fn all_notes_off(&mut self) {
        self.events.clear();
        let _ = self.events.push(SampleEvent {
            sample_offset: 0,
            kind: EventKind::AllNotesOff,
        });
    }

    /// Updates voicing targets (including `A4` concert pitch) for smooth real-time interpolation.
    #[inline]
    pub fn set_voicing(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        a4_freq_hz: f32,
    ) {
        self.engine.set_voicing_with_a4(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            a4_freq_hz,
        );
    }

    /// Reconfigures spatial listening perspective and lid opening state.
    #[inline]
    pub fn set_space(&mut self, perspective: ListenerPerspective, lid: LidState) {
        self.engine.set_space(perspective, lid);
    }

    /// Sets hall reverb decay `[0.0..=3.2 s]` and wet mix `[0.0..=1.0]`.
    #[inline]
    pub fn set_reverb(&mut self, decay_s: f32, mix: f32) {
        self.engine.set_reverb(decay_s, mix);
    }

    /// Returns active `(reverb_decay_s, reverb_mix)`.
    #[inline]
    #[must_use]
    pub const fn reverb_params(&self) -> (f32, f32) {
        self.engine.reverb_params()
    }

    /// Returns the active [`BodyArchetype`].
    #[inline]
    #[must_use]
    pub const fn body_archetype(&self) -> BodyArchetype {
        self.engine.body_archetype()
    }

    /// Synchronously switches `(body, perspective, lid)` across Stage 3 and Stage 4.
    #[inline]
    pub fn set_body_space(
        &mut self,
        body: BodyArchetype,
        perspective: ListenerPerspective,
        lid: LidState,
    ) -> bool {
        self.engine.set_body_space(body, perspective, lid)
    }

    /// Streams one slice from `staging[0..chunk_len]` into the standby spatial radiator for `body`.
    pub fn stage_body_space_chunk(
        &mut self,
        body: BodyArchetype,
        perspective: ListenerPerspective,
        lid: LidState,
        blob_off: usize,
        chunk_len: usize,
        total_len: usize,
    ) -> bool {
        let Some(chunk_end) = blob_off.checked_add(chunk_len) else {
            return false;
        };
        if chunk_len == 0
            || chunk_len > self.staging.len()
            || total_len == 0
            || chunk_end > total_len
        {
            return false;
        }
        self.engine.stage_precomputed_body_space_chunk(
            body,
            perspective,
            lid,
            blob_off,
            &self.staging[..chunk_len],
            total_len,
        )
    }

    /// Commits the streamed standby spatial radiator for `(body, perspective, lid)`.
    #[inline]
    pub fn commit_body_space(
        &mut self,
        body: BodyArchetype,
        perspective: ListenerPerspective,
        lid: LidState,
    ) -> bool {
        self.engine
            .commit_precomputed_body_space(body, perspective, lid)
    }

    /// Updates only hammer hardness smoothly (`< 50 us`), avoiding string table rebuild.
    #[inline]
    pub fn set_hammer_hardness(&mut self, hammer_hardness: f32) {
        self.engine.set_hammer_hardness(hammer_hardness);
    }

    /// Returns mutable pointer to pre-allocated staging buffer if `len_f32` fits.
    #[inline]
    pub fn staging_mut_ptr(&mut self, len_f32: usize) -> *mut f32 {
        self.staging_offset_mut_ptr(0, len_f32)
    }

    /// Returns mutable pointer to `staging[off_f32..]` if `[off_f32..off_f32 + len_f32]` fits.
    #[inline]
    pub fn staging_offset_mut_ptr(&mut self, off_f32: usize, len_f32: usize) -> *mut f32 {
        let Some(end) = off_f32.checked_add(len_f32) else {
            return core::ptr::null_mut();
        };
        if len_f32 == 0 || end > self.staging.len() {
            return core::ptr::null_mut();
        }
        // SAFETY: `off_f32 < end <= self.staging.len()`.
        unsafe { self.staging.as_mut_ptr().add(off_f32) }
    }

    /// Streams one slice from `staging[0..chunk_len]` into the standby spatial radiator.
    pub fn stage_space_chunk(
        &mut self,
        perspective: ListenerPerspective,
        lid: LidState,
        blob_off: usize,
        chunk_len: usize,
        total_len: usize,
    ) -> bool {
        let Some(chunk_end) = blob_off.checked_add(chunk_len) else {
            return false;
        };
        if chunk_len == 0
            || chunk_len > self.staging.len()
            || total_len == 0
            || chunk_end > total_len
        {
            return false;
        }
        self.engine.stage_precomputed_space_chunk(
            perspective,
            lid,
            blob_off,
            &self.staging[..chunk_len],
            total_len,
        )
    }

    /// Commits the streamed standby spatial radiator.
    #[inline]
    pub fn commit_space(&mut self, perspective: ListenerPerspective, lid: LidState) -> bool {
        self.engine.commit_precomputed_space(perspective, lid)
    }

    /// Returns `true` if a spatial crossfade is currently in flight.
    #[inline]
    #[must_use]
    pub fn is_space_crossfading(&self) -> bool {
        self.engine.space_guard().is_crossfading()
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
        a4_freq_hz: f32,
        len_f32: usize,
    ) -> bool {
        if len_f32 == 0 || len_f32 > self.staging.len() {
            return false;
        }
        let spec = VoicingSpec::from_globals_with_a4(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            a4_freq_hz,
        );
        self.engine
            .apply_precomputed_voicing(spec, &self.staging[..len_f32])
    }

    /// Renders up to `MAX_BLOCK` frames into the internal L/R buffers, updates
    /// telemetry, and carries over future (`sample_offset >= n`) events to subsequent
    /// blocks without heap allocations.
    pub fn render(&mut self, frames: usize) {
        let n = frames.min(MAX_BLOCK);
        let n_u32 = n as u32;
        let consumed = self.events.partition_for_block(n_u32);
        self.engine.render_block(
            &self.events.as_slice()[..consumed],
            &mut self.out_l.0[..n],
            &mut self.out_r.0[..n],
        );
        self.events.advance_block(n_u32, consumed);
        self.telemetry.0 = self.engine.telemetry().to_array();
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

#[cfg(test)]
#[path = "ring_ipc_tests.rs"]
mod tests;
