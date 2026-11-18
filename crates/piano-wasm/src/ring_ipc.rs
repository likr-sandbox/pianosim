//! Linear-memory zero-allocation IPC context for `AudioWorkletGlobalScope`.

use piano_engine::{
    space_blob_len_for_sample_rate, voicing_blob_len, BodyArchetype, EventQueue, PianoEngine,
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
mod tests {
    use super::*;

    fn read_out_l(ctx: &WasmContext, len: usize) -> &[f32] {
        // SAFETY: `out_l_ptr()` points to `MAX_BLOCK` initialized floats and `len <= MAX_BLOCK`.
        unsafe { core::slice::from_raw_parts(ctx.out_l_ptr(), len) }
    }

    fn read_out_r(ctx: &WasmContext, len: usize) -> &[f32] {
        // SAFETY: `out_r_ptr()` points to `MAX_BLOCK` initialized floats and `len <= MAX_BLOCK`.
        unsafe { core::slice::from_raw_parts(ctx.out_r_ptr(), len) }
    }

    fn read_telem(ctx: &WasmContext) -> &[f32] {
        // SAFETY: `telemetry_ptr()` points to `TELEMETRY_LEN` initialized floats.
        unsafe { core::slice::from_raw_parts(ctx.telemetry_ptr(), TELEMETRY_LEN) }
    }

    #[test]
    fn render_carries_over_future_events_across_multiple_blocks() {
        let mut ctx = WasmContext::new(48_000.0);
        // Fire A4 at sample 160 (offset 32 in block 2) and C4 at sample 272 (offset 16 in block 3).
        ctx.note_on(60, 50_000, 272);
        ctx.note_on(69, 55_000, 160);

        // Block 1 [0..128): strictly silent, zero active voices.
        ctx.render(128);
        assert!(read_out_l(&ctx, 128).iter().all(|&x| x == 0.0));
        assert!(read_out_r(&ctx, 128).iter().all(|&x| x == 0.0));
        assert_eq!(read_telem(&ctx)[0], 0.0);

        // Block 2 [128..256): silent before sample 32, audible energy from sample 32+, 1 voice.
        ctx.render(128);
        let l2 = read_out_l(&ctx, 128);
        assert!(l2[..32].iter().all(|&x| x == 0.0));
        assert!(l2[32..].iter().any(|&x| x.abs() > 1e-4));
        assert_eq!(read_telem(&ctx)[0], 1.0);

        // Block 3 [256..384): second note enters, 2 active voices.
        ctx.render(128);
        let l3 = read_out_l(&ctx, 128);
        assert!(l3.iter().all(|x| x.is_finite()));
        assert_eq!(read_telem(&ctx)[0], 2.0);
    }

    #[test]
    fn render_boundary_offset_equal_to_block_size_fires_in_next_block() {
        let mut ctx = WasmContext::new(48_000.0);
        ctx.note_on(64, 55_000, 128);

        ctx.render(128);
        assert!(read_out_l(&ctx, 128).iter().all(|&x| x == 0.0));
        assert_eq!(read_telem(&ctx)[0], 0.0);

        ctx.render(128);
        assert!(read_out_l(&ctx, 128).iter().any(|&x| x.abs() > 1e-4));
        assert_eq!(read_telem(&ctx)[0], 1.0);
    }

    #[test]
    fn all_notes_off_discards_queued_future_lookahead_events() {
        let mut ctx = WasmContext::new(48_000.0);
        ctx.note_on(60, 55_000, 0);
        ctx.render(128);
        assert_eq!(read_telem(&ctx)[0], 1.0);

        // Queue future lookahead notes beyond the current block, then trigger Panic/Stop.
        ctx.note_on(64, 55_000, 150);
        ctx.note_on(67, 55_000, 300);
        ctx.all_notes_off();

        for _ in 0..4 {
            ctx.render(128);
            assert!(read_out_l(&ctx, 128).iter().all(|&x| x == 0.0));
            assert!(read_out_r(&ctx, 128).iter().all(|&x| x == 0.0));
            assert_eq!(read_telem(&ctx)[0], 0.0);
        }
    }

    #[test]
    fn staging_guards_reject_overflow_oob_and_corrupted_headers() {
        let mut ctx = WasmContext::new(48_000.0);
        let cap = ctx.staging.len();

        assert!(ctx.staging_mut_ptr(0).is_null());
        assert!(ctx.staging_offset_mut_ptr(usize::MAX, 1).is_null());
        assert!(ctx.staging_offset_mut_ptr(cap, 1).is_null());
        assert!(ctx.staging_offset_mut_ptr(0, cap + 1).is_null());
        assert!(!ctx.staging_offset_mut_ptr(0, 1).is_null());

        assert!(!ctx.stage_space_chunk(
            ListenerPerspective::Player,
            LidState::Open,
            usize::MAX,
            1,
            100
        ));
        assert!(!ctx.stage_space_chunk(ListenerPerspective::Player, LidState::Open, 0, 0, 100));
        assert!(!ctx.stage_space_chunk(ListenerPerspective::Player, LidState::Open, 50, 60, 100));
        assert!(!ctx.stage_space_chunk(ListenerPerspective::Player, LidState::Open, 0, 10, 0));

        assert!(!ctx.apply_space_blob(ListenerPerspective::Player, LidState::Open, 0));
        assert!(!ctx.apply_space_blob(ListenerPerspective::Player, LidState::Open, cap + 1));
        assert!(!ctx.apply_voicing_blob(0.5, 1.0, 1.0, 1.0, 440.0, 0));
        assert!(!ctx.apply_voicing_blob(0.5, 1.0, 1.0, 1.0, 440.0, cap + 1));

        // Corrupt staging header (simulating a staging overwrite) and verify clean rejection.
        ctx.staging[0] = 0.0;
        let vc_len = voicing_blob_len(48_000.0);
        assert!(!ctx.apply_voicing_blob(0.5, 1.0, 1.0, 1.0, 440.0, vc_len));
    }
}
