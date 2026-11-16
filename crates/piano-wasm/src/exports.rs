//! Raw C-ABI exports for `AudioWorkletGlobalScope` (no JS imports required).

use core::ptr;

use crate::prep_ipc::PrepContext;
use crate::ring_ipc::WasmContext;

/// Allocates and initializes a [`WasmContext`] for the given sample rate.
#[no_mangle]
pub extern "C" fn piano_init(sample_rate: f32) -> *mut WasmContext {
    Box::into_raw(Box::new(WasmContext::new(sample_rate)))
}

/// Frees a [`WasmContext`] previously returned by [`piano_init`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_destroy(ctx: *mut WasmContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: Caller provides a valid live pointer from `piano_init` and destroys it once.
    unsafe {
        drop(Box::from_raw(ctx));
    }
}

/// Queues a sample-accurate note-on command for the next render block.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_note_on(
    ctx: *mut WasmContext,
    midi_key: u8,
    velocity_u16: u16,
    sample_offset: u32,
) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.note_on(midi_key, velocity_u16, sample_offset);
}

/// Queues a sample-accurate note-off command for the next render block.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_note_off(
    ctx: *mut WasmContext,
    midi_key: u8,
    release_vel_u16: u16,
    sample_offset: u32,
) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.note_off(midi_key, release_vel_u16, sample_offset);
}

/// Queues a sample-accurate pedal command (`0`=Sustain, `1`=UnaCorda, `2`=Sostenuto).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_pedal(
    ctx: *mut WasmContext,
    pedal_kind: u8,
    value_u16: u16,
    sample_offset: u32,
) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.pedal(pedal_kind, value_u16, sample_offset);
}

/// Queues an all-notes-off command for the next render block.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_all_notes_off(ctx: *mut WasmContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.all_notes_off();
}

/// Renders up to `MAX_BLOCK` (512) stereo frames into the context's internal buffers.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_render(ctx: *mut WasmContext, frames: usize) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.render(frames);
}

/// Returns a pointer to the left channel output buffer (`MAX_BLOCK` floats).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_out_l_ptr(ctx: *const WasmContext) -> *const f32 {
    if ctx.is_null() {
        return ptr::null();
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &*ctx };
    ctx.out_l_ptr()
}

/// Returns a pointer to the right channel output buffer (`MAX_BLOCK` floats).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_out_r_ptr(ctx: *const WasmContext) -> *const f32 {
    if ctx.is_null() {
        return ptr::null();
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &*ctx };
    ctx.out_r_ptr()
}

/// Returns a pointer to the 8-float telemetry array `[active_voices, peak_l, peak_r, rms_mono, 0..]`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_telemetry_ptr(ctx: *const WasmContext) -> *const f32 {
    if ctx.is_null() {
        return ptr::null();
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &*ctx };
    ctx.telemetry_ptr()
}

/// Sets voicing parameters with 20 ms exponential smoothing.
///
/// Returns 0 on success, or -1 if ctx is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_voicing(
    ctx: *mut WasmContext,
    hammer_hardness: f32,
    unison_detune: f32,
    stretch: f32,
    length_scale: f32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_voicing(hammer_hardness, unison_detune, stretch, length_scale);
    0
}

/// Sets listener perspective and piano lid state with 8 ms raised-cosine crossfade.
///
/// Returns 0 on success, -1 if ctx is null, or -2 if perspective or lid discriminant is invalid.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_space(ctx: *mut WasmContext, perspective: u32, lid: u32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let p_u8 = match u8::try_from(perspective) {
        Ok(v) => v,
        Err(_) => return -2,
    };
    let l_u8 = match u8::try_from(lid) {
        Ok(v) => v,
        Err(_) => return -2,
    };
    let Some(persp) = piano_spec::ListenerPerspective::from_u8(p_u8) else {
        return -2;
    };
    let Some(lid_state) = piano_spec::LidState::from_u8(l_u8) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_space(persp, lid_state);
    0
}

/// Updates only hammer hardness smoothly (`< 50 us`), avoiding Stage 2/3 rebuild.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_hammer_hardness(ctx: *mut WasmContext, hammer_hardness: f32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_hammer_hardness(hammer_hardness);
    0
}

/// Returns mutable pointer to pre-allocated staging memory at `off_f32`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_staging_offset_ptr(
    ctx: *mut WasmContext,
    off_f32: usize,
    len_f32: usize,
) -> *mut f32 {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    let ctx = unsafe { &mut *ctx };
    ctx.staging_offset_mut_ptr(off_f32, len_f32)
}

/// Streams a precomputed space blob chunk from `staging[0..chunk_len]` into standby.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_stage_space_chunk(
    ctx: *mut WasmContext,
    perspective: u32,
    lid: u32,
    blob_off: usize,
    chunk_len: usize,
    total_len: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(persp) = u8::try_from(perspective)
        .ok()
        .and_then(piano_spec::ListenerPerspective::from_u8)
    else {
        return -2;
    };
    let Some(lid_state) = u8::try_from(lid)
        .ok()
        .and_then(piano_spec::LidState::from_u8)
    else {
        return -2;
    };
    let ctx = unsafe { &mut *ctx };
    if ctx.stage_space_chunk(persp, lid_state, blob_off, chunk_len, total_len) {
        0
    } else {
        -3
    }
}

/// Commits pre-staged spatial radiator standby via instant swap or 8 ms crossfade.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_commit_space(ctx: *mut WasmContext, perspective: u32, lid: u32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(persp) = u8::try_from(perspective)
        .ok()
        .and_then(piano_spec::ListenerPerspective::from_u8)
    else {
        return -2;
    };
    let Some(lid_state) = u8::try_from(lid)
        .ok()
        .and_then(piano_spec::LidState::from_u8)
    else {
        return -2;
    };
    let ctx = unsafe { &mut *ctx };
    if ctx.commit_space(persp, lid_state) {
        0
    } else {
        -3
    }
}

/// Returns 1 while a spatial 8 ms crossfade is active, else 0.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_is_space_crossfading(ctx: *const WasmContext) -> i32 {
    if ctx.is_null() {
        return 0;
    }
    let ctx = unsafe { &*ctx };
    i32::from(ctx.is_space_crossfading())
}

/// Returns mutable pointer to pre-allocated staging memory for zero-alloc blob transfer.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_staging_ptr(ctx: *mut WasmContext, len_f32: usize) -> *mut f32 {
    if ctx.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.staging_mut_ptr(len_f32)
}

/// Installs an off-thread precomputed spatial FIR + NUPOLS blob from staging memory.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_apply_space_blob(
    ctx: *mut WasmContext,
    perspective: u32,
    lid: u32,
    len_f32: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(persp) = u8::try_from(perspective)
        .ok()
        .and_then(piano_spec::ListenerPerspective::from_u8)
    else {
        return -2;
    };
    let Some(lid_state) = u8::try_from(lid)
        .ok()
        .and_then(piano_spec::LidState::from_u8)
    else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    if ctx.apply_space_blob(persp, lid_state, len_f32) {
        0
    } else {
        -3
    }
}

/// Installs an off-thread precomputed Stage 2/3 tuning blob from staging memory.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_apply_voicing_blob(
    ctx: *mut WasmContext,
    hammer_hardness: f32,
    unison_detune: f32,
    stretch: f32,
    length_scale: f32,
    len_f32: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    if ctx.apply_voicing_blob(
        hammer_hardness,
        unison_detune,
        stretch,
        length_scale,
        len_f32,
    ) {
        0
    } else {
        -3
    }
}

/// Allocates a [`PrepContext`] for the off-thread Prep Web Worker.
#[no_mangle]
pub extern "C" fn piano_prep_init(sample_rate: f32) -> *mut PrepContext {
    Box::into_raw(Box::new(PrepContext::new(sample_rate)))
}

/// Frees a [`PrepContext`] previously returned by [`piano_prep_init`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_destroy(prep: *mut PrepContext) {
    if !prep.is_null() {
        // SAFETY: Caller owns valid pointer returned by `piano_prep_init`.
        unsafe { drop(Box::from_raw(prep)) };
    }
}

/// Precomputes Stage 4 spatial FIR + NUPOLS spectra off-thread; returns `f32` length.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_space(prep: *mut PrepContext, perspective: u32, lid: u32) -> usize {
    if prep.is_null() {
        return 0;
    }
    let Some(persp) = u8::try_from(perspective)
        .ok()
        .and_then(piano_spec::ListenerPerspective::from_u8)
    else {
        return 0;
    };
    let Some(lid_state) = u8::try_from(lid)
        .ok()
        .and_then(piano_spec::LidState::from_u8)
    else {
        return 0;
    };
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    unsafe { &mut *prep }.prepare_space(persp, lid_state)
}

/// Precomputes Stage 2/3 tuning tables off-thread; returns `f32` length.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_voicing(
    prep: *mut PrepContext,
    hammer_hardness: f32,
    unison_detune: f32,
    stretch: f32,
    length_scale: f32,
) -> usize {
    if prep.is_null() {
        return 0;
    }
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    unsafe { &mut *prep }.prepare_voicing(hammer_hardness, unison_detune, stretch, length_scale)
}

/// Returns pointer to the most recently prepared blob in [`PrepContext`] linear memory.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_out_ptr(prep: *const PrepContext) -> *const f32 {
    if prep.is_null() {
        return ptr::null();
    }
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    unsafe { &*prep }.out_ptr()
}
