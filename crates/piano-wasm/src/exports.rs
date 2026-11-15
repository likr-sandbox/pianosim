//! Raw C-ABI exports for `AudioWorkletGlobalScope` (no JS imports required).

use core::ptr;

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
