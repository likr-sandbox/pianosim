//! Wave 3 C-ABI exports for reverb controls and multi-body acoustic space switching.

use piano_engine::BodyArchetype;
use piano_spec::{LidState, ListenerPerspective};

use crate::prep_ipc::PrepContext;
use crate::ring_ipc::WasmContext;

#[inline]
fn decode_body_space(
    body_id: u32,
    perspective: u32,
    lid: u32,
) -> Option<(BodyArchetype, ListenerPerspective, LidState)> {
    let body = u8::try_from(body_id)
        .ok()
        .and_then(BodyArchetype::from_u8)?;
    let persp = u8::try_from(perspective)
        .ok()
        .and_then(ListenerPerspective::from_u8)?;
    let lid_state = u8::try_from(lid).ok().and_then(LidState::from_u8)?;
    Some((body, persp, lid_state))
}

/// Sets hall reverb decay `[0.0..=3.2 s]` and wet mix `[0.0..=1.0]`.
///
/// Returns 0 on success, or -1 if `ctx` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_reverb(ctx: *mut WasmContext, decay_s: f32, mix: f32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_reverb(decay_s, mix);
    0
}

/// Synchronously switches `(body_id, perspective, lid)` across Stage 3 and Stage 4.
///
/// Returns 0 on success, -1 if `ctx` is null, or -2 if any discriminant is invalid.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_body_space(
    ctx: *mut WasmContext,
    body_id: u32,
    perspective: u32,
    lid: u32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some((body, persp, lid_state)) = decode_body_space(body_id, perspective, lid) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    if ctx.set_body_space(body, persp, lid_state) {
        0
    } else {
        -3
    }
}

/// Precomputes Stage 4 spatial FIR + NUPOLS spectra off-thread for `(body_id, perspective, lid)`;
/// returns `f32` element length (or 0 on null/invalid discriminants).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_space_for_body(
    prep: *mut PrepContext,
    body_id: u32,
    perspective: u32,
    lid: u32,
) -> usize {
    if prep.is_null() {
        return 0;
    }
    let Some((body, persp, lid_state)) = decode_body_space(body_id, perspective, lid) else {
        return 0;
    };
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    unsafe { &mut *prep }.prepare_space_for_body(body, persp, lid_state)
}

/// Streams a precomputed space blob chunk for `(body_id, perspective, lid)` from `staging`
/// into standby.
///
/// Returns 0 on success, -1 if `ctx` is null, -2 if any discriminant is invalid, or -3 on staging failure.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_stage_body_space_chunk(
    ctx: *mut WasmContext,
    body_id: u32,
    perspective: u32,
    lid: u32,
    blob_off_f32: usize,
    chunk_len_f32: usize,
    total_len_f32: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some((body, persp, lid_state)) = decode_body_space(body_id, perspective, lid) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    if ctx.stage_body_space_chunk(
        body,
        persp,
        lid_state,
        blob_off_f32,
        chunk_len_f32,
        total_len_f32,
    ) {
        0
    } else {
        -3
    }
}

/// Commits pre-staged spatial standby for `(body_id, perspective, lid)` via instant swap or
/// 8 ms crossfade, synchronizing Stage 3 bridge admittance zero-alloc.
///
/// Returns 0 on success, -1 if `ctx` is null, -2 if any discriminant is invalid, or -3 on mismatch.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_commit_body_space(
    ctx: *mut WasmContext,
    body_id: u32,
    perspective: u32,
    lid: u32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some((body, persp, lid_state)) = decode_body_space(body_id, perspective, lid) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    if ctx.commit_body_space(body, persp, lid_state) {
        0
    } else {
        -3
    }
}
