//! Unit tests for `WasmContext` SPSC carryover, panic clearing, and staging guards.

use piano_engine::voicing_blob_len;
use piano_spec::{LidState, ListenerPerspective};

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
