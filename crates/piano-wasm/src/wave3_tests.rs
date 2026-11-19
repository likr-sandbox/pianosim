//! Wave 3 C-ABI tests for reverb controls and multi-body space staging/commit.

use core::ptr;
use core::slice;

use crate::{
    piano_all_notes_off, piano_commit_body_space, piano_destroy, piano_init,
    piano_is_space_crossfading, piano_note_off, piano_note_on, piano_out_l_ptr, piano_out_r_ptr,
    piano_prep_destroy, piano_prep_init, piano_prep_out_ptr, piano_prep_space_for_body,
    piano_render, piano_set_body_space, piano_set_reverb, piano_space_chunk_ptr,
    piano_stage_body_space_chunk,
};

#[test]
fn c_abi_reverb_and_body_space_discriminant_validation() {
    let ctx = piano_init(48_000.0);
    let prep = piano_prep_init(48_000.0);
    assert!(!ctx.is_null() && !prep.is_null());

    // Null pointer guards
    assert_eq!(piano_set_reverb(ptr::null_mut(), 1.5, 0.2), -1);
    assert_eq!(piano_set_body_space(ptr::null_mut(), 0, 0, 2), -1);
    assert_eq!(piano_prep_space_for_body(ptr::null_mut(), 0, 0, 2), 0);
    assert_eq!(
        piano_stage_body_space_chunk(ptr::null_mut(), 0, 0, 2, 0, 1, 10),
        -1
    );
    assert_eq!(piano_commit_body_space(ptr::null_mut(), 0, 0, 2), -1);
    assert!(piano_space_chunk_ptr(ptr::null_mut()).is_null());

    // Invalid discriminants return -2 (or 0 for prep_space_for_body)
    assert_eq!(piano_set_body_space(ctx, 3, 0, 2), -2);
    assert_eq!(piano_set_body_space(ctx, 0, 4, 2), -2);
    assert_eq!(piano_set_body_space(ctx, 0, 0, 3), -2);
    assert_eq!(piano_prep_space_for_body(prep, 3, 0, 2), 0);
    assert_eq!(piano_prep_space_for_body(prep, 0, 4, 2), 0);
    assert_eq!(piano_prep_space_for_body(prep, 0, 0, 3), 0);
    assert_eq!(piano_stage_body_space_chunk(ctx, 3, 0, 2, 0, 8, 100), -2);
    assert_eq!(piano_commit_body_space(ctx, 3, 0, 2), -2);

    // Valid reverb & synchronous body switching across all 3 archetypes
    assert_eq!(piano_set_reverb(ctx, 2.2, 0.30), 0);
    assert_eq!(piano_set_reverb(ctx, f32::NAN, -1.0), 0);
    assert_eq!(piano_set_reverb(ctx, 1.8, 0.22), 0);
    assert_eq!(piano_set_body_space(ctx, 1, 0, 2), 0);
    assert_eq!(piano_set_body_space(ctx, 2, 3, 1), 0);

    piano_note_on(ctx, 60, 52_000, 0);
    piano_render(ctx, 128);
    let out_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
    assert!(out_l.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
    assert!(out_l.iter().any(|&x| x.abs() > 1e-4));

    piano_prep_destroy(prep);
    piano_destroy(ctx);
}

#[test]
fn c_abi_prep_worker_streams_multibody_space_chunks_and_commits() {
    let ctx = piano_init(48_000.0);
    let prep = piano_prep_init(48_000.0);
    assert!(!ctx.is_null() && !prep.is_null());

    // Start note so audio is actively ringing when switching body.
    piano_note_on(ctx, 69, 50_000, 0);
    for _ in 0..8 {
        piano_render(ctx, 128);
    }

    // Prepare UprightStudio (body_id=1) for Player (0), Open (2).
    let len = piano_prep_space_for_body(prep, 1, 0, 2);
    assert!(len > 10_000);
    let src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), len) };

    // Oversized space chunk (> 2048 floats) must be rejected cleanly.
    assert_eq!(
        piano_stage_body_space_chunk(ctx, 1, 0, 2, 0, 2_049, len),
        -3
    );

    let mut off = 0_usize;
    while off < len {
        let n = 2_048.min(len - off);
        let dst = piano_space_chunk_ptr(ctx);
        assert!(!dst.is_null());
        unsafe { slice::from_raw_parts_mut(dst, n) }.copy_from_slice(&src[off..off + n]);
        assert_eq!(piano_stage_body_space_chunk(ctx, 1, 0, 2, off, n, len), 0);
        off += n;
    }

    // Wrong body_id commit rejected with -3, right body_id succeeds and starts crossfade.
    assert_eq!(piano_commit_body_space(ctx, 0, 0, 2), -3);
    assert_eq!(piano_commit_body_space(ctx, 1, 0, 2), 0);
    assert_eq!(piano_is_space_crossfading(ctx), 1);

    for _ in 0..16 {
        piano_render(ctx, 128);
        let l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
        let r = unsafe { slice::from_raw_parts(piano_out_r_ptr(ctx), 128) };
        assert!(l
            .iter()
            .chain(r.iter())
            .all(|x| x.is_finite() && x.abs() <= 1.0));
    }
    assert_eq!(piano_is_space_crossfading(ctx), 0);

    // Switch to VintageSaloon (body_id=2), ConcertHall (2), Closed (0) with reverb tail & panic reset.
    assert_eq!(piano_set_reverb(ctx, 2.4, 0.40), 0);
    let len2 = piano_prep_space_for_body(prep, 2, 2, 0);
    assert_eq!(len2, len);
    let src2 = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), len2) };
    let mut off2 = 0_usize;
    while off2 < len2 {
        let n = 2_048.min(len2 - off2);
        let dst = piano_space_chunk_ptr(ctx);
        assert!(!dst.is_null());
        unsafe { slice::from_raw_parts_mut(dst, n) }.copy_from_slice(&src2[off2..off2 + n]);
        assert_eq!(piano_stage_body_space_chunk(ctx, 2, 2, 0, off2, n, len2), 0);
        off2 += n;
    }
    assert_eq!(piano_commit_body_space(ctx, 2, 2, 0), 0);
    piano_note_off(ctx, 69, 32_768, 0);
    piano_render(ctx, 128);

    piano_all_notes_off(ctx);
    piano_render(ctx, 128);
    let post_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
    let post_r = unsafe { slice::from_raw_parts(piano_out_r_ptr(ctx), 128) };
    assert!(post_l.iter().chain(post_r.iter()).all(|&x| x == 0.0));

    piano_prep_destroy(prep);
    piano_destroy(ctx);
}
