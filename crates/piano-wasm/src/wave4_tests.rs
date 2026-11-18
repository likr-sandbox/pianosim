//! Wave 4 C-ABI tests for concert pitch (392..=466.2 Hz), classical temperaments (0..=3),
//! aging condition macro (0.0..=1.0), and extended voicingBlob preparation/installation.

use core::ptr;
use core::slice;

use crate::{
    piano_apply_voicing_blob_ex, piano_destroy, piano_init, piano_note_on, piano_out_l_ptr,
    piano_out_r_ptr, piano_prep_destroy, piano_prep_init, piano_prep_out_ptr,
    piano_prep_voicing_ex, piano_render, piano_set_concert_pitch, piano_set_condition,
    piano_set_temperament, piano_set_voicing_full, piano_staging_offset_ptr, piano_staging_ptr,
};

#[test]
fn c_abi_wave4_null_guards_and_temperament_validation() {
    let ctx = piano_init(48_000.0);
    let prep = piano_prep_init(48_000.0);
    assert!(!ctx.is_null() && !prep.is_null());

    // Null pointer guards
    assert_eq!(piano_set_concert_pitch(ptr::null_mut(), 415.0), -1);
    assert_eq!(piano_set_temperament(ptr::null_mut(), 1), -1);
    assert_eq!(piano_set_condition(ptr::null_mut(), 0.5), -1);
    assert_eq!(
        piano_set_voicing_full(ptr::null_mut(), 0.5, 1.0, 1.0, 1.0, 440.0, 0, 0.0),
        -1
    );
    assert_eq!(
        piano_prep_voicing_ex(ptr::null_mut(), 0.5, 1.0, 1.0, 1.0, 440.0, 0, 0.0),
        0
    );
    assert_eq!(
        piano_apply_voicing_blob_ex(ptr::null_mut(), 0.5, 1.0, 1.0, 1.0, 440.0, 0, 0.0, 100),
        -1
    );

    // Invalid temperament IDs rejected with -2 (or 0 for prep)
    assert_eq!(piano_set_temperament(ctx, 4), -2);
    assert_eq!(piano_set_temperament(ctx, u32::MAX), -2);
    assert_eq!(
        piano_set_voicing_full(ctx, 0.5, 1.0, 1.0, 1.0, 440.0, 9, 0.0),
        -2
    );
    assert_eq!(
        piano_prep_voicing_ex(prep, 0.5, 1.0, 1.0, 1.0, 440.0, 4, 0.0),
        0
    );
    assert_eq!(
        piano_apply_voicing_blob_ex(ctx, 0.5, 1.0, 1.0, 1.0, 440.0, 4, 0.0, 100),
        -2
    );

    // Valid real-time controls and NaN/out-of-range clamping without panic
    assert_eq!(piano_set_concert_pitch(ctx, 392.0), 0);
    assert_eq!(piano_set_concert_pitch(ctx, 466.2), 0);
    assert_eq!(piano_set_concert_pitch(ctx, f32::NAN), 0);
    for id in 0..=3 {
        assert_eq!(piano_set_temperament(ctx, id), 0);
    }
    assert_eq!(piano_set_condition(ctx, 0.75), 0);
    assert_eq!(piano_set_condition(ctx, f32::NAN), 0);
    assert_eq!(
        piano_set_voicing_full(ctx, 0.6, 1.2, 1.1, 0.98, 415.0, 2, 0.5),
        0
    );

    piano_note_on(ctx, 60, 50_000, 0);
    for _ in 0..8 {
        piano_render(ctx, 128);
    }
    let out_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
    let out_r = unsafe { slice::from_raw_parts(piano_out_r_ptr(ctx), 128) };
    assert!(out_l
        .iter()
        .chain(out_r.iter())
        .all(|x| x.is_finite() && x.abs() <= 1.0));
    assert!(out_l.iter().any(|&x| x.abs() > 1e-4));

    piano_prep_destroy(prep);
    piano_destroy(ctx);
}

#[test]
fn c_abi_wave4_prep_voicing_ex_chunked_staging_and_default_parity() {
    let ctx_default = piano_init(48_000.0);
    let ctx_ex = piano_init(48_000.0);
    let prep = piano_prep_init(48_000.0);
    assert!(!ctx_default.is_null() && !ctx_ex.is_null() && !prep.is_null());

    // Stream an extended voicing blob with 392.0 Hz + Werckmeister III (1) + condition=0.8
    let vc_len = piano_prep_voicing_ex(prep, 0.65, 1.25, 1.1, 1.0, 392.0, 1, 0.8);
    assert!(vc_len > 10_000);
    let src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), vc_len) };
    let mut off = 0_usize;
    while off < vc_len {
        let n = 8_192.min(vc_len - off);
        let dst = piano_staging_offset_ptr(ctx_ex, off, n);
        assert!(!dst.is_null());
        unsafe { slice::from_raw_parts_mut(dst, n) }.copy_from_slice(&src[off..off + n]);
        off += n;
    }
    assert_eq!(
        piano_apply_voicing_blob_ex(ctx_ex, 0.65, 1.25, 1.1, 1.0, 392.0, 1, 0.8, vc_len),
        0
    );

    // Invalid len_f32 rejected with -3
    assert_eq!(
        piano_apply_voicing_blob_ex(ctx_ex, 0.65, 1.25, 1.1, 1.0, 392.0, 1, 0.8, 0),
        -3
    );

    // Restoring default 440.0 + Equal (0) + condition=0.0 via staging yields clean audio
    let def_len = piano_prep_voicing_ex(prep, 0.5, 1.0, 1.0, 1.0, 440.0, 0, 0.0);
    assert_eq!(def_len, vc_len);
    let def_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), def_len) };
    let dst_def = piano_staging_ptr(ctx_ex, def_len);
    assert!(!dst_def.is_null());
    unsafe { slice::from_raw_parts_mut(dst_def, def_len) }.copy_from_slice(def_src);
    assert_eq!(
        piano_apply_voicing_blob_ex(ctx_ex, 0.5, 1.0, 1.0, 1.0, 440.0, 0, 0.0, def_len),
        0
    );

    // Compare rendered output of restored ctx_ex vs untouched ctx_default (close match; exact on fresh init)
    piano_note_on(ctx_default, 69, 52_000, 0);
    piano_note_on(ctx_ex, 69, 52_000, 0);
    piano_render(ctx_default, 128);
    piano_render(ctx_ex, 128);
    let ld = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx_default), 128) };
    let lx = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx_ex), 128) };
    let max_diff = ld
        .iter()
        .zip(lx.iter())
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(
        max_diff < 1e-6,
        "restored default C-ABI voicing diverged by {max_diff}"
    );

    piano_prep_destroy(prep);
    piano_destroy(ctx_default);
    piano_destroy(ctx_ex);
}
