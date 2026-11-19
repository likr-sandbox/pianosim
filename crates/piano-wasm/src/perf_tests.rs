//! Tests for M3 C-ABI `PerfProfile` exports, `SpaceBlob` v2, and `VoicingBlob` prep installation.

use core::ptr;
use core::slice;

use piano_engine::{
    space_blob_len_for_profile, voicing_blob_len_with_profile, PerfProfile, SPACE_BLOB_MAGIC_V2,
};

use crate::{piano_is_voicing_applying, piano_space_chunk_ptr};

use super::*;

fn stream_space_via_chunk_buf(
    ctx: *mut WasmContext,
    body_id: u32,
    persp: u32,
    lid: u32,
    src: &[f32],
) {
    let mut off = 0_usize;
    while off < src.len() {
        let n = 2_048.min(src.len() - off);
        let dst = piano_space_chunk_ptr(ctx);
        assert!(!dst.is_null());
        unsafe { slice::from_raw_parts_mut(dst, n) }.copy_from_slice(&src[off..off + n]);
        assert_eq!(
            piano_stage_body_space_chunk(ctx, body_id, persp, lid, off, n, src.len()),
            0
        );
        off += n;
    }
}

fn drain_voicing(ctx: *mut WasmContext) {
    while piano_is_voicing_applying(ctx) != 0 {
        piano_render(ctx, 128);
    }
}

#[test]
fn init_with_profile_renders_finite_audio_and_installs_profile_blobs() {
    assert!(piano_init_with_profile(48_000.0, 3).is_null());

    for preset_id in 0..=2_u32 {
        let profile = match preset_id {
            0 => PerfProfile::full(),
            1 => PerfProfile::balanced(),
            _ => PerfProfile::realtime(),
        };
        let ctx = piano_init_with_profile(48_000.0, preset_id);
        let prep = piano_prep_init(48_000.0);
        assert!(!ctx.is_null() && !prep.is_null());
        assert_eq!(piano_prep_set_perf_preset(prep, preset_id), 0);

        piano_note_on(ctx, 60, 52_000, 0);
        piano_render(ctx, 128);

        let sp_len = piano_prep_space(prep, 2, 1);
        assert_eq!(sp_len, space_blob_len_for_profile(48_000.0, &profile));
        let sp_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), sp_len) };
        assert_eq!(sp_src[0].to_bits(), SPACE_BLOB_MAGIC_V2);
        let st_sp = piano_staging_ptr(ctx, sp_len);
        assert!(!st_sp.is_null());
        unsafe { slice::from_raw_parts_mut(st_sp, sp_len) }.copy_from_slice(sp_src);
        assert_eq!(piano_apply_space_blob(ctx, 2, 1, sp_len), 0);

        let vc_len = piano_prep_voicing(prep, 0.7, 1.4, 1.2, 0.96, 442.0);
        assert_eq!(vc_len, voicing_blob_len_with_profile(48_000.0, &profile));
        let vc_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), vc_len) };
        let st_vc = piano_staging_ptr(ctx, vc_len);
        assert!(!st_vc.is_null());
        unsafe { slice::from_raw_parts_mut(st_vc, vc_len) }.copy_from_slice(vc_src);
        assert_eq!(
            piano_apply_voicing_blob(ctx, 0.7, 1.4, 1.2, 0.96, 442.0, vc_len),
            0
        );
        drain_voicing(ctx);

        piano_render(ctx, 128);
        let out_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
        let out_r = unsafe { slice::from_raw_parts(piano_out_r_ptr(ctx), 128) };
        assert!(out_l.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        assert!(out_r.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        assert!(out_l.iter().any(|&x| x.abs() > 1e-4));

        piano_prep_destroy(prep);
        piano_destroy(ctx);
    }
}

#[test]
fn set_perf_preset_and_knob_runtime_checks_and_error_codes() {
    assert_eq!(piano_set_perf_preset(ptr::null_mut(), 0), -1);
    assert_eq!(piano_set_perf_knob(ptr::null_mut(), 0, -90.0), -1);
    assert_eq!(piano_prep_set_perf_preset(ptr::null_mut(), 0), -1);
    assert_eq!(
        piano_prep_set_stage4_profile(ptr::null_mut(), 0.25, 2048, 32, 0),
        -1
    );

    let ctx = piano_init_with_profile(48_000.0, 0);
    let prep = piano_prep_init(48_000.0);
    assert_eq!(piano_set_perf_preset(ctx, 99), -2);
    assert_eq!(piano_set_perf_knob(ctx, 99, 1.0), -2);
    assert_eq!(piano_set_perf_knob(ctx, 19, 1.0), -2);
    assert_eq!(piano_prep_set_perf_preset(prep, 99), -2);

    // Full -> Balanced at runtime returns 1 (structural SpaceBlob recommended),
    // and applies all 11 runtime knobs cleanly.
    assert_eq!(piano_set_perf_preset(ctx, 1), 1);
    // Individual runtime knobs 0..=9 and 18 return 0 when no structural mismatch introduced.
    assert_eq!(piano_set_perf_knob(ctx, 0, -95.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 1, -44.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 2, 32.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 3, 4_200.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 4, 16.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 5, 16.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 6, 0.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 7, 8.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 8, 1.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 9, 1.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 9, 0.0), 0);
    assert_eq!(piano_set_perf_knob(ctx, 18, -105.0), 0);
    assert_eq!(unsafe { (*ctx).perf_profile().damper_control_decim }, 16);
    assert_eq!(
        unsafe { (*ctx).perf_profile().low_order_energy_horizon_db },
        -105.0
    );

    // Stage 4 PrepContext direct override produces matching SpaceBlob v2 length and applies.
    assert_eq!(piano_prep_set_stage4_profile(prep, 0.125, 1024, 16, 1), 0);
    let sp_len = piano_prep_space_for_body(prep, 1, 3, 2);
    let expected = space_blob_len_for_profile(
        48_000.0,
        &PerfProfile {
            ir_duration_secs: 0.125,
            spatial_tail_split_tap: 1024,
            fir_head_taps: 16,
            fir_block_fft: true,
            ..PerfProfile::full()
        },
    );
    assert_eq!(sp_len, expected);
    let sp_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), sp_len) };
    stream_space_via_chunk_buf(ctx, 1, 3, 2, sp_src);
    assert_eq!(piano_commit_body_space(ctx, 1, 3, 2), 0);

    piano_note_on(ctx, 69, 48_000, 0);
    piano_render(ctx, 128);
    let out_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
    assert!(out_l.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
    assert!(out_l.iter().any(|&x| x.abs() > 1e-4));

    piano_prep_destroy(prep);
    piano_destroy(ctx);
}

#[test]
fn perf_builder_configures_all_19_knobs_and_syncs_wasm_and_prep_contexts() {
    assert!(piano_perf_builder_new(3).is_null());
    assert_eq!(piano_perf_builder_set(ptr::null_mut(), 0, -90.0), -1);
    assert!(piano_init_with_builder(48_000.0, ptr::null()).is_null());
    assert_eq!(
        piano_prep_set_perf_builder(ptr::null_mut(), ptr::null()),
        -1
    );
    piano_perf_builder_destroy(ptr::null_mut());

    let builder = piano_perf_builder_new(1);
    assert!(!builder.is_null());
    assert_eq!(piano_perf_builder_set(builder, 19, 1.0), -2);

    // Exercise every knob 0..=18 with custom overrides.
    let overrides: [(u32, f32); 19] = [
        (0, -90.0),
        (1, -40.0),
        (2, 30.0),
        (3, 3_800.0),
        (4, 4.0),
        (5, 16.0),
        (6, 0.0),
        (7, 8.0),
        (8, 0.0),
        (9, 1.0),
        (10, 3_200.0),
        (11, -48.0),
        (12, 1.0),
        (13, 0.20),
        (14, 2_048.0),
        (15, 32.0),
        (16, 1.0),
        (17, 4.0),
        (18, -108.0),
    ];
    for (knob, val) in overrides {
        assert_eq!(piano_perf_builder_set(builder, knob, val), 0);
    }

    let ctx = piano_init_with_builder(48_000.0, builder);
    let prep = piano_prep_init(48_000.0);
    assert!(!ctx.is_null() && !prep.is_null());
    assert_eq!(piano_prep_set_perf_builder(prep, builder), 0);
    assert_eq!(piano_prep_set_perf_builder(prep, ptr::null()), -1);

    let expected_profile = unsafe { (*builder).sanitized() };
    assert_eq!(unsafe { (*prep).perf_profile() }, expected_profile);
    assert_eq!(unsafe { (*ctx).perf_profile() }, expected_profile);

    piano_perf_builder_destroy(builder);

    // Space & Voicing blobs produced by PrepContext match WasmContext dimensions exactly.
    let sp_len = piano_prep_space_for_body(prep, 0, 1, 1);
    assert_eq!(
        sp_len,
        space_blob_len_for_profile(48_000.0, &expected_profile)
    );
    let sp_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), sp_len) };
    stream_space_via_chunk_buf(ctx, 0, 1, 1, sp_src);
    assert_eq!(piano_commit_body_space(ctx, 0, 1, 1), 0);

    let vc_len = piano_prep_voicing_ex(prep, 0.65, 1.2, 1.1, 0.98, 442.0, 1, 0.15);
    assert_eq!(
        vc_len,
        voicing_blob_len_with_profile(48_000.0, &expected_profile)
    );
    let vc_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), vc_len) };
    let st_vc = piano_staging_ptr(ctx, vc_len);
    assert!(!st_vc.is_null());
    unsafe { slice::from_raw_parts_mut(st_vc, vc_len) }.copy_from_slice(vc_src);
    assert_eq!(
        piano_apply_voicing_blob_ex(ctx, 0.65, 1.2, 1.1, 0.98, 442.0, 1, 0.15, vc_len),
        0
    );
    drain_voicing(ctx);

    piano_note_on(ctx, 64, 50_000, 0);
    piano_render(ctx, 128);
    let out_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
    assert!(out_l.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
    assert!(out_l.iter().any(|&x| x.abs() > 1e-4));

    piano_prep_destroy(prep);
    piano_destroy(ctx);
}
