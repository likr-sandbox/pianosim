//! AudioWorklet WebAssembly bindings and linear-memory SPSC IPC queues.

pub mod exports;
pub mod prep_ipc;
pub mod ring_ipc;
pub mod wave3_exports;
pub mod wave4_exports;

pub use exports::{
    piano_all_notes_off, piano_apply_space_blob, piano_apply_voicing_blob, piano_commit_space,
    piano_destroy, piano_init, piano_is_space_crossfading, piano_note_off, piano_note_on,
    piano_out_l_ptr, piano_out_r_ptr, piano_pedal, piano_prep_destroy, piano_prep_init,
    piano_prep_out_ptr, piano_prep_space, piano_prep_voicing, piano_render,
    piano_set_hammer_hardness, piano_set_space, piano_set_voicing, piano_stage_space_chunk,
    piano_staging_offset_ptr, piano_staging_ptr, piano_telemetry_ptr,
};
pub use prep_ipc::PrepContext;
pub use ring_ipc::{WasmContext, MAX_BLOCK, TELEMETRY_LEN};
pub use wave3_exports::{
    piano_commit_body_space, piano_prep_space_for_body, piano_set_body_space, piano_set_reverb,
    piano_stage_body_space_chunk,
};
pub use wave4_exports::{
    piano_apply_voicing_blob_ex, piano_prep_voicing_ex, piano_set_concert_pitch,
    piano_set_condition, piano_set_temperament, piano_set_voicing_full,
};

#[cfg(test)]
mod wave3_tests;

#[cfg(test)]
mod wave4_tests;

#[cfg(test)]
mod tests {
    use core::ptr;
    use core::slice;

    use super::*;

    #[test]
    fn c_abi_full_lifecycle_render_and_telemetry() {
        let ctx = piano_init(48_000.0);
        assert!(!ctx.is_null());

        let l_ptr = piano_out_l_ptr(ctx);
        let r_ptr = piano_out_r_ptr(ctx);
        let t_ptr = piano_telemetry_ptr(ctx);
        assert!(!l_ptr.is_null() && !r_ptr.is_null() && !t_ptr.is_null());
        assert_ne!(l_ptr, r_ptr, "L and R pointers must never alias");
        assert_eq!((l_ptr as usize) % 4, 0);
        assert_eq!((r_ptr as usize) % 4, 0);
        assert_eq!((t_ptr as usize) % 4, 0);

        // Silent render baseline.
        piano_render(ctx, 128);
        let telem = unsafe { slice::from_raw_parts(t_ptr, TELEMETRY_LEN) };
        assert_eq!(telem[0], 0.0);
        assert_eq!(telem[1], 0.0);
        assert_eq!(telem[2], 0.0);
        assert_eq!(telem[3], 0.0);

        // Trigger A4 with Sustain pedal, render across MAX_BLOCK clamping boundary.
        piano_pedal(ctx, 0, u16::MAX, 0);
        piano_note_on(ctx, 69, 55_000, 16);
        piano_render(ctx, 128);

        let out_l = unsafe { slice::from_raw_parts(l_ptr, 128) };
        let out_r = unsafe { slice::from_raw_parts(r_ptr, 128) };
        assert!(out_l[..16].iter().all(|&x| x == 0.0));
        assert!(out_l[16..].iter().any(|&x| x.abs() > 1e-4));
        assert!(out_r[16..].iter().any(|&x| x.abs() > 1e-4));
        for (&l, &r) in out_l.iter().zip(out_r.iter()) {
            assert!(l.is_finite() && r.is_finite());
            assert!(l.abs() <= 1.0 && r.abs() <= 1.0);
        }

        let telem = unsafe { slice::from_raw_parts(t_ptr, TELEMETRY_LEN) };
        assert_eq!(telem[0], 1.0);
        assert!(telem[1] > 0.0 && telem[2] > 0.0 && telem[3] > 0.0);
        assert_eq!(&telem[4..], &[0.0, 0.0, 0.0, 0.0]);

        // Release note while sustain pedal holds, then test oversized frame clamping & panic.
        piano_note_off(ctx, 69, 0, 0);
        piano_render(ctx, 9_999);
        let clamped_l = unsafe { slice::from_raw_parts(l_ptr, MAX_BLOCK) };
        assert!(clamped_l.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
        let telem = unsafe { slice::from_raw_parts(t_ptr, TELEMETRY_LEN) };
        assert_eq!(telem[0], 1.0);

        piano_all_notes_off(ctx);
        for _ in 0..200 {
            piano_render(ctx, 128);
        }
        let telem = unsafe { slice::from_raw_parts(t_ptr, TELEMETRY_LEN) };
        assert_eq!(telem[0], 0.0);

        piano_destroy(ctx);
    }

    #[test]
    fn c_abi_handles_null_pointers_safely() {
        piano_note_on(ptr::null_mut(), 69, u16::MAX, 0);
        piano_note_off(ptr::null_mut(), 69, 0, 0);
        piano_pedal(ptr::null_mut(), 0, u16::MAX, 0);
        piano_all_notes_off(ptr::null_mut());
        piano_render(ptr::null_mut(), 128);
        assert!(piano_out_l_ptr(ptr::null()).is_null());
        assert!(piano_out_r_ptr(ptr::null()).is_null());
        assert!(piano_telemetry_ptr(ptr::null()).is_null());
        piano_destroy(ptr::null_mut());
    }

    #[test]
    fn c_abi_voicing_and_space_safety() {
        let ctx = piano_init(48_000.0);
        assert!(!ctx.is_null());

        // Valid voicing update (including A4 = 442.0 Hz)
        assert_eq!(piano_set_voicing(ctx, 0.8, 1.5, 1.2, 0.95, 442.0), 0);

        // Non-finite and extreme values clamped safely without panicking
        assert_eq!(
            piano_set_voicing(ctx, f32::NAN, f32::INFINITY, -10.0, 50.0, f32::NAN),
            0
        );

        // Valid space update
        assert_eq!(piano_set_space(ctx, 0, 2), 0); // Player, Open
        assert_eq!(piano_set_space(ctx, 3, 0), 0); // BinauralHrtf, Closed

        // Invalid discriminants rejected with -2
        assert_eq!(piano_set_space(ctx, 4, 0), -2);
        assert_eq!(piano_set_space(ctx, 0, 3), -2);
        assert_eq!(piano_set_space(ctx, 999, 999), -2);

        // Null safety
        assert_eq!(
            piano_set_voicing(core::ptr::null_mut(), 0.5, 1.0, 1.0, 1.0, 440.0),
            -1
        );
        assert_eq!(piano_set_space(core::ptr::null_mut(), 0, 0), -1);

        piano_destroy(ctx);
    }

    #[test]
    fn c_abi_prep_worker_transfers_space_and_voicing_without_audio_stall() {
        let ctx = piano_init(48_000.0);
        let prep = piano_prep_init(48_000.0);
        assert!(!ctx.is_null() && !prep.is_null());

        piano_note_on(ctx, 60, 50_000, 0);
        piano_render(ctx, 128);

        let sp_len = piano_prep_space(prep, 2, 1);
        assert!(sp_len > 10_000);
        let sp_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), sp_len) };
        let st_sp = piano_staging_ptr(ctx, sp_len);
        assert!(!st_sp.is_null());
        unsafe { slice::from_raw_parts_mut(st_sp, sp_len) }.copy_from_slice(sp_src);
        assert_eq!(piano_apply_space_blob(ctx, 2, 1, sp_len), 0);

        let vc_len = piano_prep_voicing(prep, 0.72, 1.8, 1.3, 0.94, 442.0);
        assert!(vc_len > 10_000);
        let vc_src = unsafe { slice::from_raw_parts(piano_prep_out_ptr(prep), vc_len) };
        let st_vc = piano_staging_ptr(ctx, vc_len);
        assert!(!st_vc.is_null());
        unsafe { slice::from_raw_parts_mut(st_vc, vc_len) }.copy_from_slice(vc_src);
        assert_eq!(
            piano_apply_voicing_blob(ctx, 0.72, 1.8, 1.3, 0.94, 442.0, vc_len),
            0
        );
        assert_eq!(piano_set_hammer_hardness(ctx, 0.4), 0);

        piano_render(ctx, 128);
        let out_l = unsafe { slice::from_raw_parts(piano_out_l_ptr(ctx), 128) };
        assert!(out_l.iter().all(|x| x.is_finite()) && out_l.iter().any(|&x| x.abs() > 1e-4));

        assert_eq!(piano_apply_space_blob(ctx, 2, 1, 0), -3);
        assert_eq!(
            piano_apply_voicing_blob(ctx, 0.72, 1.8, 1.3, 0.94, 442.0, 0),
            -3
        );
        piano_prep_destroy(prep);
        piano_destroy(ctx);
    }
}
