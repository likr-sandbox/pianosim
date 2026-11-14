//! AudioWorklet WebAssembly bindings and linear-memory SPSC IPC queues.

pub mod exports;
pub mod ring_ipc;

pub use exports::{
    piano_all_notes_off, piano_destroy, piano_init, piano_note_off, piano_note_on, piano_out_l_ptr,
    piano_out_r_ptr, piano_pedal, piano_render, piano_telemetry_ptr,
};
pub use ring_ipc::{WasmContext, MAX_BLOCK, TELEMETRY_LEN};

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
}
