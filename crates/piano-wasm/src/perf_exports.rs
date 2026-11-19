//! M3 C-ABI exports for `PerfProfile` initialization, zero-alloc audio-thread
//! runtime preset/knob switching, and Prep Worker profile configuration.

use core::ptr;

use piano_engine::{PerfPreset, PerfProfile};

use crate::prep_ipc::PrepContext;
use crate::ring_ipc::WasmContext;

#[inline]
fn decode_preset(preset_id: u32) -> Option<PerfPreset> {
    u8::try_from(preset_id).ok().and_then(PerfPreset::from_u8)
}

#[inline]
fn apply_runtime_knob(profile: &mut PerfProfile, knob_id: u32, value: f32) -> bool {
    match knob_id {
        0 => profile.energy_horizon_db = value,
        1 => profile.sav_dormancy_db = value,
        2 => {
            profile.sav_max_key_idx = if value.is_finite() {
                value.round().clamp(0.0, 52.0) as u8
            } else {
                52
            };
        }
        3 => profile.passive_res_max_hz = value,
        4 => {
            profile.hall_fdn_lines = if value.is_finite() {
                value.round().clamp(4.0, 16.0) as u8
            } else {
                8
            };
        }
        5 => {
            profile.damper_control_decim = if value.is_finite() {
                value.round().clamp(1.0, 16.0) as u8
            } else {
                1
            };
        }
        6 => profile.hertz_nonlinear = value >= 0.5,
        7 => {
            profile.active_bridge_poles = if value.is_finite() {
                value.round().clamp(4.0, 10.0) as u8
            } else {
                10
            };
        }
        8 => profile.shank_flex_enabled = value >= 0.5,
        9 => profile.polynomial_adaa = value >= 0.5,
        18 => {
            profile.low_order_energy_horizon_db = if value.is_finite() {
                value.clamp(-120.0, -40.0)
            } else {
                -120.0
            };
        }
        _ => return false,
    }
    true
}

#[inline]
fn apply_any_knob(profile: &mut PerfProfile, knob_id: u32, value: f32) -> bool {
    if knob_id <= 9 || knob_id == 18 {
        return apply_runtime_knob(profile, knob_id, value);
    }
    match knob_id {
        10 => profile.horiz_max_hz = value,
        11 => profile.dead_mode_prune_db = value,
        12 => profile.bass_phantom_sparse = value >= 0.5,
        13 => profile.ir_duration_secs = value,
        14 => {
            profile.spatial_tail_split_tap = if value.is_finite() {
                value.round().clamp(512.0, f32::from(u16::MAX)) as u16
            } else {
                u16::MAX
            };
        }
        15 => {
            profile.fir_head_taps = if value.is_finite() {
                value.round().clamp(16.0, 64.0) as u8
            } else {
                64
            };
        }
        16 => profile.fir_block_fft = value >= 0.5,
        17 => {
            profile.action_oversample = if value.is_finite() {
                value.round().clamp(2.0, 8.0) as u8
            } else {
                8
            };
        }
        _ => return false,
    }
    true
}

/// Allocates and initializes a [`WasmContext`] with `preset_id` (`0=Full`, `1=Balanced`,
/// `2=Realtime`), preallocating `staging` to full capacity for safe runtime/blob upgrades.
///
/// Returns null if `preset_id > 2`.
#[no_mangle]
pub extern "C" fn piano_init_with_profile(sample_rate: f32, preset_id: u32) -> *mut WasmContext {
    let Some(preset) = decode_preset(preset_id) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(WasmContext::new_with_profile(
        sample_rate,
        PerfProfile::from_preset(preset),
    )))
}

/// Applies runtime-tunable `PerfProfile` knobs for `preset_id` (`0..=2`) with zero allocations.
///
/// Returns `0` if full runtime parity was achieved, `1` if structural rebuild / `SpaceBlob`
/// prep is recommended, `-1` if `ctx` is null, or `-2` if `preset_id > 2`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_perf_preset(ctx: *mut WasmContext, preset_id: u32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(preset) = decode_preset(preset_id) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init*`.
    let ctx = unsafe { &mut *ctx };
    if ctx.set_perf_profile_runtime(PerfProfile::from_preset(preset)) {
        0
    } else {
        1
    }
}

/// Updates a single runtime-tunable `PerfProfile` knob (`knob_id` in `0..=9` or `18`) with zero allocations.
///
/// Returns `0` if applied cleanly, `1` if `polynomial_adaa = false` requires `PGridLuts` allocation,
/// `-1` if `ctx` is null, or `-2` if `knob_id` is not runtime-tunable.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_perf_knob(ctx: *mut WasmContext, knob_id: u32, value: f32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init*`.
    let ctx = unsafe { &mut *ctx };
    let mut next = ctx.perf_profile();
    if !apply_runtime_knob(&mut next, knob_id, value) {
        return -2;
    }
    if ctx.set_perf_profile_runtime(next) {
        0
    } else {
        1
    }
}

/// Updates both `SpacePrepContext` and `VoicingPrepContext` inside [`PrepContext`] to `preset_id`.
///
/// Returns `0` on success, `-1` if `prep` is null, or `-2` if `preset_id > 2`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_set_perf_preset(prep: *mut PrepContext, preset_id: u32) -> i32 {
    if prep.is_null() {
        return -1;
    }
    let Some(preset) = decode_preset(preset_id) else {
        return -2;
    };
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    let prep = unsafe { &mut *prep };
    prep.set_perf_profile(&PerfProfile::from_preset(preset));
    0
}

/// Directly configures Stage 4 `SpacePrepContext` profile parameters on [`PrepContext`].
///
/// Returns `0` on success or `-1` if `prep` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_set_stage4_profile(
    prep: *mut PrepContext,
    ir_duration_secs: f32,
    spatial_tail_split_tap: u32,
    fir_head_taps: u32,
    fir_block_fft: u32,
) -> i32 {
    if prep.is_null() {
        return -1;
    }
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    let prep = unsafe { &mut *prep };
    prep.set_stage4_profile(
        ir_duration_secs,
        spatial_tail_split_tap.min(u32::from(u16::MAX)) as u16,
        fir_head_taps.min(255) as u8,
        fir_block_fft != 0,
    );
    0
}

/// Allocates a mutable [`PerfProfile`] builder initialized from `preset_id` (`0..=2`).
///
/// Returns null if `preset_id > 2`. Caller must free with [`piano_perf_builder_destroy`].
#[no_mangle]
pub extern "C" fn piano_perf_builder_new(preset_id: u32) -> *mut PerfProfile {
    let Some(preset) = decode_preset(preset_id) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(PerfProfile::from_preset(preset)))
}

/// Sets any of the 19 [`PerfProfile`] knobs (`knob_id` in `0..=18`) on `builder`.
///
/// Returns `0` on success, `-1` if `builder` is null, or `-2` if `knob_id > 18`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_perf_builder_set(
    builder: *mut PerfProfile,
    knob_id: u32,
    value: f32,
) -> i32 {
    if builder.is_null() {
        return -1;
    }
    // SAFETY: `builder` is a valid non-null pointer from `piano_perf_builder_new`.
    let profile = unsafe { &mut *builder };
    if apply_any_knob(profile, knob_id, value) {
        0
    } else {
        -2
    }
}

/// Frees a [`PerfProfile`] builder allocated by [`piano_perf_builder_new`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_perf_builder_destroy(builder: *mut PerfProfile) {
    if !builder.is_null() {
        // SAFETY: `builder` was created by `Box::into_raw` in `piano_perf_builder_new`.
        unsafe {
            drop(Box::from_raw(builder));
        }
    }
}

/// Allocates and initializes a [`WasmContext`] from a custom `PerfProfile` builder.
///
/// Returns null if `builder` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_init_with_builder(
    sample_rate: f32,
    builder: *const PerfProfile,
) -> *mut WasmContext {
    if builder.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: `builder` is a valid non-null pointer from `piano_perf_builder_new`.
    let profile = unsafe { (*builder).sanitized() };
    Box::into_raw(Box::new(WasmContext::new_with_profile(
        sample_rate,
        profile,
    )))
}

/// Configures [`PrepContext`] (`SpacePrepContext` + `VoicingPrepContext`) from a `PerfProfile` builder.
///
/// Returns `0` on success or `-1` if `prep` or `builder` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_set_perf_builder(
    prep: *mut PrepContext,
    builder: *const PerfProfile,
) -> i32 {
    if prep.is_null() || builder.is_null() {
        return -1;
    }
    // SAFETY: `prep` and `builder` are valid non-null pointers.
    let prep = unsafe { &mut *prep };
    let profile = unsafe { (*builder).sanitized() };
    prep.set_perf_profile(&profile);
    0
}
