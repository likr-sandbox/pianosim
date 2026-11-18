//! Wave 4 C-ABI exports for concert pitch (`392.0..=466.2` Hz), classical
//! 12-tone temperaments (`0..=3`), deterministic aging `condition` (`0.0..=1.0`),
//! and extended `voicingBlob` precomputation/installation.

use piano_spec::{Temperament, VoicingSpec};

use crate::prep_ipc::PrepContext;
use crate::ring_ipc::WasmContext;

impl WasmContext {
    /// Updates all global voicing targets (`concert_pitch_hz`, `temperament`, `condition`)
    /// for smooth real-time interpolation.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn set_voicing_full(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        concert_pitch_hz: f32,
        temperament: Temperament,
        condition: f32,
    ) {
        self.engine.set_voicing_full(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            concert_pitch_hz,
            temperament,
            condition,
        );
    }

    /// Sets target concert pitch `A4` (`[392.0, 466.2]` Hz) with 20 ms smoothing.
    #[inline]
    pub fn set_concert_pitch(&mut self, concert_pitch_hz: f32) {
        self.engine.set_concert_pitch_hz(concert_pitch_hz);
    }

    /// Sets the active classical 12-tone [`Temperament`].
    #[inline]
    pub fn set_temperament(&mut self, temperament: Temperament) {
        self.engine.set_temperament(temperament);
    }

    /// Sets target deterministic aging `condition` (`[0.0, 1.0]`) with 20 ms smoothing.
    #[inline]
    pub fn set_condition(&mut self, condition: f32) {
        self.engine.set_condition(condition);
    }

    /// Installs an off-thread precomputed Stage 2/3 tuning table blob from staging memory
    /// including `concert_pitch_hz`, `temperament`, and `condition`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_voicing_blob_full(
        &mut self,
        hammer_hardness: f32,
        unison_detune: f32,
        stretch: f32,
        length_scale: f32,
        concert_pitch_hz: f32,
        temperament: Temperament,
        condition: f32,
        len_f32: usize,
    ) -> bool {
        if len_f32 == 0 || len_f32 > self.staging.len() {
            return false;
        }
        let spec = VoicingSpec::from_globals_full(
            hammer_hardness,
            unison_detune,
            stretch,
            length_scale,
            concert_pitch_hz,
            temperament,
            condition,
        );
        self.engine
            .apply_precomputed_voicing(spec, &self.staging[..len_f32])
    }
}

/// Sets all global voicing parameters (`concert_pitch_hz` in `[392.0, 466.2]`, `temperament_id` in `0..=3`,
/// `condition` in `[0.0, 1.0]`) with 20 ms exponential smoothing.
///
/// Returns `0` on success, `-1` if `ctx` is null, or `-2` if `temperament_id` is invalid.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_voicing_full(
    ctx: *mut WasmContext,
    hammer_hardness: f32,
    unison_detune: f32,
    stretch: f32,
    length_scale: f32,
    concert_pitch_hz: f32,
    temperament_id: u32,
    condition: f32,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(temp) = Temperament::from_u32(temperament_id) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_voicing_full(
        hammer_hardness,
        unison_detune,
        stretch,
        length_scale,
        concert_pitch_hz,
        temp,
        condition,
    );
    0
}

/// Sets target concert pitch `A4` (`[392.0, 466.2]` Hz; `NaN -> 440.0`) with 20 ms smoothing.
///
/// Returns `0` on success, or `-1` if `ctx` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_concert_pitch(ctx: *mut WasmContext, concert_pitch_hz: f32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_concert_pitch(concert_pitch_hz);
    0
}

/// Sets the active classical 12-tone temperament (`0` = Equal, `1` = Werckmeister III,
/// `2` = Kirnberger III, `3` = Vallotti).
///
/// Returns `0` on success, `-1` if `ctx` is null, or `-2` if `temperament_id > 3`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_temperament(ctx: *mut WasmContext, temperament_id: u32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(temp) = Temperament::from_u32(temperament_id) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_temperament(temp);
    0
}

/// Sets target deterministic aging `condition` (`[0.0, 1.0]`; `NaN -> 0.0`) with 20 ms smoothing.
///
/// Returns `0` on success, or `-1` if `ctx` is null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_set_condition(ctx: *mut WasmContext, condition: f32) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    ctx.set_condition(condition);
    0
}

/// Precomputes Stage 2/3 tuning tables off-thread including `concert_pitch_hz` (`[392.0, 466.2]`),
/// `temperament_id` (`0..=3`), and `condition` (`[0.0, 1.0]`).
///
/// Returns `f32` element length on success, or `0` if `prep` is null or `temperament_id > 3`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_prep_voicing_ex(
    prep: *mut PrepContext,
    hammer_hardness: f32,
    unison_detune: f32,
    stretch: f32,
    length_scale: f32,
    concert_pitch_hz: f32,
    temperament_id: u32,
    condition: f32,
) -> usize {
    if prep.is_null() {
        return 0;
    }
    let Some(temp) = Temperament::from_u32(temperament_id) else {
        return 0;
    };
    // SAFETY: `prep` is a valid non-null pointer from `piano_prep_init`.
    unsafe { &mut *prep }.prepare_voicing_full(
        hammer_hardness,
        unison_detune,
        stretch,
        length_scale,
        concert_pitch_hz,
        temp,
        condition,
    )
}

/// Installs an off-thread precomputed Stage 2/3 tuning blob from staging memory including
/// `concert_pitch_hz`, `temperament_id` (`0..=3`), and `condition` (`[0.0, 1.0]`).
///
/// Returns `0` on success, `-1` if `ctx` is null, `-2` if `temperament_id > 3`,
/// or `-3` on staging bounds / blob validation failure.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn piano_apply_voicing_blob_ex(
    ctx: *mut WasmContext,
    hammer_hardness: f32,
    unison_detune: f32,
    stretch: f32,
    length_scale: f32,
    concert_pitch_hz: f32,
    temperament_id: u32,
    condition: f32,
    len_f32: usize,
) -> i32 {
    if ctx.is_null() {
        return -1;
    }
    let Some(temp) = Temperament::from_u32(temperament_id) else {
        return -2;
    };
    // SAFETY: `ctx` is a valid non-null pointer from `piano_init`.
    let ctx = unsafe { &mut *ctx };
    if ctx.apply_voicing_blob_full(
        hammer_hardness,
        unison_detune,
        stretch,
        length_scale,
        concert_pitch_hz,
        temp,
        condition,
        len_f32,
    ) {
        0
    } else {
        -3
    }
}
