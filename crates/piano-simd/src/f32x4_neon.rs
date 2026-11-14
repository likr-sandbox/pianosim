//! `aarch64` NEON (`core::arch::aarch64::float32x4_t`) backend for `F32x4`.

use core::arch::aarch64::*;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct Inner(float32x4_t);

#[inline(always)]
pub(crate) fn splat(value: f32) -> Inner {
    Inner(unsafe { vdupq_n_f32(value) })
}

#[inline(always)]
pub(crate) fn from_array(lanes: [f32; 4]) -> Inner {
    Inner(unsafe { vld1q_f32(lanes.as_ptr()) })
}

#[inline(always)]
pub(crate) fn to_array(v: Inner) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    unsafe { vst1q_f32(out.as_mut_ptr(), v.0) };
    out
}

#[inline(always)]
pub(crate) unsafe fn load_unaligned(ptr: *const f32) -> Inner {
    // SAFETY: Caller guarantees `ptr` is valid for reading 4 `f32` values.
    Inner(unsafe { vld1q_f32(ptr) })
}

#[inline(always)]
pub(crate) unsafe fn store_unaligned(v: Inner, ptr: *mut f32) {
    // SAFETY: Caller guarantees `ptr` is valid for writing 4 `f32` values.
    unsafe { vst1q_f32(ptr, v.0) };
}

#[inline(always)]
pub(crate) unsafe fn load_aligned(ptr: *const f32) -> Inner {
    debug_assert_eq!((ptr as usize) & 15, 0);
    // SAFETY: Caller guarantees 16-byte alignment and validity for 4 `f32` values.
    Inner(unsafe { vld1q_f32(ptr) })
}

#[inline(always)]
pub(crate) unsafe fn store_aligned(v: Inner, ptr: *mut f32) {
    debug_assert_eq!((ptr as usize) & 15, 0);
    // SAFETY: Caller guarantees 16-byte alignment and validity for 4 `f32` values.
    unsafe { vst1q_f32(ptr, v.0) };
}

#[inline(always)]
pub(crate) fn add(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { vaddq_f32(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn sub(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { vsubq_f32(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn mul(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { vmulq_f32(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn div(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { vdivq_f32(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
    // `vfmaq_f32(acc, x, y)` computes `acc + x * y`.
    Inner(unsafe { vfmaq_f32(c.0, a.0, b.0) })
}

#[inline(always)]
pub(crate) fn min(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { vminq_f32(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn max(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { vmaxq_f32(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn abs(a: Inner) -> Inner {
    Inner(unsafe { vabsq_f32(a.0) })
}

#[inline(always)]
pub(crate) fn sqrt(a: Inner) -> Inner {
    Inner(unsafe { vsqrtq_f32(a.0) })
}

#[inline(always)]
pub(crate) fn reduce_sum(a: Inner) -> f32 {
    unsafe { vaddvq_f32(a.0) }
}
