//! `wasm32` SIMD128 (`core::arch::wasm32::v128`) backend for `F32x4`.

use core::arch::wasm32::*;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct Inner(v128);

#[repr(C, align(16))]
struct Aligned4([f32; 4]);

#[inline(always)]
pub(crate) fn splat(value: f32) -> Inner {
    Inner(f32x4_splat(value))
}

#[inline(always)]
pub(crate) fn from_array(lanes: [f32; 4]) -> Inner {
    Inner(f32x4(lanes[0], lanes[1], lanes[2], lanes[3]))
}

#[inline(always)]
pub(crate) fn to_array(v: Inner) -> [f32; 4] {
    let mut out = Aligned4([0.0; 4]);
    // SAFETY: `out` is 16-byte aligned and holds 16 bytes.
    unsafe { v128_store(out.0.as_mut_ptr().cast::<v128>(), v.0) };
    out.0
}

#[inline(always)]
pub(crate) unsafe fn load_unaligned(ptr: *const f32) -> Inner {
    // SAFETY: `v128_load` supports unaligned pointers; caller ensures 4 valid `f32`s.
    #[allow(clippy::cast_ptr_alignment)]
    Inner(unsafe { v128_load(ptr.cast::<v128>()) })
}

#[inline(always)]
pub(crate) unsafe fn store_unaligned(v: Inner, ptr: *mut f32) {
    // SAFETY: `v128_store` supports unaligned pointers; caller ensures 4 valid `f32`s.
    #[allow(clippy::cast_ptr_alignment)]
    unsafe {
        v128_store(ptr.cast::<v128>(), v.0);
    }
}

#[inline(always)]
pub(crate) unsafe fn load_aligned(ptr: *const f32) -> Inner {
    debug_assert_eq!((ptr as usize) & 15, 0);
    // SAFETY: Caller guarantees 16-byte alignment and 4 valid `f32`s.
    Inner(unsafe { v128_load(ptr.cast::<v128>()) })
}

#[inline(always)]
pub(crate) unsafe fn store_aligned(v: Inner, ptr: *mut f32) {
    debug_assert_eq!((ptr as usize) & 15, 0);
    // SAFETY: Caller guarantees 16-byte alignment and 4 valid `f32`s.
    unsafe { v128_store(ptr.cast::<v128>(), v.0) };
}

#[inline(always)]
pub(crate) fn add(a: Inner, b: Inner) -> Inner {
    Inner(f32x4_add(a.0, b.0))
}

#[inline(always)]
pub(crate) fn sub(a: Inner, b: Inner) -> Inner {
    Inner(f32x4_sub(a.0, b.0))
}

#[inline(always)]
pub(crate) fn mul(a: Inner, b: Inner) -> Inner {
    Inner(f32x4_mul(a.0, b.0))
}

#[inline(always)]
pub(crate) fn div(a: Inner, b: Inner) -> Inner {
    Inner(f32x4_div(a.0, b.0))
}

#[inline(always)]
pub(crate) fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
    Inner(f32x4_add(f32x4_mul(a.0, b.0), c.0))
}

#[inline(always)]
pub(crate) fn min(a: Inner, b: Inner) -> Inner {
    Inner(f32x4_min(a.0, b.0))
}

#[inline(always)]
pub(crate) fn max(a: Inner, b: Inner) -> Inner {
    Inner(f32x4_max(a.0, b.0))
}

#[inline(always)]
pub(crate) fn abs(a: Inner) -> Inner {
    Inner(f32x4_abs(a.0))
}

#[inline(always)]
pub(crate) fn sqrt(a: Inner) -> Inner {
    Inner(f32x4_sqrt(a.0))
}

#[inline(always)]
pub(crate) fn reduce_sum(a: Inner) -> f32 {
    let hi = i32x4_shuffle::<2, 3, 2, 3>(a.0, a.0);
    let sum2 = f32x4_add(a.0, hi);
    let shuf = i32x4_shuffle::<1, 1, 1, 1>(sum2, sum2);
    let sum4 = f32x4_add(sum2, shuf);
    f32x4_extract_lane::<0>(sum4)
}
