//! `x86_64` SSE/FMA (`core::arch::x86_64::__m128`) backend for `F32x4`.

use core::arch::x86_64::*;

#[repr(transparent)]
#[derive(Copy, Clone)]
pub(crate) struct Inner(__m128);

#[inline(always)]
pub(crate) fn splat(value: f32) -> Inner {
    // SAFETY: SSE is mandatory on `x86_64`.
    Inner(unsafe { _mm_set1_ps(value) })
}

#[inline(always)]
pub(crate) fn from_array(lanes: [f32; 4]) -> Inner {
    // SAFETY: `lanes` holds 4 valid `f32` values and SSE is mandatory on `x86_64`.
    Inner(unsafe { _mm_loadu_ps(lanes.as_ptr()) })
}

#[inline(always)]
pub(crate) fn to_array(v: Inner) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    // SAFETY: `out` holds 4 valid `f32` values.
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), v.0) };
    out
}

#[inline(always)]
pub(crate) unsafe fn load_unaligned(ptr: *const f32) -> Inner {
    // SAFETY: Caller guarantees `ptr` is valid for reading 4 `f32` values.
    Inner(unsafe { _mm_loadu_ps(ptr) })
}

#[inline(always)]
pub(crate) unsafe fn store_unaligned(v: Inner, ptr: *mut f32) {
    // SAFETY: Caller guarantees `ptr` is valid for writing 4 `f32` values.
    unsafe { _mm_storeu_ps(ptr, v.0) };
}

#[inline(always)]
pub(crate) unsafe fn load_aligned(ptr: *const f32) -> Inner {
    debug_assert_eq!((ptr as usize) & 15, 0);
    // SAFETY: Caller guarantees 16-byte alignment and validity for 4 `f32` values.
    Inner(unsafe { _mm_load_ps(ptr) })
}

#[inline(always)]
pub(crate) unsafe fn store_aligned(v: Inner, ptr: *mut f32) {
    debug_assert_eq!((ptr as usize) & 15, 0);
    // SAFETY: Caller guarantees 16-byte alignment and validity for 4 `f32` values.
    unsafe { _mm_store_ps(ptr, v.0) };
}

#[inline(always)]
pub(crate) fn add(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { _mm_add_ps(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn sub(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { _mm_sub_ps(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn mul(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { _mm_mul_ps(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn div(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { _mm_div_ps(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
    #[cfg(target_feature = "fma")]
    {
        // SAFETY: Guarded by `cfg(target_feature = "fma")`.
        Inner(unsafe { _mm_fmadd_ps(a.0, b.0, c.0) })
    }
    #[cfg(not(target_feature = "fma"))]
    {
        Inner(unsafe { _mm_add_ps(_mm_mul_ps(a.0, b.0), c.0) })
    }
}

#[inline(always)]
pub(crate) fn min(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { _mm_min_ps(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn max(a: Inner, b: Inner) -> Inner {
    Inner(unsafe { _mm_max_ps(a.0, b.0) })
}

#[inline(always)]
pub(crate) fn abs(a: Inner) -> Inner {
    Inner(unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), a.0) })
}

#[inline(always)]
pub(crate) fn sqrt(a: Inner) -> Inner {
    Inner(unsafe { _mm_sqrt_ps(a.0) })
}

#[inline(always)]
pub(crate) fn reduce_sum(a: Inner) -> f32 {
    // SAFETY: Baseline SSE instruction sequence; matches `(a0+a2) + (a1+a3)`.
    unsafe {
        let hi = _mm_movehl_ps(a.0, a.0);
        let sum2 = _mm_add_ps(a.0, hi);
        let shuf = _mm_shuffle_ps(sum2, sum2, 0x55);
        _mm_cvtss_f32(_mm_add_ss(sum2, shuf))
    }
}
