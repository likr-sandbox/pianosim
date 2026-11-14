//! Scalar `[f32; 4]` 4-lane reference backend and fallback implementation.

use crate::math::{abs_f32, sqrt_f32};

/// Deterministic `[f32; 4]` 4-lane reference type (`16`-byte aligned).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub(crate) struct ScalarF32x4(pub(crate) [f32; 4]);

#[allow(dead_code)]
impl ScalarF32x4 {
    #[inline(always)]
    pub(crate) const fn splat(value: f32) -> Self {
        Self([value; 4])
    }

    #[inline(always)]
    pub(crate) const fn from_array(lanes: [f32; 4]) -> Self {
        Self(lanes)
    }

    #[inline(always)]
    pub(crate) const fn to_array(self) -> [f32; 4] {
        self.0
    }

    #[inline(always)]
    pub(crate) unsafe fn load_unaligned(ptr: *const f32) -> Self {
        // SAFETY: Caller guarantees `ptr` is valid for reading 4 `f32` values.
        Self(unsafe { ptr.cast::<[f32; 4]>().read_unaligned() })
    }

    #[inline(always)]
    pub(crate) unsafe fn store_unaligned(self, ptr: *mut f32) {
        // SAFETY: Caller guarantees `ptr` is valid for writing 4 `f32` values.
        unsafe { ptr.cast::<[f32; 4]>().write_unaligned(self.0) };
    }

    #[inline(always)]
    pub(crate) unsafe fn load_aligned(ptr: *const f32) -> Self {
        debug_assert_eq!((ptr as usize) & 15, 0);
        // SAFETY: Caller guarantees 16-byte alignment and validity for 4 `f32`s.
        Self(unsafe { *ptr.cast::<[f32; 4]>() })
    }

    #[inline(always)]
    pub(crate) unsafe fn store_aligned(self, ptr: *mut f32) {
        debug_assert_eq!((ptr as usize) & 15, 0);
        // SAFETY: Caller guarantees 16-byte alignment and validity for 4 `f32`s.
        unsafe { *ptr.cast::<[f32; 4]>() = self.0 };
    }

    #[inline(always)]
    pub(crate) fn add(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]])
    }

    #[inline(always)]
    pub(crate) fn sub(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]])
    }

    #[inline(always)]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
    }

    #[inline(always)]
    pub(crate) fn div(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]])
    }

    #[inline(always)]
    pub(crate) fn mul_add(self, mul: Self, add: Self) -> Self {
        let (a, b, c) = (self.0, mul.0, add.0);
        Self([
            a[0] * b[0] + c[0],
            a[1] * b[1] + c[1],
            a[2] * b[2] + c[2],
            a[3] * b[3] + c[3],
        ])
    }

    #[inline(always)]
    pub(crate) fn min(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ])
    }

    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([
            a[0].max(b[0]),
            a[1].max(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ])
    }

    #[inline(always)]
    pub(crate) fn abs(self) -> Self {
        let a = self.0;
        Self([abs_f32(a[0]), abs_f32(a[1]), abs_f32(a[2]), abs_f32(a[3])])
    }

    #[inline(always)]
    pub(crate) fn sqrt(self) -> Self {
        let a = self.0;
        Self([
            sqrt_f32(a[0]),
            sqrt_f32(a[1]),
            sqrt_f32(a[2]),
            sqrt_f32(a[3]),
        ])
    }

    #[inline(always)]
    pub(crate) fn reduce_sum(self) -> f32 {
        let a = self.0;
        (a[0] + a[2]) + (a[1] + a[3])
    }
}
