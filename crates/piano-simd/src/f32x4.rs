//! Portable 4-lane SIMD `f32` abstraction (`F32x4`) with zero-cost native
//! mappings for `wasm32` SIMD128, `x86_64` SSE/FMA, `aarch64` NEON, and a
//! deterministic scalar fallback.

use core::fmt;
use core::ops::{Add, Div, Mul, Sub};

#[path = "f32x4_scalar.rs"]
pub(crate) mod scalar;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[path = "f32x4_wasm.rs"]
mod imp;

#[cfg(target_arch = "x86_64")]
#[path = "f32x4_x86.rs"]
mod imp;

#[cfg(target_arch = "aarch64")]
#[path = "f32x4_neon.rs"]
mod imp;

#[cfg(not(any(
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64",
    target_arch = "aarch64"
)))]
mod imp {
    use super::scalar::ScalarF32x4;

    pub(crate) type Inner = ScalarF32x4;

    #[inline(always)]
    pub(crate) fn splat(value: f32) -> Inner {
        ScalarF32x4::splat(value)
    }
    #[inline(always)]
    pub(crate) fn from_array(lanes: [f32; 4]) -> Inner {
        ScalarF32x4::from_array(lanes)
    }
    #[inline(always)]
    pub(crate) fn to_array(v: Inner) -> [f32; 4] {
        v.to_array()
    }
    #[inline(always)]
    pub(crate) unsafe fn load_unaligned(ptr: *const f32) -> Inner {
        unsafe { ScalarF32x4::load_unaligned(ptr) }
    }
    #[inline(always)]
    pub(crate) unsafe fn store_unaligned(v: Inner, ptr: *mut f32) {
        unsafe { v.store_unaligned(ptr) };
    }
    #[inline(always)]
    pub(crate) unsafe fn load_aligned(ptr: *const f32) -> Inner {
        unsafe { ScalarF32x4::load_aligned(ptr) }
    }
    #[inline(always)]
    pub(crate) unsafe fn store_aligned(v: Inner, ptr: *mut f32) {
        unsafe { v.store_aligned(ptr) };
    }
    #[inline(always)]
    pub(crate) fn add(a: Inner, b: Inner) -> Inner {
        a.add(b)
    }
    #[inline(always)]
    pub(crate) fn sub(a: Inner, b: Inner) -> Inner {
        a.sub(b)
    }
    #[inline(always)]
    pub(crate) fn mul(a: Inner, b: Inner) -> Inner {
        a.mul(b)
    }
    #[inline(always)]
    pub(crate) fn div(a: Inner, b: Inner) -> Inner {
        a.div(b)
    }
    #[inline(always)]
    pub(crate) fn mul_add(a: Inner, b: Inner, c: Inner) -> Inner {
        a.mul_add(b, c)
    }
    #[inline(always)]
    pub(crate) fn min(a: Inner, b: Inner) -> Inner {
        a.min(b)
    }
    #[inline(always)]
    pub(crate) fn max(a: Inner, b: Inner) -> Inner {
        a.max(b)
    }
    #[inline(always)]
    pub(crate) fn abs(a: Inner) -> Inner {
        a.abs()
    }
    #[inline(always)]
    pub(crate) fn sqrt(a: Inner) -> Inner {
        a.sqrt()
    }
    #[inline(always)]
    pub(crate) fn reduce_sum(a: Inner) -> f32 {
        a.reduce_sum()
    }
}

/// Portable 4-lane `f32` SIMD vector (`16` bytes, `16`-byte aligned).
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct F32x4(imp::Inner);

impl F32x4 {
    /// Number of `f32` lanes in [`F32x4`].
    pub const LANES: usize = 4;

    /// Broadcasts `value` into all four lanes.
    #[must_use]
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self(imp::splat(value))
    }

    /// Returns a vector with all lanes set to `0.0`.
    #[must_use]
    #[inline(always)]
    pub fn zero() -> Self {
        Self::splat(0.0)
    }

    /// Constructs a vector from `[lane0, lane1, lane2, lane3]`.
    #[must_use]
    #[inline(always)]
    pub fn from_array(lanes: [f32; 4]) -> Self {
        Self(imp::from_array(lanes))
    }

    /// Extracts the four lanes as `[lane0, lane1, lane2, lane3]`.
    #[must_use]
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        imp::to_array(self.0)
    }

    /// Loads four lanes from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for reading 4 contiguous `f32` values.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_unaligned(ptr: *const f32) -> Self {
        Self(unsafe { imp::load_unaligned(ptr) })
    }

    /// Stores four lanes to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be non-null and valid for writing 4 contiguous `f32` values.
    #[inline(always)]
    pub unsafe fn store_unaligned(self, ptr: *mut f32) {
        unsafe { imp::store_unaligned(self.0, ptr) };
    }

    /// Loads four lanes from a 16-byte aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be 16-byte aligned and valid for reading 4 contiguous `f32` values.
    #[must_use]
    #[inline(always)]
    pub unsafe fn load_aligned(ptr: *const f32) -> Self {
        Self(unsafe { imp::load_aligned(ptr) })
    }

    /// Stores four lanes to a 16-byte aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be 16-byte aligned and valid for writing 4 contiguous `f32` values.
    #[inline(always)]
    pub unsafe fn store_aligned(self, ptr: *mut f32) {
        unsafe { imp::store_aligned(self.0, ptr) };
    }

    /// Safe convenience loader from a slice (`slice.len() >= 4`).
    #[must_use]
    #[inline(always)]
    pub fn from_slice(slice: &[f32]) -> Self {
        assert!(slice.len() >= 4, "F32x4::from_slice requires >= 4 elements");
        // SAFETY: `slice` contains at least 4 contiguous `f32` elements.
        unsafe { Self::load_unaligned(slice.as_ptr()) }
    }

    /// Safe convenience store into a slice (`slice.len() >= 4`).
    #[inline(always)]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        assert!(
            slice.len() >= 4,
            "F32x4::write_to_slice requires >= 4 elements"
        );
        // SAFETY: `slice` contains at least 4 contiguous `f32` elements.
        unsafe { self.store_unaligned(slice.as_mut_ptr()) };
    }

    /// Lane-wise addition `self + rhs`.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self(imp::add(self.0, rhs.0))
    }

    /// Lane-wise subtraction `self - rhs`.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        Self(imp::sub(self.0, rhs.0))
    }

    /// Lane-wise multiplication `self * rhs`.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self(imp::mul(self.0, rhs.0))
    }

    /// Lane-wise division `self / rhs`.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    #[inline(always)]
    pub fn div(self, rhs: Self) -> Self {
        Self(imp::div(self.0, rhs.0))
    }

    /// Lane-wise fused multiply-add `self * mul + add` (uses FMA where available).
    #[must_use]
    #[inline(always)]
    pub fn mul_add(self, mul: Self, add: Self) -> Self {
        Self(imp::mul_add(self.0, mul.0, add.0))
    }

    /// Lane-wise minimum.
    #[must_use]
    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        Self(imp::min(self.0, rhs.0))
    }

    /// Lane-wise maximum.
    #[must_use]
    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        Self(imp::max(self.0, rhs.0))
    }

    /// Lane-wise absolute value.
    #[must_use]
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(imp::abs(self.0))
    }

    /// Lane-wise square root.
    #[must_use]
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(imp::sqrt(self.0))
    }

    /// Horizontal sum of all four lanes.
    #[must_use]
    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        imp::reduce_sum(self.0)
    }
}

impl Default for F32x4 {
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for F32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

impl fmt::Debug for F32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("F32x4").field(&self.to_array()).finish()
    }
}

impl Add for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        F32x4::add(self, rhs)
    }
}

impl Sub for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        F32x4::sub(self, rhs)
    }
}

impl Mul for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        F32x4::mul(self, rhs)
    }
}

impl Div for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        F32x4::div(self, rhs)
    }
}
