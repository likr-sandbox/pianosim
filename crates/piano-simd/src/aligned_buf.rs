//! 64-byte aligned `f32` buffer allocated strictly at construction time.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut};
use core::slice;

use crate::f32x4::F32x4;

const LANES_PER_CACHE_LINE: usize = 16;

const _: () = {
    assert!(size_of::<CacheLineF32>() == 64);
    assert!(align_of::<CacheLineF32>() == 64);
    assert!(size_of::<F32x4>() == 16);
    assert!(align_of::<F32x4>() == 16);
};

/// One 64-byte cache line holding 16 contiguous `f32` values.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug)]
struct CacheLineF32([f32; LANES_PER_CACHE_LINE]);

impl CacheLineF32 {
    const ZERO: Self = Self([0.0; LANES_PER_CACHE_LINE]);
}

/// Heap buffer guaranteed to start on a 64-byte boundary (`ptr as usize % 64 == 0`).
///
/// Memory is allocated exclusively during construction (`new`, `zeroed`, `filled`,
/// `from_slice`) and padded up to a multiple of 16 `f32` values (4 [`F32x4`] vectors).
#[derive(Clone, Debug)]
pub struct AlignedBuf {
    blocks: Box<[CacheLineF32]>,
    len: usize,
}

impl AlignedBuf {
    /// Allocates a zero-initialized buffer of `len` `f32` elements on a 64-byte boundary.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::zeroed(len)
    }

    /// Allocates a zero-initialized buffer of `len` `f32` elements on a 64-byte boundary.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let block_count = len.div_ceil(LANES_PER_CACHE_LINE);
        let blocks = if block_count == 0 {
            Vec::new().into_boxed_slice()
        } else {
            vec![CacheLineF32::ZERO; block_count].into_boxed_slice()
        };
        Self { blocks, len }
    }

    /// Allocates a 64-byte aligned buffer of `len` elements initialized to `value`,
    /// keeping trailing SIMD padding lanes zeroed.
    #[must_use]
    pub fn filled(len: usize, value: f32) -> Self {
        let mut buf = Self::zeroed(len);
        buf.as_mut_slice().fill(value);
        buf
    }

    /// Allocates a 64-byte aligned buffer initialized from `src`, zero-padding any tail.
    #[must_use]
    pub fn from_slice(src: &[f32]) -> Self {
        let mut buf = Self::zeroed(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    /// Logical number of `f32` elements requested at construction.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if `len() == 0`.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocated capacity in `f32` elements (multiple of 16 and 4).
    #[must_use]
    #[inline]
    pub fn padded_len(&self) -> usize {
        self.blocks.len() * LANES_PER_CACHE_LINE
    }

    /// Number of complete 4-lane [`F32x4`] vectors within `len()`.
    #[must_use]
    #[inline]
    pub const fn lane_vec_len(&self) -> usize {
        self.len / F32x4::LANES
    }

    /// Number of 4-lane [`F32x4`] vectors across the padded 64-byte blocks.
    #[must_use]
    #[inline]
    pub fn padded_vec_len(&self) -> usize {
        self.padded_len() / F32x4::LANES
    }

    /// Returns a 64-byte aligned pointer to the start of the buffer.
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.blocks.as_ptr().cast::<f32>()
    }

    /// Returns a mutable 64-byte aligned pointer to the start of the buffer.
    #[must_use]
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.blocks.as_mut_ptr().cast::<f32>()
    }

    /// Returns the logical `&[f32]` view (`len()` elements). Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `self.len <= self.padded_len()` and `CacheLineF32` is tightly packed `f32`s.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Returns the logical `&mut [f32]` view (`len()` elements). Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: `self.len <= self.padded_len()` and `CacheLineF32` is tightly packed `f32`s.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Returns the full padded `&[f32]` view (`padded_len()` elements). Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_padded_slice(&self) -> &[f32] {
        // SAFETY: All `padded_len()` elements are initialized within `self.blocks`.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.padded_len()) }
    }

    /// Returns the full padded `&mut [f32]` view (`padded_len()` elements). Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_padded_mut_slice(&mut self) -> &mut [f32] {
        let cap = self.padded_len();
        // SAFETY: All `cap` elements are initialized within `self.blocks`.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), cap) }
    }

    /// Zeroes all logical and padding elements without reallocating.
    #[inline]
    pub fn clear(&mut self) {
        self.blocks.fill(CacheLineF32::ZERO);
    }

    /// Fills logical elements `[0..len()]` with `value` and zeroes tail padding.
    #[inline]
    pub fn fill(&mut self, value: f32) {
        self.clear();
        self.as_mut_slice().fill(value);
    }

    /// Views the first `len() / 4` aligned chunks as `&[F32x4]`. Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_f32x4_slice(&self) -> &[F32x4] {
        // SAFETY: Base pointer is 64B aligned (>= 16B) and `lane_vec_len() * 4 <= padded_len()`.
        unsafe { slice::from_raw_parts(self.blocks.as_ptr().cast::<F32x4>(), self.lane_vec_len()) }
    }

    /// Views the first `len() / 4` aligned chunks as `&mut [F32x4]`. Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_f32x4_slice_mut(&mut self) -> &mut [F32x4] {
        let count = self.lane_vec_len();
        // SAFETY: Base pointer is 64B aligned (>= 16B) and `count * 4 <= padded_len()`.
        unsafe { slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<F32x4>(), count) }
    }

    /// Views all padded 4-lane vectors (`padded_vec_len()` elements). Zero-allocation.
    #[must_use]
    #[inline]
    pub fn as_padded_f32x4_slice(&self) -> &[F32x4] {
        // SAFETY: Base pointer is 64B aligned and `padded_vec_len() * 4 == padded_len()`.
        unsafe {
            slice::from_raw_parts(self.blocks.as_ptr().cast::<F32x4>(), self.padded_vec_len())
        }
    }

    /// Views all padded 4-lane vectors mutably (`padded_vec_len()` elements).
    #[must_use]
    #[inline]
    pub fn as_padded_f32x4_slice_mut(&mut self) -> &mut [F32x4] {
        let count = self.padded_vec_len();
        // SAFETY: Base pointer is 64B aligned and `count * 4 == padded_len()`.
        unsafe { slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<F32x4>(), count) }
    }

    /// Reads 4-lane vector `vec_idx` from the padded SIMD slice.
    #[must_use]
    #[inline]
    pub fn load_f32x4(&self, vec_idx: usize) -> F32x4 {
        self.as_padded_f32x4_slice()[vec_idx]
    }

    /// Writes 4-lane vector `vec_idx` into the padded SIMD slice.
    #[inline]
    pub fn store_f32x4(&mut self, vec_idx: usize, value: F32x4) {
        self.as_padded_f32x4_slice_mut()[vec_idx] = value;
    }
}

impl Deref for AlignedBuf {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuf {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl AsRef<[f32]> for AlignedBuf {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl AsMut<[f32]> for AlignedBuf {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}
