//! Zero-allocation-in-process power-of-two complex Radix-2 FFT and IFFT.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::f64::consts::TAU;
use core::ops::{Add, Mul, Sub};

use crate::aligned_buf::AlignedBuf;
use crate::f32x4::F32x4;
use crate::math::sin_cos_f64;

/// Single-precision complex number with C-compatible `[re, im]` layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// Zero complex number `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Constructs `re + i * im`.
    #[must_use]
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Constructs `mag * (cos(angle) + i * sin(angle))` at initialization time.
    #[must_use]
    #[inline]
    pub fn from_polar(mag: f32, angle_rad: f64) -> Self {
        let (s, c) = sin_cos_f64(angle_rad);
        Self {
            re: mag * (c as f32),
            im: mag * (s as f32),
        }
    }

    /// Complex conjugate `re - i * im`.
    #[must_use]
    #[inline(always)]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude `re^2 + im^2`.
    #[must_use]
    #[inline(always)]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Scales both components by `s`.
    #[must_use]
    #[inline(always)]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Add for Complex {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Precomputed in-place Cooley-Tukey Radix-2 complex FFT / IFFT plan.
///
/// All heap allocations (bit-reversal swap pairs and `f64`-computed twiddle tables)
/// occur once inside [`Radix2Fft::new`] / [`Radix2Fft::try_new`]. Stages with
/// `half >= 4` execute 4 butterflies per [`F32x4`] SIMD iteration.
#[derive(Clone, Debug)]
pub struct Radix2Fft {
    len: usize,
    inv_len: f32,
    swap_pairs: Box<[(u32, u32)]>,
    tw_re: AlignedBuf,
    tw_im: AlignedBuf,
}

impl Radix2Fft {
    /// Builds a Radix-2 FFT plan for power-of-two `len >= 1`.
    ///
    /// # Panics
    /// Panics if `len == 0` or `!len.is_power_of_two()`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::try_new(len).expect("Radix2Fft length must be a power of two >= 1")
    }

    /// Checked constructor returning `None` if `len == 0` or not a power of two.
    #[must_use]
    pub fn try_new(len: usize) -> Option<Self> {
        if len == 0 || !len.is_power_of_two() {
            return None;
        }

        let log2_n = len.trailing_zeros();
        let mut swap_pairs = Vec::new();
        for i in 0..len {
            let j = reverse_bits(i, log2_n);
            if i < j {
                swap_pairs.push((i as u32, j as u32));
            }
        }

        // Stages `half = 1, 2` use exact trivial twiddles (`1` and `-i`); store `half >= 4`.
        let simd_tw_len = len.saturating_sub(4);
        let mut tw_re = Vec::with_capacity(simd_tw_len);
        let mut tw_im = Vec::with_capacity(simd_tw_len);
        let mut half = 4usize;
        while half < len {
            let step = half << 1;
            let base_angle = -TAU / (step as f64);
            for k in 0..half {
                let (s, c) = sin_cos_f64(base_angle * (k as f64));
                tw_re.push(c as f32);
                tw_im.push(s as f32);
            }
            half = step;
        }

        Some(Self {
            len,
            inv_len: 1.0 / (len as f32),
            swap_pairs: swap_pairs.into_boxed_slice(),
            tw_re: AlignedBuf::from_slice(&tw_re),
            tw_im: AlignedBuf::from_slice(&tw_im),
        })
    }

    /// Transform size `N`.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` only if `len() == 0` (always `false` for valid plans).
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Computes the unnormalized forward DFT in-place (`buf.len() == self.len()`).
    ///
    /// Zero allocations and panic-free in release builds.
    pub fn forward(&self, buf: &mut [Complex]) {
        if buf.len() != self.len {
            debug_assert_eq!(buf.len(), self.len);
            return;
        }

        for &(i, j) in &self.swap_pairs {
            buf.swap(i as usize, j as usize);
        }

        if self.len >= 2 {
            stage_half1(buf);
        }
        if self.len >= 4 {
            stage_half2(buf);
        }

        let tw_re = self.tw_re.as_slice();
        let tw_im = self.tw_im.as_slice();
        let mut half = 4usize;
        let mut tw_offset = 0usize;
        while half < self.len {
            let step = half << 1;
            let stage_re = &tw_re[tw_offset..tw_offset + half];
            let stage_im = &tw_im[tw_offset..tw_offset + half];
            stage_simd_half_ge_4(buf, half, step, stage_re, stage_im);
            tw_offset += half;
            half = step;
        }
    }

    /// Computes the `1 / N`-normalized inverse DFT in-place (`buf.len() == self.len()`).
    ///
    /// Uses `IFFT(X) = (1/N) * conj(FFT(conj(X)))` with zero allocations.
    pub fn inverse(&self, buf: &mut [Complex]) {
        if buf.len() != self.len {
            debug_assert_eq!(buf.len(), self.len);
            return;
        }

        for c in buf.iter_mut() {
            c.im = -c.im;
        }

        self.forward(buf);

        let inv = self.inv_len;
        for c in buf.iter_mut() {
            c.re *= inv;
            c.im = -c.im * inv;
        }
    }
}

#[inline]
fn stage_half1(buf: &mut [Complex]) {
    for pair in buf.chunks_exact_mut(2) {
        let u = pair[0];
        let v = pair[1];
        pair[0] = u + v;
        pair[1] = u - v;
    }
}

#[inline]
fn stage_half2(buf: &mut [Complex]) {
    for quad in buf.chunks_exact_mut(4) {
        let u0 = quad[0];
        let v0 = quad[2];
        quad[0] = u0 + v0;
        quad[2] = u0 - v0;

        let u1 = quad[1];
        // Multiply `quad[3]` by `W_4^1 = -i`: `(re + i im) * (-i) = im - i re`.
        let v1 = Complex::new(quad[3].im, -quad[3].re);
        quad[1] = u1 + v1;
        quad[3] = u1 - v1;
    }
}

#[inline]
fn stage_simd_half_ge_4(
    buf: &mut [Complex],
    half: usize,
    step: usize,
    tw_re: &[f32],
    tw_im: &[f32],
) {
    debug_assert!(half >= 4 && half.is_multiple_of(4));
    for chunk in buf.chunks_exact_mut(step) {
        let (left, right) = chunk.split_at_mut(half);
        let mut k = 0;
        while k < half {
            // SAFETY: `k + 4 <= half` (`half` is a multiple of 4), `tw_*` offsets are 16B aligned.
            unsafe {
                let wr = F32x4::load_aligned(tw_re.as_ptr().add(k));
                let wi = F32x4::load_aligned(tw_im.as_ptr().add(k));

                let l_ptr = left.as_mut_ptr().add(k).cast::<f32>();
                let r_ptr = right.as_mut_ptr().add(k).cast::<f32>();

                let (ar, ai) = load_complex4(l_ptr);
                let (br, bi) = load_complex4(r_ptr);

                let tr = wr * br - wi * bi;
                let ti = wr.mul_add(bi, wi * br);

                store_complex4(l_ptr, ar + tr, ai + ti);
                store_complex4(r_ptr, ar - tr, ai - ti);
            }
            k += 4;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn load_complex4(ptr: *const f32) -> (F32x4, F32x4) {
    let c01 = unsafe { F32x4::load_unaligned(ptr) };
    let c23 = unsafe { F32x4::load_unaligned(ptr.add(4)) };
    let t0 = c01.unpack_lo(c23);
    let t1 = c01.unpack_hi(c23);
    (t0.unpack_lo(t1), t0.unpack_hi(t1))
}

#[inline(always)]
pub(crate) unsafe fn store_complex4(ptr: *mut f32, re: F32x4, im: F32x4) {
    unsafe {
        re.unpack_lo(im).store_unaligned(ptr);
        re.unpack_hi(im).store_unaligned(ptr.add(4));
    }
}

#[inline]
fn reverse_bits(x: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS - bits)
    }
}
