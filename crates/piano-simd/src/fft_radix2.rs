//! Zero-allocation-in-process power-of-two complex Radix-2 FFT and IFFT.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::f64::consts::TAU;
use core::ops::{Add, Mul, Sub};

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
/// occur once inside [`Radix2Fft::new`] / [`Radix2Fft::try_new`].
#[derive(Clone, Debug)]
pub struct Radix2Fft {
    len: usize,
    inv_len: f32,
    swap_pairs: Box<[(u32, u32)]>,
    twiddles: Box<[Complex]>,
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

        let mut twiddles = Vec::with_capacity(len.saturating_sub(1));
        let mut half = 1usize;
        while half < len {
            let step = half << 1;
            let base_angle = -TAU / (step as f64);
            for k in 0..half {
                let (s, c) = sin_cos_f64(base_angle * (k as f64));
                twiddles.push(Complex::new(c as f32, s as f32));
            }
            half = step;
        }

        Some(Self {
            len,
            inv_len: 1.0 / (len as f32),
            swap_pairs: swap_pairs.into_boxed_slice(),
            twiddles: twiddles.into_boxed_slice(),
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

        let mut half = 1usize;
        let mut tw_offset = 0usize;
        while half < self.len {
            let step = half << 1;
            let stage_tw = &self.twiddles[tw_offset..tw_offset + half];
            for chunk in buf.chunks_exact_mut(step) {
                let (left, right) = chunk.split_at_mut(half);
                for ((a, b), &w) in left.iter_mut().zip(right.iter_mut()).zip(stage_tw.iter()) {
                    let t = w * *b;
                    let u = *a;
                    *a = u + t;
                    *b = u - t;
                }
            }
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
fn reverse_bits(x: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        x.reverse_bits() >> (usize::BITS - bits)
    }
}
