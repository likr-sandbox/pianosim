//! Zero-allocation real-to-complex forward and complex-to-real inverse FFT
//! built over an `N / 2`-point [`Radix2Fft`].

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::TAU;

use crate::fft_radix2::{Complex, Radix2Fft};
use crate::math::sin_cos_f64;

/// Real-valued FFT/IFFT plan for power-of-two size `N >= 2`.
///
/// The complex spectrum slice contains `N / 2 + 1` Hermitian bins `[0..=N/2]`.
/// All scratch memory and twiddle factors are allocated solely at construction.
#[derive(Clone, Debug)]
pub struct RealFft {
    len: usize,
    cfft: Radix2Fft,
    half_twiddles: Box<[Complex]>,
    scratch: Box<[Complex]>,
}

impl RealFft {
    /// Builds a real FFT plan for power-of-two `len >= 2`.
    ///
    /// # Panics
    /// Panics if `len < 2` or `!len.is_power_of_two()`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::try_new(len).expect("RealFft length must be a power of two >= 2")
    }

    /// Checked constructor returning `None` if `len < 2` or not a power of two.
    #[must_use]
    pub fn try_new(len: usize) -> Option<Self> {
        if len < 2 || !len.is_power_of_two() {
            return None;
        }

        let half = len / 2;
        let cfft = Radix2Fft::try_new(half)?;
        let mut half_twiddles = Vec::with_capacity(half);
        let base_angle = -TAU / (len as f64);
        for k in 0..half {
            let (s, c) = sin_cos_f64(base_angle * (k as f64));
            half_twiddles.push(Complex::new(c as f32, s as f32));
        }

        let scratch = vec![Complex::ZERO; half].into_boxed_slice();
        Some(Self {
            len,
            cfft,
            half_twiddles: half_twiddles.into_boxed_slice(),
            scratch,
        })
    }

    /// Real-domain transform length `N`.
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

    /// Complex spectrum length `N / 2 + 1` (DC through Nyquist inclusive).
    #[must_use]
    #[inline]
    pub const fn spectrum_len(&self) -> usize {
        self.len / 2 + 1
    }

    /// Computes the forward real-to-complex DFT (`input.len() == N`, `spectrum.len() == N/2 + 1`).
    ///
    /// Zero allocations and panic-free in release builds.
    pub fn forward(&mut self, input: &[f32], spectrum: &mut [Complex]) {
        let half = self.len / 2;
        if input.len() != self.len || spectrum.len() != half + 1 {
            debug_assert_eq!(input.len(), self.len);
            debug_assert_eq!(spectrum.len(), half + 1);
            return;
        }

        for (dst, pair) in self.scratch.iter_mut().zip(input.chunks_exact(2)) {
            *dst = Complex::new(pair[0], pair[1]);
        }

        self.cfft.forward(&mut self.scratch);

        let z0 = self.scratch[0];
        spectrum[0] = Complex::new(z0.re + z0.im, 0.0);
        spectrum[half] = Complex::new(z0.re - z0.im, 0.0);

        for (k, slot) in spectrum.iter_mut().enumerate().take(half).skip(1) {
            let a = self.scratch[k];
            let b = self.scratch[half - k].conj();
            let ze = (a + b).scale(0.5);
            let diff = a - b;
            // Z_o[k] = -i / 2 * (Z[k] - conj(Z[M - k]))
            let zo = Complex::new(0.5 * diff.im, -0.5 * diff.re);
            *slot = ze + self.half_twiddles[k] * zo;
        }
    }

    /// Computes the `1 / N`-normalized inverse complex-to-real DFT
    /// (`spectrum.len() == N/2 + 1`, `output.len() == N`).
    ///
    /// Zero allocations and panic-free in release builds.
    pub fn inverse(&mut self, spectrum: &[Complex], output: &mut [f32]) {
        let half = self.len / 2;
        if spectrum.len() != half + 1 || output.len() != self.len {
            debug_assert_eq!(spectrum.len(), half + 1);
            debug_assert_eq!(output.len(), self.len);
            return;
        }

        let dc = spectrum[0].re;
        let nyq = spectrum[half].re;
        self.scratch[0] = Complex::new(0.5 * (dc + nyq), 0.5 * (dc - nyq));

        for k in 1..half {
            let a = spectrum[k];
            let b = spectrum[half - k].conj();
            let ze = (a + b).scale(0.5);
            let diff = a - b;
            // i * Z_o[k] = conj(W_N^k) * (i / 2) * (X[k] - conj(X[M - k]))
            let i_half_diff = Complex::new(-0.5 * diff.im, 0.5 * diff.re);
            let i_zo = self.half_twiddles[k].conj() * i_half_diff;
            self.scratch[k] = ze + i_zo;
        }

        self.cfft.inverse(&mut self.scratch);

        for (src, pair) in self.scratch.iter().zip(output.chunks_exact_mut(2)) {
            pair[0] = src.re;
            pair[1] = src.im;
        }
    }
}
