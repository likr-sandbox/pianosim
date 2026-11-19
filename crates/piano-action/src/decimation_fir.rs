//! Multi-rate Blackman-sinc polyphase decimation FIR filter (`16 / 8 / 4` taps)
//! downsampling `8x / 4x / 2x` oversampled hammer-string contact forces to host rate.

use core::f64::consts::{PI, TAU};

use crate::math::{sin_cos_f64, snap_oversample};

/// Maximum supported oversampling factor (`8x`).
pub const MAX_OVERSAMPLE_FACTOR: usize = 8;
/// Maximum supported FIR tap count (`16` taps at `8x`).
pub const MAX_FIR_TAPS: usize = 16;

/// DC-normalized (`sum == 1.0`, strictly non-negative) Blackman-sinc polyphase
/// decimation filter supporting `8x` (16-tap), `4x` (8-tap), and `2x` (4-tap).
#[derive(Copy, Clone, Debug)]
pub struct DecimationFir {
    factor: u8,
    taps: u8,
    coeffs: [f32; MAX_FIR_TAPS],
}

/// Backward-compatible alias for [`DecimationFir`].
pub type DecimationFir16 = DecimationFir;

impl DecimationFir {
    /// Precomputes the `8x` 16-tap Blackman-sinc kernel with `sum == 1.0`.
    #[must_use]
    pub fn new() -> Self {
        Self::for_oversample(8)
    }

    /// Precomputes a DC-normalized symmetric Blackman-sinc kernel for `oversample` (`2 | 4 | 8`).
    #[must_use]
    pub fn for_oversample(oversample: u8) -> Self {
        let factor = snap_oversample(oversample);
        let taps = usize::from(factor) * 2;
        let cutoff_norm = 0.44 / f64::from(factor);

        let mut raw = [0.0_f64; MAX_FIR_TAPS];
        let mut sum = 0.0_f64;
        let center = 0.5 * ((taps - 1) as f64);

        for (i, slot) in raw[..taps].iter_mut().enumerate() {
            let x = (i as f64) - center;
            let sinc = if x == 0.0 {
                2.0 * cutoff_norm
            } else {
                let s = sin_cos_f64(2.0 * PI * cutoff_norm * x).0;
                s / (PI * x)
            };
            let phase = TAU * ((i as f64) + 0.5) / (taps as f64);
            let c1 = sin_cos_f64(phase).1;
            let c2 = sin_cos_f64(2.0 * phase).1;
            let blackman = 0.42 - 0.50 * c1 + 0.08 * c2;
            let w = (sinc * blackman).max(0.0);
            *slot = w;
            sum += w;
        }

        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        let mut coeffs = [0.0_f32; MAX_FIR_TAPS];
        for (dst, &src) in coeffs[..taps].iter_mut().zip(raw[..taps].iter()) {
            *dst = (src * inv_sum) as f32;
        }
        Self {
            factor,
            taps: taps as u8,
            coeffs,
        }
    }

    /// Configured oversampling factor (`2`, `4`, or `8`).
    #[must_use]
    #[inline]
    pub const fn factor(&self) -> u8 {
        self.factor
    }

    /// Active FIR tap count (`4`, `8`, or `16`).
    #[must_use]
    #[inline]
    pub const fn taps(&self) -> u8 {
        self.taps
    }

    /// Returns the 16-slot FIR tap array (active taps in `0..self.taps()`).
    #[must_use]
    #[inline]
    pub const fn coeffs(&self) -> &[f32; MAX_FIR_TAPS] {
        &self.coeffs
    }

    /// Filters `factor` new oversampled forces against the previous `factor` substeps in `prev_sub`.
    #[inline]
    pub fn process(
        &self,
        prev_sub: &mut [f32; MAX_OVERSAMPLE_FACTOR],
        cur: &[f32; MAX_OVERSAMPLE_FACTOR],
    ) -> f32 {
        let os = usize::from(self.factor);
        let mut acc = 0.0_f32;
        for k in 0..os {
            acc += self.coeffs[k] * cur[os - 1 - k];
            acc += self.coeffs[os + k] * prev_sub[os - 1 - k];
        }
        prev_sub[..os].copy_from_slice(&cur[..os]);
        if os < MAX_OVERSAMPLE_FACTOR {
            prev_sub[os..].fill(0.0);
        }
        acc
    }

    /// Filters 8 new oversampled forces against the previous 8 substeps in `prev_sub`.
    #[inline]
    pub fn process_8(
        &self,
        prev_sub: &mut [f32; MAX_OVERSAMPLE_FACTOR],
        cur: &[f32; MAX_OVERSAMPLE_FACTOR],
    ) -> f32 {
        self.process(prev_sub, cur)
    }
}

impl Default for DecimationFir {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimation_fir_is_dc_normalized_symmetric_and_non_negative_across_factors() {
        for &os in &[8_u8, 4, 2] {
            let fir = DecimationFir::for_oversample(os);
            assert_eq!(fir.factor(), os);
            let taps = usize::from(fir.taps());
            assert_eq!(taps, usize::from(os) * 2);
            let c = fir.coeffs();
            let sum: f32 = c[..taps].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "os={os} sum={sum}");
            for i in 0..usize::from(os) {
                assert!((c[i] - c[taps - 1 - i]).abs() < 1e-7);
            }
            assert!(c[..taps].iter().all(|&x| x >= 0.0));
            assert!(c[taps..].iter().all(|&x| x == 0.0));

            let mut prev = [0.0_f32; MAX_OVERSAMPLE_FACTOR];
            let mut ones = [0.0_f32; MAX_OVERSAMPLE_FACTOR];
            ones[..usize::from(os)].fill(1.0);
            let y0 = fir.process(&mut prev, &ones);
            let y1 = fir.process(&mut prev, &ones);
            assert!(y0 > 0.0 && y0 < 1.0, "os={os} y0={y0}");
            assert!((y1 - 1.0).abs() < 1e-6, "os={os} y1={y1}");

            let zeros = [0.0_f32; MAX_OVERSAMPLE_FACTOR];
            let t0 = fir.process(&mut prev, &zeros);
            let t1 = fir.process(&mut prev, &zeros);
            assert!(t0 > 0.0, "os={os} t0={t0}");
            assert_eq!(t1, 0.0);
        }
    }
}
