//! 64-byte aligned split-complex (`re` / `im`) buffers and `F32x4`-vectorized
//! frequency-domain multiply-accumulate kernels for partitioned convolution.

use core::mem::{align_of, size_of};

use crate::aligned_buf::AlignedBuf;
use crate::f32x4::F32x4;
use crate::fft_radix2::{load_complex4, store_complex4, Complex};

const _: () = {
    assert!(size_of::<Complex>() == 8);
    assert!(align_of::<Complex>() == 4);
};

/// 64-byte aligned split-complex (`re`, `im`) storage for frequency-domain spectra.
///
/// All heap memory is allocated strictly at construction (`new` / `zeroed`).
#[derive(Clone, Debug)]
pub struct SplitComplexBuf {
    re: AlignedBuf,
    im: AlignedBuf,
}

impl SplitComplexBuf {
    /// Allocates a zero-initialized split-complex buffer of `len` bins.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::zeroed(len)
    }

    /// Allocates a zero-initialized split-complex buffer of `len` bins.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        Self {
            re: AlignedBuf::zeroed(len),
            im: AlignedBuf::zeroed(len),
        }
    }

    /// Logical number of complex bins.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.re.len()
    }

    /// Returns `true` if `len() == 0`.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.re.is_empty()
    }

    /// Padded per-plane capacity in `f32` elements (multiple of 16).
    #[must_use]
    #[inline]
    pub fn padded_len(&self) -> usize {
        self.re.padded_len()
    }

    /// Zeroes both `re` and `im` planes (including padding) without reallocating.
    #[inline]
    pub fn clear(&mut self) {
        self.re.clear();
        self.im.clear();
    }

    /// Returns the logical real plane `&[f32]`.
    #[must_use]
    #[inline]
    pub fn re(&self) -> &[f32] {
        self.re.as_slice()
    }

    /// Returns the logical imaginary plane `&[f32]`.
    #[must_use]
    #[inline]
    pub fn im(&self) -> &[f32] {
        self.im.as_slice()
    }

    /// Returns the mutable logical real plane `&mut [f32]`.
    #[must_use]
    #[inline]
    pub fn re_mut(&mut self) -> &mut [f32] {
        self.re.as_mut_slice()
    }

    /// Returns the mutable logical imaginary plane `&mut [f32]`.
    #[must_use]
    #[inline]
    pub fn im_mut(&mut self) -> &mut [f32] {
        self.im.as_mut_slice()
    }

    /// Returns immutable `(re, im)` logical plane slices.
    #[must_use]
    #[inline]
    pub fn planes(&self) -> (&[f32], &[f32]) {
        (self.re.as_slice(), self.im.as_slice())
    }

    /// Returns mutable `(re, im)` logical plane slices simultaneously.
    #[must_use]
    #[inline]
    pub fn planes_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (self.re.as_mut_slice(), self.im.as_mut_slice())
    }

    /// Returns immutable `(re, im)` padded plane slices (`padded_len()` elements each).
    #[must_use]
    #[inline]
    pub fn padded_planes(&self) -> (&[f32], &[f32]) {
        (self.re.as_padded_slice(), self.im.as_padded_slice())
    }

    /// Returns mutable `(re, im)` padded plane slices (`padded_len()` elements each).
    #[must_use]
    #[inline]
    pub fn padded_planes_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (self.re.as_padded_mut_slice(), self.im.as_padded_mut_slice())
    }

    /// Deinterleaves `src` (`Complex` AoS) into `re` and `im` planes using `F32x4`.
    ///
    /// Zero-allocation and panic-free. Copies `self.len().min(src.len())` bins.
    pub fn load_from_complex(&mut self, src: &[Complex]) {
        let n = self.len().min(src.len());
        let simd_len = n & !3;
        let re = self.re.as_mut_slice();
        let im = self.im.as_mut_slice();
        let mut i = 0;
        while i < simd_len {
            // SAFETY: `i + 4 <= simd_len <= n <= src.len()`, and `Complex` is `#[repr(C)]` `[f32; 2]`.
            unsafe {
                let (rv, iv) = load_complex4(src.as_ptr().add(i).cast::<f32>());
                rv.store_unaligned(re.as_mut_ptr().add(i));
                iv.store_unaligned(im.as_mut_ptr().add(i));
            }
            i += 4;
        }
        while i < n {
            re[i] = src[i].re;
            im[i] = src[i].im;
            i += 1;
        }
    }

    /// Interleaves `re` and `im` planes into `dst` (`Complex` AoS) using `F32x4`.
    ///
    /// Zero-allocation and panic-free. Writes `self.len().min(dst.len())` bins.
    pub fn store_to_complex(&self, dst: &mut [Complex]) {
        let n = self.len().min(dst.len());
        let simd_len = n & !3;
        let re = self.re.as_slice();
        let im = self.im.as_slice();
        let mut i = 0;
        while i < simd_len {
            // SAFETY: `i + 4 <= simd_len <= n <= dst.len()`, and `Complex` is `#[repr(C)]` `[f32; 2]`.
            unsafe {
                let rv = F32x4::load_unaligned(re.as_ptr().add(i));
                let iv = F32x4::load_unaligned(im.as_ptr().add(i));
                store_complex4(dst.as_mut_ptr().add(i).cast::<f32>(), rv, iv);
            }
            i += 4;
        }
        while i < n {
            dst[i] = Complex::new(re[i], im[i]);
            i += 1;
        }
    }

    /// Accumulates `self += x * h` bin-wise across the padded 4-lane vectors.
    #[inline]
    pub fn fma_accumulate(&mut self, x: &Self, h: &Self) {
        let (acc_re, acc_im) = self.padded_planes_mut();
        let (x_re, x_im) = x.padded_planes();
        let (h_re, h_im) = h.padded_planes();
        complex_fma_accumulate(acc_re, acc_im, x_re, x_im, h_re, h_im);
    }

    /// Accumulates both `acc_l += x * hl` and `acc_r += x * hr` sharing one load of `x`.
    #[inline]
    pub fn fma_accumulate_lr(acc_l: &mut Self, acc_r: &mut Self, x: &Self, hl: &Self, hr: &Self) {
        let (al_re, al_im) = acc_l.padded_planes_mut();
        let (ar_re, ar_im) = acc_r.padded_planes_mut();
        let (x_re, x_im) = x.padded_planes();
        let (hl_re, hl_im) = hl.padded_planes();
        let (hr_re, hr_im) = hr.padded_planes();
        complex_fma_accumulate_lr(
            al_re, al_im, ar_re, ar_im, x_re, x_im, hl_re, hl_im, hr_re, hr_im,
        );
    }
}

/// Accumulates complex products `acc += x * h` over 4 frequency bins per `F32x4` step:
/// - `acc_re[k] += x_re[k] * h_re[k] - x_im[k] * h_im[k]`
/// - `acc_im[k] += x_re[k] * h_im[k] + x_im[k] * h_re[k]`
///
/// Zero-allocation and panic-free; processes `min` of all six slice lengths.
pub fn complex_fma_accumulate(
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
) {
    let n = acc_re
        .len()
        .min(acc_im.len())
        .min(x_re.len())
        .min(x_im.len())
        .min(h_re.len())
        .min(h_im.len());
    let simd_len = n & !3;
    let mut i = 0;

    while i < simd_len {
        // SAFETY: `i + 4 <= simd_len <= n` for every slice; mutable slices cannot alias inputs.
        unsafe {
            let ar = F32x4::load_unaligned(acc_re.as_ptr().add(i));
            let ai = F32x4::load_unaligned(acc_im.as_ptr().add(i));
            let xr = F32x4::load_unaligned(x_re.as_ptr().add(i));
            let xi = F32x4::load_unaligned(x_im.as_ptr().add(i));
            let hr = F32x4::load_unaligned(h_re.as_ptr().add(i));
            let hi = F32x4::load_unaligned(h_im.as_ptr().add(i));

            let out_re = ar + (xr * hr - xi * hi);
            let out_im = xr.mul_add(hi, xi.mul_add(hr, ai));
            out_re.store_unaligned(acc_re.as_mut_ptr().add(i));
            out_im.store_unaligned(acc_im.as_mut_ptr().add(i));
        }
        i += 4;
    }

    while i < n {
        let xr = x_re[i];
        let xi = x_im[i];
        let hr = h_re[i];
        let hi = h_im[i];
        acc_re[i] += xr * hr - xi * hi;
        acc_im[i] += xr * hi + xi * hr;
        i += 1;
    }
}

/// Accumulates stereo products `acc_l += x * hl` and `acc_r += x * hr` sharing one load of `x`.
#[allow(clippy::too_many_arguments)]
pub fn complex_fma_accumulate_lr(
    al_re: &mut [f32],
    al_im: &mut [f32],
    ar_re: &mut [f32],
    ar_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    hl_re: &[f32],
    hl_im: &[f32],
    hr_re: &[f32],
    hr_im: &[f32],
) {
    let n = al_re
        .len()
        .min(al_im.len())
        .min(ar_re.len())
        .min(ar_im.len())
        .min(x_re.len())
        .min(x_im.len())
        .min(hl_re.len())
        .min(hl_im.len())
        .min(hr_re.len())
        .min(hr_im.len());
    let simd_len = n & !3;
    let mut i = 0;

    while i < simd_len {
        // SAFETY: `i + 4 <= simd_len <= n` for every slice; mutable accumulators cannot alias inputs.
        unsafe {
            let xr = F32x4::load_unaligned(x_re.as_ptr().add(i));
            let xi = F32x4::load_unaligned(x_im.as_ptr().add(i));

            let alr = F32x4::load_unaligned(al_re.as_ptr().add(i));
            let ali = F32x4::load_unaligned(al_im.as_ptr().add(i));
            let hlr = F32x4::load_unaligned(hl_re.as_ptr().add(i));
            let hli = F32x4::load_unaligned(hl_im.as_ptr().add(i));
            (alr + (xr * hlr - xi * hli)).store_unaligned(al_re.as_mut_ptr().add(i));
            xr.mul_add(hli, xi.mul_add(hlr, ali))
                .store_unaligned(al_im.as_mut_ptr().add(i));

            let arr = F32x4::load_unaligned(ar_re.as_ptr().add(i));
            let ari = F32x4::load_unaligned(ar_im.as_ptr().add(i));
            let hrr = F32x4::load_unaligned(hr_re.as_ptr().add(i));
            let hri = F32x4::load_unaligned(hr_im.as_ptr().add(i));
            (arr + (xr * hrr - xi * hri)).store_unaligned(ar_re.as_mut_ptr().add(i));
            xr.mul_add(hri, xi.mul_add(hrr, ari))
                .store_unaligned(ar_im.as_mut_ptr().add(i));
        }
        i += 4;
    }

    while i < n {
        let xr = x_re[i];
        let xi = x_im[i];
        let (hlr, hli) = (hl_re[i], hl_im[i]);
        al_re[i] += xr * hlr - xi * hli;
        al_im[i] += xr * hli + xi * hlr;
        let (hrr, hri) = (hr_re[i], hr_im[i]);
        ar_re[i] += xr * hrr - xi * hri;
        ar_im[i] += xr * hri + xi * hrr;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real_fft::RealFft;

    #[test]
    fn split_complex_buf_alignment_load_store_and_clear() {
        for &len in &[0, 1, 3, 4, 7, 17, 129, 2049] {
            let buf = SplitComplexBuf::new(len);
            assert_eq!(buf.len(), len);
            assert_eq!(buf.is_empty(), len == 0);
            assert_eq!(buf.re().as_ptr() as usize % 64, 0);
            assert_eq!(buf.im().as_ptr() as usize % 64, 0);
        }

        let src: Vec<Complex> = (0..19)
            .map(|k| Complex::new((k as f32) * 0.5 - 3.0, -(k as f32) * 0.25 + 1.0))
            .collect();
        let mut split = SplitComplexBuf::zeroed(19);
        split.load_from_complex(&src);
        for (k, c) in src.iter().enumerate() {
            assert_eq!(split.re()[k], c.re);
            assert_eq!(split.im()[k], c.im);
        }

        let mut roundtrip = vec![Complex::ZERO; 19];
        split.store_to_complex(&mut roundtrip);
        assert_eq!(roundtrip, src);

        split.clear();
        assert!(split.re().iter().all(|&v| v == 0.0));
        assert!(split.im().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn complex_fma_accumulate_matches_scalar_complex_arithmetic() {
        let n = 131usize;
        let x: Vec<Complex> = (0..n)
            .map(|k| Complex::new(((k % 11) as f32) - 5.0, ((k % 7) as f32) * 0.3 - 1.0))
            .collect();
        let h: Vec<Complex> = (0..n)
            .map(|k| Complex::new(((k % 5) as f32) * 0.2 + 0.1, -((k % 9) as f32) * 0.15))
            .collect();
        let init: Vec<Complex> = (0..n)
            .map(|k| Complex::new((k as f32) * 0.01, -(k as f32) * 0.005))
            .collect();

        let mut sx = SplitComplexBuf::new(n);
        let mut sh = SplitComplexBuf::new(n);
        let mut sacc = SplitComplexBuf::new(n);
        sx.load_from_complex(&x);
        sh.load_from_complex(&h);
        sacc.load_from_complex(&init);

        let (ar, ai) = sacc.planes_mut();
        complex_fma_accumulate(ar, ai, sx.re(), sx.im(), sh.re(), sh.im());
        // Accumulate a second partition via convenience method as well.
        sacc.fma_accumulate(&sx, &sh);

        for k in 0..n {
            let expected = init[k] + (x[k] * h[k]) + (x[k] * h[k]);
            assert!((sacc.re()[k] - expected.re).abs() < 1e-5);
            assert!((sacc.im()[k] - expected.im).abs() < 1e-5);
        }
    }

    #[test]
    fn real_fft_split_matches_interleaved_and_roundtrips() {
        let mut seed = 0x55aa_1234_9876_fedc_u64;
        for &n in &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let mut rfft = RealFft::new(n);
            let bins = rfft.spectrum_len();
            let input: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (((seed >> 40) as u32) as f32 / 8_388_608.0) - 1.0
                })
                .collect();

            let mut spec_aos = vec![Complex::ZERO; bins];
            let mut split = SplitComplexBuf::new(bins);
            rfft.forward(&input, &mut spec_aos);
            {
                let (re, im) = split.planes_mut();
                rfft.forward_split(&input, re, im);
            }

            for (k, c) in spec_aos.iter().enumerate() {
                assert!((split.re()[k] - c.re).abs() < 1e-6);
                assert!((split.im()[k] - c.im).abs() < 1e-6);
            }

            let mut recon = vec![0.0_f32; n];
            rfft.inverse_split(split.re(), split.im(), &mut recon);
            let max_err = input
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_err < 1e-5,
                "RealFft split round-trip err {max_err} at N={n}"
            );
        }
    }
}
