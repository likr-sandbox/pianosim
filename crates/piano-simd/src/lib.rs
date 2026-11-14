//! Portable 4-lane SIMD abstractions, 64-byte aligned buffers, FFT, and ADAA LUTs.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod aligned_buf;
pub mod f32x4;
pub mod fft_radix2;
pub mod lut_adaa;
mod math;
pub mod real_fft;

pub use aligned_buf::AlignedBuf;
pub use f32x4::F32x4;
pub use fft_radix2::{Complex, Radix2Fft};
pub use lut_adaa::{AdaaPowerLut, AdaaState, ADAA_DIFF_EPS};
pub use real_fft::RealFft;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f32x4::scalar::ScalarF32x4;
    use crate::math::{pow_nonneg_f64, sin_cos_f64, sqrt_f32};
    use core::f64::consts::TAU;

    fn pseudo_unit(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let bits = ((*seed >> 40) as u32) as f32 / 16_777_216.0;
        bits * 2.0 - 1.0
    }

    #[test]
    fn nostd_math_matches_std_precision() {
        for i in -32..=32 {
            let angle = f64::from(i) * 0.371_234_567;
            let (s, c) = sin_cos_f64(angle);
            let (es, ec) = angle.sin_cos();
            assert!((s - es).abs() < 1e-14, "sin mismatch at {angle}");
            assert!((c - ec).abs() < 1e-14, "cos mismatch at {angle}");
        }

        for &base in &[0.0_f64, 1e-12, 0.001, 0.25, 0.73, 1.0, 2.5, 13.7] {
            for &exp in &[0.5_f64, 2.2, 2.8, 3.5, 4.2] {
                let got = pow_nonneg_f64(base, exp);
                let expected = base.powf(exp);
                let tol = 1e-13 * expected.abs().max(1.0);
                assert!((got - expected).abs() <= tol, "pow({base}, {exp})");
            }
        }

        for &x in &[0.0_f32, 1e-40, 1e-12, 0.25, 1.0, 2.0, 1234.5, 1e12] {
            let got = sqrt_f32(x);
            let expected = x.sqrt();
            assert!((got - expected).abs() <= 1e-6 * expected.max(1e-38));
        }
        assert!(sqrt_f32(-1.0).is_nan());
    }

    #[test]
    fn f32x4_matches_scalar_oracle_and_memory_ops() {
        let a_vals = [-3.5_f32, 0.25, 4.0, -0.0];
        let b_vals = [1.25_f32, -2.0, 2.5, 9.0];
        let c_vals = [0.5_f32, 3.0, -1.5, 4.0];

        let va = F32x4::from_array(a_vals);
        let vb = F32x4::from_array(b_vals);
        let vc = F32x4::from_array(c_vals);
        let sa = ScalarF32x4::from_array(a_vals);
        let sb = ScalarF32x4::from_array(b_vals);
        let sc = ScalarF32x4::from_array(c_vals);

        assert_eq!(F32x4::splat(2.5).to_array(), [2.5; 4]);
        assert_eq!(F32x4::zero().to_array(), [0.0; 4]);
        assert_eq!((va + vb).to_array(), sa.add(sb).to_array());
        assert_eq!((va - vb).to_array(), sa.sub(sb).to_array());
        assert_eq!((va * vb).to_array(), sa.mul(sb).to_array());
        assert_eq!((va / vb).to_array(), sa.div(sb).to_array());
        assert_eq!(va.min(vb).to_array(), sa.min(sb).to_array());
        assert_eq!(va.max(vb).to_array(), sa.max(sb).to_array());
        assert_eq!(va.abs().to_array(), sa.abs().to_array());
        assert_eq!(va.reduce_sum(), sa.reduce_sum());

        let fma_got = va.mul_add(vb, vc).to_array();
        let fma_ref = sa.mul_add(sb, sc).to_array();
        for i in 0..4 {
            assert!((fma_got[i] - fma_ref[i]).abs() < 1e-6);
        }

        let pos = [0.0_f32, 0.25, 2.0, 144.0];
        let sq_got = F32x4::from_array(pos).sqrt().to_array();
        let sq_ref = ScalarF32x4::from_array(pos).sqrt().to_array();
        for i in 0..4 {
            assert!((sq_got[i] - sq_ref[i]).abs() <= 1e-6 * sq_ref[i].max(1.0));
        }

        // Unaligned load/store across odd slice offsets.
        let mut unaligned = [0.0_f32; 9];
        unsafe {
            va.store_unaligned(unaligned.as_mut_ptr().add(3));
            let loaded = F32x4::load_unaligned(unaligned.as_ptr().add(3));
            assert_eq!(loaded.to_array(), a_vals);
        }

        // 64-byte aligned load/store via AlignedBuf.
        let mut aligned = AlignedBuf::new(8);
        unsafe {
            vb.store_aligned(aligned.as_mut_ptr());
            let loaded = F32x4::load_aligned(aligned.as_ptr());
            assert_eq!(loaded.to_array(), b_vals);
        }
        va.write_to_slice(&mut aligned[4..8]);
        assert_eq!(F32x4::from_slice(&aligned[4..8]), va);
    }

    #[test]
    fn aligned_buf_enforces_64_byte_alignment_and_lane_views() {
        for &len in &[0, 1, 3, 4, 5, 15, 16, 17, 63, 64, 128, 1024, 26_200] {
            let mut buf = AlignedBuf::new(len);
            assert_eq!(buf.len(), len);
            assert_eq!(buf.is_empty(), len == 0);
            assert_eq!(buf.as_ptr() as usize % 64, 0, "align fail at len={len}");
            assert_eq!(buf.as_mut_ptr() as usize % 64, 0);
            assert_eq!(buf.padded_len() % 16, 0);
            assert!(buf.as_slice().iter().all(|&x| x == 0.0));
        }

        let mut buf = AlignedBuf::filled(10, 3.25);
        assert_eq!(buf.padded_len(), 16);
        assert!(buf[..10].iter().all(|&x| x == 3.25));
        assert!(buf.as_padded_slice()[10..].iter().all(|&x| x == 0.0));

        assert_eq!(buf.as_f32x4_slice().len(), 2);
        assert_eq!(buf.as_padded_f32x4_slice().len(), 4);
        buf.store_f32x4(1, F32x4::splat(-1.5));
        assert_eq!(buf.load_f32x4(1), F32x4::splat(-1.5));

        buf.fill(2.0);
        assert!(buf.as_slice().iter().all(|&x| x == 2.0));
        assert!(buf.as_padded_slice()[10..].iter().all(|&x| x == 0.0));

        buf.clear();
        assert!(buf.as_padded_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn radix2_fft_roundtrip_and_parseval_sizes_1_to_4096() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for pow in 0..=12 {
            let n = 1usize << pow;
            let fft = Radix2Fft::new(n);
            let orig: Vec<Complex> = (0..n)
                .map(|_| Complex::new(pseudo_unit(&mut seed), pseudo_unit(&mut seed)))
                .collect();
            let mut buf = orig.clone();

            fft.forward(&mut buf);
            let time_energy: f64 = orig.iter().map(|c| f64::from(c.norm_sq())).sum();
            let freq_energy: f64 =
                buf.iter().map(|c| f64::from(c.norm_sq())).sum::<f64>() / (n as f64);
            assert!(
                (time_energy - freq_energy).abs() <= 1e-4 * time_energy.max(1.0),
                "Parseval mismatch at N={n}"
            );

            fft.inverse(&mut buf);
            let max_err = orig
                .iter()
                .zip(buf.iter())
                .map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs()))
                .fold(0.0_f32, f32::max);
            assert!(
                max_err < 1e-5,
                "Radix2Fft round-trip err {max_err} at N={n}"
            );
        }
    }

    #[test]
    fn real_fft_roundtrip_and_spectral_parity_64_to_4096() {
        let mut seed = 0x1234_5678_9abc_def0_u64;
        for &n in &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let mut rfft = RealFft::new(n);
            let cfft = Radix2Fft::new(n);
            let input: Vec<f32> = (0..n).map(|_| pseudo_unit(&mut seed)).collect();
            let mut spectrum = vec![Complex::ZERO; rfft.spectrum_len()];
            let mut recon = vec![0.0_f32; n];

            rfft.forward(&input, &mut spectrum);

            // Cross-check RealFft bins against full complex Radix2Fft.
            let mut cbuf: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
            cfft.forward(&mut cbuf);
            for k in 0..rfft.spectrum_len() {
                let err = (spectrum[k].re - cbuf[k].re)
                    .abs()
                    .max((spectrum[k].im - cbuf[k].im).abs());
                assert!(
                    err < 2e-4,
                    "RealFft vs Complex FFT mismatch {err} at N={n}, bin={k}"
                );
            }

            rfft.inverse(&spectrum, &mut recon);
            let max_err = input
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_err < 1e-5, "RealFft round-trip err {max_err} at N={n}");
        }

        // Exact bin sinusoid tone check at N=256, bin=17.
        let n = 256usize;
        let bin = 17usize;
        let mut rfft = RealFft::new(n);
        let tone: Vec<f32> = (0..n)
            .map(|i| (TAU * (bin as f64) * (i as f64) / (n as f64)).cos() as f32)
            .collect();
        let mut spec = vec![Complex::ZERO; rfft.spectrum_len()];
        rfft.forward(&tone, &mut spec);
        assert!((spec[bin].re - (n as f32) * 0.5).abs() < 1e-3);
        assert!(spec[bin].im.abs() < 1e-3);
        for (k, c) in spec.iter().enumerate() {
            if k != bin {
                assert!(c.norm_sq().sqrt() < 1e-3, "unexpected leakage at bin {k}");
            }
        }
    }

    #[test]
    fn checked_constructors_reject_invalid_inputs() {
        assert!(Radix2Fft::try_new(0).is_none());
        assert!(Radix2Fft::try_new(6).is_none());
        assert!(Radix2Fft::try_new(1).is_some());
        assert!(RealFft::try_new(0).is_none());
        assert!(RealFft::try_new(1).is_none());
        assert!(RealFft::try_new(100).is_none());
        assert!(AdaaPowerLut::try_new(0.0, 1.0, 16).is_none());
        assert!(AdaaPowerLut::try_new(-2.0, 1.0, 16).is_none());
        assert!(AdaaPowerLut::try_new(2.5, 0.0, 16).is_none());
        assert!(AdaaPowerLut::try_new(2.5, 1.0, 1).is_none());
        assert!(AdaaPowerLut::try_new(f32::NAN, 1.0, 16).is_none());
    }

    #[test]
    fn adaa_lut_accuracy_derivative_consistency_and_guard() {
        for &(p, x_max) in &[(2.2_f32, 0.02_f32), (2.8, 1.0), (3.5, 5.0)] {
            let lut = AdaaPowerLut::new(p, x_max, 4096);
            assert_eq!(lut.eval_f(-0.2), 0.0);
            assert_eq!(lut.eval_f1(-0.2), 0.0);
            assert_eq!(lut.eval_f(0.0), 0.0);
            assert_eq!(lut.eval_f1(0.0), 0.0);

            for i in 1..20 {
                let x = (i as f32) * (x_max / 20.0);
                let expected_f = x.powf(p);
                let expected_f1 = x.powf(p + 1.0) / (p + 1.0);
                let scale_f = expected_f.abs().max(1.0);
                let scale_f1 = expected_f1.abs().max(1.0);
                assert!((lut.eval_f(x) - expected_f).abs() < 1e-4 * scale_f);
                assert!((lut.eval_f1(x) - expected_f1).abs() < 1e-4 * scale_f1);
            }

            // Derivative consistency: (F1(x + h) - F1(x - h)) / (2h) ~= f(x)
            let h = x_max * 5.0e-4_f32;
            for i in 2..19 {
                let x = (i as f32) * (x_max / 20.0);
                let num_deriv = (lut.eval_f1(x + h) - lut.eval_f1(x - h)) / (2.0 * h);
                let f_val = lut.eval_f(x);
                let tol = 1e-3 * f_val.abs().max(1.0);
                assert!(
                    (num_deriv - f_val).abs() < tol,
                    "derivative mismatch at p={p}, x={x}: {num_deriv} vs {f_val}"
                );
            }
        }

        let lut = AdaaPowerLut::new(2.8, 1.0, 4096);
        // Extension above x_max preserves dF1/dx == f(x_max).
        assert!((lut.eval_f(1.5) - 1.0).abs() < 1e-6);
        assert!((lut.adaa1(1.8, 1.2) - 1.0).abs() < 1e-5);

        // Guard triggers cleanly when |x - x_prev| < 1e-6 (including x == x_prev == 0).
        assert_eq!(lut.adaa1(0.0, 0.0), 0.0);
        let x0 = 0.65_f32;
        let x1 = x0 + 4.0e-7_f32;
        let guarded = lut.adaa1(x1, x0);
        assert!(guarded.is_finite());
        assert!((guarded - lut.eval_f(0.5 * (x0 + x1))).abs() < 1e-7);

        let mut state = AdaaState::new();
        let y0 = state.process(&lut, 0.3);
        let y1 = state.process(&lut, 0.3 + 2.0e-7);
        assert!(y0.is_finite() && y1.is_finite());
        state.reset();
        assert_eq!(state.process(&lut, 0.0), 0.0);
    }
}
