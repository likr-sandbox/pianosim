//! Tier configuration constants, SIMD slice addition, and RMS-matched
//! late-partition stereo collapse helpers for `NupolsTier` and `SpacePrepContext`.

use piano_simd::{F32x4, SplitComplexBuf};

use crate::ir_synth::path_index;
use crate::math::sqrt_f64;

pub(crate) const IN_RING_LEN: usize = 16_384;
pub(crate) const IN_RING_MASK: usize = IN_RING_LEN - 1;
pub(crate) const OUT_RING_LEN: usize = 16_384;
pub(crate) const OUT_RING_MASK: usize = OUT_RING_LEN - 1;
pub(crate) const MAX_TIER_FFT_LEN: usize = 8_192;

const COLLAPSE_SIGNS: [f32; 8] = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0];

#[derive(Clone, Copy)]
pub(crate) struct TierConfig {
    pub(crate) block_size: usize,
    pub(crate) ir_offset: usize,
    pub(crate) end_limit: usize,
}

pub(crate) const SUB_TIER_32_CFG: TierConfig = TierConfig {
    block_size: 32,
    ir_offset: 32,
    end_limit: 64,
};

pub(crate) const SUB_TIER_16_CFG: TierConfig = TierConfig {
    block_size: 16,
    ir_offset: 16,
    end_limit: 64,
};

pub(crate) const TIER_CONFIGS: [TierConfig; 4] = [
    TierConfig {
        block_size: 64,
        ir_offset: 64,
        end_limit: 512,
    },
    TierConfig {
        block_size: 256,
        ir_offset: 512,
        end_limit: 2_048,
    },
    TierConfig {
        block_size: 1_024,
        ir_offset: 2_048,
        end_limit: 8_192,
    },
    TierConfig {
        block_size: 4_096,
        ir_offset: 8_192,
        end_limit: usize::MAX,
    },
];

#[inline]
pub(crate) fn compute_p_count(cfg: TierConfig, ir_len: usize) -> usize {
    let end = ir_len.min(cfg.end_limit);
    if end > cfg.ir_offset {
        (end - cfg.ir_offset).div_ceil(cfg.block_size)
    } else {
        0
    }
}

#[inline]
pub(crate) fn compute_mimo_p_count(cfg: TierConfig, p_count: usize, split_tap: u16) -> usize {
    if split_tap == u16::MAX {
        p_count
    } else {
        let st = split_tap as usize;
        if st <= cfg.ir_offset {
            0
        } else {
            (st - cfg.ir_offset).div_ceil(cfg.block_size).min(p_count)
        }
    }
}

/// Collapse 9 zone spectra for channel `ch` into one RMS-energy-matched diffuse spectrum.
pub(crate) fn collapse_zone_spectra_into(
    zone_spectra: &[SplitComplexBuf],
    spec_len: usize,
    ch: usize,
    out_re: &mut [f32],
    out_im: &mut [f32],
) {
    let inv_8 = 1.0_f64 / 8.0;
    let z8_w = 0.35_f64;
    let z8_sq_w = 0.35_f64 / 8.0;
    for k in 0..spec_len {
        let mut s_re = 0.0_f64;
        let mut s_im = 0.0_f64;
        let mut p_sum = 0.0_f64;
        for (z, &sgn) in COLLAPSE_SIGNS.iter().enumerate() {
            let (zr, zi) = zone_spectra[path_index(z, ch)].planes();
            let re = f64::from(zr[k]);
            let im = f64::from(zi[k]);
            let sf = f64::from(sgn);
            s_re += sf * re;
            s_im += sf * im;
            p_sum += re * re + im * im;
        }
        let (r8, i8) = zone_spectra[path_index(8, ch)].planes();
        let re8 = f64::from(r8[k]);
        let im8 = f64::from(i8[k]);
        s_re += z8_w * re8;
        s_im += z8_w * im8;
        let rms = sqrt_f64(p_sum * inv_8 + z8_sq_w * (re8 * re8 + im8 * im8));
        let s_mag = sqrt_f64(s_re * s_re + s_im * s_im);
        if s_mag >= 1.0e-12 {
            let scale = rms / s_mag;
            out_re[k] = (s_re * scale) as f32;
            out_im[k] = (s_im * scale) as f32;
        } else {
            out_re[k] = rms as f32;
            out_im[k] = 0.0;
        }
    }
}

#[inline]
pub(crate) fn add_simd_slice(dst: &mut [f32], src: &[f32]) {
    let mut i = 0;
    while i + 4 <= dst.len() {
        let d = F32x4::from_slice(&dst[i..i + 4]);
        let s = F32x4::from_slice(&src[i..i + 4]);
        (d + s).write_to_slice(&mut dst[i..i + 4]);
        i += 4;
    }
    while i < dst.len() {
        dst[i] += src[i];
        i += 1;
    }
}
