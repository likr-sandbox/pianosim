//! 16-tap Blackman-Nuttall polyphase `4x` upsampler for low-band modal IRs.

use piano_simd::F32x4;

use crate::math::{sin_cos_f64, sinc_f64, PI};
use crate::modal_synth::NUM_IR_PATHS;

pub(crate) const DECIM: usize = 4;
pub(crate) const PAD_LEFT: usize = 8;
pub(crate) const PAD_RIGHT: usize = 9;
const UPSAMPLE_TAPS: usize = 16;

pub(crate) fn upsample_4x_add(
    sub_flat: &[f32],
    sub_stride: usize,
    ir_len: usize,
    ir_flat: &mut [f32],
) {
    let mut k1 = [0.0_f32; UPSAMPLE_TAPS];
    let mut k2 = [0.0_f32; UPSAMPLE_TAPS];
    let mut k3 = [0.0_f32; UPSAMPLE_TAPS];
    build_polyphase_row(0.25, &mut k1);
    build_polyphase_row(0.50, &mut k2);
    build_polyphase_row(0.75, &mut k3);

    let full_quads = ir_len / DECIM;
    let rem = ir_len % DECIM;

    let mut path = 0;
    while path < NUM_IR_PATHS {
        let src = &sub_flat[path * sub_stride..(path + 1) * sub_stride];
        let dst = &mut ir_flat[path * ir_len..(path + 1) * ir_len];

        let mut m = 0;
        while m < full_quads {
            let s = &src[m + 1..m + 1 + UPSAMPLE_TAPS];
            let base = m * DECIM;
            dst[base] += src[m + PAD_LEFT];
            dst[base + 1] += dot16(s, &k1);
            dst[base + 2] += dot16(s, &k2);
            dst[base + 3] += dot16(s, &k3);
            m += 1;
        }
        if rem > 0 {
            let s = &src[m + 1..m + 1 + UPSAMPLE_TAPS];
            let base = m * DECIM;
            dst[base] += src[m + PAD_LEFT];
            if rem > 1 {
                dst[base + 1] += dot16(s, &k1);
            }
            if rem > 2 {
                dst[base + 2] += dot16(s, &k2);
            }
        }
        path += 1;
    }
}

fn build_polyphase_row(phase: f64, out: &mut [f32; UPSAMPLE_TAPS]) {
    let mut tmp = [0.0_f64; UPSAMPLE_TAPS];
    let mut sum = 0.0_f64;
    let mut tap = 0;
    while tap < UPSAMPLE_TAPS {
        let k_rel = (tap as f64) - 7.0;
        let x = phase - k_rel;
        let u = (x + 8.0) / 16.0;
        let (_, c1) = sin_cos_f64(2.0 * PI * u);
        let (_, c2) = sin_cos_f64(4.0 * PI * u);
        let (_, c3) = sin_cos_f64(6.0 * PI * u);
        let win = 0.363_581_9 - 0.489_177_5 * c1 + 0.136_599_5 * c2 - 0.010_641_1 * c3;
        let w = sinc_f64(x) * win;
        tmp[tap] = w;
        sum += w;
        tap += 1;
    }
    let inv = 1.0 / sum;
    let mut i = 0;
    while i < UPSAMPLE_TAPS {
        out[i] = (tmp[i] * inv) as f32;
        i += 1;
    }
}

#[inline]
fn dot16(s: &[f32], k: &[f32; UPSAMPLE_TAPS]) -> f32 {
    let a0 = F32x4::from_slice(&s[0..4]) * F32x4::from_slice(&k[0..4]);
    let a1 = F32x4::from_slice(&s[4..8]) * F32x4::from_slice(&k[4..8]);
    let a2 = F32x4::from_slice(&s[8..12]) * F32x4::from_slice(&k[8..12]);
    let a3 = F32x4::from_slice(&s[12..16]) * F32x4::from_slice(&k[12..16]);
    ((a0 + a1) + (a2 + a3)).reduce_sum()
}
