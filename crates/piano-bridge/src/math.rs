//! Pure-Rust `no_std` mathematical helpers and coupled-form pole computation
//! for initialization-time precomputation in `piano-bridge`.

use core::f64::consts::{FRAC_2_PI, LN_10, LOG2_E, TAU};

const LN_1000: f64 = 3.0 * LN_10;
const PIO2_HI: f64 = 1.570_796_326_734_125_6;
const PIO2_LO: f64 = 6.077_100_506_506_192e-11;
const LN2_HI: f64 = 0.693_147_180_485_539_1;
const LN2_LO: f64 = 7.440_614_790_965_583e-11;

const SIN_COEFFS: [f64; 8] = [
    1.0,
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5_040.0,
    1.0 / 362_880.0,
    -1.0 / 39_916_800.0,
    1.0 / 6_227_020_800.0,
    -1.0 / 1_307_674_368_000.0,
];
const COS_COEFFS: [f64; 8] = [
    1.0,
    -0.5,
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40_320.0,
    -1.0 / 3_628_800.0,
    1.0 / 479_001_600.0,
    -1.0 / 87_178_291_200.0,
];
const EXP_COEFFS: [f64; 14] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5_040.0,
    1.0 / 40_320.0,
    1.0 / 362_880.0,
    1.0 / 3_628_800.0,
    1.0 / 39_916_800.0,
    1.0 / 479_001_600.0,
    1.0 / 6_227_020_800.0,
];
const ONE_MINUS_EXP_NEG_DIV_ALPHA: [f64; 6] =
    [1.0, -0.5, 1.0 / 6.0, -1.0 / 24.0, 1.0 / 120.0, -1.0 / 720.0];

#[inline]
fn eval_horner(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    let mut i = coeffs.len();
    while i > 0 {
        i -= 1;
        acc = acc * x + coeffs[i];
    }
    acc
}

#[inline]
fn round_f64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

#[inline]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    let (scaled, post_scale) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, 1.0_f64 / 134_217_728.0)
    } else {
        (x, 1.0_f64)
    };
    let mut y = f64::from_bits((scaled.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..6 {
        y = 0.5 * (y + scaled / y);
    }
    y * post_scale
}

pub(crate) fn sin_cos_f64(angle: f64) -> (f64, f64) {
    if !angle.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let q = round_f64(angle * FRAC_2_PI);
    let qf = q as f64;
    let r = (angle - qf * PIO2_HI) - qf * PIO2_LO;
    let r2 = r * r;
    let s = r * eval_horner(r2, &SIN_COEFFS);
    let c = eval_horner(r2, &COS_COEFFS);
    match q.rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

pub(crate) fn exp_f64(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y > 709.78 {
        return f64::INFINITY;
    }
    if y < -745.13 {
        return 0.0;
    }
    let k = round_f64(y * LOG2_E);
    let kf = k as f64;
    let r = (y - kf * LN2_HI) - kf * LN2_LO;
    let poly = eval_horner(r, &EXP_COEFFS);
    if k < -1022 {
        let sub_scale = f64::from_bits(((k + 1077) as u64) << 52);
        let tiny_scale = f64::from_bits((1023_u64 - 55) << 52);
        (poly * sub_scale) * tiny_scale
    } else {
        poly * f64::from_bits(((k + 1023) as u64) << 52)
    }
}

#[inline]
fn one_minus_exp_neg_f64(alpha: f64) -> f64 {
    if alpha <= 0.0 {
        0.0
    } else if alpha < 1.0e-3 {
        alpha * eval_horner(alpha, &ONE_MINUS_EXP_NEG_DIV_ALPHA)
    } else {
        1.0 - exp_f64(-alpha)
    }
}

/// Computes unit-circle-consistent coupled-form pole `(d, s)` in `f32`
/// where `(1 - d)^2 + s^2 == r_hold^2`.
#[inline]
pub(crate) fn coupled_pole_f32(
    freq_hz: f64,
    sample_rate: f64,
    t60_hold: f64,
) -> Option<(f32, f32)> {
    if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz >= 0.49 * sample_rate {
        return None;
    }
    let theta = TAU * freq_hz / sample_rate;
    let (sin_t, cos_t) = sin_cos_f64(theta);
    let (one_minus_rh, r_hold) = if t60_hold.is_infinite() && t60_hold.is_sign_positive() {
        (0.0, 1.0)
    } else {
        let one_minus = one_minus_exp_neg_f64(LN_1000 / (t60_hold * sample_rate));
        (one_minus, 1.0 - one_minus)
    };
    let s_f32 = (r_hold * sin_t) as f32;
    let s_ratio = (f64::from(s_f32) / r_hold).clamp(-1.0, 1.0);
    let cos_mag = sqrt_f64((1.0 - s_ratio * s_ratio).max(0.0));
    let d_f32 = if cos_t >= 0.0 {
        (one_minus_rh + r_hold * (s_ratio * s_ratio) / (1.0 + cos_mag)) as f32
    } else {
        (1.0 + r_hold * cos_mag) as f32
    };
    Some((d_f32, s_f32))
}
