//! Pure-Rust `no_std` mathematical helpers for initialization-time precomputation
//! and scalar fallback operations.

use core::f64::consts::{FRAC_2_PI, LN_2, LOG2_E, SQRT_2};

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

const LN_ODD_COEFFS: [f64; 10] = [
    1.0,
    1.0 / 3.0,
    1.0 / 5.0,
    1.0 / 7.0,
    1.0 / 9.0,
    1.0 / 11.0,
    1.0 / 13.0,
    1.0 / 15.0,
    1.0 / 17.0,
    1.0 / 19.0,
];

const EXP_COEFFS: [f64; 13] = [
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
];

#[inline]
fn eval_horner(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for &c in coeffs.iter().rev() {
        acc = acc * x + c;
    }
    acc
}

/// Returns the IEEE-754 absolute value of `x` without requiring `std`.
#[inline(always)]
pub(crate) fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

/// Computes `sqrt(x)` in pure Rust (`no_std`) to IEEE-754 `f32` precision.
#[inline]
pub(crate) fn sqrt_f32(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }

    // Scale subnormals into the normal range before Newton-Raphson refinement.
    let (scaled, post_scale) = if x < f32::MIN_POSITIVE {
        (x * 16_777_216.0, 1.0_f64 / 4096.0)
    } else {
        (x, 1.0_f64)
    };

    let seed = f32::from_bits((scaled.to_bits() >> 1) + 0x1fc0_0000);
    let xd = f64::from(scaled);
    let mut y = f64::from(seed);
    for _ in 0..4 {
        y = 0.5 * (y + xd / y);
    }
    (y * post_scale) as f32
}

#[inline]
fn round_f64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

/// Computes `(sin(angle), cos(angle))` in `f64` for init-time twiddle tables.
pub(crate) fn sin_cos_f64(angle: f64) -> (f64, f64) {
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

/// Computes `ln(x)` for positive finite `f64` values using IEEE-754 decomposition.
pub(crate) fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    let mut val = x;
    let mut exp_bias: i32 = 0;
    if val < f64::MIN_POSITIVE {
        // 2^54 brings any f64 subnormal into normal range.
        val *= 18_014_398_509_481_984.0;
        exp_bias = -54;
    }

    let bits = val.to_bits();
    let mut k = ((bits >> 52) as i32) - 1023 + exp_bias;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT_2 {
        m *= 0.5;
        k += 1;
    }

    let u = (m - 1.0) / (m + 1.0);
    let series = eval_horner(u * u, &LN_ODD_COEFFS);
    f64::from(k) * LN_2 + 2.0 * u * series
}

/// Computes `exp(y)` in `f64` for init-time power evaluation.
pub(crate) fn exp_f64(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
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
        let shift = (k + 1077) as u64;
        let sub_scale = f64::from_bits(shift << 52);
        let tiny_scale = f64::from_bits((1023_u64 - 55) << 52);
        (poly * sub_scale) * tiny_scale
    } else {
        let pow2k = f64::from_bits(((k + 1023) as u64) << 52);
        poly * pow2k
    }
}

/// Computes `base.powf(exp)` for `base >= 0.0` at init time without `std`.
#[inline]
pub(crate) fn pow_nonneg_f64(base: f64, exp: f64) -> f64 {
    if base <= 0.0 {
        if exp == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        exp_f64(exp * ln_f64(base))
    }
}
