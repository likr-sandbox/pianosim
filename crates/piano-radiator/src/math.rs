//! Pure-Rust `no_std` mathematical helpers and `10x10` linear solver.

pub(crate) const PI: f64 = core::f64::consts::PI;
pub(crate) const FRAC_PI_2: f64 = core::f64::consts::FRAC_PI_2;
pub(crate) const FRAC_PI_4: f64 = core::f64::consts::FRAC_PI_4;
pub(crate) const LN_2: f64 = core::f64::consts::LN_2;

// Exact 24-bit leading chunks for Cody-Waite reduction of `pi/2` and `ln(2)`.
const PIO2_HI: f64 = f64::from_bits(0x3ff9_21fb_5440_0000);
const PIO2_MID: f64 = f64::from_bits(0x3dd0_b461_1a60_0000);
const PIO2_LO: f64 = f64::from_bits(0x3ba3_198a_2e03_7073);

const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fefa_0000);
const LN2_LO: f64 = f64::from_bits(0x3d7c_f79a_bc9e_3b3a);

const INV_LN2: f64 = core::f64::consts::LOG2_E;
const SQRT_2: f64 = core::f64::consts::SQRT_2;
const TAN_PI_8: f64 = SQRT_2 - 1.0;
const TAN_3PI_8: f64 = SQRT_2 + 1.0;

#[inline]
pub(crate) fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

#[inline]
pub(crate) fn min_f64(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub(crate) fn max_f64(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
pub(crate) fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    max_f64(lo, min_f64(hi, x))
}

#[inline]
pub(crate) fn eval_horner(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0;
    let mut idx = coeffs.len();
    while idx > 0 {
        idx -= 1;
        acc = acc * x + coeffs[idx];
    }
    acc
}

#[inline]
pub(crate) fn floor_f64(x: f64) -> f64 {
    if !x.is_finite() || abs_f64(x) >= 4_503_599_627_370_496.0 {
        return x;
    }
    let trunc = (x as i64) as f64;
    if x < trunc {
        trunc - 1.0
    } else {
        trunc
    }
}

#[inline]
pub(crate) fn round_f64(x: f64) -> f64 {
    if !x.is_finite() || abs_f64(x) >= 4_503_599_627_370_496.0 {
        return x;
    }
    if x >= 0.0 {
        floor_f64(x + 0.5)
    } else {
        -floor_f64(-x + 0.5)
    }
}

pub(crate) fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut val = x;
    let mut scale = 1.0;
    if val < 2.225_073_858_507_201_4e-308 {
        val *= 1.801_439_850_948_198_4e16; // 2^54
        scale = 7.450_580_596_923_828e-9; // 2^-27
    }
    let mut y = f64::from_bits((val.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    let mut iter = 0;
    while iter < 6 {
        y = 0.5 * (y + val / y);
        iter += 1;
    }
    y * scale
}

pub(crate) fn sin_cos_f64(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    let q_f = round_f64(x / FRAC_PI_2);
    let q = (q_f as i64).rem_euclid(4);
    let r = ((x - q_f * PIO2_HI) - q_f * PIO2_MID) - q_f * PIO2_LO;
    let r2 = r * r;

    const SIN_POLY: [f64; 8] = [
        1.0,
        -1.0 / 6.0,
        1.0 / 120.0,
        -1.0 / 5_040.0,
        1.0 / 362_880.0,
        -1.0 / 39_916_800.0,
        1.0 / 6_227_020_800.0,
        -1.0 / 1_307_674_368_000.0,
    ];
    const COS_POLY: [f64; 9] = [
        1.0,
        -1.0 / 2.0,
        1.0 / 24.0,
        -1.0 / 720.0,
        1.0 / 40_320.0,
        -1.0 / 3_628_800.0,
        1.0 / 479_001_600.0,
        -1.0 / 87_178_291_200.0,
        1.0 / 20_922_789_888_000.0,
    ];

    let s = r * eval_horner(r2, &SIN_POLY);
    let c = eval_horner(r2, &COS_POLY);
    match q {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[inline]
pub(crate) fn tan_f64(x: f64) -> f64 {
    let (s, c) = sin_cos_f64(x);
    s / c
}

fn atan_small(x: f64) -> f64 {
    const ATAN_POLY: [f64; 20] = [
        1.0,
        -1.0 / 3.0,
        1.0 / 5.0,
        -1.0 / 7.0,
        1.0 / 9.0,
        -1.0 / 11.0,
        1.0 / 13.0,
        -1.0 / 15.0,
        1.0 / 17.0,
        -1.0 / 19.0,
        1.0 / 21.0,
        -1.0 / 23.0,
        1.0 / 25.0,
        -1.0 / 27.0,
        1.0 / 29.0,
        -1.0 / 31.0,
        1.0 / 33.0,
        -1.0 / 35.0,
        1.0 / 37.0,
        -1.0 / 39.0,
    ];
    x * eval_horner(x * x, &ATAN_POLY)
}

pub(crate) fn atan_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else if x.is_sign_positive() {
            FRAC_PI_2
        } else {
            -FRAC_PI_2
        };
    }
    let ax = abs_f64(x);
    let res = if ax > TAN_3PI_8 {
        FRAC_PI_2 - atan_small(1.0 / ax)
    } else if ax > TAN_PI_8 {
        FRAC_PI_4 + atan_small((ax - 1.0) / (ax + 1.0))
    } else {
        atan_small(ax)
    };
    if x < 0.0 {
        -res
    } else {
        res
    }
}

pub(crate) fn atan2_f64(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        atan_f64(y / x)
    } else if x < 0.0 {
        if y >= 0.0 {
            atan_f64(y / x) + PI
        } else {
            atan_f64(y / x) - PI
        }
    } else if y > 0.0 {
        FRAC_PI_2
    } else if y < 0.0 {
        -FRAC_PI_2
    } else {
        0.0
    }
}

pub(crate) fn asin_f64(x: f64) -> f64 {
    let clamped = clamp_f64(x, -1.0, 1.0);
    let denom = sqrt_f64(max_f64(0.0, 1.0 - clamped * clamped));
    if denom < 1.0e-15 {
        if clamped >= 0.0 {
            FRAC_PI_2
        } else {
            -FRAC_PI_2
        }
    } else {
        atan_f64(clamped / denom)
    }
}

#[cfg(test)]
pub(crate) fn acos_f64(x: f64) -> f64 {
    FRAC_PI_2 - asin_f64(x)
}

pub(crate) fn exp_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() {
            f64::NAN
        } else if x > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }
    if x >= 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x <= -745.133_219_101_941_2 {
        return 0.0;
    }
    let k_f = round_f64(x * INV_LN2);
    let k = k_f as i64;
    let r = (x - k_f * LN2_HI) - k_f * LN2_LO;

    const EXP_POLY: [f64; 14] = [
        1.0,
        1.0,
        1.0 / 2.0,
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
    let er = eval_horner(r, &EXP_POLY);

    if k < -1022 {
        let pow2_k = f64::from_bits(((k + 1023 + 64) as u64) << 52);
        (er * pow2_k) * 5.421_010_862_427_522e-20 // 2^-64
    } else {
        let pow2_k = f64::from_bits(((k + 1023) as u64) << 52);
        er * pow2_k
    }
}

pub(crate) fn ln_f64(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if !x.is_finite() {
        return f64::INFINITY;
    }

    let mut val = x;
    let mut e_offset: i64 = 0;
    if val < 2.225_073_858_507_201_4e-308 {
        val *= 1.801_439_850_948_198_4e16; // 2^54
        e_offset = -54;
    }
    let bits = val.to_bits();
    let mut exp = (((bits >> 52) & 0x7ff) as i64) - 1023 + e_offset;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let u = (m - 1.0) / (m + 1.0);
    const LN_POLY: [f64; 9] = [
        1.0,
        1.0 / 3.0,
        1.0 / 5.0,
        1.0 / 7.0,
        1.0 / 9.0,
        1.0 / 11.0,
        1.0 / 13.0,
        1.0 / 15.0,
        1.0 / 17.0,
    ];
    let ln_m = 2.0 * u * eval_horner(u * u, &LN_POLY);
    (exp as f64) * LN_2 + ln_m
}

pub(crate) fn powf_f64(base: f64, exp: f64) -> f64 {
    if exp == 0.0 {
        return 1.0;
    }
    if base <= 0.0 {
        return if base == 0.0 && exp > 0.0 {
            0.0
        } else {
            f64::NAN
        };
    }
    exp_f64(exp * ln_f64(base))
}

#[inline]
pub(crate) fn sinc_f64(x: f64) -> f64 {
    let ax = abs_f64(x);
    if ax < 1.0e-8 {
        let px = PI * x;
        1.0 - (px * px) / 6.0
    } else {
        let px = PI * x;
        let (s, _) = sin_cos_f64(px);
        s / px
    }
}

/// Invert a `10x10` matrix using Gauss-Jordan elimination with partial pivoting.
pub(crate) fn invert_10x10(mut a: [[f64; 10]; 10]) -> Option<[[f64; 10]; 10]> {
    let mut inv = [[0.0_f64; 10]; 10];
    let mut i = 0;
    while i < 10 {
        inv[i][i] = 1.0;
        i += 1;
    }

    let mut col = 0;
    while col < 10 {
        let mut pivot_row = col;
        let mut max_abs = abs_f64(a[col][col]);
        let mut row = col + 1;
        while row < 10 {
            let v = abs_f64(a[row][col]);
            if v > max_abs {
                max_abs = v;
                pivot_row = row;
            }
            row += 1;
        }
        if max_abs < 1.0e-24 || !max_abs.is_finite() {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }

        let inv_diag = 1.0 / a[col][col];
        let mut c = 0;
        while c < 10 {
            a[col][c] *= inv_diag;
            inv[col][c] *= inv_diag;
            c += 1;
        }

        let mut r = 0;
        while r < 10 {
            if r != col {
                let factor = a[r][col];
                if factor != 0.0 {
                    let mut c2 = 0;
                    while c2 < 10 {
                        a[r][c2] -= factor * a[col][c2];
                        inv[r][c2] -= factor * inv[col][c2];
                        c2 += 1;
                    }
                }
            }
            r += 1;
        }
        col += 1;
    }
    Some(inv)
}
