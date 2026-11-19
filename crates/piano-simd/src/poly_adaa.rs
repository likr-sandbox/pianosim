//! Analytical degree-6 Bernstein-to-monomial polynomial approximation of
//! unilateral power contact `f(u) = max(0, u)^p` and its exact antiderivative
//! for zero-LUT, SIMD-friendly 1st-order ADAA and Newton K-method solving.

use core::f64::consts::PI;

use crate::lut_adaa::AdaaState;
use crate::math::{abs_f32, pow_nonneg_f64, sin_cos_f64};

/// Finite-difference guard threshold `|u - u_prev| < 1e-4` for polynomial ADAA midpoint fallback.
pub const POLY_ADAA_DIFF_EPS: f32 = 1.0e-4;

const CHEB_SAMPLES: usize = 64;
const COMB6: [f64; 7] = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];

/// Analytical monotone, non-negative, convex degree-6 polynomial approximation
/// `P(u) = u^2 * (c2 + c3*u + c4*u^2 + c5*u^3 + c6*u^4)` to `u^p` on `[0, 1]`.
///
/// Clamps to `0.0` for `u <= 0.0` and saturates at `f_at_1` for `u >= 1.0`,
/// with exact `C^1` analytical antiderivative `P_1(u)` extended linearly above `1.0`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PolyPowerApprox {
    exponent: f32,
    c: [f32; 5],
    c_int: [f32; 5],
    f_at_1: f32,
    f1_at_1: f32,
}

impl PolyPowerApprox {
    /// Constructs a convex, monotone, non-negative degree-6 polynomial fitting `u^p`
    /// on `[0, 1]` via Chebyshev-node active-set least squares on Bernstein weights.
    #[must_use]
    pub fn new(p: f32) -> Self {
        let p_clamped = if p.is_finite() {
            p.clamp(2.0, 4.0)
        } else {
            2.8
        };
        let p_f64 = f64::from(p_clamped);
        let w = fit_convex_bernstein_weights(p_f64);
        let c_f64 = bernstein_to_monomial_tail(&w);

        let mut c = [0.0_f32; 5];
        let mut c_int = [0.0_f32; 5];
        let mut sum_f = 0.0_f64;
        let mut sum_f1 = 0.0_f64;
        for i in 0..5 {
            c[i] = c_f64[i] as f32;
            let ci = c_f64[i] / ((i + 3) as f64);
            c_int[i] = ci as f32;
            sum_f += c_f64[i];
            sum_f1 += ci;
        }

        Self {
            exponent: p_clamped,
            c,
            c_int,
            f_at_1: sum_f as f32,
            f1_at_1: sum_f1 as f32,
        }
    }

    /// Configured power exponent `p`.
    #[must_use]
    #[inline]
    pub const fn exponent(&self) -> f32 {
        self.exponent
    }

    /// Monomial coefficients `[c2, c3, c4, c5, c6]`.
    #[must_use]
    #[inline]
    pub const fn coeffs(&self) -> [f32; 5] {
        self.c
    }

    /// Value `P(1.0)` at the upper saturation boundary.
    #[must_use]
    #[inline]
    pub const fn f_at_1(&self) -> f32 {
        self.f_at_1
    }

    /// Exact antiderivative value `P_1(1.0)` at the upper saturation boundary.
    #[must_use]
    #[inline]
    pub const fn f1_at_1(&self) -> f32 {
        self.f1_at_1
    }

    /// Evaluates `P(u)` for normalized compression `u` (`0.0` for `u <= 0.0`, `f_at_1` for `u >= 1.0`).
    #[must_use]
    #[inline]
    pub fn eval_f(&self, u: f32) -> f32 {
        if u <= 0.0 || u.is_nan() {
            return 0.0;
        }
        if u >= 1.0 {
            return self.f_at_1;
        }
        let c = &self.c;
        let poly = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * c[4])));
        u * u * poly
    }

    /// Evaluates `(P(u), P'(u))` simultaneously in Horner form.
    #[must_use]
    #[inline]
    pub fn eval_f_and_df(&self, u: f32) -> (f32, f32) {
        if u <= 0.0 || u.is_nan() {
            return (0.0, 0.0);
        }
        if u > 1.0 {
            return (self.f_at_1, 0.0);
        }
        let c = &self.c;
        let poly = c[0] + u * (c[1] + u * (c[2] + u * (c[3] + u * c[4])));
        let dpoly =
            2.0 * c[0] + u * (3.0 * c[1] + u * (4.0 * c[2] + u * (5.0 * c[3] + u * (6.0 * c[4]))));
        let f = if u == 1.0 { self.f_at_1 } else { u * u * poly };
        (f, u * dpoly)
    }

    /// Evaluates the exact analytical first antiderivative `P_1(u)`.
    #[must_use]
    #[inline]
    pub fn eval_f1(&self, u: f32) -> f32 {
        if u <= 0.0 || u.is_nan() {
            return 0.0;
        }
        if u >= 1.0 {
            return self.f1_at_1 + self.f_at_1 * (u - 1.0);
        }
        let d = &self.c_int;
        let q = d[0] + u * (d[1] + u * (d[2] + u * (d[3] + u * d[4])));
        u * u * u * q
    }

    /// Evaluates first-order ADAA output across step `u_prev -> u`.
    ///
    /// Falls back to midpoint `eval_f(0.5 * (u + u_prev))` whenever `|u - u_prev| < 1e-4`.
    #[must_use]
    #[inline]
    pub fn adaa1(&self, u: f32, u_prev: f32) -> f32 {
        if (u <= 0.0 || u.is_nan()) && (u_prev <= 0.0 || u_prev.is_nan()) {
            return 0.0;
        }
        let du = u - u_prev;
        if abs_f32(du) >= POLY_ADAA_DIFF_EPS && du.is_finite() {
            (self.eval_f1(u) - self.eval_f1(u_prev)) / du
        } else {
            self.eval_f(0.5 * (u + u_prev))
        }
    }

    /// Advances stateful 1st-order ADAA by one sample and updates `state`.
    #[inline]
    pub fn process(&self, u: f32, state: &mut AdaaState) -> f32 {
        let out = self.adaa1(u, state.prev());
        state.sync_x(u);
        out
    }
}

fn fit_convex_bernstein_weights(p: f64) -> [f64; 7] {
    let mut a = [[0.0_f64; 4]; 4];
    let mut b = [0.0_f64; 4];

    for idx in 0..CHEB_SAMPLES {
        let theta = PI * ((idx as f64) + 0.5) / (CHEB_SAMPLES as f64);
        let (_, cos_t) = sin_cos_f64(theta);
        let u = 0.5 * (1.0 - cos_t);
        let um = 1.0 - u;

        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let u5 = u4 * u;
        let u6 = u3 * u3;

        let um1 = um;
        let um2 = um1 * um1;
        let um3 = um2 * um1;
        let um4 = um2 * um2;

        let phi = [
            COMB6[2] * u2 * um4,
            COMB6[3] * u3 * um3,
            COMB6[4] * u4 * um2,
            COMB6[5] * u5 * um1,
        ];
        let rhs = pow_nonneg_f64(u, p) - COMB6[6] * u6;

        for i in 0..4 {
            b[i] += phi[i] * rhs;
            for j in 0..4 {
                a[i][j] += phi[i] * phi[j];
            }
        }
    }

    let mut w_mid = solve_4x4(a, b);
    if w_mid[0] <= 0.0 {
        let a3 = [
            [a[1][1], a[1][2], a[1][3]],
            [a[2][1], a[2][2], a[2][3]],
            [a[3][1], a[3][2], a[3][3]],
        ];
        let b3 = [b[1], b[2], b[3]];
        let sub = solve_3x3(a3, b3);
        w_mid = [0.0, sub[0], sub[1], sub[2]];
    }

    let mut w = [0.0, 0.0, w_mid[0], w_mid[1], w_mid[2], w_mid[3], 1.0];
    enforce_bernstein_convexity(&mut w);
    w
}

fn enforce_bernstein_convexity(w: &mut [f64; 7]) {
    let valid = |arr: &[f64; 7]| -> bool {
        for k in 1..7 {
            if arr[k] < arr[k - 1] - 1.0e-15 {
                return false;
            }
        }
        for k in 0..5 {
            if arr[k + 2] - 2.0 * arr[k + 1] + arr[k] < -1.0e-15 {
                return false;
            }
        }
        true
    };
    if valid(w) {
        return;
    }
    // Project onto convex monotone cone via iterative coordinate clamping.
    for k in 2..6 {
        w[k] = w[k].clamp(w[k - 1], 1.0);
    }
    for _ in 0..16 {
        for i in 2..6 {
            let lo = (2.0 * w[i - 1] - w[i - 2]).max(2.0 * w[i + 1] - w[i + 2]);
            let hi = 0.5 * (w[i - 1] + w[i + 1]);
            let lo_m = lo.max(w[i - 1]);
            let hi_m = hi.min(w[i + 1]);
            if lo_m <= hi_m {
                w[i] = w[i].clamp(lo_m, hi_m);
            } else {
                w[i] = 0.5 * (lo_m + hi_m);
            }
        }
    }
}

fn bernstein_to_monomial_tail(w: &[f64; 7]) -> [f64; 5] {
    let mut c = [0.0_f64; 5];
    c[0] = 15.0 * w[2];
    c[1] = -60.0 * w[2] + 20.0 * w[3];
    c[2] = 90.0 * w[2] - 60.0 * w[3] + 15.0 * w[4];
    c[3] = -60.0 * w[2] + 60.0 * w[3] - 30.0 * w[4] + 6.0 * w[5];
    c[4] = 15.0 * w[2] - 20.0 * w[3] + 15.0 * w[4] - 6.0 * w[5] + w[6];
    c
}

#[inline]
fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

fn solve_4x4(a: [[f64; 4]; 4], b: [f64; 4]) -> [f64; 4] {
    let mut m = [[0.0_f64; 5]; 4];
    for r in 0..4 {
        m[r][..4].copy_from_slice(&a[r]);
        m[r][4] = b[r];
    }
    for col in 0..4 {
        let mut piv = col;
        let mut max_v = abs_f64(m[col][col]);
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            let v = abs_f64(row[col]);
            if v > max_v {
                max_v = v;
                piv = r;
            }
        }
        if piv != col {
            m.swap(col, piv);
        }
        let inv = 1.0 / m[col][col];
        for val in &mut m[col][col..5] {
            *val *= inv;
        }
        let pivot_row = m[col];
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let f = row[col];
                for (dst, &src) in row[col..5].iter_mut().zip(&pivot_row[col..5]) {
                    *dst -= f * src;
                }
            }
        }
    }
    [m[0][4], m[1][4], m[2][4], m[3][4]]
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut m = [[0.0_f64; 4]; 3];
    for r in 0..3 {
        m[r][..3].copy_from_slice(&a[r]);
        m[r][3] = b[r];
    }
    for col in 0..3 {
        let mut piv = col;
        let mut max_v = abs_f64(m[col][col]);
        for (r, row) in m.iter().enumerate().skip(col + 1) {
            let v = abs_f64(row[col]);
            if v > max_v {
                max_v = v;
                piv = r;
            }
        }
        if piv != col {
            m.swap(col, piv);
        }
        let inv = 1.0 / m[col][col];
        for val in &mut m[col][col..4] {
            *val *= inv;
        }
        let pivot_row = m[col];
        for (r, row) in m.iter_mut().enumerate() {
            if r != col {
                let f = row[col];
                for (dst, &src) in row[col..4].iter_mut().zip(&pivot_row[col..4]) {
                    *dst -= f * src;
                }
            }
        }
    }
    [m[0][3], m[1][3], m[2][3]]
}
