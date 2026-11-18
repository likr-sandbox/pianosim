//! Lookup-table first-order ADAA (Antiderivative Antialiasing) for `f(x) = x^p`
//! and `F_1(x) = x^(p+1)/(p+1)` on `x >= 0`.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::math::{abs_f32, pow_nonneg_f64};

/// Finite-difference guard threshold `|x - x_prev| < 1e-6` for ADAA midpoint fallback.
pub const ADAA_DIFF_EPS: f32 = 1.0e-6;

/// Precomputed linear-interpolation LUT for unilateral power contact `f(x) = max(0, x)^p`
/// and its first antiderivative `F_1(x) = max(0, x)^(p+1) / (p+1)`.
///
/// All heap allocation happens once in [`AdaaPowerLut::new`] / [`AdaaPowerLut::try_new`].
#[derive(Clone, Debug)]
pub struct AdaaPowerLut {
    exponent: f32,
    x_max: f32,
    dx: f32,
    inv_dx: f32,
    last_idx: usize,
    f_table: Box<[f32]>,
    f1_table: Box<[f32]>,
}

impl AdaaPowerLut {
    /// Builds a uniform LUT over `[0.0, x_max]` with `table_size >= 2` samples.
    ///
    /// # Panics
    /// Panics if `exponent <= 0.0`, `x_max <= 0.0`, `table_size < 2`, or any input is non-finite.
    #[must_use]
    pub fn new(exponent: f32, x_max: f32, table_size: usize) -> Self {
        Self::try_new(exponent, x_max, table_size)
            .expect("AdaaPowerLut requires finite exponent > 0, x_max > 0, and table_size >= 2")
    }

    /// Checked constructor returning `None` for invalid parameters.
    #[must_use]
    pub fn try_new(exponent: f32, x_max: f32, table_size: usize) -> Option<Self> {
        if !exponent.is_finite()
            || !x_max.is_finite()
            || exponent <= 0.0
            || x_max <= 0.0
            || table_size < 2
        {
            return None;
        }

        let last_idx = table_size - 1;
        let dx = f64::from(x_max) / (last_idx as f64);
        let p = f64::from(exponent);
        let p1 = p + 1.0;

        let mut f_table = Vec::with_capacity(table_size);
        let mut f1_table = Vec::with_capacity(table_size);

        for i in 0..table_size {
            let x = (i as f64) * dx;
            let fx = pow_nonneg_f64(x, p);
            let f1x = pow_nonneg_f64(x, p1) / p1;
            f_table.push(fx as f32);
            f1_table.push(f1x as f32);
        }

        Some(Self {
            exponent,
            x_max,
            dx: x_max / (last_idx as f32),
            inv_dx: (last_idx as f32) / x_max,
            last_idx,
            f_table: f_table.into_boxed_slice(),
            f1_table: f1_table.into_boxed_slice(),
        })
    }

    /// Configured power exponent `p`.
    #[must_use]
    #[inline]
    pub const fn exponent(&self) -> f32 {
        self.exponent
    }

    /// Upper domain bound `x_max` of the precomputed table.
    #[must_use]
    #[inline]
    pub const fn x_max(&self) -> f32 {
        self.x_max
    }

    /// Number of tabulated grid points.
    #[must_use]
    #[inline]
    pub fn table_size(&self) -> usize {
        self.f_table.len()
    }

    /// Evaluates `f(x) = x^p` via linear interpolation on `[0, x_max]`.
    ///
    /// Returns `0.0` for `x <= 0.0` (unilateral contact) and clamps smoothly to
    /// `f(x_max)` for `x >= x_max`. Zero-allocation and panic-free.
    #[must_use]
    #[inline]
    pub fn eval_f(&self, x: f32) -> f32 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= self.x_max {
            return self.f_table[self.last_idx];
        }
        sample_lut(&self.f_table, x, self.inv_dx, self.last_idx)
    }

    /// Evaluates first antiderivative `F_1(x) = x^(p+1) / (p+1)` via C^1 cubic Hermite
    /// interpolation using the tabulated values (`f1_table`) and slopes (`f_table`).
    ///
    /// Returns `0.0` for `x <= 0.0` and extends with constant slope `f(x_max)` for
    /// `x > x_max` so `dF_1/dx == f(x)` remains consistent above `x_max`.
    #[must_use]
    #[inline]
    pub fn eval_f1(&self, x: f32) -> f32 {
        if x <= 0.0 {
            return 0.0;
        }
        if x >= self.x_max {
            let f_end = self.f_table[self.last_idx];
            let f1_end = self.f1_table[self.last_idx];
            return f1_end + f_end * (x - self.x_max);
        }
        sample_hermite_f1(
            &self.f1_table,
            &self.f_table,
            x,
            self.inv_dx,
            self.dx,
            self.last_idx,
        )
    }

    /// Evaluates first-order ADAA output across step `x_prev -> x`.
    ///
    /// Falls back to midpoint `eval_f(0.5 * (x + x_prev))` whenever `|x - x_prev| < 1e-6`.
    #[must_use]
    #[inline]
    pub fn adaa1(&self, x: f32, x_prev: f32) -> f32 {
        let dx = x - x_prev;
        if abs_f32(dx) < ADAA_DIFF_EPS {
            self.eval_f(0.5 * (x + x_prev))
        } else {
            (self.eval_f1(x) - self.eval_f1(x_prev)) / dx
        }
    }

    /// Evaluates linearly interpolated `f(x) = (1 - t) * self.f(x) + t * other.f(x)`.
    ///
    /// Clamps `t` safely to `[0.0, 1.0]` (`NaN` maps to `0.0`). Zero-allocation and panic-free.
    #[must_use]
    #[inline]
    pub fn eval_f_lerp(&self, other: &Self, t: f32, x: f32) -> f32 {
        let t = clamp_unit(t);
        if x <= 0.0 || x.is_nan() {
            return 0.0;
        }
        if self.last_idx == other.last_idx && self.x_max == other.x_max {
            if x >= self.x_max {
                return lerp_f32(
                    self.f_table[self.last_idx],
                    other.f_table[other.last_idx],
                    t,
                );
            }
            let (f_lo, f_hi) =
                sample_lut_pair(&self.f_table, &other.f_table, x, self.inv_dx, self.last_idx);
            lerp_f32(f_lo, f_hi, t)
        } else {
            lerp_f32(self.eval_f(x), other.eval_f(x), t)
        }
    }

    /// Evaluates linearly interpolated antiderivative
    /// `F_1(x) = (1 - t) * self.F_1(x) + t * other.F_1(x)`.
    ///
    /// Clamps `t` safely to `[0.0, 1.0]` (`NaN` maps to `0.0`). Zero-allocation and panic-free.
    #[must_use]
    #[inline]
    pub fn eval_f1_lerp(&self, other: &Self, t: f32, x: f32) -> f32 {
        let t = clamp_unit(t);
        let (f1_lo, f1_hi) = eval_f1_pair(self, other, x);
        lerp_f32(f1_lo, f1_hi, t)
    }

    /// Evaluates first-order ADAA across `x_prev -> x` for the interpolated nonlinearity
    /// `(1 - t) * self + t * other` using the exact consistent antiderivative
    /// `(1 - t) * self.F_1 + t * other.F_1`.
    ///
    /// Falls back to midpoint `self.eval_f_lerp(other, t, 0.5 * (x + x_prev))` when
    /// `|x - x_prev| < ADAA_DIFF_EPS` (or non-finite `x - x_prev`), matching `dF_1_lerp/dx`.
    #[must_use]
    #[inline]
    pub fn adaa1_lerp(&self, other: &Self, t: f32, x: f32, x_prev: f32) -> f32 {
        if (x <= 0.0 || x.is_nan()) && (x_prev <= 0.0 || x_prev.is_nan()) {
            return 0.0;
        }
        let t = clamp_unit(t);
        let dx = x - x_prev;
        if abs_f32(dx) >= ADAA_DIFF_EPS && dx.is_finite() {
            let (cur_lo, cur_hi) = eval_f1_pair(self, other, x);
            let (prev_lo, prev_hi) = eval_f1_pair(self, other, x_prev);
            lerp_f32(cur_lo - prev_lo, cur_hi - prev_hi, t) / dx
        } else {
            self.eval_f_lerp(other, t, 0.5 * (x + x_prev))
        }
    }
}

/// Evaluates first-order ADAA across `x_prev -> x` between two adjacent LUTs at blend `t`.
#[must_use]
#[inline]
pub fn adaa1_lerp(
    lut_lo: &AdaaPowerLut,
    lut_hi: &AdaaPowerLut,
    t: f32,
    x: f32,
    x_prev: f32,
) -> f32 {
    lut_lo.adaa1_lerp(lut_hi, t, x, x_prev)
}

#[inline]
fn clamp_unit(t: f32) -> f32 {
    if t >= 1.0 {
        1.0
    } else if t > 0.0 {
        t
    } else {
        0.0
    }
}

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    (1.0 - t) * a + t * b
}

#[inline]
fn sample_lut(table: &[f32], x: f32, inv_dx: f32, last_idx: usize) -> f32 {
    let pos = x * inv_dx;
    let idx = (pos as usize).min(last_idx - 1);
    let frac = pos - (idx as f32);
    let y0 = table[idx];
    let y1 = table[idx + 1];
    y0 + frac * (y1 - y0)
}

#[inline]
fn sample_lut_pair(
    table_lo: &[f32],
    table_hi: &[f32],
    x: f32,
    inv_dx: f32,
    last_idx: usize,
) -> (f32, f32) {
    let pos = x * inv_dx;
    let idx = (pos as usize).min(last_idx - 1);
    let frac = pos - (idx as f32);
    let lo0 = table_lo[idx];
    let lo1 = table_lo[idx + 1];
    let hi0 = table_hi[idx];
    let hi1 = table_hi[idx + 1];
    (lo0 + frac * (lo1 - lo0), hi0 + frac * (hi1 - hi0))
}

#[inline]
fn sample_hermite_f1(
    f1_table: &[f32],
    f_table: &[f32],
    x: f32,
    inv_dx: f32,
    dx: f32,
    last_idx: usize,
) -> f32 {
    let pos = x * inv_dx;
    let idx = (pos as usize).min(last_idx - 1);
    let t = pos - (idx as f32);
    let u = 1.0 - t;
    let y0 = f1_table[idx];
    let y1 = f1_table[idx + 1];
    let m0 = f_table[idx] * dx;
    let m1 = f_table[idx + 1] * dx;
    (y0 + t * t * (3.0 - 2.0 * t) * (y1 - y0) + t * u * (u * m0 - t * m1)).max(0.0)
}

#[inline]
fn sample_hermite_f1_pair(lo: &AdaaPowerLut, hi: &AdaaPowerLut, x: f32) -> (f32, f32) {
    let pos = x * lo.inv_dx;
    let idx = (pos as usize).min(lo.last_idx - 1);
    let t = pos - (idx as f32);
    let u = 1.0 - t;
    let dx = lo.dx;
    let h01 = t * t * (3.0 - 2.0 * t);
    let tu = t * u;

    let lo_y0 = lo.f1_table[idx];
    let lo_y1 = lo.f1_table[idx + 1];
    let lo_m0 = lo.f_table[idx] * dx;
    let lo_m1 = lo.f_table[idx + 1] * dx;
    let val_lo = (lo_y0 + h01 * (lo_y1 - lo_y0) + tu * (u * lo_m0 - t * lo_m1)).max(0.0);

    let hi_y0 = hi.f1_table[idx];
    let hi_y1 = hi.f1_table[idx + 1];
    let hi_m0 = hi.f_table[idx] * dx;
    let hi_m1 = hi.f_table[idx + 1] * dx;
    let val_hi = (hi_y0 + h01 * (hi_y1 - hi_y0) + tu * (u * hi_m0 - t * hi_m1)).max(0.0);

    (val_lo, val_hi)
}

#[inline]
fn eval_f1_pair(lo: &AdaaPowerLut, hi: &AdaaPowerLut, x: f32) -> (f32, f32) {
    if x <= 0.0 || x.is_nan() {
        return (0.0, 0.0);
    }
    if lo.last_idx == hi.last_idx && lo.x_max == hi.x_max {
        if x >= lo.x_max {
            let dx_over = x - lo.x_max;
            let f1_lo = lo.f1_table[lo.last_idx] + lo.f_table[lo.last_idx] * dx_over;
            let f1_hi = hi.f1_table[hi.last_idx] + hi.f_table[hi.last_idx] * dx_over;
            return (f1_lo, f1_hi);
        }
        sample_hermite_f1_pair(lo, hi, x)
    } else {
        (lo.eval_f1(x), hi.eval_f1(x))
    }
}

/// Stateful per-voice 1st-order ADAA processor storing `x[n - 1]`.
#[derive(Copy, Clone, Debug, Default)]
pub struct AdaaState {
    x_prev: f32,
}

impl AdaaState {
    /// Creates zero-initialized ADAA state.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self { x_prev: 0.0 }
    }

    /// Resets stored previous sample to `0.0`.
    #[inline]
    pub fn reset(&mut self) {
        self.x_prev = 0.0;
    }

    /// Returns `true` when stored previous sample `x_prev` is `0.0`.
    #[must_use]
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.x_prev == 0.0
    }

    /// Updates `x_prev` directly when stepping within a linear equilibrium region.
    #[inline]
    pub fn sync_x(&mut self, x: f32) {
        self.x_prev = x;
    }

    /// Advances one sample and returns the antialiased nonlinearity output.
    #[inline]
    pub fn process(&mut self, lut: &AdaaPowerLut, x: f32) -> f32 {
        let out = lut.adaa1(x, self.x_prev);
        self.x_prev = x;
        out
    }

    /// Advances one sample across two linearly interpolated LUTs and returns the
    /// antialiased nonlinearity output.
    #[inline]
    pub fn process_lerp(
        &mut self,
        lut_lo: &AdaaPowerLut,
        lut_hi: &AdaaPowerLut,
        t: f32,
        x: f32,
    ) -> f32 {
        let out = lut_lo.adaa1_lerp(lut_hi, t, x, self.x_prev);
        self.x_prev = x;
        out
    }
}
