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

    /// Evaluates first antiderivative `F_1(x) = x^(p+1) / (p+1)` via linear interpolation.
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
        sample_lut(&self.f1_table, x, self.inv_dx, self.last_idx)
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

    /// Advances one sample and returns the antialiased nonlinearity output.
    #[inline]
    pub fn process(&mut self, lut: &AdaaPowerLut, x: f32) -> f32 {
        let out = lut.adaa1(x, self.x_prev);
        self.x_prev = x;
        out
    }
}
