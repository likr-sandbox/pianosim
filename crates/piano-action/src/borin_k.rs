//! Pre-tabulated solver for the dimensionless Borin K-method nonlinear equation
//! `y_tilde + y_tilde^p = p_tilde` (`y_tilde >= 0`, `p_tilde >= 0`).

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::k_method::compute_lambda_for_p;
use crate::math::pow_nonneg_f64;

/// Default upper bound of the tabulated dimensionless `p_tilde` domain.
pub const BORIN_P_MAX: f32 = 64.0;
/// Default LUT resolution for `BorinKLut`.
pub const BORIN_LUT_SIZE: usize = 2048;

/// Pre-tabulated solver for the dimensionless Borin K-method equation
/// `y_tilde + y_tilde^p = p_tilde` (`y_tilde >= 0`, `p_tilde >= 0`).
///
/// All allocation occurs once inside [`BorinKLut::new`]. Above `p_max`, a
/// 2-step `no_std` Newton fallback preserves accuracy without saturation.
#[derive(Clone, Debug)]
pub struct BorinKLut {
    exponent_p: f32,
    inv_p: f64,
    p_max: f32,
    inv_dp: f32,
    last_idx: usize,
    y_table: Box<[f32]>,
}

impl BorinKLut {
    /// Builds a uniform LUT over `[0.0, p_max]` solving `y + y^p = p_tilde` in `f64`.
    #[must_use]
    pub fn new(exponent_p: f32, p_max: f32, table_size: usize) -> Self {
        let p = f64::from(exponent_p.clamp(1.2, 6.0));
        let safe_p_max = if p_max.is_finite() && p_max > 0.1 {
            p_max
        } else {
            BORIN_P_MAX
        };
        let size = table_size.max(16);
        let last_idx = size - 1;
        let dp = f64::from(safe_p_max) / (last_idx as f64);

        let mut y_table = Vec::with_capacity(size);
        for i in 0..size {
            let pt = (i as f64) * dp;
            let y = solve_dimless_newton_f64(pt, p);
            y_table.push(y as f32);
        }

        Self {
            exponent_p: p as f32,
            inv_p: 1.0 / p,
            p_max: safe_p_max,
            inv_dp: (last_idx as f32) / safe_p_max,
            last_idx,
            y_table: y_table.into_boxed_slice(),
        }
    }

    /// Configured power exponent `p`.
    #[must_use]
    #[inline]
    pub const fn exponent_p(&self) -> f32 {
        self.exponent_p
    }

    /// Heap table payload size in bytes.
    #[must_use]
    #[inline]
    pub fn table_bytes(&self) -> usize {
        self.y_table.len() * core::mem::size_of::<f32>()
    }

    /// Solves `y_tilde + y_tilde^p = p_tilde` (`0.0` if `p_tilde <= 0`).
    /// Uses LUT linear interpolation inside `[0, p_max]` and a safe 2-step Newton
    /// fallback above `p_max`. Zero heap allocation and panic-free.
    #[must_use]
    #[inline]
    pub fn solve_dimless(&self, p_tilde: f32) -> f32 {
        if !p_tilde.is_finite() || p_tilde <= 0.0 {
            return 0.0;
        }
        if p_tilde >= self.p_max {
            return self.newton_fallback_above(p_tilde);
        }
        let pos = p_tilde * self.inv_dp;
        let idx = (pos as usize).min(self.last_idx - 1);
        let frac = pos - (idx as f32);
        let y0 = self.y_table[idx];
        let y1 = self.y_table[idx + 1];
        y0 + frac * (y1 - y0)
    }

    /// Solves `y_tilde + y_tilde^p = p_tilde` interpolated across `(self, other)` at blend `t`.
    /// Zero heap allocation and panic-free (`NaN` in `t` clamps to `0.0`).
    #[must_use]
    #[inline]
    pub fn solve_dimless_lerp(&self, other: &Self, t: f32, p_tilde: f32) -> f32 {
        if !p_tilde.is_finite() || p_tilde <= 0.0 {
            return 0.0;
        }
        let t_clamped = if t >= 1.0 {
            1.0
        } else if t > 0.0 {
            t
        } else {
            0.0
        };
        if self.last_idx == other.last_idx && self.p_max == other.p_max && p_tilde < self.p_max {
            let pos = p_tilde * self.inv_dp;
            let idx = (pos as usize).min(self.last_idx - 1);
            let frac = pos - (idx as f32);
            let lo0 = self.y_table[idx];
            let lo1 = self.y_table[idx + 1];
            let hi0 = other.y_table[idx];
            let hi1 = other.y_table[idx + 1];
            let y_lo = lo0 + frac * (lo1 - lo0);
            let y_hi = hi0 + frac * (hi1 - hi0);
            (1.0 - t_clamped) * y_lo + t_clamped * y_hi
        } else {
            let y_lo = self.solve_dimless(p_tilde);
            let y_hi = other.solve_dimless(p_tilde);
            (1.0 - t_clamped) * y_lo + t_clamped * y_hi
        }
    }

    /// Computes scaling factor `lambda = delta_scale * (delta_scale / (k_loop * k_inst))^(1/(p-1))`.
    #[must_use]
    #[inline]
    pub fn compute_lambda(&self, delta_scale_m: f32, k_loop_m_per_n: f32, k_inst_n: f32) -> f32 {
        compute_lambda_for_p(self.exponent_p, delta_scale_m, k_loop_m_per_n, k_inst_n)
    }

    #[inline(never)]
    fn newton_fallback_above(&self, p_tilde: f32) -> f32 {
        let pt = f64::from(p_tilde);
        let p = f64::from(self.exponent_p);
        let mut y = pow_nonneg_f64(pt, self.inv_p);
        for _ in 0..2 {
            let yp_minus_1 = pow_nonneg_f64(y, p - 1.0);
            let f = y + y * yp_minus_1 - pt;
            let df = 1.0 + p * yp_minus_1;
            if df > 0.0 {
                y = (y - f / df).max(0.0);
            }
        }
        y as f32
    }
}

#[inline]
fn solve_dimless_newton_f64(p_tilde: f64, p: f64) -> f64 {
    if p_tilde <= 0.0 {
        return 0.0;
    }
    let mut y = if p_tilde < 1.0 {
        p_tilde
    } else {
        pow_nonneg_f64(p_tilde, 1.0 / p)
    };
    for _ in 0..8 {
        let yp_minus_1 = pow_nonneg_f64(y, p - 1.0);
        let f = y + y * yp_minus_1 - p_tilde;
        let df = 1.0 + p * yp_minus_1;
        let next = (y - f / df).max(0.0);
        if (next - y).abs() <= 1.0e-13 * next.max(1.0) {
            return next;
        }
        y = next;
    }
    y
}
