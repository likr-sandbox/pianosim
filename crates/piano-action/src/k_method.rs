//! Borin K-method implicit nonlinear loop solver (`delta_y + K_loop * F = p_free`)
//! paired with 1st-order ADAA (`piano_simd::AdaaPowerLut`) and 0.5-sample
//! group-delay feedback phase alignment.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use piano_simd::{AdaaPowerLut, AdaaState};

use crate::math::pow_nonneg_f64;

/// Default upper bound of the tabulated dimensionless `p_tilde` domain.
pub const BORIN_P_MAX: f32 = 16.0;
/// Default LUT resolution for `BorinKLut`.
pub const BORIN_LUT_SIZE: usize = 1024;

/// Pre-tabulated solver for the dimensionless Borin K-method equation
/// `y_tilde + y_tilde^p = p_tilde` (`y_tilde >= 0`, `p_tilde >= 0`).
///
/// All allocation occurs once inside [`BorinKLut::new`]. Above `p_max`, a
/// 2-step `no_std` Newton fallback preserves accuracy without saturation.
#[derive(Clone, Debug)]
pub struct BorinKLut {
    exponent_p: f32,
    inv_p: f64,
    inv_p_minus_1: f64,
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
            inv_p_minus_1: 1.0 / (p - 1.0),
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

    /// Computes scaling factor `lambda = delta_scale * (delta_scale / (k_loop * k_inst))^(1/(p-1))`.
    #[must_use]
    #[inline]
    pub fn compute_lambda(&self, delta_scale_m: f32, k_loop_m_per_n: f32, k_inst_n: f32) -> f32 {
        let denom = f64::from(k_loop_m_per_n) * f64::from(k_inst_n);
        if !denom.is_finite() || denom <= 0.0 || delta_scale_m <= 0.0 {
            return 0.0;
        }
        let ratio = f64::from(delta_scale_m) / denom;
        (f64::from(delta_scale_m) * pow_nonneg_f64(ratio, self.inv_p_minus_1)) as f32
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

/// Parameters for one coupled `8x` oversampled K-method + Stulov + ADAA substep.
#[derive(Copy, Clone, Debug)]
pub struct CoupledSubstepInput {
    pub delta_prev_m: f32,
    pub p_end_m: f32,
    pub k_loop_end: f32,
    pub delta_scale_m: f32,
    pub f_scale_eff_n: f32,
    pub eps_eff: f32,
    pub a_fast_sub: f32,
    pub lambda_end: f32,
}

/// Output of one coupled substep: unilateral contact force (`N`) and normalized
/// compression `u_pred` (`delta_pred / delta_scale`) for compaction updates.
#[derive(Copy, Clone, Debug)]
pub struct CoupledSubstepOutput {
    pub force_n: f32,
    pub u_norm_mid: f32,
}

/// Resolves one `8x` hammer-string substep using:
/// 1. Pre-tabulated Borin K-method implicit loop resolution at endpoint `p_end_m`,
/// 2. 1st-order ADAA (`AdaaPowerLut` + `AdaaState`) providing exact 0.5-sample
///    midpoint alignment across `[delta_prev, delta_end]`, and
/// 3. Fast intra-strike Stulov relaxation with unilateral force clamp `F >= 0`.
#[inline]
pub fn solve_coupled_substep(
    adaa_lut: &AdaaPowerLut,
    borin_lut: &BorinKLut,
    adaa_state: &mut AdaaState,
    z_fast: &mut f32,
    inp: CoupledSubstepInput,
) -> CoupledSubstepOutput {
    let inv_scale = 1.0 / inp.delta_scale_m;
    let c_inst = 1.0 - inp.eps_eff * (1.0 - inp.a_fast_sub);
    let k_inst = inp.f_scale_eff_n * c_inst;
    let f_hist = inp.f_scale_eff_n * inp.eps_eff * inp.a_fast_sub * (*z_fast);

    let delta_end = if inp.p_end_m <= 0.0 || inp.k_loop_end <= 0.0 || inp.lambda_end <= 0.0 {
        0.0
    } else if k_inst * adaa_lut.eval_f(inp.p_end_m * inv_scale) <= f_hist {
        // Below Stulov hysteresis lift-off compression: free separation with F = 0.
        0.0
    } else {
        let p_eff = inp.p_end_m + inp.k_loop_end * f_hist;
        let y_tilde = borin_lut.solve_dimless(p_eff / inp.lambda_end);
        (inp.lambda_end * y_tilde).max(0.0)
    };

    let u_end = delta_end * inv_scale;
    let phi_adaa = adaa_state.process(adaa_lut, u_end);

    let z_next = inp.a_fast_sub * (*z_fast) + (1.0 - inp.a_fast_sub) * phi_adaa;
    *z_fast = z_next;

    let force_n = (inp.f_scale_eff_n * (phi_adaa - inp.eps_eff * z_next)).max(0.0);
    let u_norm_mid = (0.5 * (inp.delta_prev_m.max(0.0) + delta_end) * inv_scale).max(0.0);

    CoupledSubstepOutput {
        force_n,
        u_norm_mid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borin_k_lut_converges_across_lut_and_newton_fallback_ranges() {
        for &p in &[2.2_f32, 2.8, 3.5] {
            let lut = BorinKLut::new(p, BORIN_P_MAX, BORIN_LUT_SIZE);
            assert_eq!(lut.solve_dimless(-1.0), 0.0);
            assert_eq!(lut.solve_dimless(0.0), 0.0);

            for &pt in &[1.0e-4_f32, 0.02, 0.25, 1.0, 4.5, 15.9, 25.0, 120.0] {
                let y = lut.solve_dimless(pt);
                let residual =
                    (f64::from(y) + f64::from(y).powf(f64::from(p)) - f64::from(pt)).abs();
                assert!(
                    residual < 1.5e-4 * f64::from(pt).max(1.0),
                    "p={p}, pt={pt}, y={y}, res={residual}"
                );
            }
        }
    }

    #[test]
    fn coupled_substep_satisfies_k_method_equation_and_unilateral_clamp() {
        let p = 2.8_f32;
        let adaa_lut = AdaaPowerLut::new(p, 1.0, 1024);
        let borin_lut = BorinKLut::new(p, BORIN_P_MAX, BORIN_LUT_SIZE);
        let delta_scale = 2.0e-3_f32;
        let f_scale = 800.0_f32;
        let k_loop = 8.0e-9_f32;
        let eps = 0.28_f32;
        let a_fast = 0.928_f32;
        let c_inst = 1.0 - eps * (1.0 - a_fast);
        let lambda_end = borin_lut.compute_lambda(delta_scale, k_loop, f_scale * c_inst);

        // Zero or negative gap yields exact zero force.
        let mut adaa = AdaaState::new();
        let mut z_fast = 0.0_f32;
        let out_neg = solve_coupled_substep(
            &adaa_lut,
            &borin_lut,
            &mut adaa,
            &mut z_fast,
            CoupledSubstepInput {
                delta_prev_m: -2.0e-5,
                p_end_m: -1.0e-5,
                k_loop_end: k_loop,
                delta_scale_m: delta_scale,
                f_scale_eff_n: f_scale,
                eps_eff: eps,
                a_fast_sub: a_fast,
                lambda_end,
            },
        );
        assert_eq!(out_neg.force_n, 0.0);

        // Positive penetration satisfies implicit loop equilibrium closely.
        let mut adaa2 = AdaaState::new();
        let mut z2 = 0.0_f32;
        let delta_prev = 5.0e-4_f32;
        let p_end = 5.4e-4_f32;
        adaa2.process(&adaa_lut, delta_prev / delta_scale);
        let out_pos = solve_coupled_substep(
            &adaa_lut,
            &borin_lut,
            &mut adaa2,
            &mut z2,
            CoupledSubstepInput {
                delta_prev_m: delta_prev,
                p_end_m: p_end,
                k_loop_end: k_loop,
                delta_scale_m: delta_scale,
                f_scale_eff_n: f_scale,
                eps_eff: eps,
                a_fast_sub: a_fast,
                lambda_end,
            },
        );
        assert!(out_pos.force_n > 0.0 && out_pos.force_n.is_finite());
        let p_mid = 0.5 * (delta_prev + p_end);
        let delta_mid = p_mid - 0.5 * k_loop * out_pos.force_n;
        let f_check = f_scale
            * ((delta_mid / delta_scale).powf(p)
                - eps * (a_fast * 0.0 + (1.0 - a_fast) * (delta_mid / delta_scale).powf(p)));
        assert!(
            (out_pos.force_n - f_check).abs() / f_check.max(1.0) < 0.02,
            "force={}, f_check={f_check}",
            out_pos.force_n
        );
    }
}
