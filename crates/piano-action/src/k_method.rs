//! Borin K-method implicit nonlinear loop solver (`delta_y + K_loop * F = p_free`)
//! paired with 1st-order ADAA (`piano_simd::AdaaPowerLut` or `PolyPowerApprox`)
//! and 0.5-sample group-delay feedback phase alignment.

use piano_simd::{AdaaPowerLut, AdaaState};

pub use crate::borin_k::{BorinKLut, BORIN_LUT_SIZE, BORIN_P_MAX};
pub use crate::k_method_poly::{solve_borin_poly, solve_coupled_substep_poly};
use crate::math::pow_nonneg_f64;

/// Computes scaling factor `lambda = delta_scale * (delta_scale / (k_loop * k_inst))^(1/(p-1))`
/// dynamically for continuous exponent `exponent_p`. Zero-alloc and `#![no_std]` compatible.
#[must_use]
#[inline]
pub fn compute_lambda_for_p(
    exponent_p: f32,
    delta_scale_m: f32,
    k_loop_m_per_n: f32,
    k_inst_n: f32,
) -> f32 {
    let p = f64::from(exponent_p.clamp(1.2, 6.0));
    let denom = f64::from(k_loop_m_per_n) * f64::from(k_inst_n);
    if !denom.is_finite() || denom <= 0.0 || delta_scale_m <= 0.0 {
        return 0.0;
    }
    let ratio = f64::from(delta_scale_m) / denom;
    (f64::from(delta_scale_m) * pow_nonneg_f64(ratio, 1.0 / (p - 1.0))) as f32
}

/// Parameters for one coupled oversampled K-method + Stulov + ADAA substep.
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

/// Resolves one hammer-string substep using:
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
    if inp.p_end_m <= 0.0 && adaa_state.is_zero() && *z_fast >= 0.0 {
        *z_fast *= inp.a_fast_sub;
        return CoupledSubstepOutput {
            force_n: 0.0,
            u_norm_mid: (0.5 * inp.delta_prev_m.max(0.0) * inv_scale).max(0.0),
        };
    }

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

/// Resolves one hammer-string substep using linearly interpolated `(lo, hi)` LUTs
/// across the shared 24-point `p`-grid at blend weight `p_t in [0.0, 1.0]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn solve_coupled_substep_lerp(
    adaa_lo: &AdaaPowerLut,
    adaa_hi: &AdaaPowerLut,
    borin_lo: &BorinKLut,
    borin_hi: &BorinKLut,
    p_t: f32,
    adaa_state: &mut AdaaState,
    z_fast: &mut f32,
    inp: CoupledSubstepInput,
) -> CoupledSubstepOutput {
    let inv_scale = 1.0 / inp.delta_scale_m;
    if inp.p_end_m <= 0.0 && adaa_state.is_zero() && *z_fast >= 0.0 {
        *z_fast *= inp.a_fast_sub;
        return CoupledSubstepOutput {
            force_n: 0.0,
            u_norm_mid: (0.5 * inp.delta_prev_m.max(0.0) * inv_scale).max(0.0),
        };
    }

    let c_inst = 1.0 - inp.eps_eff * (1.0 - inp.a_fast_sub);
    let k_inst = inp.f_scale_eff_n * c_inst;
    let f_hist = inp.f_scale_eff_n * inp.eps_eff * inp.a_fast_sub * (*z_fast);

    let delta_end = if inp.p_end_m <= 0.0
        || inp.k_loop_end <= 0.0
        || inp.lambda_end <= 0.0
        || k_inst * adaa_lo.eval_f_lerp(adaa_hi, p_t, inp.p_end_m * inv_scale) <= f_hist
    {
        0.0
    } else {
        let p_eff = inp.p_end_m + inp.k_loop_end * f_hist;
        let y_tilde = borin_lo.solve_dimless_lerp(borin_hi, p_t, p_eff / inp.lambda_end);
        (inp.lambda_end * y_tilde).max(0.0)
    };

    let u_end = delta_end * inv_scale;
    let phi_adaa = adaa_state.process_lerp(adaa_lo, adaa_hi, p_t, u_end);

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
