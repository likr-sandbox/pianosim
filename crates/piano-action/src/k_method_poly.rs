//! Zero-LUT Borin K-method + 1st-order ADAA substep solver backed by
//! `piano_simd::PolyPowerApprox` and safeguarded Newton iteration (FTO #2 compliant).

use piano_simd::{AdaaState, PolyPowerApprox};

use crate::k_method::{CoupledSubstepInput, CoupledSubstepOutput};

#[inline]
fn sqrt_seed_f32(x: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    let safe_x = x.max(f32::MIN_POSITIVE);
    let y = f32::from_bits((safe_x.to_bits() >> 1) + 0x1fc0_0000);
    0.5 * (y + safe_x / y)
}

/// Solves `u + alpha * P(u) = p_u` directly for normalized compression `u >= 0`
/// using [`PolyPowerApprox`] without any LUT lookups or runtime `powf`.
///
/// Exploits convexity (`P''(u) >= 0`) and exact analytical saturation above `u = 1.0`
/// via 1 bracket safeguard + safeguarded Newton iterations.
#[must_use]
#[inline]
pub fn solve_borin_poly(poly: &PolyPowerApprox, p_u: f32, alpha: f32) -> f32 {
    if !p_u.is_finite() || p_u <= 0.0 {
        return 0.0;
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return p_u;
    }
    let f1 = poly.f_at_1();
    let p_sat = 1.0 + alpha * f1;
    if p_u >= p_sat {
        return p_u - alpha * f1;
    }

    let u_hi = p_u.min(1.0);
    let mut lo = 0.0_f32;
    let mut hi = u_hi;
    let mut u = sqrt_seed_f32(p_u / alpha.max(1.0e-12)).min(u_hi);
    if u <= lo || u >= hi {
        u = 0.5 * (lo + hi);
    }

    for step in 0..6 {
        let (p_val, dp_val) = poly.eval_f_and_df(u);
        let g = u + alpha * p_val - p_u;
        if g < 0.0 {
            lo = u;
        } else {
            hi = u;
        }
        if step > 0 && g.abs() <= 1.0e-6 * p_u.max(1.0e-6) {
            return u;
        }
        let dg = 1.0 + alpha * dp_val;
        let next = u - g / dg;
        u = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }
    u
}

/// Resolves one oversampled hammer-string substep using [`PolyPowerApprox`]
/// (zero LUT lookups, zero runtime `powf`, P1-5 free-flight fast-path).
#[inline]
pub fn solve_coupled_substep_poly(
    poly: &PolyPowerApprox,
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

    let u_end = if inp.p_end_m <= 0.0
        || inp.k_loop_end <= 0.0
        || k_inst * poly.eval_f(inp.p_end_m * inv_scale) <= f_hist
    {
        0.0
    } else {
        let p_eff = inp.p_end_m + inp.k_loop_end * f_hist;
        let alpha = (inp.k_loop_end * k_inst) * inv_scale;
        solve_borin_poly(poly, p_eff * inv_scale, alpha).max(0.0)
    };

    let delta_end = u_end * inp.delta_scale_m;
    let phi_adaa = poly.process(u_end, adaa_state);

    let z_next = inp.a_fast_sub * (*z_fast) + (1.0 - inp.a_fast_sub) * phi_adaa;
    *z_fast = z_next;

    let force_n = (inp.f_scale_eff_n * (phi_adaa - inp.eps_eff * z_next)).max(0.0);
    let u_norm_mid = (0.5 * (inp.delta_prev_m.max(0.0) + delta_end) * inv_scale).max(0.0);

    CoupledSubstepOutput {
        force_n,
        u_norm_mid,
    }
}
