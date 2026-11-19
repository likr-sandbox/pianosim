//! Three-mechanism physical transverse string T60 damping laws (`no_std`):
//! 1. Air viscosity boundary-layer drag (`alpha_air ~ r_eff * sqrt(omega) / mu`)
//! 2. Internal thermoelastic / viscoelastic bending loss (`alpha_thermo ~ omega^2`)
//! 3. Copper-winding interfacial friction for wound bass strings (`k < 26`).

use core::f64::consts::{PI, TAU};

use piano_spec::scale_design::{BICHORD_KEYS, MONOCHORD_KEYS};
use piano_spec::{StringPhysicalParams, STEEL_DENSITY_KG_M3};

use crate::math::sqrt_f64;

const LN_1000: f64 = 6.907_755_278_982_137;
const COPPER_DENSITY_KG_M3: f64 = 8_960.0;
const WOUND_KEY_BREAK: usize = MONOCHORD_KEYS + BICHORD_KEYS;

// Calibrated physical damping coefficients (with agraffe/hitch-pin boundary loss folded into air/wire scale).
const AIR_VISC_COEFF: f64 = 0.165;
const THERMOELASTIC_COEFF: f64 = 1.65e-9;
const WINDING_FRICTION_COEFF: f64 = 0.22;
const WINDING_REF_OMEGA: f64 = TAU * 27.5;

/// Computes the analytical effective outer string radius `r_eff` in meters, accounting for
/// the copper overwinding layer on bass strings (`mu > pi * rho_steel * r_core^2`).
#[inline]
#[must_use]
pub(crate) fn effective_outer_radius_m(spec: &StringPhysicalParams) -> f64 {
    let r_core = spec.core_radius_m.max(1.0e-6);
    let mu = spec.linear_density_kg_m.max(1.0e-6);
    let mu_core = PI * STEEL_DENSITY_KG_M3 * r_core * r_core;
    let mu_cu = (mu - mu_core).max(0.0);
    if mu_cu > 0.0 {
        sqrt_f64(r_core * r_core + mu_cu / (PI * COPPER_DENSITY_KG_M3))
    } else {
        r_core
    }
}

/// Computes the three-mechanism physical decay rate `alpha` (`1/s`) for partial `f_j`.
#[inline]
#[must_use]
pub(crate) fn transverse_partial_alpha(
    key_idx: usize,
    spec: &StringPhysicalParams,
    f_j: f64,
    bridge_coupled: bool,
) -> f64 {
    if !f_j.is_finite() || f_j <= 0.0 {
        return 1.0e-6;
    }
    let omega = TAU * f_j;
    let mu = spec.linear_density_kg_m.max(1.0e-6);
    let r_core = spec.core_radius_m.max(1.0e-6);
    let r_eff = effective_outer_radius_m(spec);

    // 1. Air viscosity (+ termination viscous boundary loss).
    let coupling_scale = if bridge_coupled { 0.85 } else { 1.0 };
    let alpha_air = coupling_scale * AIR_VISC_COEFF * r_eff * sqrt_f64(omega) / mu;

    // 2. Internal thermoelasticity (~ omega^2).
    let alpha_thermo = THERMOELASTIC_COEFF * omega * omega;

    // 3. Copper-winding interfacial friction for wound bass strings (k < 26).
    let alpha_wind = if key_idx < WOUND_KEY_BREAK {
        let mu_core = PI * STEEL_DENSITY_KG_M3 * r_core * r_core;
        let cu_frac = ((mu - mu_core) / mu).clamp(0.0, 1.0);
        WINDING_FRICTION_COEFF * cu_frac * sqrt_f64(omega / WINDING_REF_OMEGA)
    } else {
        0.0
    };

    (alpha_air + alpha_thermo + alpha_wind).max(1.0e-6)
}

#[inline]
#[must_use]
pub(crate) fn t60_transverse_partial(
    key_idx: usize,
    spec: &StringPhysicalParams,
    f_j: f64,
    hold_mult: f64,
    bridge_coupled: bool,
) -> f64 {
    let alpha = transverse_partial_alpha(key_idx, spec, f_j, bridge_coupled);
    (LN_1000 / alpha) * hold_mult
}

#[cfg(test)]
#[path = "damping_law_tests.rs"]
mod tests;
