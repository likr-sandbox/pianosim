//! Shared 24-point `p in [2.10, 3.50]` lookup-table grid for ADAA power contact
//! and Borin K-method nonlinear resolution (`384 KB` total across all 88 keys).

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use piano_simd::AdaaPowerLut;

use crate::k_method::{BorinKLut, BORIN_LUT_SIZE, BORIN_P_MAX};

/// Number of uniform grid points over `[P_GRID_MIN, P_GRID_MAX]`.
pub const P_GRID_SIZE: usize = 24;
/// Minimum hammer felt exponent covered by the shared `p`-grid.
pub const P_GRID_MIN: f32 = 2.10;
/// Maximum hammer felt exponent covered by the shared `p`-grid.
pub const P_GRID_MAX: f32 = 3.50;
/// Resolution of each shared `AdaaPowerLut`.
pub const ADAA_LUT_SIZE: usize = 1024;
/// Normalized indentation upper bound of each shared `AdaaPowerLut`.
pub const ADAA_LUT_U_MAX: f32 = 1.0;

/// Shared 24-point `p`-grid of [`AdaaPowerLut`] and [`BorinKLut`] owned once by `ActionBank`.
#[derive(Clone, Debug)]
pub struct PGridLuts {
    inv_dp: f32,
    p_values: [f32; P_GRID_SIZE],
    adaa_luts: Box<[AdaaPowerLut]>,
    borin_luts: Box<[BorinKLut]>,
}

impl PGridLuts {
    /// Precomputes the 24 uniform grid tables over `p in [2.10, 3.50]`.
    #[must_use]
    pub fn new() -> Self {
        let last = (P_GRID_SIZE - 1) as f64;
        let p_min = f64::from(P_GRID_MIN);
        let p_max = f64::from(P_GRID_MAX);
        let dp = (p_max - p_min) / last;

        let mut p_values = [0.0_f32; P_GRID_SIZE];
        let mut adaa_luts = Vec::with_capacity(P_GRID_SIZE);
        let mut borin_luts = Vec::with_capacity(P_GRID_SIZE);

        for (i, slot) in p_values.iter_mut().enumerate() {
            let p = (p_min + (i as f64) * dp) as f32;
            *slot = p;
            adaa_luts.push(AdaaPowerLut::new(p, ADAA_LUT_U_MAX, ADAA_LUT_SIZE));
            borin_luts.push(BorinKLut::new(p, BORIN_P_MAX, BORIN_LUT_SIZE));
        }

        Self {
            inv_dp: ((P_GRID_SIZE - 1) as f32) / (P_GRID_MAX - P_GRID_MIN),
            p_values,
            adaa_luts: adaa_luts.into_boxed_slice(),
            borin_luts: borin_luts.into_boxed_slice(),
        }
    }

    /// Zero-alloc, panic-free, NaN-safe bracket lookup returning `(idx_lo, idx_hi, t)`.
    #[must_use]
    #[inline]
    pub fn lookup(&self, exponent_p: f32) -> (usize, usize, f32) {
        if exponent_p.is_nan() || exponent_p <= P_GRID_MIN {
            return (0, 1, 0.0);
        }
        if exponent_p >= P_GRID_MAX {
            return (P_GRID_SIZE - 2, P_GRID_SIZE - 1, 1.0);
        }
        let pos = (exponent_p - P_GRID_MIN) * self.inv_dp;
        let idx_lo = (pos as usize).min(P_GRID_SIZE - 2);
        let idx_hi = idx_lo + 1;
        let t = (pos - (idx_lo as f32)).clamp(0.0, 1.0);
        (idx_lo, idx_hi, t)
    }

    /// Returns the `(adaa_lo, adaa_hi, borin_lo, borin_hi)` table pair for `(idx_lo, idx_hi)`.
    #[must_use]
    #[inline]
    pub fn luts_for(
        &self,
        idx_lo: usize,
        idx_hi: usize,
    ) -> (&AdaaPowerLut, &AdaaPowerLut, &BorinKLut, &BorinKLut) {
        let lo = idx_lo.min(P_GRID_SIZE - 1);
        let hi = idx_hi.min(P_GRID_SIZE - 1);
        (
            &self.adaa_luts[lo],
            &self.adaa_luts[hi],
            &self.borin_luts[lo],
            &self.borin_luts[hi],
        )
    }

    /// Returns tabulated exponent `p` for grid index `idx`.
    #[must_use]
    #[inline]
    pub fn grid_p(&self, idx: usize) -> f32 {
        self.p_values[idx.min(P_GRID_SIZE - 1)]
    }

    /// Total heap LUT payload size in bytes across all 24 grid entries (`384 KB`).
    #[must_use]
    pub fn table_bytes(&self) -> usize {
        let adaa_bytes: usize = self
            .adaa_luts
            .iter()
            .map(|l| l.table_size() * 2 * core::mem::size_of::<f32>())
            .sum();
        let borin_bytes: usize = self.borin_luts.iter().map(BorinKLut::table_bytes).sum();
        adaa_bytes + borin_bytes
    }
}

impl Default for PGridLuts {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use piano_simd::AdaaState;

    use super::*;
    use crate::k_method::{
        compute_lambda_for_p, solve_coupled_substep, solve_coupled_substep_lerp,
        CoupledSubstepInput,
    };

    #[test]
    fn shared_p_grid_size_is_384kb_and_lookup_handles_bounds_and_nan() {
        let grid = PGridLuts::new();
        assert_eq!(grid.table_bytes(), 384 * 1024);

        assert_eq!(grid.lookup(P_GRID_MIN), (0, 1, 0.0));
        assert_eq!(grid.lookup(1.5), (0, 1, 0.0));
        assert_eq!(grid.lookup(f32::NAN), (0, 1, 0.0));
        assert_eq!(grid.lookup(f32::NEG_INFINITY), (0, 1, 0.0));
        assert_eq!(
            grid.lookup(P_GRID_MAX),
            (P_GRID_SIZE - 2, P_GRID_SIZE - 1, 1.0)
        );
        assert_eq!(grid.lookup(4.2), (P_GRID_SIZE - 2, P_GRID_SIZE - 1, 1.0));
        assert_eq!(
            grid.lookup(f32::INFINITY),
            (P_GRID_SIZE - 2, P_GRID_SIZE - 1, 1.0)
        );

        let p_mid = 0.5 * (grid.grid_p(10) + grid.grid_p(11));
        let (lo, hi, t) = grid.lookup(p_mid);
        assert_eq!((lo, hi), (10, 11));
        assert!((t - 0.5).abs() < 1e-5, "t={t}");
    }

    #[test]
    fn shared_p_grid_interpolation_matches_exact_luts_closely() {
        let grid = PGridLuts::new();
        for &p in &[2.25_f32, 2.55, 2.83, 3.17, 3.45] {
            let (lo, hi, t) = grid.lookup(p);
            let (adaa_lo, adaa_hi, borin_lo, borin_hi) = grid.luts_for(lo, hi);
            let adaa_exact = AdaaPowerLut::new(p, ADAA_LUT_U_MAX, ADAA_LUT_SIZE);
            let borin_exact = BorinKLut::new(p, BORIN_P_MAX, BORIN_LUT_SIZE);

            for &pt in &[1.0e-3_f32, 0.05, 0.4, 2.0, 12.0, 40.0, 90.0] {
                let y_lerp = borin_lo.solve_dimless_lerp(borin_hi, t, pt);
                let y_exact = borin_exact.solve_dimless(pt);
                let err = (y_lerp - y_exact).abs() / y_exact.max(1e-3);
                assert!(
                    err < 1.0e-3,
                    "p={p} pt={pt}: lerp={y_lerp} exact={y_exact} rel_err={err}"
                );
            }

            let delta_scale = 1.8e-3_f32;
            let f_scale = 750.0_f32;
            let k_loop = 8.0e-9_f32;
            let eps = 0.28_f32;
            let a_fast = 0.928_f32;
            let c_inst = 1.0 - eps * (1.0 - a_fast);
            let lambda = compute_lambda_for_p(p, delta_scale, k_loop, f_scale * c_inst);

            let mut adaa_lerp = AdaaState::new();
            let mut adaa_ex = AdaaState::new();
            let (mut z_lerp, mut z_ex) = (0.0_f32, 0.0_f32);
            let mut prev_d = 0.0_f32;
            for step in 1..=24 {
                let p_end = (step as f32) * 3.5e-5;
                let inp = CoupledSubstepInput {
                    delta_prev_m: prev_d,
                    p_end_m: p_end,
                    k_loop_end: k_loop,
                    delta_scale_m: delta_scale,
                    f_scale_eff_n: f_scale,
                    eps_eff: eps,
                    a_fast_sub: a_fast,
                    lambda_end: lambda,
                };
                let out_l = solve_coupled_substep_lerp(
                    adaa_lo,
                    adaa_hi,
                    borin_lo,
                    borin_hi,
                    t,
                    &mut adaa_lerp,
                    &mut z_lerp,
                    inp,
                );
                let out_e =
                    solve_coupled_substep(&adaa_exact, &borin_exact, &mut adaa_ex, &mut z_ex, inp);
                let diff = (out_l.force_n - out_e.force_n).abs();
                assert!(
                    diff < 6.0e-3 * out_e.force_n.max(1.0),
                    "p={p} step={step}: lerp={} exact={} diff={diff}",
                    out_l.force_n,
                    out_e.force_n
                );
                prev_d = p_end - k_loop * out_e.force_n;
            }
        }
    }
}
