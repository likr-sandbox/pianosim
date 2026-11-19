//! Local `2x/4x/8x` multirate hammer-string loop active ONLY while a hammer
//! is in flight/contact or draining its polyphase FIR tail.

use piano_simd::{AdaaPowerLut, AdaaState, PolyPowerApprox};

use crate::agraffe_delay::AgraffeDelayLine;
pub use crate::decimation_fir::{DecimationFir, DecimationFir16, MAX_OVERSAMPLE_FACTOR};
use crate::felt_stulov::{FeltSharedCoeffs, FeltState, KeyFeltSpec};
use crate::k_method::{
    compute_lambda_for_p, solve_coupled_substep, solve_coupled_substep_lerp,
    solve_coupled_substep_poly, BorinKLut, CoupledSubstepInput,
};
use crate::shank_timoshenko::{ShankParams, ShankState, HAMMER_LET_OFF_Y0_M};

const OVERSAMPLE_FACTOR: usize = MAX_OVERSAMPLE_FACTOR;

/// Dispatch target for nonlinear felt contact inside oversampled substeps.
#[derive(Copy, Clone)]
pub enum HammerContactSolver<'a> {
    /// Single-table LUT path (exact `p_t <= 1e-5` or `p_t >= 1 - 1e-5` fast-path).
    LutSingle {
        adaa: &'a AdaaPowerLut,
        borin: &'a BorinKLut,
    },
    /// Linearly interpolated `(lo, hi)` `p`-grid LUT path.
    LutLerp {
        adaa_lo: &'a AdaaPowerLut,
        adaa_hi: &'a AdaaPowerLut,
        borin_lo: &'a BorinKLut,
        borin_hi: &'a BorinKLut,
        p_t: f32,
    },
    /// Zero-LUT analytical Bernstein-monomial polynomial ADAA + Newton K-method.
    Poly { poly: &'a PolyPowerApprox },
}

/// Per-key mutable oversampled strike state.
#[derive(Copy, Clone, Debug)]
pub struct HammerOversampleState {
    pub shank: ShankState,
    pub adaa_state: AdaaState,
    pub delta_prev_m: f32,
    pub prev_sub_forces: [f32; OVERSAMPLE_FACTOR],
    pub tail_samples_left: u8,
    pub last_u_norm: f32,
    pub last_host_force_n: f32,
}

impl Default for HammerOversampleState {
    fn default() -> Self {
        Self {
            shank: ShankState::default(),
            adaa_state: AdaaState::new(),
            delta_prev_m: HAMMER_LET_OFF_Y0_M,
            prev_sub_forces: [0.0; OVERSAMPLE_FACTOR],
            tail_samples_left: 0,
            last_u_norm: 0.0,
            last_host_force_n: 0.0,
        }
    }
}

impl HammerOversampleState {
    /// Clears strike and decimation tail state while preserving `flex_enabled`.
    pub fn reset(&mut self) {
        self.shank.reset();
        self.adaa_state.reset();
        self.delta_prev_m = HAMMER_LET_OFF_Y0_M;
        self.prev_sub_forces = [0.0; OVERSAMPLE_FACTOR];
        self.tail_samples_left = 0;
        self.last_u_norm = 0.0;
        self.last_host_force_n = 0.0;
    }

    /// Launches a new strike at `v_h0` (`m/s`), resetting ADAA phase tracking
    /// while preserving polyphase FIR history across rapid re-strikes.
    #[inline]
    pub fn launch(&mut self, v_h0: f32) {
        self.shank.launch(v_h0);
        self.adaa_state.reset();
        self.delta_prev_m = HAMMER_LET_OFF_Y0_M;
        self.tail_samples_left = 0;
        self.last_u_norm = 0.0;
        self.last_host_force_n = 0.0;
    }

    /// Returns `true` if hammer physics OR polyphase decimation tail is active.
    #[must_use]
    #[inline]
    pub const fn is_active(&self) -> bool {
        self.shank.active || self.tail_samples_left > 0
    }

    /// Executes `fir.factor()` oversampled substeps using LUT interpolation and returns
    /// the decimated host-rate strike force (`N`) onto the string.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_host_sample(
        &mut self,
        shank_params: &ShankParams,
        felt_spec: &KeyFeltSpec,
        felt_coeffs: &FeltSharedCoeffs,
        felt_state: &mut FeltState,
        adaa_lo: &AdaaPowerLut,
        adaa_hi: &AdaaPowerLut,
        borin_lo: &BorinKLut,
        borin_hi: &BorinKLut,
        p_t: f32,
        fir: &DecimationFir,
        memory_enabled: bool,
        una_corda_mod: (f32, f32),
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32 {
        self.step_host_sample_with_agraffe(
            shank_params,
            felt_spec,
            felt_coeffs,
            felt_state,
            adaa_lo,
            adaa_hi,
            borin_lo,
            borin_hi,
            p_t,
            fir,
            memory_enabled,
            una_corda_mod,
            y_str_m,
            v_str_mps,
            r_port_strike,
            None,
            1.0,
        )
    }

    /// Executes `fir.factor()` oversampled substeps with optional short-string agraffe
    /// reflection feedback and returns the decimated host-rate strike force (`N`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_host_sample_with_agraffe(
        &mut self,
        shank_params: &ShankParams,
        felt_spec: &KeyFeltSpec,
        felt_coeffs: &FeltSharedCoeffs,
        felt_state: &mut FeltState,
        adaa_lo: &AdaaPowerLut,
        adaa_hi: &AdaaPowerLut,
        borin_lo: &BorinKLut,
        borin_hi: &BorinKLut,
        p_t: f32,
        fir: &DecimationFir,
        memory_enabled: bool,
        una_corda_mod: (f32, f32),
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
        agraffe: Option<&mut AgraffeDelayLine>,
        uc_choir_scale: f32,
    ) -> f32 {
        let solver = if p_t <= 1.0e-5 {
            HammerContactSolver::LutSingle {
                adaa: adaa_lo,
                borin: borin_lo,
            }
        } else if p_t >= 1.0 - 1.0e-5 {
            HammerContactSolver::LutSingle {
                adaa: adaa_hi,
                borin: borin_hi,
            }
        } else {
            HammerContactSolver::LutLerp {
                adaa_lo,
                adaa_hi,
                borin_lo,
                borin_hi,
                p_t,
            }
        };
        self.step_host_sample_with_solver(
            shank_params,
            felt_spec,
            felt_coeffs,
            felt_state,
            solver,
            fir,
            memory_enabled,
            una_corda_mod,
            y_str_m,
            v_str_mps,
            r_port_strike,
            agraffe,
            uc_choir_scale,
        )
    }

    /// Executes `fir.factor()` (`2 | 4 | 8`) oversampled substeps using `solver` and returns
    /// the decimated host-rate strike force (`N`) onto the string.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_host_sample_with_solver(
        &mut self,
        shank_params: &ShankParams,
        felt_spec: &KeyFeltSpec,
        felt_coeffs: &FeltSharedCoeffs,
        felt_state: &mut FeltState,
        solver: HammerContactSolver<'_>,
        fir: &DecimationFir,
        memory_enabled: bool,
        una_corda_mod: (f32, f32),
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
        mut agraffe: Option<&mut AgraffeDelayLine>,
        uc_choir_scale: f32,
    ) -> f32 {
        if !self.is_active() {
            self.last_u_norm = 0.0;
            self.last_host_force_n = 0.0;
            return 0.0;
        }

        let safe_y = if y_str_m.is_finite() { y_str_m } else { 0.0 };
        let safe_v = if v_str_mps.is_finite() {
            v_str_mps
        } else {
            0.0
        };
        let safe_r = if r_port_strike.is_finite() && r_port_strike >= 0.0 {
            r_port_strike
        } else {
            0.0
        };

        let os_u8 = fir.factor();
        let os = usize::from(os_u8);
        let os_f32 = f32::from(os_u8);
        let mut sub_forces = [0.0_f32; OVERSAMPLE_FACTOR];
        let dt_sub = shank_params.dt_sub;
        let dt_host = os_f32 * dt_sub;

        if self.shank.active {
            if self.shank.elapsed_host_samples == 0 {
                self.delta_prev_m = self.shank.y_h() - safe_y;
            }
            let r_sub = safe_r / os_f32;
            // Include host-rate ZOH/FIR port series compliance so high-register strikes
            // (where K_felt * r_port * dt_host > 1) stay strictly Nyquist-passive across os->1x.
            let k_str_end = 0.5 * r_sub * dt_sub + 0.68 * safe_r * dt_host;

            let (f_base_n, eps_base) = felt_state.effective_params(felt_spec, memory_enabled);
            let f_scale_eff_n = f_base_n * una_corda_mod.0;
            let eps_eff = if una_corda_mod.1 > 0.0 {
                (eps_base + 2.0 * una_corda_mod.1).min(0.68)
            } else {
                eps_base
            };
            let c_inst = 1.0 - eps_eff * (1.0 - felt_coeffs.a_fast_sub);
            let k_ham_nom = self.shank.predict_free(shank_params).k_ham;
            let lambda_end = match solver {
                HammerContactSolver::Poly { .. } => 0.0,
                HammerContactSolver::LutSingle { .. } | HammerContactSolver::LutLerp { .. } => {
                    compute_lambda_for_p(
                        felt_spec.exponent_p,
                        felt_spec.delta_scale_m,
                        k_ham_nom + k_str_end,
                        f_scale_eff_n * c_inst,
                    )
                }
            };

            let mut y_s_loc = safe_y;
            let mut v_s_loc = safe_v;
            let mut u_norm_peak = 0.0_f32;

            for slot in &mut sub_forces[..os] {
                let pred = self.shank.predict_free(shank_params);
                let (y_agg, v_agg) = if let Some(ref mut agg) = agraffe {
                    agg.step_substep_before(dt_sub, uc_choir_scale)
                } else {
                    (0.0, 0.0)
                };
                let y_s_free = y_s_loc + (v_s_loc + v_agg) * dt_sub + y_agg;
                let p_end_m = pred.y_h_free - y_s_free;
                let k_loop_end = pred.k_ham + k_str_end;

                let inp = CoupledSubstepInput {
                    delta_prev_m: self.delta_prev_m,
                    p_end_m,
                    k_loop_end,
                    delta_scale_m: felt_spec.delta_scale_m,
                    f_scale_eff_n,
                    eps_eff,
                    a_fast_sub: felt_coeffs.a_fast_sub,
                    lambda_end,
                };
                let out = match solver {
                    HammerContactSolver::LutSingle { adaa, borin } => solve_coupled_substep(
                        adaa,
                        borin,
                        &mut self.adaa_state,
                        &mut felt_state.z_fast,
                        inp,
                    ),
                    HammerContactSolver::LutLerp {
                        adaa_lo,
                        adaa_hi,
                        borin_lo,
                        borin_hi,
                        p_t,
                    } => solve_coupled_substep_lerp(
                        adaa_lo,
                        adaa_hi,
                        borin_lo,
                        borin_hi,
                        p_t,
                        &mut self.adaa_state,
                        &mut felt_state.z_fast,
                        inp,
                    ),
                    HammerContactSolver::Poly { poly } => solve_coupled_substep_poly(
                        poly,
                        &mut self.adaa_state,
                        &mut felt_state.z_fast,
                        inp,
                    ),
                };

                self.shank.apply_force(shank_params, pred, out.force_n);
                if let Some(ref mut agg) = agraffe {
                    agg.push_force(out.force_n);
                }
                let v_s_next = v_s_loc + r_sub * out.force_n;
                y_s_loc += 0.5 * (v_s_loc + v_s_next) * dt_sub;
                v_s_loc = v_s_next;
                self.delta_prev_m = self.shank.y_h() - (y_s_loc + y_agg);
                felt_state.accumulate_substep_compaction(felt_spec, out.u_norm_mid, memory_enabled);
                u_norm_peak = u_norm_peak.max(out.u_norm_mid);
                *slot = out.force_n;
            }

            self.last_u_norm = u_norm_peak.clamp(0.0, 1.0);
            if memory_enabled && felt_state.z_mem > 0.0 {
                felt_state.z_mem *= felt_coeffs.a_mem_host;
            }
            if !self.shank.finish_host_sample(shank_params) {
                self.tail_samples_left = 2;
            }
        } else {
            self.tail_samples_left = self.tail_samples_left.saturating_sub(1);
            self.last_u_norm *= 0.5;
        }

        let f_fir = fir.process(&mut self.prev_sub_forces, &sub_forces).max(0.0);
        self.last_host_force_n = f_fir;
        f_fir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rapid_restrike_launch_preserves_fir_history_while_reset_clears_it() {
        let fir = DecimationFir::new();
        let mut st = HammerOversampleState {
            prev_sub_forces: [2.5; OVERSAMPLE_FACTOR],
            ..Default::default()
        };
        st.launch(1.8);
        assert_eq!(st.prev_sub_forces, [2.5; OVERSAMPLE_FACTOR]);

        // Next host step with zero new sub-forces drains the preserved FIR tail smoothly.
        let drained = fir.process(&mut st.prev_sub_forces, &[0.0; OVERSAMPLE_FACTOR]);
        assert!(
            drained > 0.0,
            "expected continuous FIR tail drain, got {drained}"
        );

        st.prev_sub_forces = [2.5; OVERSAMPLE_FACTOR];
        st.reset();
        assert_eq!(st.prev_sub_forces, [0.0; OVERSAMPLE_FACTOR]);
    }
}
