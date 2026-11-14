//! Local `8x` multirate hammer-string loop (`384 kHz` @ `48 kHz`) active ONLY
//! while a hammer is in flight/contact or draining its 16-tap polyphase FIR tail.

use core::f64::consts::{PI, TAU};

use piano_simd::{AdaaPowerLut, AdaaState};

use crate::felt_stulov::{FeltSharedCoeffs, FeltState, KeyFeltSpec};
use crate::k_method::{solve_coupled_substep, BorinKLut, CoupledSubstepInput};
use crate::math::sin_cos_f64;
use crate::shank_timoshenko::{ShankParams, ShankState, HAMMER_LET_OFF_Y0_M};

const FIR_TAPS: usize = 16;
const OVERSAMPLE_FACTOR: usize = 8;
const FIR_CUTOFF_NORM: f64 = 0.44 / 8.0;

/// DC-normalized (`sum == 1.0`, strictly non-negative) 16-tap Blackman-sinc
/// polyphase decimation filter downsampling `8x` forces to host rate.
#[derive(Copy, Clone, Debug)]
pub struct DecimationFir16 {
    coeffs: [f32; FIR_TAPS],
}

impl DecimationFir16 {
    /// Precomputes the symmetric 16-tap Blackman-sinc kernel with `sum == 1.0`.
    #[must_use]
    pub fn new() -> Self {
        let mut raw = [0.0_f64; FIR_TAPS];
        let mut sum = 0.0_f64;
        let center = 0.5 * ((FIR_TAPS - 1) as f64);

        for (i, slot) in raw.iter_mut().enumerate() {
            let x = (i as f64) - center;
            let sinc = if x == 0.0 {
                2.0 * FIR_CUTOFF_NORM
            } else {
                let s = sin_cos_f64(2.0 * PI * FIR_CUTOFF_NORM * x).0;
                s / (PI * x)
            };
            let phase = TAU * ((i as f64) + 0.5) / (FIR_TAPS as f64);
            let c1 = sin_cos_f64(phase).1;
            let c2 = sin_cos_f64(2.0 * phase).1;
            let blackman = 0.42 - 0.50 * c1 + 0.08 * c2;
            let w = (sinc * blackman).max(0.0);
            *slot = w;
            sum += w;
        }

        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        let mut coeffs = [0.0_f32; FIR_TAPS];
        for (dst, &src) in coeffs.iter_mut().zip(raw.iter()) {
            *dst = (src * inv_sum) as f32;
        }
        Self { coeffs }
    }

    /// Returns the 16 FIR taps.
    #[must_use]
    #[inline]
    pub const fn coeffs(&self) -> &[f32; FIR_TAPS] {
        &self.coeffs
    }

    /// Filters 8 new oversampled forces against the previous 8 substeps in `prev_sub`.
    #[inline]
    pub fn process_8(&self, prev_sub: &mut [f32; OVERSAMPLE_FACTOR], cur: &[f32; 8]) -> f32 {
        let mut acc = 0.0_f32;
        for k in 0..OVERSAMPLE_FACTOR {
            acc += self.coeffs[k] * cur[7 - k];
            acc += self.coeffs[8 + k] * prev_sub[7 - k];
        }
        *prev_sub = *cur;
        acc
    }
}

impl Default for DecimationFir16 {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-key mutable `8x` oversampled strike state.
#[derive(Copy, Clone, Debug)]
pub struct HammerOversampleState {
    pub shank: ShankState,
    pub adaa_state: AdaaState,
    pub delta_prev_m: f32,
    pub prev_sub_forces: [f32; OVERSAMPLE_FACTOR],
    pub tail_samples_left: u8,
}

impl Default for HammerOversampleState {
    fn default() -> Self {
        Self {
            shank: ShankState::default(),
            adaa_state: AdaaState::new(),
            delta_prev_m: HAMMER_LET_OFF_Y0_M,
            prev_sub_forces: [0.0; OVERSAMPLE_FACTOR],
            tail_samples_left: 0,
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
    }

    /// Launches a new strike at `v_h0` (`m/s`), resetting ADAA phase tracking.
    #[inline]
    pub fn launch(&mut self, v_h0: f32) {
        self.shank.launch(v_h0);
        self.adaa_state.reset();
        self.delta_prev_m = HAMMER_LET_OFF_Y0_M;
        self.prev_sub_forces = [0.0; OVERSAMPLE_FACTOR];
        self.tail_samples_left = 0;
    }

    /// Returns `true` if hammer physics OR polyphase decimation tail is active.
    #[must_use]
    #[inline]
    pub const fn is_active(&self) -> bool {
        self.shank.active || self.tail_samples_left > 0
    }

    /// Executes 8 oversampled substeps (`384 kHz` @ `48 kHz`) and returns the
    /// decimated host-rate strike force (`N`) onto the string.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_host_sample(
        &mut self,
        shank_params: &ShankParams,
        felt_spec: &KeyFeltSpec,
        felt_coeffs: &FeltSharedCoeffs,
        felt_state: &mut FeltState,
        adaa_lut: &AdaaPowerLut,
        borin_lut: &BorinKLut,
        fir: &DecimationFir16,
        memory_enabled: bool,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32 {
        if !self.is_active() {
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

        let mut sub_forces = [0.0_f32; OVERSAMPLE_FACTOR];
        let dt_sub = shank_params.dt_sub;
        let dt_host = (OVERSAMPLE_FACTOR as f32) * dt_sub;
        if self.shank.active {
            if self.shank.elapsed_host_samples == 0 {
                self.delta_prev_m = self.shank.y_h() - safe_y;
            }
            let r_sub = safe_r * (1.0 / 8.0);
            // Include host-rate ZOH/FIR port series compliance so high-register strikes
            // (where K_felt * r_port * dt_host > 1) stay strictly Nyquist-passive across 8x->1x.
            let k_str_end = 0.5 * r_sub * dt_sub + 0.68 * safe_r * dt_host;

            let (f_scale_eff_n, eps_eff) = felt_state.effective_params(felt_spec, memory_enabled);
            let c_inst = 1.0 - eps_eff * (1.0 - felt_coeffs.a_fast_sub);
            let k_ham_nom = self.shank.predict_free(shank_params).k_ham;
            let lambda_end = borin_lut.compute_lambda(
                felt_spec.delta_scale_m,
                k_ham_nom + k_str_end,
                f_scale_eff_n * c_inst,
            );

            let mut y_s_loc = safe_y;
            let mut v_s_loc = safe_v;

            for slot in &mut sub_forces {
                let pred = self.shank.predict_free(shank_params);
                let y_s_free = y_s_loc + v_s_loc * dt_sub;
                let p_end_m = pred.y_h_free - y_s_free;
                let k_loop_end = pred.k_ham + k_str_end;

                let out = solve_coupled_substep(
                    adaa_lut,
                    borin_lut,
                    &mut self.adaa_state,
                    &mut felt_state.z_fast,
                    CoupledSubstepInput {
                        delta_prev_m: self.delta_prev_m,
                        p_end_m,
                        k_loop_end,
                        delta_scale_m: felt_spec.delta_scale_m,
                        f_scale_eff_n,
                        eps_eff,
                        a_fast_sub: felt_coeffs.a_fast_sub,
                        lambda_end,
                    },
                );

                self.shank.apply_force(shank_params, pred, out.force_n);
                let v_s_next = v_s_loc + r_sub * out.force_n;
                y_s_loc += 0.5 * (v_s_loc + v_s_next) * dt_sub;
                v_s_loc = v_s_next;
                self.delta_prev_m = self.shank.y_h() - y_s_loc;
                felt_state.accumulate_substep_compaction(felt_spec, out.u_norm_mid, memory_enabled);
                *slot = out.force_n;
            }

            if memory_enabled && felt_state.z_mem > 0.0 {
                felt_state.z_mem *= felt_coeffs.a_mem_host;
            }
            if !self.shank.finish_host_sample(shank_params) {
                self.tail_samples_left = 2;
            }
        } else {
            self.tail_samples_left = self.tail_samples_left.saturating_sub(1);
        }

        let f_fir = fir
            .process_8(&mut self.prev_sub_forces, &sub_forces)
            .max(0.0);
        let _ = (dt_host, safe_v);
        f_fir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimation_fir_is_dc_normalized_symmetric_and_non_negative() {
        let fir = DecimationFir16::new();
        let c = fir.coeffs();
        let sum: f32 = c.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        for i in 0..8 {
            assert!((c[i] - c[15 - i]).abs() < 1e-7);
        }
        assert!(c.iter().all(|&x| x >= 0.0));

        // Constant unit input converges to exact 1.0 output.
        let mut prev = [0.0_f32; 8];
        let ones = [1.0_f32; 8];
        let y0 = fir.process_8(&mut prev, &ones);
        let y1 = fir.process_8(&mut prev, &ones);
        assert!(y0 > 0.0 && y0 < 1.0);
        assert!((y1 - 1.0).abs() < 1e-6, "y1={y1}");

        // Draining with zeros hits exact 0.0 after 2 host steps.
        let zeros = [0.0_f32; 8];
        let t0 = fir.process_8(&mut prev, &zeros);
        let t1 = fir.process_8(&mut prev, &zeros);
        assert!(t0 > 0.0);
        assert_eq!(t1, 0.0);
    }
}
