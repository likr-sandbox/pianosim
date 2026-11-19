//! Stage 1 physical piano action: pedals, 1st-order Timoshenko hammer shank,
//! two-timescale Stulov wool felt, Borin K-method + 1st-order ADAA contact solver,
//! `2x/4x/8x` oversampled strike loop, and FTO #3 unilateral hysteretic damper pad.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::boxed::Box;

use piano_spec::{PerfProfile, VoicingSpec, CHOIR_LAYOUTS, NUM_KEYS};

mod action_bank_ops;
pub mod agraffe_delay;
pub mod borin_k;
pub mod damper_pad;
mod damper_pad_symp;
pub mod decimation_fir;
pub mod felt_stulov;
pub mod k_method;
pub mod k_method_poly;
pub mod keybed;
mod math;
pub mod oversample;
pub mod p_grid;
pub mod pedals;
mod pedals_clearance;
pub mod shank_timoshenko;

#[cfg(test)]
mod tests_common;
#[cfg(test)]
mod tests_damper;
#[cfg(test)]
mod tests_escapement;
#[cfg(test)]
mod tests_hammer;
#[cfg(test)]
mod tests_m3_action;

use action_bank_ops::KeyActionSlot;
use damper_pad::DamperPadBank;
use decimation_fir::DecimationFir;
use felt_stulov::FeltSharedCoeffs;
use keybed::KeybedBank;
use oversample::HammerContactSolver;
use p_grid::PGridLuts;
use pedals::PedalBank;

/// Complete Stage 1 physical piano action bank for all 88 keys and 230 strings.
///
/// All LUTs and per-key buffers are allocated once inside [`ActionBank::new`] /
/// [`ActionBank::new_with_profile`]. Every sample-rate and event method executes
/// with zero heap allocations, zero locks, zero syscalls, and zero panics.
#[derive(Clone, Debug)]
pub struct ActionBank {
    pub(crate) sample_rate: f32,
    pub(crate) memory_enabled: bool,
    pub(crate) perf: PerfProfile,
    pub(crate) felt_coeffs: FeltSharedCoeffs,
    pub(crate) p_grid: Option<PGridLuts>,
    pub(crate) decimation_fir: DecimationFir,
    pub(crate) pedals: PedalBank,
    pub(crate) damper_pad: DamperPadBank,
    pub(crate) keybed: KeybedBank,
    pub(crate) keys: Box<[KeyActionSlot]>,
    pub(crate) hammer_voicing: VoicingSpec,
    pub(crate) key_hardness_applied: [f32; NUM_KEYS],
}

impl ActionBank {
    /// Constructs and pre-tabulates the Stage 1 action bank for `sample_rate` and `voicing`
    /// using [`PerfProfile::full()`].
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::new_with_profile(sample_rate, voicing, &PerfProfile::full())
    }

    /// Configured host audio sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns `true` if `key_idx`'s hammer or polyphase FIR tail is active.
    #[must_use]
    #[inline]
    pub fn is_hammer_active(&self, key_idx: usize) -> bool {
        self.keys
            .get(key_idx)
            .is_some_and(|slot| slot.hammer.is_active())
    }

    /// Current smoothed damper clearance (`m`) for `key_idx`.
    #[must_use]
    #[inline]
    pub fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        self.pedals.damper_clearance_m(key_idx)
    }

    /// Returns `Some(clearance_m)` when `key_idx`'s damper is settled open
    /// (`>= clear_threshold_m`) with no active smoothing or peel-off pulse.
    #[must_use]
    #[inline]
    pub fn is_damper_settled_open(&self, key_idx: usize, clear_threshold_m: f32) -> Option<f32> {
        self.pedals
            .is_damper_settled_open(key_idx, clear_threshold_m)
    }

    /// Current continuous sustain pedal position (`0.0..=1.0`).
    #[must_use]
    #[inline]
    pub fn sustain(&self) -> f32 {
        self.pedals.sustain()
    }

    /// Una Corda choir strike weights `[w0, w1, w2]` for `key_idx`.
    #[must_use]
    #[inline]
    pub fn strike_weights(&self, key_idx: usize) -> [f32; 3] {
        self.pedals.strike_weights(key_idx)
    }

    /// Instantaneous damper peel-off adhesion force (`N`, `>= 0`) for `key_idx`.
    #[must_use]
    #[inline]
    pub fn damper_peel_force_n(&self, key_idx: usize) -> f32 {
        self.pedals.damper_peel_force_n(key_idx)
    }

    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and peel-off adhesion forces (`N`)
    /// into `out_peel_n[..n]`. Returns `(min_clear_m, any_peel_nonzero)`.
    #[inline]
    pub fn step_key_clearance_tile_with_peel(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
        out_peel_n: &mut [f32; 8],
    ) -> (f32, bool) {
        self.pedals
            .step_key_clearance_tile_with_peel(key_idx, t_len, out_clears, out_peel_n)
    }

    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and returning the tile minimum.
    #[inline]
    pub fn step_key_clearance_tile(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
    ) -> f32 {
        self.pedals
            .step_key_clearance_tile(key_idx, t_len, out_clears)
    }

    /// Advances the 1-pole damper rail smoother by one host sample and returns `y_clear_m`.
    #[inline]
    pub fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        self.pedals.step_key_clearance(key_idx)
    }

    /// Steps one host sample of the oversampled hammer-string strike interaction,
    /// returning the decimated upward force (`N >= 0`) applied to the string choir.
    #[inline]
    pub fn step_hammer_sample(
        &mut self,
        key_idx: usize,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32 {
        let uc_mod = self.pedals.una_corda_felt_mod(key_idx);
        let w = self.pedals.strike_weights(key_idx);
        let Some(slot) = self.keys.get_mut(key_idx) else {
            return 0.0;
        };
        let w_sum = (w[0] + w[1] + w[2]).max(1.0);
        let n_str = f32::from(CHOIR_LAYOUTS[key_idx].strings_per_key.max(1));
        let uc_choir_scale = n_str / w_sum;
        let solver = if self.perf.polynomial_adaa {
            HammerContactSolver::Poly {
                poly: &slot.poly_adaa,
            }
        } else if let Some(ref p_grid) = self.p_grid {
            let (p_lo, p_hi, p_t) = p_grid.lookup(slot.felt_spec.exponent_p);
            let (adaa_lo, adaa_hi, borin_lo, borin_hi) = p_grid.luts_for(p_lo, p_hi);
            if p_t <= 1.0e-5 {
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
            }
        } else {
            HammerContactSolver::Poly {
                poly: &slot.poly_adaa,
            }
        };
        slot.hammer.step_host_sample_with_solver(
            &slot.shank_params,
            &slot.felt_spec,
            &self.felt_coeffs,
            &mut slot.felt_state,
            solver,
            &self.decimation_fir,
            self.memory_enabled,
            uc_mod,
            y_str_m,
            v_str_mps,
            r_port_strike,
            Some(&mut slot.agraffe),
            uc_choir_scale,
        )
    }

    /// Overrides `key_idx`'s oversampled agraffe round-trip delay and wave admittance.
    #[inline]
    pub fn set_agraffe_delay_samples(
        &mut self,
        key_idx: usize,
        delay_substeps: f32,
        wave_admittance: f32,
    ) {
        if let Some(slot) = self.keys.get_mut(key_idx) {
            slot.agraffe
                .set_delay_params(delay_substeps, wave_admittance);
        }
    }

    /// Returns the most recent peak normalized felt compression (`[0.0, 1.0]`) for `key_idx`.
    #[must_use]
    #[inline]
    pub fn hammer_compression_norm(&self, key_idx: usize) -> f32 {
        self.keys
            .get(key_idx)
            .map_or(0.0, |slot| slot.hammer.last_u_norm)
    }

    /// Returns `(last_host_force_n, prev_sub_forces)` from the most recent strike step of `key_idx`.
    #[must_use]
    #[inline]
    pub fn last_hammer_forces(&self, key_idx: usize) -> (f32, [f32; 8]) {
        self.keys.get(key_idx).map_or((0.0, [0.0; 8]), |slot| {
            (slot.hammer.last_host_force_n, slot.hammer.prev_sub_forces)
        })
    }

    /// Steps one host sample of the unilateral hysteretic damper pad for `(key_idx, slot_idx)`
    /// (`slot_idx = 2 * s_idx` vertical, `2 * s_idx + 1` horizontal),
    /// returning the passive damping force `F_damp` (`N`) applied **ON the string**.
    #[inline]
    pub fn step_damper_sample(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        self.damper_pad.step_damper_sample(
            key_idx,
            slot_idx,
            y_damp_m,
            v_damp_mps,
            r_port_damp,
            y_clear_m,
        )
    }

    /// Evaluates one host sample for both V (`2 * str_idx`) and H (`2 * str_idx + 1`)
    /// damper slots of `(key_idx, str_idx)`, sharing geometric clearance computations.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn step_damper_pair(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        y_dv_m: f32,
        v_dv_mps: f32,
        r_dv: f32,
        y_dh_m: f32,
        v_dh_mps: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> (f32, f32) {
        self.damper_pad.step_damper_pair(
            key_idx, str_idx, y_dv_m, v_dv_mps, r_dv, y_dh_m, v_dh_mps, r_dh, y_clear_m,
        )
    }

    /// Returns exact closed-damper linear-equilibrium V/H velocity gains when felt memory has settled.
    #[inline]
    pub fn symp_linear_gains_if_settled(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> Option<(f32, f32)> {
        self.damper_pad
            .symp_linear_gains_if_settled(key_idx, str_idx, r_dv, r_dh, y_clear_m)
    }

    /// Advances quiescent damper felt relaxation over `sub` samples and returns linear V/H gains.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn symp_linear_gains_for_chunk(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        yc0: f32,
        yc_end: f32,
        sub: usize,
    ) -> Option<(f32, f32)> {
        self.damper_pad
            .symp_linear_gains_for_chunk(key_idx, str_idx, r_dv, r_dh, yc0, yc_end, sub)
    }

    /// Relaxes damper felt hysteresis states `(z_pos, z_neg)` across `sub` open-gap samples.
    #[inline]
    pub fn relax_open_gap_chunk(&mut self, str_idx: usize, sub: usize) {
        self.damper_pad.relax_open_gap_chunk(str_idx, sub);
    }

    /// Advances active keybed rail impacts by one host sample, returning the total
    /// Stage 4 Zone 8 mechanical keybed bottoming force (`N`).
    #[inline]
    pub fn step_keybed_force_sample(&mut self) -> f32 {
        self.keybed.step_sample()
    }
}
