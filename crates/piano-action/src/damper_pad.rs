//! FTO #3 compliant unilateral geometric-clearance hysteretic damper contact pad
//! (`460` orthogonal V/H per-string slots across `0..70` damped keys, passive at `44.1/48/96 kHz`).

use piano_simd::{AdaaPowerLut, AdaaState};
use piano_spec::{build_scale_design, VoicingSpec, CHOIR_LAYOUTS, DAMPED_KEYS, TOTAL_STRINGS};

use crate::damper_pad_symp::{passive_damp_force, smooth_felt_engage};
use crate::math::exp_f64;

/// Total independent unilateral damper pad slots (`2 * s_idx` = V, `2 * s_idx + 1` = H).
pub const DAMPER_PAD_SLOTS: usize = TOTAL_STRINGS * 2;

const DAMPER_EXPONENT_P: f32 = 2.2;
const DAMPER_LUT_SIZE: usize = 512;
const DAMPER_LUT_X_MAX: f32 = 2.0;
const RESTING_PRELOAD_M: f32 = -3.0e-4;
pub(crate) const ENGAGE_ONSET_M: f32 = 3.0e-6;
#[cfg(test)]
const ENGAGE_HALF_M: f32 = 8.0e-5;
pub(crate) const INV_ENGAGE_HALF_M: f32 = 1.0 / 8.0e-5;
pub(crate) const MIN_CLOSED_LINEAR_CLEAR_M: f32 = -2.0e-7;
const TAU_DAMP_FELT_S: f64 = 1.2e-3;
pub(crate) const EPS_DAMP_HYST: f32 = 0.25;
const DAMPER_C_PER_MASS: f64 = 380.0;
pub(crate) const HYST_DAMP_GAIN: f32 = 1.2;
pub(crate) const MAX_PASSIVE_PORT_RATIO: f32 = 2.4;

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct DamperStringSlot {
    pub(crate) adaa_pos: AdaaState,
    pub(crate) adaa_neg: AdaaState,
    pub(crate) z_pos: f32,
    pub(crate) z_neg: f32,
}

impl DamperStringSlot {
    #[inline]
    pub(crate) fn reset_to_preload(&mut self, u_eq: f32, phi_eq: f32) {
        self.adaa_pos.sync_x(u_eq);
        self.adaa_neg.sync_x(u_eq);
        self.z_pos = phi_eq;
        self.z_neg = phi_eq;
    }
}

/// Unilateral geometric-clearance damper contact bank covering all `460` orthogonal V/H slots.
#[derive(Clone, Debug)]
pub struct DamperPadBank {
    pub(crate) lut: AdaaPowerLut,
    pub(crate) a_damp_felt: f32,
    pub(crate) c_nom_key: [f32; DAMPED_KEYS],
    pub(crate) f_scale_key: [f32; DAMPED_KEYS],
    pub(crate) slots: [DamperStringSlot; DAMPER_PAD_SLOTS],
}

fn fill_damper_key_coeffs(
    voicing: &VoicingSpec,
    c_nom_key: &mut [f32; DAMPED_KEYS],
    f_scale_key: &mut [f32; DAMPED_KEYS],
) {
    let specs = build_scale_design(voicing);
    for k in 0..DAMPED_KEYS {
        let p = specs[k];
        let m_wire = p.linear_density_kg_m * p.speaking_length_m;
        let z0 = 2.0 * m_wire * p.f0_et_hz;
        // Scale viscous damper by string mass with wave-impedance cap to avoid node-pinning.
        let c_target = (DAMPER_C_PER_MASS * m_wire).min(6.0 * z0);
        c_nom_key[k] = c_target as f32;
        // Normalized Stulov nonlinear contact impedance boost proportional to tension/mass.
        f_scale_key[k] = (0.06 * DAMPER_C_PER_MASS * m_wire) as f32;
    }
}

impl DamperPadBank {
    /// Initializes damper pad LUT and static per-key mechanical parameters.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let sr = f64::from(sample_rate.max(8_000.0));
        let a_damp_felt = exp_f64(-1.0 / (TAU_DAMP_FELT_S * sr)) as f32;
        let lut = AdaaPowerLut::new(DAMPER_EXPONENT_P, DAMPER_LUT_X_MAX, DAMPER_LUT_SIZE);

        let mut c_nom_key = [0.0_f32; DAMPED_KEYS];
        let mut f_scale_key = [0.0_f32; DAMPED_KEYS];
        fill_damper_key_coeffs(voicing, &mut c_nom_key, &mut f_scale_key);

        let mut bank = Self {
            lut,
            a_damp_felt,
            c_nom_key,
            f_scale_key,
            slots: [DamperStringSlot::default(); DAMPER_PAD_SLOTS],
        };
        bank.reset();
        bank
    }

    /// Updates per-key damper mechanical impedance coefficients in-place without reallocating.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        fill_damper_key_coeffs(voicing, &mut self.c_nom_key, &mut self.f_scale_key);
    }

    /// Resets all 460 V/H per-string ADAA and Stulov relaxation slots to closed-damper preload equilibrium.
    pub fn reset(&mut self) {
        let u_eq = -RESTING_PRELOAD_M * 1000.0_f32;
        let phi_eq = self.lut.eval_f(u_eq);
        for slot in &mut self.slots {
            slot.reset_to_preload(u_eq, phi_eq);
        }
    }

    /// Clears ADAA and hysteresis preload state for `key_idx`'s choir strings when lifted open (`+5 mm`).
    #[inline]
    pub fn lift_key_open(&mut self, key_idx: usize) {
        if key_idx >= DAMPED_KEYS {
            return;
        }
        let layout = CHOIR_LAYOUTS[key_idx];
        let start = usize::from(layout.first_string_idx) * 2;
        let end = start + usize::from(layout.strings_per_key) * 2;
        for slot in &mut self.slots[start..end] {
            slot.reset_to_preload(0.0, 0.0);
        }
    }

    /// Evaluates one host-rate unilateral damper contact step for `(key_idx, slot_idx)`
    /// (`slot_idx = 2 * s_idx` vertical, `2 * s_idx + 1` horizontal),
    /// returning the passive force `F_damp` (`N`) applied **ON the string**.
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
        if key_idx >= DAMPED_KEYS
            || slot_idx >= DAMPER_PAD_SLOTS
            || !y_damp_m.is_finite()
            || !v_damp_mps.is_finite()
            || !r_port_damp.is_finite()
            || !y_clear_m.is_finite()
        {
            return 0.0;
        }

        let slot = &mut self.slots[slot_idx];
        let inv_scale = 1000.0_f32;

        if y_damp_m == 0.0 && v_damp_mps == 0.0 {
            let u_eq = (-y_clear_m).max(0.0) * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            slot.adaa_pos.sync_x(u_eq);
            slot.adaa_neg.sync_x(u_eq);
            let a = self.a_damp_felt;
            slot.z_pos = a * slot.z_pos + (1.0 - a) * phi_eq;
            slot.z_neg = a * slot.z_neg + (1.0 - a) * phi_eq;
            return 0.0;
        }

        let safe_r = r_port_damp.max(0.0);
        let lin_lim = (-0.85 * y_clear_m).min(8.0e-6);
        if y_clear_m <= MIN_CLOSED_LINEAR_CLEAR_M && y_damp_m.abs() <= lin_lim {
            let u_eq = -y_clear_m * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            slot.adaa_pos.sync_x(u_eq + y_damp_m * inv_scale);
            slot.adaa_neg.sync_x(u_eq - y_damp_m * inv_scale);
            let a = self.a_damp_felt;
            let z_next = a * slot.z_pos + (1.0 - a) * phi_eq;
            (slot.z_pos, slot.z_neg) = (z_next, z_next);
            let (c_raw, _) = self.eval_preload_c_vh(key_idx, y_clear_m, phi_eq, z_next, z_next);
            return passive_damp_force(c_raw, safe_r, v_damp_mps);
        }

        let delta_pos = (y_damp_m - y_clear_m).max(0.0);
        let delta_neg = (-y_damp_m - y_clear_m).max(0.0);
        let s_sum = delta_pos + delta_neg;
        let u_pos = delta_pos * inv_scale;
        let u_neg = delta_neg * inv_scale;
        let a = self.a_damp_felt;
        if s_sum <= ENGAGE_ONSET_M && slot.z_pos <= 1.0e-5 && slot.z_neg <= 1.0e-5 {
            slot.adaa_pos.sync_x(u_pos);
            slot.adaa_neg.sync_x(u_neg);
            slot.z_pos *= a;
            slot.z_neg *= a;
            return 0.0;
        }

        let phi_pos = if u_pos == 0.0 && slot.adaa_pos.is_zero() {
            0.0
        } else {
            slot.adaa_pos.process(&self.lut, u_pos)
        };
        let phi_neg = if u_neg == 0.0 && slot.adaa_neg.is_zero() {
            0.0
        } else {
            slot.adaa_neg.process(&self.lut, u_neg)
        };

        slot.z_pos = a * slot.z_pos + (1.0 - a) * phi_pos;
        slot.z_neg = a * slot.z_neg + (1.0 - a) * phi_neg;

        if s_sum <= 0.0 {
            return 0.0;
        }

        let felt_hyst = (phi_pos - EPS_DAMP_HYST * slot.z_pos).max(0.0)
            + (phi_neg - EPS_DAMP_HYST * slot.z_neg).max(0.0);

        let engage = smooth_felt_engage(s_sum);
        if engage <= 0.0 {
            return 0.0;
        }
        let c_raw = engage
            * (self.c_nom_key[key_idx] * (1.0 + HYST_DAMP_GAIN * felt_hyst)
                + self.f_scale_key[key_idx] * felt_hyst);
        passive_damp_force(c_raw, safe_r, v_damp_mps)
    }

    /// Evaluates one host sample for both V (`2 * str_idx`) and H (`2 * str_idx + 1`)
    /// slots of `(key_idx, str_idx)`, sharing geometric clearance computations.
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
        if key_idx >= DAMPED_KEYS
            || str_idx >= TOTAL_STRINGS
            || !y_clear_m.is_finite()
            || !y_dv_m.is_finite()
            || !v_dv_mps.is_finite()
            || !r_dv.is_finite()
            || !y_dh_m.is_finite()
            || !v_dh_mps.is_finite()
            || !r_dh.is_finite()
        {
            return (0.0, 0.0);
        }

        let sv_idx = str_idx * 2;
        let inv_scale = 1000.0_f32;
        let lin_lim = (-0.85 * y_clear_m).min(8.0e-6);
        if y_clear_m <= MIN_CLOSED_LINEAR_CLEAR_M
            && y_dv_m.abs() <= lin_lim
            && y_dh_m.abs() <= lin_lim
        {
            let u_eq = -y_clear_m * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            let a = self.a_damp_felt;
            let one_m_a_phi = (1.0 - a) * phi_eq;
            let (zv, zh) = {
                let [sv, sh] = &mut self.slots[sv_idx..sv_idx + 2] else {
                    return (0.0, 0.0);
                };
                sv.adaa_pos.sync_x(u_eq + y_dv_m * inv_scale);
                sv.adaa_neg.sync_x(u_eq - y_dv_m * inv_scale);
                let zv = a * sv.z_pos + one_m_a_phi;
                (sv.z_pos, sv.z_neg) = (zv, zv);
                sh.adaa_pos.sync_x(u_eq + y_dh_m * inv_scale);
                sh.adaa_neg.sync_x(u_eq - y_dh_m * inv_scale);
                let zh = a * sh.z_pos + one_m_a_phi;
                (sh.z_pos, sh.z_neg) = (zh, zh);
                (zv, zh)
            };
            let (cv, ch) = self.eval_preload_c_vh(key_idx, y_clear_m, phi_eq, zv, zh);
            let (srv, srh) = (r_dv.max(0.0), r_dh.max(0.0));
            let fv = if y_dv_m == 0.0 && v_dv_mps == 0.0 {
                0.0
            } else {
                passive_damp_force(cv, srv, v_dv_mps)
            };
            let fh = if y_dh_m == 0.0 && v_dh_mps == 0.0 {
                0.0
            } else {
                passive_damp_force(ch, srh, v_dh_mps)
            };
            return (fv, fh);
        }

        let fv = self.step_damper_sample(key_idx, sv_idx, y_dv_m, v_dv_mps, r_dv, y_clear_m);
        let fh = self.step_damper_sample(key_idx, sv_idx + 1, y_dh_m, v_dh_mps, r_dh, y_clear_m);
        (fv, fh)
    }
}

#[cfg(test)]
#[path = "damper_pad_tests.rs"]
mod tests;
