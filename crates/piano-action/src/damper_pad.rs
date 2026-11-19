//! FTO #3 compliant unilateral geometric-clearance hysteretic damper contact pad
//! (`460` orthogonal V/H per-string slots across `0..70` damped keys, padded to `464`
//! for 64-byte-aligned `F32x4` SoA SIMD and decimated `{1, 2, 4, 8, 16}` control updates).

use piano_spec::{
    build_scale_design, StringPhysicalParams, VoicingSpec, CHOIR_LAYOUTS, DAMPED_KEYS, NUM_KEYS,
    TOTAL_STRINGS,
};

use crate::damper_pad_simd::DamperPadSoa;
use crate::damper_pad_symp::{passive_damp_force, pow_usize, smooth_felt_engage};
use crate::math::exp_f64;
use crate::pedals::CLEARANCE_OPEN_M;
use crate::pedals_clearance::snap_control_decim;

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

/// Unilateral geometric-clearance damper contact bank covering all `460` orthogonal V/H slots
/// (`464`-padded 64B-aligned SoA).
#[derive(Clone, Debug)]
pub struct DamperPadBank {
    pub(crate) lut: piano_simd::AdaaPowerLut,
    pub(crate) a_damp_felt: f32,
    pub(crate) a_damp_felt_decim: f32,
    pub(crate) control_decim: u8,
    pub(crate) c_nom_key: [f32; DAMPED_KEYS],
    pub(crate) f_scale_key: [f32; DAMPED_KEYS],
    pub(crate) soa: DamperPadSoa,
}

#[inline]
pub(crate) fn compute_damper_key_coeffs(phys: &StringPhysicalParams) -> (f32, f32) {
    let m_wire = phys.linear_density_kg_m * phys.speaking_length_m;
    let z0 = 2.0 * m_wire * phys.f0_et_hz;
    // Scale viscous damper by string mass with wave-impedance cap to avoid node-pinning.
    let c_target = (DAMPER_C_PER_MASS * m_wire).min(6.0 * z0);
    // Normalized Stulov nonlinear contact impedance boost proportional to tension/mass.
    let f_scale = 0.06 * DAMPER_C_PER_MASS * m_wire;
    (c_target as f32, f_scale as f32)
}

impl DamperPadBank {
    /// Initializes damper pad LUT and static per-key mechanical parameters (`decim = 1`).
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::new_with_decim(sample_rate, voicing, 1)
    }

    /// Initializes damper pad LUT, SoA buffers, and control-rate decimation (`{1,2,4,8,16}`).
    #[must_use]
    pub fn new_with_decim(sample_rate: f32, voicing: &VoicingSpec, control_decim: u8) -> Self {
        let decim = snap_control_decim(control_decim);
        let sr = f64::from(sample_rate.max(8_000.0));
        let a_damp_felt = exp_f64(-1.0 / (TAU_DAMP_FELT_S * sr)) as f32;
        let a_damp_felt_decim = pow_usize(a_damp_felt, usize::from(decim));
        let lut =
            piano_simd::AdaaPowerLut::new(DAMPER_EXPONENT_P, DAMPER_LUT_X_MAX, DAMPER_LUT_SIZE);

        let mut bank = Self {
            lut,
            a_damp_felt,
            a_damp_felt_decim,
            control_decim: decim,
            c_nom_key: [0.0_f32; DAMPED_KEYS],
            f_scale_key: [0.0_f32; DAMPED_KEYS],
            soa: DamperPadSoa::new(),
        };
        let specs = build_scale_design(voicing);
        bank.update_voicing_in_place_from_specs(&specs);
        bank.reset();
        bank
    }

    /// Configured damper control-rate decimation factor (`1`, `2`, `4`, `8`, or `16`).
    #[must_use]
    #[inline]
    pub const fn control_decim(&self) -> u8 {
        self.control_decim
    }

    /// Updates damper pad control-rate decimation (`{1, 2, 4, 8, 16}`) with zero allocations.
    #[inline]
    pub fn set_control_decim(&mut self, control_decim: u8) {
        let decim = snap_control_decim(control_decim);
        if decim == self.control_decim {
            return;
        }
        self.control_decim = decim;
        self.a_damp_felt_decim = pow_usize(self.a_damp_felt, usize::from(decim));
        self.soa.decim_phase.fill(0);
    }

    /// Updates per-key damper mechanical impedance coefficients for key `key_idx` from
    /// precomputed `phys` without invoking `build_scale_design`.
    #[inline]
    pub fn update_voicing_key(&mut self, key_idx: usize, phys: &StringPhysicalParams) {
        if key_idx >= DAMPED_KEYS {
            return;
        }
        let (c_nom, f_scale) = compute_damper_key_coeffs(phys);
        self.c_nom_key[key_idx] = c_nom;
        self.f_scale_key[key_idx] = f_scale;
        self.soa.set_key_coeffs(key_idx, c_nom, f_scale);
    }

    /// Updates all damped keys from precomputed `specs` without calling `build_scale_design`.
    pub fn update_voicing_in_place_from_specs(&mut self, specs: &[StringPhysicalParams; NUM_KEYS]) {
        for (k, phys) in specs.iter().enumerate().take(DAMPED_KEYS) {
            self.update_voicing_key(k, phys);
        }
    }

    /// Updates per-key damper mechanical impedance coefficients in-place without reallocating.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        let specs = build_scale_design(voicing);
        self.update_voicing_in_place_from_specs(&specs);
    }

    /// Resets all 460 V/H per-string ADAA and Stulov relaxation slots to closed-damper preload equilibrium.
    pub fn reset(&mut self) {
        let u_eq = -RESTING_PRELOAD_M * 1000.0_f32;
        let phi_eq = self.lut.eval_f(u_eq);
        self.soa.reset_all(u_eq, phi_eq, RESTING_PRELOAD_M);
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
        for slot in start..end {
            self.soa.reset_slot(slot, 0.0, 0.0, CLEARANCE_OPEN_M);
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
        if self.control_decim <= 1 {
            self.step_damper_sample_exact(
                key_idx,
                slot_idx,
                y_damp_m,
                v_damp_mps,
                r_port_damp,
                y_clear_m,
            )
        } else {
            self.step_damper_sample_decim(
                key_idx,
                slot_idx,
                y_damp_m,
                v_damp_mps,
                r_port_damp,
                y_clear_m,
            )
        }
    }

    #[inline]
    fn step_damper_sample_exact(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        let inv_scale = 1000.0_f32;
        if y_damp_m == 0.0 && v_damp_mps == 0.0 {
            let u_eq = (-y_clear_m).max(0.0) * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            self.soa.adaa_pos_x[slot_idx] = u_eq;
            self.soa.adaa_neg_x[slot_idx] = u_eq;
            let a = self.a_damp_felt;
            self.soa.z_pos[slot_idx] = a * self.soa.z_pos[slot_idx] + (1.0 - a) * phi_eq;
            self.soa.z_neg[slot_idx] = a * self.soa.z_neg[slot_idx] + (1.0 - a) * phi_eq;
            return 0.0;
        }

        let safe_r = r_port_damp.max(0.0);
        let lin_lim = (-0.85 * y_clear_m).min(8.0e-6);
        if y_clear_m <= MIN_CLOSED_LINEAR_CLEAR_M && y_damp_m.abs() <= lin_lim {
            let u_eq = -y_clear_m * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            self.soa.adaa_pos_x[slot_idx] = u_eq + y_damp_m * inv_scale;
            self.soa.adaa_neg_x[slot_idx] = u_eq - y_damp_m * inv_scale;
            let a = self.a_damp_felt;
            let z_next = a * self.soa.z_pos[slot_idx] + (1.0 - a) * phi_eq;
            self.soa.z_pos[slot_idx] = z_next;
            self.soa.z_neg[slot_idx] = z_next;
            let (c_raw, _) = self.eval_preload_c_vh(key_idx, y_clear_m, phi_eq, z_next, z_next);
            return passive_damp_force(c_raw, safe_r, v_damp_mps);
        }

        let delta_pos = (y_damp_m - y_clear_m).max(0.0);
        let delta_neg = (-y_damp_m - y_clear_m).max(0.0);
        let s_sum = delta_pos + delta_neg;
        let u_pos = delta_pos * inv_scale;
        let u_neg = delta_neg * inv_scale;
        let a = self.a_damp_felt;
        let zp = self.soa.z_pos[slot_idx];
        let zn = self.soa.z_neg[slot_idx];
        if s_sum <= ENGAGE_ONSET_M && zp <= 1.0e-5 && zn <= 1.0e-5 {
            self.soa.adaa_pos_x[slot_idx] = u_pos;
            self.soa.adaa_neg_x[slot_idx] = u_neg;
            self.soa.z_pos[slot_idx] = zp * a;
            self.soa.z_neg[slot_idx] = zn * a;
            return 0.0;
        }

        let prev_pos = self.soa.adaa_pos_x[slot_idx];
        let phi_pos = if u_pos == 0.0 && prev_pos == 0.0 {
            0.0
        } else {
            self.lut.adaa1(u_pos, prev_pos)
        };
        self.soa.adaa_pos_x[slot_idx] = u_pos;

        let prev_neg = self.soa.adaa_neg_x[slot_idx];
        let phi_neg = if u_neg == 0.0 && prev_neg == 0.0 {
            0.0
        } else {
            self.lut.adaa1(u_neg, prev_neg)
        };
        self.soa.adaa_neg_x[slot_idx] = u_neg;

        let zp_next = a * zp + (1.0 - a) * phi_pos;
        let zn_next = a * zn + (1.0 - a) * phi_neg;
        self.soa.z_pos[slot_idx] = zp_next;
        self.soa.z_neg[slot_idx] = zn_next;

        if s_sum <= 0.0 {
            return 0.0;
        }

        let felt_hyst = (phi_pos - EPS_DAMP_HYST * zp_next).max(0.0)
            + (phi_neg - EPS_DAMP_HYST * zn_next).max(0.0);
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
        if self.control_decim <= 1 {
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
                self.soa.adaa_pos_x[sv_idx] = u_eq + y_dv_m * inv_scale;
                self.soa.adaa_neg_x[sv_idx] = u_eq - y_dv_m * inv_scale;
                let zv = a * self.soa.z_pos[sv_idx] + one_m_a_phi;
                self.soa.z_pos[sv_idx] = zv;
                self.soa.z_neg[sv_idx] = zv;
                self.soa.adaa_pos_x[sv_idx + 1] = u_eq + y_dh_m * inv_scale;
                self.soa.adaa_neg_x[sv_idx + 1] = u_eq - y_dh_m * inv_scale;
                let zh = a * self.soa.z_pos[sv_idx + 1] + one_m_a_phi;
                self.soa.z_pos[sv_idx + 1] = zh;
                self.soa.z_neg[sv_idx + 1] = zh;
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
        } else {
            let pv = self.soa.decim_phase[sv_idx];
            let ph = self.soa.decim_phase[sv_idx + 1];
            let (srv, srh) = (r_dv.max(0.0), r_dh.max(0.0));
            if pv > 0
                && ph > 0
                && self.soa.mode_linear[sv_idx] > 0.5
                && self.soa.mode_linear[sv_idx + 1] > 0.5
                && self.soa.r_cached[sv_idx] == srv
                && self.soa.r_cached[sv_idx + 1] == srh
            {
                let ycv = self.soa.yc_cached[sv_idx];
                let ych = self.soa.yc_cached[sv_idx + 1];
                if y_dv_m.abs() <= (-0.85 * ycv).min(8.0e-6)
                    && y_dh_m.abs() <= (-0.85 * ych).min(8.0e-6)
                {
                    let nv = pv + 1;
                    let nh = ph + 1;
                    self.soa.decim_phase[sv_idx] = if nv >= self.control_decim { 0 } else { nv };
                    self.soa.decim_phase[sv_idx + 1] =
                        if nh >= self.control_decim { 0 } else { nh };
                    return (
                        self.soa.g_eff_cached[sv_idx] * v_dv_mps,
                        self.soa.g_eff_cached[sv_idx + 1] * v_dh_mps,
                    );
                }
            }
        }

        let fv = self.step_damper_sample(key_idx, sv_idx, y_dv_m, v_dv_mps, r_dv, y_clear_m);
        let fh = self.step_damper_sample(key_idx, sv_idx + 1, y_dh_m, v_dh_mps, r_dh, y_clear_m);
        (fv, fh)
    }
}

#[cfg(test)]
#[path = "damper_pad_tests.rs"]
mod tests;
