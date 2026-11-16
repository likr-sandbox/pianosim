//! FTO #3 compliant unilateral geometric-clearance hysteretic damper contact pad
//! (`460` orthogonal V/H per-string slots across `0..70` damped keys, passive at `44.1/48/96 kHz`).

use piano_simd::{AdaaPowerLut, AdaaState};
use piano_spec::{build_scale_design, VoicingSpec, DAMPED_KEYS, TOTAL_STRINGS};

use crate::math::exp_f64;

/// Total independent unilateral damper pad slots (`2 * s_idx` = V, `2 * s_idx + 1` = H).
pub const DAMPER_PAD_SLOTS: usize = TOTAL_STRINGS * 2;

const DAMPER_EXPONENT_P: f32 = 2.2;
const DAMPER_LUT_SIZE: usize = 512;
const DAMPER_LUT_X_MAX: f32 = 2.0;
const RESTING_PRELOAD_M: f32 = -3.0e-4;
const ENGAGE_HALF_M: f32 = 8.0e-5;
const TAU_DAMP_FELT_S: f64 = 1.2e-3;
const EPS_DAMP_HYST: f32 = 0.25;
const DAMPER_C_PER_MASS: f64 = 380.0;
const HYST_DAMP_GAIN: f32 = 1.2;
const MAX_PASSIVE_PORT_RATIO: f32 = 2.4;

#[derive(Copy, Clone, Debug, Default)]
struct DamperStringSlot {
    adaa_pos: AdaaState,
    adaa_neg: AdaaState,
    z_pos: f32,
    z_neg: f32,
}

impl DamperStringSlot {
    #[inline]
    fn reset_to_preload(&mut self, u_eq: f32, phi_eq: f32) {
        self.adaa_pos.sync_x(u_eq);
        self.adaa_neg.sync_x(u_eq);
        self.z_pos = phi_eq;
        self.z_neg = phi_eq;
    }
}

/// Unilateral geometric-clearance damper contact bank covering all `460` orthogonal V/H slots.
#[derive(Clone, Debug)]
pub struct DamperPadBank {
    lut: AdaaPowerLut,
    a_damp_felt: f32,
    c_nom_key: [f32; DAMPED_KEYS],
    f_scale_key: [f32; DAMPED_KEYS],
    slots: [DamperStringSlot; DAMPER_PAD_SLOTS],
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
        if y_clear_m <= -5.0e-5 && y_damp_m.abs() <= 8.0e-6 {
            let u_eq = -y_clear_m * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            slot.adaa_pos.sync_x(u_eq + y_damp_m * inv_scale);
            slot.adaa_neg.sync_x(u_eq - y_damp_m * inv_scale);
            let a = self.a_damp_felt;
            let z_next = a * slot.z_pos + (1.0 - a) * phi_eq;
            slot.z_pos = z_next;
            slot.z_neg = z_next;
            let felt_hyst = 2.0 * (phi_eq - EPS_DAMP_HYST * z_next).max(0.0);
            let s_sum = -2.0 * y_clear_m;
            let engage = s_sum / (s_sum + ENGAGE_HALF_M);
            let c_raw = self.c_nom_key[key_idx] * engage * (1.0 + HYST_DAMP_GAIN * felt_hyst)
                + self.f_scale_key[key_idx] * felt_hyst;
            let c_eff = if c_raw * safe_r > MAX_PASSIVE_PORT_RATIO && safe_r > 0.0 {
                MAX_PASSIVE_PORT_RATIO / safe_r
            } else {
                c_raw
            };
            return (-c_eff * v_damp_mps) / (1.0 + 0.5 * safe_r * c_eff);
        }

        let delta_pos = (y_damp_m - y_clear_m).max(0.0);
        let delta_neg = (-y_damp_m - y_clear_m).max(0.0);
        let s_sum = delta_pos + delta_neg;

        let u_pos = delta_pos * inv_scale;
        let u_neg = delta_neg * inv_scale;
        let phi_pos = slot.adaa_pos.process(&self.lut, u_pos);
        let phi_neg = slot.adaa_neg.process(&self.lut, u_neg);

        let a = self.a_damp_felt;
        slot.z_pos = a * slot.z_pos + (1.0 - a) * phi_pos;
        slot.z_neg = a * slot.z_neg + (1.0 - a) * phi_neg;

        if s_sum <= 0.0 {
            return 0.0;
        }

        let felt_hyst = (phi_pos - EPS_DAMP_HYST * slot.z_pos).max(0.0)
            + (phi_neg - EPS_DAMP_HYST * slot.z_neg).max(0.0);

        let engage = s_sum / (s_sum + ENGAGE_HALF_M);
        let c_raw = self.c_nom_key[key_idx] * engage * (1.0 + HYST_DAMP_GAIN * felt_hyst)
            + self.f_scale_key[key_idx] * felt_hyst;
        let c_eff = if c_raw * safe_r > MAX_PASSIVE_PORT_RATIO && safe_r > 0.0 {
            MAX_PASSIVE_PORT_RATIO / safe_r
        } else {
            c_raw
        };

        (-c_eff * v_damp_mps) / (1.0 + 0.5 * safe_r * c_eff)
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
        if y_clear_m <= -5.0e-5 && y_dv_m.abs() <= 8.0e-6 && y_dh_m.abs() <= 8.0e-6 {
            let u_eq = -y_clear_m * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            let a = self.a_damp_felt;
            let one_m_a_phi = (1.0 - a) * phi_eq;
            let s_sum = -2.0 * y_clear_m;
            let engage = s_sum / (s_sum + ENGAGE_HALF_M);
            let c_nom_e = self.c_nom_key[key_idx] * engage;
            let f_scale = self.f_scale_key[key_idx];

            let [sv, sh] = &mut self.slots[sv_idx..sv_idx + 2] else {
                return (0.0, 0.0);
            };
            sv.adaa_pos.sync_x(u_eq + y_dv_m * inv_scale);
            sv.adaa_neg.sync_x(u_eq - y_dv_m * inv_scale);
            let zv = a * sv.z_pos + one_m_a_phi;
            sv.z_pos = zv;
            sv.z_neg = zv;

            sh.adaa_pos.sync_x(u_eq + y_dh_m * inv_scale);
            sh.adaa_neg.sync_x(u_eq - y_dh_m * inv_scale);
            let zh = a * sh.z_pos + one_m_a_phi;
            sh.z_pos = zh;
            sh.z_neg = zh;

            let hv = 2.0 * (phi_eq - EPS_DAMP_HYST * zv).max(0.0);
            let hh = 2.0 * (phi_eq - EPS_DAMP_HYST * zh).max(0.0);
            let cv = c_nom_e * (1.0 + HYST_DAMP_GAIN * hv) + f_scale * hv;
            let ch = c_nom_e * (1.0 + HYST_DAMP_GAIN * hh) + f_scale * hh;
            let srv = r_dv.max(0.0);
            let srh = r_dh.max(0.0);
            let cev = if cv * srv > MAX_PASSIVE_PORT_RATIO && srv > 0.0 {
                MAX_PASSIVE_PORT_RATIO / srv
            } else {
                cv
            };
            let ceh = if ch * srh > MAX_PASSIVE_PORT_RATIO && srh > 0.0 {
                MAX_PASSIVE_PORT_RATIO / srh
            } else {
                ch
            };
            let fv = if y_dv_m == 0.0 && v_dv_mps == 0.0 {
                0.0
            } else {
                (-cev * v_dv_mps) / (1.0 + 0.5 * srv * cev)
            };
            let fh = if y_dh_m == 0.0 && v_dh_mps == 0.0 {
                0.0
            } else {
                (-ceh * v_dh_mps) / (1.0 + 0.5 * srh * ceh)
            };
            return (fv, fh);
        }

        let fv = self.step_damper_sample(key_idx, sv_idx, y_dv_m, v_dv_mps, r_dv, y_clear_m);
        let fh = self.step_damper_sample(key_idx, sv_idx + 1, y_dh_m, v_dh_mps, r_dh, y_clear_m);
        (fv, fh)
    }

    /// Returns pre-evaluated linear equilibrium V/H damper force-per-velocity gains `(gv, gh)`
    /// for `(key_idx, str_idx)` when `y_clear_m <= -5e-5` and felt hysteresis state `z` has
    /// converged to its exact fixed point `phi(y_clear_m)` (within `1e-5`).
    #[inline]
    pub fn symp_linear_gains_if_settled(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> Option<(f32, f32)> {
        if key_idx >= DAMPED_KEYS
            || str_idx >= TOTAL_STRINGS
            || !y_clear_m.is_finite()
            || y_clear_m > -5.0e-5
            || !r_dv.is_finite()
            || !r_dh.is_finite()
        {
            return None;
        }
        let u_eq = -y_clear_m * 1000.0_f32;
        let phi_eq = self.lut.eval_f(u_eq);
        let sv_idx = str_idx * 2;
        let [sv, sh] = &mut self.slots[sv_idx..sv_idx + 2] else {
            return None;
        };
        if (sv.z_pos - phi_eq).abs() > 1.0e-5
            || (sv.z_neg - phi_eq).abs() > 1.0e-5
            || (sh.z_pos - phi_eq).abs() > 1.0e-5
            || (sh.z_neg - phi_eq).abs() > 1.0e-5
        {
            return None;
        }
        sv.adaa_pos.sync_x(u_eq);
        sv.adaa_neg.sync_x(u_eq);
        sh.adaa_pos.sync_x(u_eq);
        sh.adaa_neg.sync_x(u_eq);
        let (zv, zh) = (sv.z_pos, sh.z_pos);
        Some(self.eval_symp_linear_gains(key_idx, r_dv, r_dh, y_clear_m, phi_eq, zv, zh))
    }

    /// Advances quiescent felt relaxation `z` over `sub` samples at midpoint clearance `yc_mid <= -5e-5`
    /// and returns exact linearized V/H velocity gains `(gv, gh)` for chunk-SIMD sympathetic damping.
    #[inline]
    pub fn symp_linear_gains_for_chunk(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        yc_mid: f32,
        sub: usize,
    ) -> Option<(f32, f32)> {
        if key_idx >= DAMPED_KEYS
            || str_idx >= TOTAL_STRINGS
            || !yc_mid.is_finite()
            || yc_mid > -5.0e-5
            || !r_dv.is_finite()
            || !r_dh.is_finite()
        {
            return None;
        }
        let u_eq = -yc_mid * 1000.0_f32;
        let phi_eq = self.lut.eval_f(u_eq);
        let sv_idx = str_idx * 2;
        let mut a_pow = 1.0_f32;
        let mut mul = self.a_damp_felt;
        let mut n = sub;
        while n > 0 {
            if (n & 1) != 0 {
                a_pow *= mul;
            }
            mul *= mul;
            n >>= 1;
        }
        let one_m_ap = 1.0 - a_pow;
        let [sv, sh] = &mut self.slots[sv_idx..sv_idx + 2] else {
            return None;
        };
        let zv_old = sv.z_pos;
        let zv_new = a_pow * zv_old + one_m_ap * phi_eq;
        sv.z_pos = zv_new;
        sv.z_neg = zv_new;
        sv.adaa_pos.sync_x(u_eq);
        sv.adaa_neg.sync_x(u_eq);

        let zh_old = sh.z_pos;
        let zh_new = a_pow * zh_old + one_m_ap * phi_eq;
        sh.z_pos = zh_new;
        sh.z_neg = zh_new;
        sh.adaa_pos.sync_x(u_eq);
        sh.adaa_neg.sync_x(u_eq);

        Some(self.eval_symp_linear_gains(
            key_idx,
            r_dv,
            r_dh,
            yc_mid,
            phi_eq,
            0.5 * (zv_old + zv_new),
            0.5 * (zh_old + zh_new),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn eval_symp_linear_gains(
        &self,
        key_idx: usize,
        r_dv: f32,
        r_dh: f32,
        y_clear_m: f32,
        phi_eq: f32,
        zv: f32,
        zh: f32,
    ) -> (f32, f32) {
        let s_sum = -2.0 * y_clear_m;
        let engage = s_sum / (s_sum + ENGAGE_HALF_M);
        let c_nom_e = self.c_nom_key[key_idx] * engage;
        let f_scale = self.f_scale_key[key_idx];
        let hv = 2.0 * (phi_eq - EPS_DAMP_HYST * zv).max(0.0);
        let hh = 2.0 * (phi_eq - EPS_DAMP_HYST * zh).max(0.0);
        let cv = c_nom_e * (1.0 + HYST_DAMP_GAIN * hv) + f_scale * hv;
        let ch = c_nom_e * (1.0 + HYST_DAMP_GAIN * hh) + f_scale * hh;
        let srv = r_dv.max(0.0);
        let srh = r_dh.max(0.0);
        let cev = if cv * srv > MAX_PASSIVE_PORT_RATIO && srv > 0.0 {
            MAX_PASSIVE_PORT_RATIO / srv
        } else {
            cv
        };
        let ceh = if ch * srh > MAX_PASSIVE_PORT_RATIO && srh > 0.0 {
            MAX_PASSIVE_PORT_RATIO / srh
        } else {
            ch
        };
        (
            -cev / (1.0 + 0.5 * srv * cev),
            -ceh / (1.0 + 0.5 * srh * ceh),
        )
    }
}

#[cfg(test)]
#[path = "damper_pad_tests.rs"]
mod tests;
