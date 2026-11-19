//! 464-padded (`116 x 4`) 64-byte-aligned SoA storage and `F32x4` SIMD kernels
//! for [`DamperPadBank`](crate::damper_pad::DamperPadBank).

use piano_simd::{AdaaPowerLut, AlignedBuf, F32x4};
use piano_spec::{CHOIR_LAYOUTS, DAMPED_KEYS};

use crate::damper_pad::{
    DamperPadBank, DAMPER_PAD_SLOTS, ENGAGE_ONSET_M, EPS_DAMP_HYST, HYST_DAMP_GAIN,
    INV_ENGAGE_HALF_M, MAX_PASSIVE_PORT_RATIO, MIN_CLOSED_LINEAR_CLEAR_M,
};
use crate::damper_pad_symp::{passive_damp_force, passive_damp_gain, smooth_felt_engage};

/// Padded damper slot count (`116 * 4 = 464`, exact multiple of 16 for 64-byte cache lines).
pub const DAMPER_PAD_PADDED: usize = 464;
/// Number of 4-lane [`F32x4`] vectors covering all padded damper slots (`464 / 4 = 116`).
pub const DAMPER_PAD_LANES: usize = DAMPER_PAD_PADDED / F32x4::LANES;

/// 64-byte-aligned Structure-of-Arrays (SoA) state for all 460 V/H damper slots (padded to 464).
#[derive(Clone, Debug)]
pub(crate) struct DamperPadSoa {
    pub(crate) c_nom: AlignedBuf,
    pub(crate) f_scale: AlignedBuf,
    pub(crate) yc_cached: AlignedBuf,
    pub(crate) engage_cached: AlignedBuf,
    pub(crate) g_eff_cached: AlignedBuf,
    pub(crate) c_hyst_cached: AlignedBuf,
    pub(crate) f_bias_cached: AlignedBuf,
    pub(crate) adaa_pos_x: AlignedBuf,
    pub(crate) adaa_neg_x: AlignedBuf,
    pub(crate) z_pos: AlignedBuf,
    pub(crate) z_neg: AlignedBuf,
    pub(crate) r_cached: AlignedBuf,
    pub(crate) mode_linear: AlignedBuf,
    pub(crate) active_mask: AlignedBuf,
    pub(crate) decim_phase: [u8; DAMPER_PAD_PADDED],
}

impl DamperPadSoa {
    pub(crate) fn new() -> Self {
        let mut active_mask = AlignedBuf::new(DAMPER_PAD_PADDED);
        let damped_slots = CHOIR_LAYOUTS[DAMPED_KEYS]
            .first_string_idx
            .saturating_mul(2) as usize;
        active_mask.as_mut_slice()[..damped_slots.min(DAMPER_PAD_SLOTS)].fill(1.0);

        Self {
            c_nom: AlignedBuf::new(DAMPER_PAD_PADDED),
            f_scale: AlignedBuf::new(DAMPER_PAD_PADDED),
            yc_cached: AlignedBuf::new(DAMPER_PAD_PADDED),
            engage_cached: AlignedBuf::new(DAMPER_PAD_PADDED),
            g_eff_cached: AlignedBuf::new(DAMPER_PAD_PADDED),
            c_hyst_cached: AlignedBuf::new(DAMPER_PAD_PADDED),
            f_bias_cached: AlignedBuf::new(DAMPER_PAD_PADDED),
            adaa_pos_x: AlignedBuf::new(DAMPER_PAD_PADDED),
            adaa_neg_x: AlignedBuf::new(DAMPER_PAD_PADDED),
            z_pos: AlignedBuf::new(DAMPER_PAD_PADDED),
            z_neg: AlignedBuf::new(DAMPER_PAD_PADDED),
            r_cached: AlignedBuf::new(DAMPER_PAD_PADDED),
            mode_linear: AlignedBuf::new(DAMPER_PAD_PADDED),
            active_mask,
            decim_phase: [0; DAMPER_PAD_PADDED],
        }
    }

    pub(crate) fn reset_all(&mut self, u_eq: f32, phi_eq: f32, yc_rest: f32) {
        self.adaa_pos_x.as_mut_slice()[..DAMPER_PAD_SLOTS].fill(u_eq);
        self.adaa_neg_x.as_mut_slice()[..DAMPER_PAD_SLOTS].fill(u_eq);
        self.z_pos.as_mut_slice()[..DAMPER_PAD_SLOTS].fill(phi_eq);
        self.z_neg.as_mut_slice()[..DAMPER_PAD_SLOTS].fill(phi_eq);
        self.yc_cached.as_mut_slice()[..DAMPER_PAD_SLOTS].fill(yc_rest);
        self.engage_cached.clear();
        self.g_eff_cached.clear();
        self.f_bias_cached.clear();
        self.r_cached.clear();
        self.mode_linear.clear();
        self.c_hyst_cached.as_mut_slice()[..DAMPER_PAD_SLOTS]
            .copy_from_slice(&self.c_nom.as_slice()[..DAMPER_PAD_SLOTS]);
        self.decim_phase.fill(0);
    }

    #[inline]
    pub(crate) fn reset_slot(&mut self, slot: usize, u_eq: f32, phi_eq: f32, yc_rest: f32) {
        self.adaa_pos_x.as_mut_slice()[slot] = u_eq;
        self.adaa_neg_x.as_mut_slice()[slot] = u_eq;
        self.z_pos.as_mut_slice()[slot] = phi_eq;
        self.z_neg.as_mut_slice()[slot] = phi_eq;
        self.yc_cached.as_mut_slice()[slot] = yc_rest;
        self.engage_cached.as_mut_slice()[slot] = 0.0;
        self.g_eff_cached.as_mut_slice()[slot] = 0.0;
        self.f_bias_cached.as_mut_slice()[slot] = 0.0;
        self.c_hyst_cached.as_mut_slice()[slot] = self.c_nom.as_slice()[slot];
        self.mode_linear.as_mut_slice()[slot] = 0.0;
        self.decim_phase[slot] = 0;
    }

    #[inline]
    pub(crate) fn set_key_coeffs(&mut self, key_idx: usize, c_nom: f32, f_scale: f32) {
        if key_idx >= DAMPED_KEYS {
            return;
        }
        let layout = CHOIR_LAYOUTS[key_idx];
        let start = usize::from(layout.first_string_idx) * 2;
        let end = (start + usize::from(layout.strings_per_key) * 2).min(DAMPER_PAD_SLOTS);
        self.c_nom.as_mut_slice()[start..end].fill(c_nom);
        self.f_scale.as_mut_slice()[start..end].fill(f_scale);
        for slot in start..end {
            if self.engage_cached.as_slice()[slot] == 0.0 {
                self.c_hyst_cached.as_mut_slice()[slot] = c_nom;
            }
        }
    }
}

/// Branchless 4-lane SIMD `C^2`-continuous Hunt-Crossley felt engagement curve.
#[inline(always)]
pub(crate) fn smooth_felt_engage_x4(s_sum: F32x4) -> F32x4 {
    let zero = F32x4::zero();
    let s_eff = (s_sum - F32x4::splat(ENGAGE_ONSET_M)).max(zero);
    let u = (s_eff * F32x4::splat(INV_ENGAGE_HALF_M)).min(F32x4::splat(1.0e4));
    let u3 = u * u * u;
    u3 / (F32x4::splat(1.0) + u3)
}

/// Branchless 4-lane SIMD passive port-limited damping conductance (`<= 0`, `N/(m/s)`).
#[inline(always)]
pub(crate) fn passive_damp_gain_x4(c_raw: F32x4, safe_r: F32x4) -> F32x4 {
    let r_pos = safe_r.max(F32x4::splat(1.0e-12));
    let c_cap = F32x4::splat(MAX_PASSIVE_PORT_RATIO) / r_pos;
    let c_eff = c_raw.min(c_cap);
    let denom = F32x4::splat(0.5).mul_add(safe_r * c_eff, F32x4::splat(1.0));
    (F32x4::zero() - c_eff) / denom
}

#[inline]
fn eval_adaa_lane4(lut: &AdaaPowerLut, u: F32x4, prev_buf: &mut AlignedBuf, lane: usize) -> F32x4 {
    let ua = u.to_array();
    let pa = prev_buf.load_f32x4(lane).to_array();
    let out = [
        if ua[0] == 0.0 && pa[0] == 0.0 {
            0.0
        } else {
            lut.adaa1(ua[0], pa[0])
        },
        if ua[1] == 0.0 && pa[1] == 0.0 {
            0.0
        } else {
            lut.adaa1(ua[1], pa[1])
        },
        if ua[2] == 0.0 && pa[2] == 0.0 {
            0.0
        } else {
            lut.adaa1(ua[2], pa[2])
        },
        if ua[3] == 0.0 && pa[3] == 0.0 {
            0.0
        } else {
            lut.adaa1(ua[3], pa[3])
        },
    ];
    prev_buf.store_f32x4(lane, u);
    F32x4::from_array(out)
}

impl DamperPadBank {
    /// Control-rate `F32x4` SIMD update for 4 contiguous damper slots at `lane` (`0..116`).
    ///
    /// Evaluates geometric clearance, ADAA felt hysteresis, and port-limited conductance,
    /// caching `(yc_cached, engage_cached, c_hyst_cached, g_eff_cached)` for intermediate substeps.
    #[inline]
    pub(crate) fn step_damper_control_lane4(
        &mut self,
        lane: usize,
        y_damp: F32x4,
        r_port: F32x4,
        y_clear: F32x4,
    ) -> F32x4 {
        let zero = F32x4::zero();
        let delta_pos = (y_damp - y_clear).max(zero);
        let delta_neg = (zero - y_damp - y_clear).max(zero);
        let s_sum = delta_pos + delta_neg;
        let inv_scale = F32x4::splat(1000.0);
        let u_pos = delta_pos * inv_scale;
        let u_neg = delta_neg * inv_scale;

        let phi_pos = eval_adaa_lane4(&self.lut, u_pos, &mut self.soa.adaa_pos_x, lane);
        let phi_neg = eval_adaa_lane4(&self.lut, u_neg, &mut self.soa.adaa_neg_x, lane);

        let a_v = F32x4::splat(self.a_damp_felt_decim);
        let one_m_a = F32x4::splat(1.0 - self.a_damp_felt_decim);
        let zp = a_v.mul_add(self.soa.z_pos.load_f32x4(lane), one_m_a * phi_pos);
        let zn = a_v.mul_add(self.soa.z_neg.load_f32x4(lane), one_m_a * phi_neg);
        self.soa.z_pos.store_f32x4(lane, zp);
        self.soa.z_neg.store_f32x4(lane, zn);

        let eps = F32x4::splat(EPS_DAMP_HYST);
        let felt_hyst = (phi_pos - eps * zp).max(zero) + (phi_neg - eps * zn).max(zero);
        let c_nom = self.soa.c_nom.load_f32x4(lane);
        let f_scale = self.soa.f_scale.load_f32x4(lane);
        let c_hyst = c_nom * (F32x4::splat(1.0) + F32x4::splat(HYST_DAMP_GAIN) * felt_hyst)
            + f_scale * felt_hyst;

        let engage = smooth_felt_engage_x4(s_sum);
        let safe_r = r_port.max(zero);
        let g_eff = passive_damp_gain_x4(engage * c_hyst, safe_r);

        self.soa.yc_cached.store_f32x4(lane, y_clear);
        self.soa.engage_cached.store_f32x4(lane, engage);
        self.soa.c_hyst_cached.store_f32x4(lane, c_hyst);
        self.soa.g_eff_cached.store_f32x4(lane, g_eff);
        self.soa.r_cached.store_f32x4(lane, safe_r);
        self.soa.f_bias_cached.store_f32x4(lane, zero);
        self.soa.mode_linear.store_f32x4(lane, zero);

        g_eff
    }

    /// Fast substep `F32x4` SIMD evaluation preserving sample-accurate FTO #3 unilateral
    /// contact `delta = max(|y_s| - yc_cached, 0.0)` against cached control-rate state.
    #[inline]
    pub(crate) fn step_damper_substep_lane4(
        &self,
        lane: usize,
        y_damp: F32x4,
        v_damp: F32x4,
        r_port: F32x4,
    ) -> F32x4 {
        let zero = F32x4::zero();
        let yc = self.soa.yc_cached.load_f32x4(lane);
        let c_hyst = self.soa.c_hyst_cached.load_f32x4(lane);
        let mask = self.soa.active_mask.load_f32x4(lane);
        let delta_pos = (y_damp - yc).max(zero);
        let delta_neg = (zero - y_damp - yc).max(zero);
        let s_sum = delta_pos + delta_neg;
        let engage = smooth_felt_engage_x4(s_sum);
        let g_eff = passive_damp_gain_x4(engage * c_hyst, r_port.max(zero));
        g_eff * v_damp * mask
    }

    /// Evaluates all active 4-slot lanes (`116` `F32x4` vectors = `464` padded slots) for one
    /// host sample using decimated control-rate updates at `sample_idx % control_decim == 0`
    /// and sample-accurate unilateral contact on intermediate substeps.
    pub fn step_damper_batch_simd(
        &mut self,
        y_damp_464: &[f32],
        v_damp_464: &[f32],
        r_port_464: &[f32],
        y_clear_464: &[f32],
        f_out_464: &mut [f32],
        sample_idx: usize,
    ) {
        let n_lanes = DAMPER_PAD_LANES
            .min(y_damp_464.len() / F32x4::LANES)
            .min(v_damp_464.len() / F32x4::LANES)
            .min(r_port_464.len() / F32x4::LANES)
            .min(y_clear_464.len() / F32x4::LANES)
            .min(f_out_464.len() / F32x4::LANES);

        let decim = usize::from(self.control_decim.max(1));
        let is_control_step = decim <= 1 || sample_idx.is_multiple_of(decim);

        for lane in 0..n_lanes {
            let mask = self.soa.active_mask.load_f32x4(lane);
            let m_arr = mask.to_array();
            let off = lane * F32x4::LANES;
            if m_arr[0] == 0.0 && m_arr[1] == 0.0 && m_arr[2] == 0.0 && m_arr[3] == 0.0 {
                F32x4::zero().write_to_slice(&mut f_out_464[off..off + 4]);
                continue;
            }
            let yd = F32x4::from_slice(&y_damp_464[off..off + 4]);
            let vd = F32x4::from_slice(&v_damp_464[off..off + 4]);
            let rp = F32x4::from_slice(&r_port_464[off..off + 4]);

            let f_out = if is_control_step {
                let yc = F32x4::from_slice(&y_clear_464[off..off + 4]);
                let g_eff = self.step_damper_control_lane4(lane, yd, rp, yc);
                g_eff * vd * mask
            } else {
                self.step_damper_substep_lane4(lane, yd, vd, rp)
            };
            f_out.write_to_slice(&mut f_out_464[off..off + 4]);
        }
    }

    /// Enables or disables a slot (`0..460`) in the [`Self::step_damper_batch_simd`] mask.
    #[inline]
    pub fn set_slot_active_mask(&mut self, slot_idx: usize, active: bool) {
        if slot_idx < DAMPER_PAD_SLOTS {
            self.soa.active_mask.as_mut_slice()[slot_idx] = if active { 1.0 } else { 0.0 };
        }
    }

    #[inline]
    pub(crate) fn step_damper_sample_decim(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        let phase = self.soa.decim_phase[slot_idx];
        let next = phase + 1;
        self.soa.decim_phase[slot_idx] = if next >= self.control_decim { 0 } else { next };
        let safe_r = r_port_damp.max(0.0);

        if phase > 0 {
            if y_damp_m == 0.0 && v_damp_mps == 0.0 {
                return 0.0;
            }
            let yc = self.soa.yc_cached[slot_idx];
            if self.soa.mode_linear[slot_idx] > 0.5 && self.soa.r_cached[slot_idx] == safe_r {
                let lin_lim = (-0.85 * yc).min(8.0e-6);
                if y_damp_m.abs() <= lin_lim {
                    return self.soa.g_eff_cached[slot_idx] * v_damp_mps;
                }
            }
            let s_sum = (y_damp_m - yc).max(0.0) + (-y_damp_m - yc).max(0.0);
            if s_sum <= ENGAGE_ONSET_M {
                return 0.0;
            }
            let engage = smooth_felt_engage(s_sum);
            let c_raw = engage * self.soa.c_hyst_cached[slot_idx];
            return passive_damp_force(c_raw, safe_r, v_damp_mps);
        }

        let inv_scale = 1000.0_f32;
        let a = self.a_damp_felt_decim;
        let c_nom = self.c_nom_key[key_idx];
        let f_scale = self.f_scale_key[key_idx];
        if y_damp_m == 0.0 && v_damp_mps == 0.0 {
            let u_eq = (-y_clear_m).max(0.0) * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            self.soa.adaa_pos_x[slot_idx] = u_eq;
            self.soa.adaa_neg_x[slot_idx] = u_eq;
            self.soa.z_pos[slot_idx] = a * self.soa.z_pos[slot_idx] + (1.0 - a) * phi_eq;
            self.soa.z_neg[slot_idx] = a * self.soa.z_neg[slot_idx] + (1.0 - a) * phi_eq;
            self.soa.yc_cached[slot_idx] = y_clear_m;
            self.soa.engage_cached[slot_idx] = 0.0;
            self.soa.c_hyst_cached[slot_idx] = c_nom;
            self.soa.g_eff_cached[slot_idx] = 0.0;
            self.soa.r_cached[slot_idx] = safe_r;
            self.soa.mode_linear[slot_idx] = 0.0;
            return 0.0;
        }

        let lin_lim = (-0.85 * y_clear_m).min(8.0e-6);
        if y_clear_m <= MIN_CLOSED_LINEAR_CLEAR_M && y_damp_m.abs() <= lin_lim {
            let u_eq = -y_clear_m * inv_scale;
            let phi_eq = self.lut.eval_f(u_eq);
            self.soa.adaa_pos_x[slot_idx] = u_eq + y_damp_m * inv_scale;
            self.soa.adaa_neg_x[slot_idx] = u_eq - y_damp_m * inv_scale;
            let z_next = a * self.soa.z_pos[slot_idx] + (1.0 - a) * phi_eq;
            self.soa.z_pos[slot_idx] = z_next;
            self.soa.z_neg[slot_idx] = z_next;
            let engage = smooth_felt_engage(-2.0 * y_clear_m);
            let h = 2.0 * (phi_eq - EPS_DAMP_HYST * z_next).max(0.0);
            let c_hyst = c_nom * (1.0 + HYST_DAMP_GAIN * h) + f_scale * h;
            let g_eff = passive_damp_gain(engage * c_hyst, safe_r);
            self.soa.yc_cached[slot_idx] = y_clear_m;
            self.soa.engage_cached[slot_idx] = engage;
            self.soa.c_hyst_cached[slot_idx] = c_hyst;
            self.soa.g_eff_cached[slot_idx] = g_eff;
            self.soa.r_cached[slot_idx] = safe_r;
            self.soa.mode_linear[slot_idx] = 1.0;
            return g_eff * v_damp_mps;
        }

        let delta_pos = (y_damp_m - y_clear_m).max(0.0);
        let delta_neg = (-y_damp_m - y_clear_m).max(0.0);
        let s_sum = delta_pos + delta_neg;
        let u_pos = delta_pos * inv_scale;
        let u_neg = delta_neg * inv_scale;
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

        let zp_next = a * self.soa.z_pos[slot_idx] + (1.0 - a) * phi_pos;
        let zn_next = a * self.soa.z_neg[slot_idx] + (1.0 - a) * phi_neg;
        self.soa.z_pos[slot_idx] = zp_next;
        self.soa.z_neg[slot_idx] = zn_next;

        let felt_hyst = (phi_pos - EPS_DAMP_HYST * zp_next).max(0.0)
            + (phi_neg - EPS_DAMP_HYST * zn_next).max(0.0);
        let c_hyst = c_nom * (1.0 + HYST_DAMP_GAIN * felt_hyst) + f_scale * felt_hyst;
        let engage = smooth_felt_engage(s_sum);
        let g_eff = passive_damp_gain(engage * c_hyst, safe_r);
        self.soa.yc_cached[slot_idx] = y_clear_m;
        self.soa.engage_cached[slot_idx] = engage;
        self.soa.c_hyst_cached[slot_idx] = c_hyst;
        self.soa.g_eff_cached[slot_idx] = g_eff;
        self.soa.r_cached[slot_idx] = safe_r;
        self.soa.mode_linear[slot_idx] = 0.0;
        if engage <= 0.0 {
            0.0
        } else {
            g_eff * v_damp_mps
        }
    }
}
