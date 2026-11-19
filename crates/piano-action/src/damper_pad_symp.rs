//! Sympathetic linear-equilibrium gains, open-gap relaxation, and Hunt-Crossley
//! engagement helpers for [`DamperPadBank`](crate::damper_pad::DamperPadBank).

use piano_spec::{DAMPED_KEYS, TOTAL_STRINGS};

use crate::damper_pad::{
    DamperPadBank, DAMPER_PAD_SLOTS, ENGAGE_ONSET_M, EPS_DAMP_HYST, HYST_DAMP_GAIN,
    INV_ENGAGE_HALF_M, MAX_PASSIVE_PORT_RATIO, MIN_CLOSED_LINEAR_CLEAR_M,
};
use crate::pedals::CLEARANCE_OPEN_M;

/// Smooth `C^2`-continuous Hunt-Crossley wool-fiber engagement curve (`g = g' = g'' = 0` at `3 um`),
/// preventing `1x` host-rate corner buzzing during half-pedal grazing contact.
#[inline(always)]
pub(crate) fn smooth_felt_engage(s_sum: f32) -> f32 {
    let s_eff = s_sum - ENGAGE_ONSET_M;
    if s_eff <= 0.0 {
        return 0.0;
    }
    let u = s_eff * INV_ENGAGE_HALF_M;
    if u >= 1.0e4 {
        return 1.0;
    }
    let u3 = u * u * u;
    u3 / (1.0 + u3)
}

#[inline(always)]
pub(crate) fn pow_usize(mut base: f32, mut exp: usize) -> f32 {
    let mut acc = 1.0_f32;
    while exp > 0 {
        if (exp & 1) != 0 {
            acc *= base;
        }
        base *= base;
        exp >>= 1;
    }
    acc
}

#[inline(always)]
pub(crate) fn passive_damp_force(c_raw: f32, safe_r: f32, v_mps: f32) -> f32 {
    let c_eff = if c_raw * safe_r > MAX_PASSIVE_PORT_RATIO && safe_r > 0.0 {
        MAX_PASSIVE_PORT_RATIO / safe_r
    } else {
        c_raw
    };
    (-c_eff * v_mps) / (1.0 + 0.5 * safe_r * c_eff)
}

#[inline(always)]
pub(crate) fn passive_damp_gain(c_raw: f32, safe_r: f32) -> f32 {
    let c_eff = if c_raw * safe_r > MAX_PASSIVE_PORT_RATIO && safe_r > 0.0 {
        MAX_PASSIVE_PORT_RATIO / safe_r
    } else {
        c_raw
    };
    -c_eff / (1.0 + 0.5 * safe_r * c_eff)
}

impl DamperPadBank {
    /// Relaxes damper felt hysteresis states `(z_pos, z_neg)` across `sub` contact-free samples
    /// when positive damper clearance (`y_clear > 0`) exceeds string excursion.
    #[inline]
    pub fn relax_open_gap_chunk(&mut self, str_idx: usize, sub: usize) {
        let sv_idx = str_idx * 2;
        if sv_idx + 2 > DAMPER_PAD_SLOTS {
            return;
        }
        let sh_idx = sv_idx + 1;
        self.soa.adaa_pos_x[sv_idx] = 0.0;
        self.soa.adaa_neg_x[sv_idx] = 0.0;
        self.soa.adaa_pos_x[sh_idx] = 0.0;
        self.soa.adaa_neg_x[sh_idx] = 0.0;
        self.soa.yc_cached[sv_idx] = CLEARANCE_OPEN_M;
        self.soa.yc_cached[sh_idx] = CLEARANCE_OPEN_M;
        self.soa.mode_linear[sv_idx] = 0.0;
        self.soa.mode_linear[sh_idx] = 0.0;
        self.soa.decim_phase[sv_idx] = 0;
        self.soa.decim_phase[sh_idx] = 0;
        let zv_p = self.soa.z_pos[sv_idx];
        let zv_n = self.soa.z_neg[sv_idx];
        let zh_p = self.soa.z_pos[sh_idx];
        let zh_n = self.soa.z_neg[sh_idx];
        if zv_p != 0.0 || zv_n != 0.0 || zh_p != 0.0 || zh_n != 0.0 {
            let a_pow = pow_usize(self.a_damp_felt, sub);
            let flush = |z: f32| if z.abs() < 1.0e-20 { 0.0 } else { z };
            self.soa.z_pos[sv_idx] = flush(zv_p * a_pow);
            self.soa.z_neg[sv_idx] = flush(zv_n * a_pow);
            self.soa.z_pos[sh_idx] = flush(zh_p * a_pow);
            self.soa.z_neg[sh_idx] = flush(zh_n * a_pow);
        }
    }

    /// Returns pre-evaluated linear equilibrium V/H damper force-per-velocity gains `(gv, gh)`
    /// for `(key_idx, str_idx)` when `y_clear_m <= -2e-7` and felt hysteresis state `z` has
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
            || y_clear_m > MIN_CLOSED_LINEAR_CLEAR_M
            || !r_dv.is_finite()
            || !r_dh.is_finite()
        {
            return None;
        }
        let u_eq = -y_clear_m * 1000.0_f32;
        let phi_eq = self.lut.eval_f(u_eq);
        let sv_idx = str_idx * 2;
        let sh_idx = sv_idx + 1;
        let zv = self.soa.z_pos[sv_idx];
        let zh = self.soa.z_pos[sh_idx];
        if (zv - phi_eq).abs() > 1.0e-5
            || (self.soa.z_neg[sv_idx] - phi_eq).abs() > 1.0e-5
            || (zh - phi_eq).abs() > 1.0e-5
            || (self.soa.z_neg[sh_idx] - phi_eq).abs() > 1.0e-5
        {
            return None;
        }
        self.soa.adaa_pos_x[sv_idx] = u_eq;
        self.soa.adaa_neg_x[sv_idx] = u_eq;
        self.soa.adaa_pos_x[sh_idx] = u_eq;
        self.soa.adaa_neg_x[sh_idx] = u_eq;
        Some(self.eval_symp_linear_gains(key_idx, r_dv, r_dh, y_clear_m, phi_eq, zv, zh))
    }

    /// Advances quiescent felt relaxation `z` over `sub` samples across `[yc0, yc_end]`
    /// (`yc0.min(yc_end) <= -2e-7`) and returns Simpson-averaged passive V/H velocity gains.
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
        if key_idx >= DAMPED_KEYS
            || str_idx >= TOTAL_STRINGS
            || !yc0.is_finite()
            || !yc_end.is_finite()
            || yc0.min(yc_end) > MIN_CLOSED_LINEAR_CLEAR_M
            || !r_dv.is_finite()
            || !r_dh.is_finite()
        {
            return None;
        }
        let yc_mid = 0.5 * (yc0 + yc_end);
        let (c0, cm, c1) = (yc0.min(0.0), yc_mid.min(0.0), yc_end.min(0.0));
        let (p0, pm, p1) = (
            self.lut.eval_f(-c0 * 1000.0),
            self.lut.eval_f(-cm * 1000.0),
            self.lut.eval_f(-c1 * 1000.0),
        );
        let phi_avg = (p0 + 4.0 * pm + p1) * (1.0 / 6.0);
        let a_pow = pow_usize(self.a_damp_felt, sub);
        let one_m_ap = 1.0 - a_pow;
        let sv_idx = str_idx * 2;
        let sh_idx = sv_idx + 1;
        let (zv0, zh0) = (self.soa.z_pos[sv_idx], self.soa.z_pos[sh_idx]);
        let (zv1, zh1) = (
            a_pow * zv0 + one_m_ap * phi_avg,
            a_pow * zh0 + one_m_ap * phi_avg,
        );
        self.soa.reset_slot(sv_idx, -c1 * 1000.0, zv1, c1);
        self.soa.reset_slot(sh_idx, -c1 * 1000.0, zh1, c1);
        let (gv0, gh0) = self.eval_symp_linear_gains(key_idx, r_dv, r_dh, c0, p0, zv0, zh0);
        let (gvm, ghm) = self.eval_symp_linear_gains(
            key_idx,
            r_dv,
            r_dh,
            cm,
            pm,
            0.5 * (zv0 + zv1),
            0.5 * (zh0 + zh1),
        );
        let (gv1, gh1) = self.eval_symp_linear_gains(key_idx, r_dv, r_dh, c1, p1, zv1, zh1);
        Some((
            (gv0 + 4.0 * gvm + gv1) * (1.0 / 6.0),
            (gh0 + 4.0 * ghm + gh1) * (1.0 / 6.0),
        ))
    }

    #[inline(always)]
    pub(crate) fn eval_preload_c_vh(
        &self,
        key_idx: usize,
        y_clear_m: f32,
        phi_eq: f32,
        zv: f32,
        zh: f32,
    ) -> (f32, f32) {
        let engage = smooth_felt_engage(-2.0 * y_clear_m);
        let (c_nom, f_scale) = (self.c_nom_key[key_idx], self.f_scale_key[key_idx]);
        let hv = 2.0 * (phi_eq - EPS_DAMP_HYST * zv).max(0.0);
        let hh = 2.0 * (phi_eq - EPS_DAMP_HYST * zh).max(0.0);
        (
            engage * (c_nom * (1.0 + HYST_DAMP_GAIN * hv) + f_scale * hv),
            engage * (c_nom * (1.0 + HYST_DAMP_GAIN * hh) + f_scale * hh),
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub(crate) fn eval_symp_linear_gains(
        &self,
        key_idx: usize,
        r_dv: f32,
        r_dh: f32,
        y_clear_m: f32,
        phi_eq: f32,
        zv: f32,
        zh: f32,
    ) -> (f32, f32) {
        let (cv, ch) = self.eval_preload_c_vh(key_idx, y_clear_m, phi_eq, zv, zh);
        (
            passive_damp_force(cv, r_dv.max(0.0), 1.0),
            passive_damp_force(ch, r_dh.max(0.0), 1.0),
        )
    }
}
