//! Scaled Auxiliary Variable (SAV) unified nonlinear tension + phantom coupling
//! with exact Cayley orthogonal step and Risse-Helie-Bilbao drift regulation (`no_std`).

use alloc::vec::Vec;

use piano_spec::{StringPhysicalParams, VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

use crate::choir_map::{ChoirMap, StringMapEntry};
#[cfg(test)]
use crate::modal_soa::sqrt_f64;
use crate::modal_soa::ModalSoa;
use crate::phantom::{
    convolve_transverse_slopes, MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_MODES,
};
#[allow(unused_imports)]
pub(crate) use crate::sav_kernels::{
    apply_cayley_u, eval_p_vec, eval_p_vec_slice, load_nl_state_n, load_vh_state, prepare_vh,
    step_sav_no_phantom_spec, step_sav_phantom_spec_n, store_nl_state_n, store_vh_u,
};
use crate::sav_spec::{SavEnergySlopes, SavStringSpec};

pub(crate) const C0_SAV: f64 = 1.0;
pub(crate) const LAMBDA_REG_DEFAULT: f64 = 400.0;
pub(crate) const DELTA_U_SAV: f64 = 1.0e-10;

/// Risse-Helie-Bilbao drift-regulated SAV solver across all 230 strings (`0` runtime allocs).
#[derive(Clone, Debug)]
pub(crate) struct SavBank {
    pub(crate) specs: Vec<SavStringSpec>,
    pub(crate) slopes: Vec<SavEnergySlopes>,
    pub r_sav: [f64; TOTAL_STRINGS],
    ts: f64,
}

impl SavBank {
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        gamma_s: &[f64; TOTAL_STRINGS],
        beta_s: &[f64; TOTAL_STRINGS],
    ) -> Self {
        let mut str_specs = Vec::with_capacity(map.strings().len());
        let mut str_slopes = Vec::with_capacity(map.strings().len());
        for (s, e) in map.strings().iter().enumerate() {
            let (sp, sl) =
                SavStringSpec::build(e, sample_rate, voicing, specs, gamma_s[s], beta_s[s]);
            str_specs.push(sp);
            str_slopes.push(sl);
        }
        Self {
            specs: str_specs,
            slopes: str_slopes,
            r_sav: [1.0; TOTAL_STRINGS],
            ts: 1.0 / sample_rate,
        }
    }

    #[inline(always)]
    pub fn spec(&self, s_idx: usize) -> &SavStringSpec {
        &self.specs[s_idx]
    }

    #[inline(always)]
    pub fn ts(&self) -> f64 {
        self.ts
    }

    #[inline(always)]
    pub fn string_n_vh(&self, s_idx: usize) -> usize {
        self.specs[s_idx].n_vh
    }

    #[inline(always)]
    pub fn string_prefix_bounds(
        &self,
        s_idx: usize,
        longitudinal_enabled: bool,
    ) -> (usize, usize, usize, usize) {
        let sp = &self.specs[s_idx];
        let m_ph = if longitudinal_enabled { sp.m_ph } else { 0 };
        let nl_vecs = if m_ph > 0 { sp.nl_vecs } else { 0 };
        (sp.nv_vecs, sp.nh_vecs, nl_vecs, m_ph)
    }

    #[inline]
    pub fn reset_all(&mut self) {
        self.r_sav.fill(1.0);
    }

    #[inline]
    pub fn reset_string(&mut self, s_idx: usize) {
        self.r_sav[s_idx] = 1.0;
    }

    /// Evaluates scaled nonlinear potential `E_nl` and returns physical `beta_s * E_nl` (Joules).
    #[must_use]
    pub fn string_nonlinear_energy_joules(
        &self,
        s_idx: usize,
        entry: &StringMapEntry,
        soa: &ModalSoa,
        beta_s: f64,
        longitudinal_enabled: bool,
    ) -> f64 {
        let sp = &self.specs[s_idx];
        let sl = &self.slopes[s_idx];
        if sp.n_vh == 0 {
            return 0.0;
        }
        let v_sl = soa.v.as_slice();
        let (vb, hb, lb) = (
            entry.vert_vecs.start * 4,
            entry.horiz_vecs.start * 4,
            entry.long_vecs.start * 4,
        );
        let mut d_v = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        let mut d_h = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        for a in 0..sl.nv_c {
            d_v[a] = sl.sigma_v[a] * f64::from(v_sl[vb + a]);
        }
        for a in 0..sl.nh_c {
            d_h[a] = sl.sigma_h[a] * f64::from(v_sl[hb + a]);
        }
        let m_ph = if longitudinal_enabled { sp.m_ph } else { 0 };
        let (s_val, phi_tilde) = convolve_transverse_slopes(sp.length_m, &d_v, &d_h, m_ph);
        let mut sum_res_sq = 0.0_f64;
        for (idx, &phi) in phi_tilde[1..=m_ph].iter().enumerate() {
            let res = phi + sl.alpha_l[idx] * f64::from(v_sl[lb + idx]);
            sum_res_sq += res * res;
        }
        beta_s * 0.5 * (sp.kappa_0 * s_val * s_val + sp.kappa_ph * sum_res_sq)
    }

    #[cfg(test)]
    pub(crate) fn eval_psi(
        &self,
        s_idx: usize,
        entry: &StringMapEntry,
        soa: &ModalSoa,
        longitudinal_enabled: bool,
    ) -> f64 {
        let e_nl =
            self.string_nonlinear_energy_joules(s_idx, entry, soa, 1.0, longitudinal_enabled);
        sqrt_f64(2.0 * e_nl + C0_SAV)
    }

    /// Advances string `s_idx`'s nonlinear state `(u_c, r_sav)` with default `lambda_reg = 400.0`.
    #[inline]
    pub fn step_string_sav(
        &mut self,
        s_idx: usize,
        entry: &StringMapEntry,
        soa: &mut ModalSoa,
        longitudinal_enabled: bool,
    ) {
        self.step_string_sav_with_lambda(
            s_idx,
            entry,
            soa,
            longitudinal_enabled,
            LAMBDA_REG_DEFAULT,
        );
    }

    #[inline]
    pub(crate) fn step_string_sav_with_lambda(
        &mut self,
        s_idx: usize,
        entry: &StringMapEntry,
        soa: &mut ModalSoa,
        longitudinal_enabled: bool,
        lambda_reg: f64,
    ) {
        let sp = &self.specs[s_idx];
        if sp.n_vh == 0 {
            return;
        }
        let (nv, nh) = (sp.nv_vecs, sp.nh_vecs);
        let nl = if longitudinal_enabled && sp.m_ph > 0 {
            sp.nl_vecs
        } else {
            0
        };
        let (vb, hb, lb) = (
            entry.vert_vecs.start,
            entry.horiz_vecs.start,
            entry.long_vecs.start,
        );

        let (mut uv, vv) = load_vh_state(soa, vb, nv);
        let (mut uh, vh) = load_vh_state(soa, hb, nh);

        let mut r_n = self.r_sav[s_idx];
        if nl == 0 {
            step_sav_no_phantom_spec(
                sp, self.ts, &mut r_n, &mut uv, &vv, &mut uh, &vh, lambda_reg,
            );
        } else if nl == 1 {
            let (mut ul, vl) = load_nl_state_n::<1>(soa, lb, 1);
            step_sav_phantom_spec_n::<1>(
                sp, self.ts, 1, &mut r_n, &mut uv, &vv, &mut uh, &vh, &mut ul, &vl, lambda_reg,
            );
            soa.u.as_f32x4_slice_mut()[lb] = ul[0];
        } else {
            let nl = nl.min(MAX_PHANTOM_LONG_VECS);
            let (mut ul, vl) = load_nl_state_n::<MAX_PHANTOM_LONG_VECS>(soa, lb, nl);
            step_sav_phantom_spec_n::<MAX_PHANTOM_LONG_VECS>(
                sp, self.ts, nl, &mut r_n, &mut uv, &vv, &mut uh, &vh, &mut ul, &vl, lambda_reg,
            );
            soa.u.as_f32x4_slice_mut()[lb..lb + nl].copy_from_slice(&ul[..nl]);
        }
        self.r_sav[s_idx] = r_n;
        store_vh_u(soa, vb, nv, &uv);
        store_vh_u(soa, hb, nh, &uh);
    }
}

#[cfg(test)]
#[path = "sav_tests.rs"]
mod tests;
