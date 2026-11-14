//! Scaled Auxiliary Variable (SAV) unified nonlinear tension + phantom coupling
//! with exact Cayley orthogonal step and Risse-Helie-Bilbao drift regulation (`no_std`).

use alloc::vec::Vec;

use piano_simd::F32x4;
use piano_spec::{StringPhysicalParams, VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

use crate::choir_map::{ChoirMap, StringMapEntry};
#[cfg(test)]
use crate::modal_soa::sqrt_f64;
use crate::modal_soa::ModalSoa;
use crate::phantom::{
    accum_phantom_grad_vh_simd_n, convolve_transverse_slopes, convolve_transverse_slopes_f32_n,
    MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_MODES, MAX_PHANTOM_TRANSVERSE_VECS,
};
use crate::sav_spec::{SavEnergySlopes, SavStringSpec};

pub(crate) const C0_SAV: f64 = 1.0;
pub(crate) const LAMBDA_REG_DEFAULT: f64 = 400.0;
pub(crate) const DELTA_U_SAV: f64 = 1.0e-10;

/// Risse-Helie-Bilbao drift-regulated SAV solver across all 230 strings (`0` runtime allocs).
#[derive(Clone, Debug)]
pub(crate) struct SavBank {
    specs: Vec<SavStringSpec>,
    slopes: Vec<SavEnergySlopes>,
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

#[inline(always)]
pub(crate) fn load_vh_state(soa: &ModalSoa, b: usize, len: usize) -> ([F32x4; 2], [F32x4; 2]) {
    let (u_s, v_s) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
    if len >= 2 {
        assert!(b + 1 < u_s.len() && b + 1 < v_s.len());
        ([u_s[b], u_s[b + 1]], [v_s[b], v_s[b + 1]])
    } else if len == 1 {
        ([u_s[b], F32x4::zero()], [v_s[b], F32x4::zero()])
    } else {
        ([F32x4::zero(); 2], [F32x4::zero(); 2])
    }
}

#[inline(always)]
fn store_vh_u(soa: &mut ModalSoa, b: usize, len: usize, u: &[F32x4; 2]) {
    let u_s = soa.u.as_f32x4_slice_mut();
    if len >= 2 {
        assert!(b + 1 < u_s.len());
        u_s[b] = u[0];
        u_s[b + 1] = u[1];
    } else if len == 1 {
        u_s[b] = u[0];
    }
}

#[inline(always)]
pub(crate) fn load_nl_state_n<const N: usize>(
    soa: &ModalSoa,
    b: usize,
    nl: usize,
) -> ([F32x4; N], [F32x4; N]) {
    let (mut u, mut v) = ([F32x4::zero(); N], [F32x4::zero(); N]);
    if N == 1 {
        u[0] = soa.u.as_f32x4_slice()[b];
        v[0] = soa.v.as_f32x4_slice()[b];
    } else {
        let n = nl.min(N);
        u[..n].copy_from_slice(&soa.u.as_f32x4_slice()[b..b + n]);
        v[..n].copy_from_slice(&soa.v.as_f32x4_slice()[b..b + n]);
    }
    (u, v)
}

#[inline(always)]
pub(crate) fn store_nl_state_n<const N: usize>(
    soa: &mut ModalSoa,
    b: usize,
    nl: usize,
    u: &[F32x4; N],
    v: &[F32x4; N],
) {
    if N == 1 {
        soa.u.as_f32x4_slice_mut()[b] = u[0];
        soa.v.as_f32x4_slice_mut()[b] = v[0];
    } else {
        let n = nl.min(N);
        soa.u.as_f32x4_slice_mut()[b..b + n].copy_from_slice(&u[..n]);
        soa.v.as_f32x4_slice_mut()[b..b + n].copy_from_slice(&v[..n]);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn step_sav_no_phantom_spec(
    sp: &SavStringSpec,
    ts: f64,
    r_n_ref: &mut f64,
    uv: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vv: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    uh: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vh: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    lambda_reg: f64,
) {
    let (uv_m, mut dv4, uh_m, mut dh4, u_sq4, d_sq4) = prepare_vh(sp, uv, vv, uh, vh);
    let den_u = f64::from(u_sq4.reduce_sum()) + DELTA_U_SAV;
    let s_val = sp.quarter_l * f64::from(d_sq4.reduce_sum());
    let e_nl = 0.5 * sp.kappa_0 * s_val * s_val;
    let inv_u = 1.0 / den_u;
    let (psi, inv_psi) = psi_and_inv(2.0 * e_nl + C0_SAV);
    let r_n = *r_n_ref;
    let reg = lambda_reg * (r_n - psi) * inv_u;

    let half_ts = 0.5 * ts;
    let kg = F32x4::splat((half_ts * inv_psi * 0.5 * sp.length_m * sp.kappa_0 * s_val) as f32);
    let kr = F32x4::splat((half_ts * reg) as f32);
    let (mut dot4, mut p2_4) = (F32x4::zero(), F32x4::zero());
    eval_p_vec(
        kg,
        kr,
        &sp.omega_sigma_v4,
        &mut dv4,
        &uv_m,
        &mut dot4,
        &mut p2_4,
    );
    eval_p_vec(
        kg,
        kr,
        &sp.omega_sigma_h4,
        &mut dh4,
        &uh_m,
        &mut dot4,
        &mut p2_4,
    );

    let r_bar = (r_n + f64::from(dot4.reduce_sum())) / (1.0 + f64::from(p2_4.reduce_sum()));
    *r_n_ref = 2.0 * r_bar - r_n;
    let trb4 = F32x4::splat((2.0 * r_bar) as f32);
    apply_cayley_u(uv, &dv4, trb4);
    apply_cayley_u(uh, &dh4, trb4);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn step_sav_phantom_spec_n<const NL_MAX: usize>(
    sp: &SavStringSpec,
    ts: f64,
    nl: usize,
    r_n_ref: &mut f64,
    uv: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vv: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    uh: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vh: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    ul: &mut [F32x4; NL_MAX],
    vl: &[F32x4; NL_MAX],
    lambda_reg: f64,
) {
    let n = if NL_MAX == 1 { 1 } else { nl.min(NL_MAX) };
    let (uv_m, dv4, uh_m, dh4, mut u_sq4, d_sq4) = prepare_vh(sp, uv, vv, uh, vh);

    let mut ul_m = [F32x4::zero(); NL_MAX];
    for i in 0..n {
        let um = ul[i] * sp.mask_l4[i];
        ul_m[i] = um;
        u_sq4 = um.mul_add(um, u_sq4);
    }
    let den_u = f64::from(u_sq4.reduce_sum()) + DELTA_U_SAV;
    let inv_u = 1.0 / den_u;
    let phi4 = convolve_transverse_slopes_f32_n::<NL_MAX>(&dv4, &dh4);
    let c_phi_scale4 = F32x4::splat((sp.quarter_l * sp.kappa_ph) as f32);
    let mut r2_4 = F32x4::zero();
    let mut c_phi4 = [F32x4::zero(); NL_MAX];
    for i in 0..n {
        let res = sp.half_ql_mask4[i] * phi4[i] + sp.alpha_l4[i] * vl[i];
        r2_4 = res.mul_add(res, r2_4);
        c_phi4[i] = res * c_phi_scale4;
    }

    let s_val = sp.quarter_l * f64::from(d_sq4.reduce_sum());
    let sum_res_sq = f64::from(r2_4.reduce_sum());
    let e_nl = 0.5 * (sp.kappa_0 * s_val * s_val + sp.kappa_ph * sum_res_sq);
    let (psi, inv_psi) = psi_and_inv(2.0 * e_nl + C0_SAV);
    let r_n = *r_n_ref;
    let reg = lambda_reg * (r_n - psi) * inv_u;

    let c_sum = (0.5 * sp.length_m * sp.kappa_0 * s_val) as f32;
    let (mut gv4, mut gh4) = accum_phantom_grad_vh_simd_n::<NL_MAX>(&dv4, &dh4, c_sum, &c_phi4);

    let half_ts = 0.5 * ts;
    let kg = F32x4::splat((half_ts * inv_psi) as f32);
    let kr = F32x4::splat((half_ts * reg) as f32);
    let (mut dot4, mut p2_4) = (F32x4::zero(), F32x4::zero());
    eval_p_vec(
        kg,
        kr,
        &sp.omega_sigma_v4,
        &mut gv4,
        &uv_m,
        &mut dot4,
        &mut p2_4,
    );
    eval_p_vec(
        kg,
        kr,
        &sp.omega_sigma_h4,
        &mut gh4,
        &uh_m,
        &mut dot4,
        &mut p2_4,
    );
    eval_p_vec_slice(
        kg,
        kr,
        &sp.omega_alpha_l_norm4,
        &mut c_phi4,
        &ul_m,
        &mut dot4,
        &mut p2_4,
    );

    let r_bar = (r_n + f64::from(dot4.reduce_sum())) / (1.0 + f64::from(p2_4.reduce_sum()));
    *r_n_ref = 2.0 * r_bar - r_n;
    let trb4 = F32x4::splat((2.0 * r_bar) as f32);
    apply_cayley_u(uv, &gv4, trb4);
    apply_cayley_u(uh, &gh4, trb4);
    apply_cayley_u(ul, &c_phi4, trb4);
}

pub(crate) type PhVecs = [F32x4; MAX_PHANTOM_TRANSVERSE_VECS];

#[inline(always)]
pub(crate) fn prepare_vh(
    sp: &SavStringSpec,
    uv: &PhVecs,
    vv: &PhVecs,
    uh: &PhVecs,
    vh: &PhVecs,
) -> (PhVecs, PhVecs, PhVecs, PhVecs, F32x4, F32x4) {
    let mut uv_m = [F32x4::zero(); MAX_PHANTOM_TRANSVERSE_VECS];
    let (mut dv4, mut uh_m, mut dh4) = (uv_m, uv_m, uv_m);
    let (mut u_sq4, mut d_sq4) = (F32x4::zero(), F32x4::zero());
    for i in 0..MAX_PHANTOM_TRANSVERSE_VECS {
        let um_v = uv[i] * sp.mask_v4[i];
        let dv = sp.sigma_v4[i] * vv[i];
        uv_m[i] = um_v;
        dv4[i] = dv;
        let um_h = uh[i] * sp.mask_h4[i];
        let dh = sp.sigma_h4[i] * vh[i];
        uh_m[i] = um_h;
        dh4[i] = dh;
        u_sq4 = um_v.mul_add(um_v, um_h.mul_add(um_h, u_sq4));
        d_sq4 = dv.mul_add(dv, dh.mul_add(dh, d_sq4));
    }
    (uv_m, dv4, uh_m, dh4, u_sq4, d_sq4)
}

#[inline(always)]
pub(crate) fn eval_p_vec<const N: usize>(
    kg: F32x4,
    kr: F32x4,
    w: &[F32x4; N],
    g_out: &mut [F32x4; N],
    u_m: &[F32x4; N],
    dot4: &mut F32x4,
    p2_4: &mut F32x4,
) {
    eval_p_vec_slice(kg, kr, w, g_out, u_m, dot4, p2_4);
}

#[inline(always)]
pub(crate) fn eval_p_vec_slice<const N: usize>(
    kg: F32x4,
    kr: F32x4,
    w: &[F32x4],
    g_out: &mut [F32x4; N],
    u_m: &[F32x4; N],
    dot4: &mut F32x4,
    p2_4: &mut F32x4,
) {
    for ((g, &wi), &ui) in g_out.iter_mut().zip(w.iter()).zip(u_m.iter()) {
        let p = kg * (wi * *g) - kr * ui;
        *g = p;
        *dot4 = p.mul_add(ui, *dot4);
        *p2_4 = p.mul_add(p, *p2_4);
    }
}

#[inline(always)]
pub(crate) fn apply_cayley_u<const N: usize>(u: &mut [F32x4; N], p: &[F32x4; N], trb4: F32x4) {
    for i in 0..N {
        u[i] = u[i] - trb4 * p[i];
    }
}

#[inline(always)]
fn psi_and_inv(x: f64) -> (f64, f64) {
    let seed = F32x4::splat(x as f32).sqrt().to_array()[0];
    let inv0 = f64::from(1.0_f32 / seed);
    let inv_psi = inv0 * (1.5 - 0.5 * x * inv0 * inv0);
    (x * inv_psi, inv_psi)
}

#[cfg(test)]
#[path = "sav_tests.rs"]
mod tests;
