//! Two-string interleaved SAV stepping for parallelizing per-sample critical paths (`no_std`).

use piano_simd::F32x4;

use crate::choir_map::StringMapEntry;
use crate::modal_soa::ModalSoa;
use crate::phantom::{
    accum_phantom_grad_vh_simd_n, convolve_transverse_slopes_f32_n, BASE_PHANTOM_LONG_VECS,
};
use crate::phantom_sparse::{
    accum_phantom_grad_vh_simd_2x2, convolve_transverse_slopes_f32_2x2, SPARSE_PHANTOM_LONG_VECS,
};
use crate::sav_drift::{
    apply_cayley_u, eval_p_vec_slice, prepare_vh, SavBank, C0_SAV, DELTA_U_SAV, LAMBDA_REG_DEFAULT,
};
use crate::sav_pair_tile::step_sav_pair_tile;
use crate::sav_spec::SavStringSpec;
use crate::sav_tile::step_ringing_string_tile;

/// Batches ringing SAV strings with identical `(nv>=2, nh>=2, nl)` layouts into pairs
/// so independent `sqrt`/`div`/`reduce_sum` chains execute in parallel across the two strings.
pub(crate) struct SavPairQueue {
    pending: [Option<usize>; BASE_PHANTOM_LONG_VECS + 1],
    long_on: bool,
    sav_on: bool,
}

impl SavPairQueue {
    #[inline(always)]
    pub fn new(long_on: bool, sav_on: bool) -> Self {
        Self {
            pending: [None; BASE_PHANTOM_LONG_VECS + 1],
            long_on,
            sav_on,
        }
    }

    #[inline(always)]
    pub fn push_ringing(
        &mut self,
        soa: &mut ModalSoa,
        sav: &mut SavBank,
        strings: &[StringMapEntry],
        s_idx: usize,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        if !self.sav_on || sav.string_n_vh(s_idx) == 0 {
            step_ringing_string_tile(
                soa,
                sav,
                s_idx,
                &strings[s_idx],
                self.long_on,
                self.sav_on,
                out_l,
                out_r,
            );
            return;
        }
        let (nv, nh, nl, m_ph) = sav.string_prefix_bounds(s_idx, self.long_on);
        if nv != 2 || nh != 2 || (m_ph > 0 && nl > BASE_PHANTOM_LONG_VECS) {
            step_ringing_string_tile(
                soa,
                sav,
                s_idx,
                &strings[s_idx],
                self.long_on,
                self.sav_on,
                out_l,
                out_r,
            );
            return;
        }
        let bucket = if m_ph == 0 {
            0
        } else {
            nl.min(BASE_PHANTOM_LONG_VECS)
        };
        if let Some(s0) = self.pending[bucket].take() {
            step_sav_pair_tile(
                soa,
                sav,
                s0,
                &strings[s0],
                s_idx,
                &strings[s_idx],
                bucket,
                self.long_on,
                out_l,
                out_r,
            );
        } else {
            self.pending[bucket] = Some(s_idx);
        }
    }

    #[inline(always)]
    pub fn flush(
        &mut self,
        soa: &mut ModalSoa,
        sav: &mut SavBank,
        strings: &[StringMapEntry],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        for slot in &mut self.pending {
            if let Some(s_idx) = slot.take() {
                step_ringing_string_tile(
                    soa,
                    sav,
                    s_idx,
                    &strings[s_idx],
                    self.long_on,
                    self.sav_on,
                    out_l,
                    out_r,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn cayley_step_vh<const NL: usize>(
    sp: &SavStringSpec,
    kg: F32x4,
    kr: F32x4,
    rn: f64,
    gv: &mut [F32x4; 2],
    uvm: &[F32x4; 2],
    gh: &mut [F32x4; 2],
    uhm: &[F32x4; 2],
    cp: Option<(&mut [F32x4; NL], &[F32x4; NL])>,
) -> (f64, F32x4) {
    let (mut d, mut p2) = (F32x4::zero(), F32x4::zero());
    eval_p_vec_slice(kg, kr, &sp.omega_sigma_v4, gv, uvm, &mut d, &mut p2);
    eval_p_vec_slice(kg, kr, &sp.omega_sigma_h4, gh, uhm, &mut d, &mut p2);
    if let Some((cp_mut, ulm)) = cp {
        eval_p_vec_slice(
            kg,
            kr,
            &sp.omega_alpha_l_norm4,
            cp_mut,
            ulm,
            &mut d,
            &mut p2,
        );
    }
    let rb = (rn + f64::from(d.reduce_sum())) / (1.0 + f64::from(p2.reduce_sum()));
    (2.0 * rb - rn, F32x4::splat((2.0 * rb) as f32))
}

#[inline(always)]
fn psi_and_inv_pair(x0: f64, x1: f64) -> ((f64, f64), (f64, f64)) {
    let seeds = F32x4::from_array([x0 as f32, x1 as f32, 1.0, 1.0]).sqrt();
    let recips = (F32x4::splat(1.0) / seeds).to_array();
    let inv0 = f64::from(recips[0]);
    let inv_psi0 = inv0 * (1.5 - 0.5 * x0 * inv0 * inv0);
    let inv1 = f64::from(recips[1]);
    let inv_psi1 = inv1 * (1.5 - 0.5 * x1 * inv1 * inv1);
    ((x0 * inv_psi0, inv_psi0), (x1 * inv_psi1, inv_psi1))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn eval_no_phantom_pair(
    sp0: &SavStringSpec,
    sp1: &SavStringSpec,
    ts: f64,
    r0_ref: &mut f64,
    r1_ref: &mut f64,
    uv0: &mut [F32x4; 2],
    vv0: &[F32x4; 2],
    uh0: &mut [F32x4; 2],
    vh0: &[F32x4; 2],
    uv1: &mut [F32x4; 2],
    vv1: &[F32x4; 2],
    uh1: &mut [F32x4; 2],
    vh1: &[F32x4; 2],
) {
    let (uvm0, mut dv0, uhm0, mut dh0, usq0, dsq0) = prepare_vh(sp0, uv0, vv0, uh0, vh0);
    let (uvm1, mut dv1, uhm1, mut dh1, usq1, dsq1) = prepare_vh(sp1, uv1, vv1, uh1, vh1);
    let inv_u0 = 1.0 / (f64::from(usq0.reduce_sum()) + DELTA_U_SAV);
    let inv_u1 = 1.0 / (f64::from(usq1.reduce_sum()) + DELTA_U_SAV);
    let s0 = sp0.quarter_l * f64::from(dsq0.reduce_sum());
    let s1 = sp1.quarter_l * f64::from(dsq1.reduce_sum());
    let ((psi0, ip0), (psi1, ip1)) = psi_and_inv_pair(
        sp0.kappa_0 * s0 * s0 + C0_SAV,
        sp1.kappa_0 * s1 * s1 + C0_SAV,
    );
    let (rn0, rn1) = (*r0_ref, *r1_ref);
    let half_ts = 0.5 * ts;
    let kg0 = F32x4::splat((half_ts * ip0 * 0.5 * sp0.length_m * sp0.kappa_0 * s0) as f32);
    let kr0 = F32x4::splat((half_ts * LAMBDA_REG_DEFAULT * (rn0 - psi0) * inv_u0) as f32);
    let kg1 = F32x4::splat((half_ts * ip1 * 0.5 * sp1.length_m * sp1.kappa_0 * s1) as f32);
    let kr1 = F32x4::splat((half_ts * LAMBDA_REG_DEFAULT * (rn1 - psi1) * inv_u1) as f32);
    let (r0n, t0) = cayley_step_vh::<0>(sp0, kg0, kr0, rn0, &mut dv0, &uvm0, &mut dh0, &uhm0, None);
    let (r1n, t1) = cayley_step_vh::<0>(sp1, kg1, kr1, rn1, &mut dv1, &uvm1, &mut dh1, &uhm1, None);
    *r0_ref = r0n;
    *r1_ref = r1n;
    apply_cayley_u(uv0, &dv0, t0);
    apply_cayley_u(uh0, &dh0, t0);
    apply_cayley_u(uv1, &dv1, t1);
    apply_cayley_u(uh1, &dh1, t1);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn eval_phantom_pair<const NL_MAX: usize>(
    sp0: &SavStringSpec,
    sp1: &SavStringSpec,
    ts: f64,
    nl: usize,
    r0_ref: &mut f64,
    r1_ref: &mut f64,
    uv0: &mut [F32x4; 2],
    vv0: &[F32x4; 2],
    uh0: &mut [F32x4; 2],
    vh0: &[F32x4; 2],
    ul0: &mut [F32x4; NL_MAX],
    vl0: &[F32x4; NL_MAX],
    uv1: &mut [F32x4; 2],
    vv1: &[F32x4; 2],
    uh1: &mut [F32x4; 2],
    vh1: &[F32x4; 2],
    ul1: &mut [F32x4; NL_MAX],
    vl1: &[F32x4; NL_MAX],
) {
    let n = if NL_MAX == 1 { 1 } else { nl.min(NL_MAX) };
    let (uvm0, dv0, uhm0, dh0, mut usq0, dsq0) = prepare_vh(sp0, uv0, vv0, uh0, vh0);
    let (uvm1, dv1, uhm1, dh1, mut usq1, dsq1) = prepare_vh(sp1, uv1, vv1, uh1, vh1);
    let (mut ulm0, mut ulm1) = ([F32x4::zero(); NL_MAX], [F32x4::zero(); NL_MAX]);
    for i in 0..n {
        let m0 = ul0[i] * sp0.mask_l4[i];
        ulm0[i] = m0;
        usq0 = m0.mul_add(m0, usq0);
        let m1 = ul1[i] * sp1.mask_l4[i];
        ulm1[i] = m1;
        usq1 = m1.mul_add(m1, usq1);
    }
    let inv_u0 = 1.0 / (f64::from(usq0.reduce_sum()) + DELTA_U_SAV);
    let inv_u1 = 1.0 / (f64::from(usq1.reduce_sum()) + DELTA_U_SAV);
    let phi0 = if NL_MAX == SPARSE_PHANTOM_LONG_VECS {
        let p = convolve_transverse_slopes_f32_2x2(&dv0, &dh0);
        core::array::from_fn(|i| p[i])
    } else {
        convolve_transverse_slopes_f32_n::<NL_MAX>(&dv0, &dh0)
    };
    let phi1 = if NL_MAX == SPARSE_PHANTOM_LONG_VECS {
        let p = convolve_transverse_slopes_f32_2x2(&dv1, &dh1);
        core::array::from_fn(|i| p[i])
    } else {
        convolve_transverse_slopes_f32_n::<NL_MAX>(&dv1, &dh1)
    };
    let cs0 = F32x4::splat((sp0.quarter_l * sp0.kappa_ph) as f32);
    let cs1 = F32x4::splat((sp1.quarter_l * sp1.kappa_ph) as f32);
    let (mut r2_0, mut r2_1) = (F32x4::zero(), F32x4::zero());
    let (mut cp0, mut cp1) = ([F32x4::zero(); NL_MAX], [F32x4::zero(); NL_MAX]);
    for i in 0..n {
        let a0 = sp0.half_ql_mask4[i] * phi0[i] + sp0.alpha_l4[i] * vl0[i];
        r2_0 = a0.mul_add(a0, r2_0);
        cp0[i] = a0 * cs0;
        let a1 = sp1.half_ql_mask4[i] * phi1[i] + sp1.alpha_l4[i] * vl1[i];
        r2_1 = a1.mul_add(a1, r2_1);
        cp1[i] = a1 * cs1;
    }
    let s0 = sp0.quarter_l * f64::from(dsq0.reduce_sum());
    let s1 = sp1.quarter_l * f64::from(dsq1.reduce_sum());
    let e0 = sp0.kappa_0 * s0 * s0 + sp0.kappa_ph * f64::from(r2_0.reduce_sum());
    let e1 = sp1.kappa_0 * s1 * s1 + sp1.kappa_ph * f64::from(r2_1.reduce_sum());
    let ((psi0, ip0), (psi1, ip1)) = psi_and_inv_pair(e0 + C0_SAV, e1 + C0_SAV);
    let (rn0, rn1) = (*r0_ref, *r1_ref);
    let c_sum0 = (0.5 * sp0.length_m * sp0.kappa_0 * s0) as f32;
    let c_sum1 = (0.5 * sp1.length_m * sp1.kappa_0 * s1) as f32;
    let (mut gv0, mut gh0) = if NL_MAX == SPARSE_PHANTOM_LONG_VECS {
        let cp = [cp0[0], cp0[1]];
        accum_phantom_grad_vh_simd_2x2(&dv0, &dh0, c_sum0, &cp)
    } else {
        accum_phantom_grad_vh_simd_n::<NL_MAX>(&dv0, &dh0, c_sum0, &cp0)
    };
    let (mut gv1, mut gh1) = if NL_MAX == SPARSE_PHANTOM_LONG_VECS {
        let cp = [cp1[0], cp1[1]];
        accum_phantom_grad_vh_simd_2x2(&dv1, &dh1, c_sum1, &cp)
    } else {
        accum_phantom_grad_vh_simd_n::<NL_MAX>(&dv1, &dh1, c_sum1, &cp1)
    };
    let half_ts = 0.5 * ts;
    let kg0 = F32x4::splat((half_ts * ip0) as f32);
    let kr0 = F32x4::splat((half_ts * LAMBDA_REG_DEFAULT * (rn0 - psi0) * inv_u0) as f32);
    let kg1 = F32x4::splat((half_ts * ip1) as f32);
    let kr1 = F32x4::splat((half_ts * LAMBDA_REG_DEFAULT * (rn1 - psi1) * inv_u1) as f32);
    let (r0n, t0) = cayley_step_vh(
        sp0,
        kg0,
        kr0,
        rn0,
        &mut gv0,
        &uvm0,
        &mut gh0,
        &uhm0,
        Some((&mut cp0, &ulm0)),
    );
    let (r1n, t1) = cayley_step_vh(
        sp1,
        kg1,
        kr1,
        rn1,
        &mut gv1,
        &uvm1,
        &mut gh1,
        &uhm1,
        Some((&mut cp1, &ulm1)),
    );
    *r0_ref = r0n;
    *r1_ref = r1n;
    apply_cayley_u(uv0, &gv0, t0);
    apply_cayley_u(uh0, &gh0, t0);
    apply_cayley_u(ul0, &cp0, t0);
    apply_cayley_u(uv1, &gv1, t1);
    apply_cayley_u(uh1, &gh1, t1);
    apply_cayley_u(ul1, &cp1, t1);
}
