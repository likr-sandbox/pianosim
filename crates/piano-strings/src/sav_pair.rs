//! Two-string interleaved SAV stepping for parallelizing per-sample critical paths (`no_std`).

use piano_simd::F32x4;

use crate::choir_map::StringMapEntry;
use crate::modal_soa::ModalSoa;
use crate::phantom::{
    accum_phantom_grad_vh_simd_n, convolve_transverse_slopes_f32_n, MAX_PHANTOM_LONG_VECS,
};
use crate::sav_drift::{
    apply_cayley_u, eval_p_vec, eval_p_vec_slice, load_nl_state_n, load_vh_state, prepare_vh,
    store_nl_state_n, SavBank, C0_SAV, DELTA_U_SAV, LAMBDA_REG_DEFAULT,
};
use crate::sav_spec::SavStringSpec;
use crate::sav_tile::{
    step_2_slices, step_nl_slices, step_ringing_string_tile, step_tails, store_vh_state,
};

/// Batches ringing SAV strings with identical `(nv>=2, nh>=2, nl)` layouts into pairs
/// so independent `sqrt`/`div`/`reduce_sum` chains execute in parallel across the two strings.
pub(crate) struct SavPairQueue {
    pending: [Option<usize>; MAX_PHANTOM_LONG_VECS + 1],
    long_on: bool,
    sav_on: bool,
}

impl SavPairQueue {
    #[inline(always)]
    pub fn new(long_on: bool, sav_on: bool) -> Self {
        Self {
            pending: [None; MAX_PHANTOM_LONG_VECS + 1],
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
        if nv < 2 || nh < 2 {
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
            nl.min(MAX_PHANTOM_LONG_VECS)
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
    eval_p_vec(kg, kr, &sp.omega_sigma_v4, gv, uvm, &mut d, &mut p2);
    eval_p_vec(kg, kr, &sp.omega_sigma_h4, gh, uhm, &mut d, &mut p2);
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
    let phi0 = convolve_transverse_slopes_f32_n::<NL_MAX>(&dv0, &dh0);
    let phi1 = convolve_transverse_slopes_f32_n::<NL_MAX>(&dv1, &dh1);
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
    let (mut gv0, mut gh0) = accum_phantom_grad_vh_simd_n::<NL_MAX>(
        &dv0,
        &dh0,
        (0.5 * sp0.length_m * sp0.kappa_0 * s0) as f32,
        &cp0,
    );
    let (mut gv1, mut gh1) = accum_phantom_grad_vh_simd_n::<NL_MAX>(
        &dv1,
        &dh1,
        (0.5 * sp1.length_m * sp1.kappa_0 * s1) as f32,
        &cp1,
    );
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

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_no_phantom_pair_tile(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    longitudinal_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    for e in [e0, e1] {
        let (vb, hb) = (e.vert_vecs.start, e.horiz_vecs.start);
        let (r1, r2) = if longitudinal_enabled {
            (hb + 2..e.long_vecs.end, 0..0)
        } else {
            (hb + 2..e.horiz_vecs.end, e.long_vecs.clone())
        };
        step_tails(soa, vb + 2..e.vert_vecs.end, r1, r2, e, out_l, out_r);
    }
    let (vb0, hb0, vb1, hb1) = (
        e0.vert_vecs.start,
        e0.horiz_vecs.start,
        e1.vert_vecs.start,
        e1.horiz_vecs.start,
    );
    let (mut uv0, mut vv0) = load_vh_state(soa, vb0, 2);
    let (mut uh0, mut vh0) = load_vh_state(soa, hb0, 2);
    let (mut uv1, mut vv1) = load_vh_state(soa, vb1, 2);
    let (mut uh1, mut vh1) = load_vh_state(soa, hb1, 2);
    let (sp0, sp1, ts) = (sav.spec(s0), sav.spec(s1), sav.ts());
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    let steps = out_l.len().min(out_r.len());
    {
        let (d_s, s_s, c_s) = (
            soa.d_coeff.as_f32x4_slice(),
            soa.s_coeff.as_f32x4_slice(),
            soa.c_bridge.as_f32x4_slice(),
        );
        let min_len = d_s.len().min(s_s.len()).min(c_s.len());
        assert!(
            vb0 + 2 <= min_len && hb0 + 2 <= min_len && vb1 + 2 <= min_len && hb1 + 2 <= min_len
        );
        for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
            let av0 = step_2_slices(&mut uv0, &mut vv0, vb0, d_s, s_s, c_s);
            let ah0 = step_2_slices(&mut uh0, &mut vh0, hb0, d_s, s_s, c_s);
            let av1 = step_2_slices(&mut uv1, &mut vv1, vb1, d_s, s_s, c_s);
            let ah1 = step_2_slices(&mut uh1, &mut vh1, hb1, d_s, s_s, c_s);
            eval_no_phantom_pair(
                sp0, sp1, ts, &mut r0, &mut r1, &mut uv0, &vv0, &mut uh0, &vh0, &mut uv1, &vv1,
                &mut uh1, &vh1,
            );
            let f0 = (av0 + ah0).reduce_sum();
            let f1 = (av1 + ah1).reduce_sum();
            *lo += f0 * e0.pan_l + f1 * e1.pan_l;
            *ro += f0 * e0.pan_r + f1 * e1.pan_r;
        }
    }
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_state(soa, vb0, 2, &uv0, &vv0);
    store_vh_state(soa, hb0, 2, &uh0, &vh0);
    store_vh_state(soa, vb1, 2, &uv1, &vv1);
    store_vh_state(soa, hb1, 2, &uh1, &vh1);
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_phantom_pair_tile<const NL_MAX: usize>(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    nl: usize,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let n = if NL_MAX == 1 { 1 } else { nl.min(NL_MAX) };
    for e in [e0, e1] {
        let (vb, hb, lb) = (e.vert_vecs.start, e.horiz_vecs.start, e.long_vecs.start);
        step_tails(
            soa,
            vb + 2..e.vert_vecs.end,
            hb + 2..e.horiz_vecs.end,
            lb + n..e.long_vecs.end,
            e,
            out_l,
            out_r,
        );
    }
    let (vb0, hb0, lb0) = (e0.vert_vecs.start, e0.horiz_vecs.start, e0.long_vecs.start);
    let (vb1, hb1, lb1) = (e1.vert_vecs.start, e1.horiz_vecs.start, e1.long_vecs.start);
    let (mut uv0, mut vv0) = load_vh_state(soa, vb0, 2);
    let (mut uh0, mut vh0) = load_vh_state(soa, hb0, 2);
    let (mut ul0, mut vl0) = load_nl_state_n::<NL_MAX>(soa, lb0, n);
    let (mut uv1, mut vv1) = load_vh_state(soa, vb1, 2);
    let (mut uh1, mut vh1) = load_vh_state(soa, hb1, 2);
    let (mut ul1, mut vl1) = load_nl_state_n::<NL_MAX>(soa, lb1, n);
    let (sp0, sp1, ts) = (sav.spec(s0), sav.spec(s1), sav.ts());
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    let steps = out_l.len().min(out_r.len());
    {
        let (d_s, s_s, c_s) = (
            soa.d_coeff.as_f32x4_slice(),
            soa.s_coeff.as_f32x4_slice(),
            soa.c_bridge.as_f32x4_slice(),
        );
        let min_len = d_s.len().min(s_s.len()).min(c_s.len());
        assert!(
            vb0 + 2 <= min_len
                && hb0 + 2 <= min_len
                && lb0 + n <= min_len
                && vb1 + 2 <= min_len
                && hb1 + 2 <= min_len
                && lb1 + n <= min_len
        );
        for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
            let av0 = step_2_slices(&mut uv0, &mut vv0, vb0, d_s, s_s, c_s);
            let ah0 = step_2_slices(&mut uh0, &mut vh0, hb0, d_s, s_s, c_s);
            let al0 = step_nl_slices::<NL_MAX>(&mut ul0, &mut vl0, lb0, n, d_s, s_s, c_s);
            let av1 = step_2_slices(&mut uv1, &mut vv1, vb1, d_s, s_s, c_s);
            let ah1 = step_2_slices(&mut uh1, &mut vh1, hb1, d_s, s_s, c_s);
            let al1 = step_nl_slices::<NL_MAX>(&mut ul1, &mut vl1, lb1, n, d_s, s_s, c_s);
            eval_phantom_pair::<NL_MAX>(
                sp0, sp1, ts, n, &mut r0, &mut r1, &mut uv0, &vv0, &mut uh0, &vh0, &mut ul0, &vl0,
                &mut uv1, &vv1, &mut uh1, &vh1, &mut ul1, &vl1,
            );
            let f0 = (av0 + ah0 + al0).reduce_sum();
            let f1 = (av1 + ah1 + al1).reduce_sum();
            *lo += f0 * e0.pan_l + f1 * e1.pan_l;
            *ro += f0 * e0.pan_r + f1 * e1.pan_r;
        }
    }
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_state(soa, vb0, 2, &uv0, &vv0);
    store_vh_state(soa, hb0, 2, &uh0, &vh0);
    store_nl_state_n::<NL_MAX>(soa, lb0, n, &ul0, &vl0);
    store_vh_state(soa, vb1, 2, &uv1, &vv1);
    store_vh_state(soa, hb1, 2, &uh1, &vh1);
    store_nl_state_n::<NL_MAX>(soa, lb1, n, &ul1, &vl1);
}

#[allow(clippy::too_many_arguments)]
fn step_sav_pair_tile(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    bucket: usize,
    longitudinal_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    if bucket == 0 {
        step_no_phantom_pair_tile(soa, sav, s0, e0, s1, e1, longitudinal_enabled, out_l, out_r);
    } else if bucket == 1 {
        step_phantom_pair_tile::<1>(soa, sav, s0, e0, s1, e1, 1, out_l, out_r);
    } else {
        step_phantom_pair_tile::<MAX_PHANTOM_LONG_VECS>(
            soa, sav, s0, e0, s1, e1, bucket, out_l, out_r,
        );
    }
}
