//! SIMD SAV Cayley step kernels and modal state load/store helpers (`no_std`).

use piano_simd::F32x4;

use crate::modal_soa::ModalSoa;
use crate::phantom::{
    accum_phantom_grad_vh_simd_n, convolve_transverse_slopes_f32_n, BASE_PHANTOM_TRANSVERSE_VECS,
    MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_VECS,
};
use crate::phantom_3x6::{accum_phantom_grad_vh_simd_3x6, convolve_transverse_slopes_f32_3x6};
use crate::phantom_sparse::{
    accum_phantom_grad_vh_simd_2x2, convolve_transverse_slopes_f32_2x2, SPARSE_PHANTOM_LONG_VECS,
};
use crate::sav_drift::{C0_SAV, DELTA_U_SAV};
use crate::sav_spec::SavStringSpec;

pub(crate) type PhVecs = [F32x4; BASE_PHANTOM_TRANSVERSE_VECS];

#[inline(always)]
pub(crate) fn load_vh_state_n<const NT_MAX: usize>(
    soa: &ModalSoa,
    b: usize,
    len: usize,
) -> ([F32x4; NT_MAX], [F32x4; NT_MAX]) {
    let (u_s, v_s) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
    let (mut u, mut v) = ([F32x4::zero(); NT_MAX], [F32x4::zero(); NT_MAX]);
    let n = len.min(NT_MAX);
    if n > 0 {
        assert!(b + n <= u_s.len() && b + n <= v_s.len());
        u[..n].copy_from_slice(&u_s[b..b + n]);
        v[..n].copy_from_slice(&v_s[b..b + n]);
    }
    (u, v)
}

#[inline(always)]
pub(crate) fn store_vh_u_n<const NT_MAX: usize>(
    soa: &mut ModalSoa,
    b: usize,
    len: usize,
    u: &[F32x4; NT_MAX],
) {
    let u_s = soa.u.as_f32x4_slice_mut();
    let n = len.min(NT_MAX);
    if n > 0 {
        assert!(b + n <= u_s.len());
        u_s[b..b + n].copy_from_slice(&u[..n]);
    }
}

#[inline(always)]
pub(crate) fn load_vh_state(soa: &ModalSoa, b: usize, len: usize) -> ([F32x4; 2], [F32x4; 2]) {
    load_vh_state_n::<2>(soa, b, len)
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
    uv: &mut [F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    vv: &[F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    uh: &mut [F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    vh: &[F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    lambda_reg: f64,
) {
    step_sav_no_phantom_spec_n::<BASE_PHANTOM_TRANSVERSE_VECS>(
        sp, ts, r_n_ref, uv, vv, uh, vh, lambda_reg,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn step_sav_no_phantom_spec_n<const NT_MAX: usize>(
    sp: &SavStringSpec,
    ts: f64,
    r_n_ref: &mut f64,
    uv: &mut [F32x4; NT_MAX],
    vv: &[F32x4; NT_MAX],
    uh: &mut [F32x4; NT_MAX],
    vh: &[F32x4; NT_MAX],
    lambda_reg: f64,
) {
    let (uv_m, mut dv4, uh_m, mut dh4, u_sq4, d_sq4) = prepare_vh_n::<NT_MAX>(sp, uv, vv, uh, vh);
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
    eval_p_vec_slice(
        kg,
        kr,
        &sp.omega_sigma_v4,
        &mut dv4,
        &uv_m,
        &mut dot4,
        &mut p2_4,
    );
    eval_p_vec_slice(
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
    uv: &mut [F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    vv: &[F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    uh: &mut [F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    vh: &[F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
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
    let phi4 = if NL_MAX == SPARSE_PHANTOM_LONG_VECS {
        let p2 = convolve_transverse_slopes_f32_2x2(&dv4, &dh4);
        core::array::from_fn(|i| p2[i])
    } else {
        convolve_transverse_slopes_f32_n::<NL_MAX>(&dv4, &dh4)
    };
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
    let (mut gv4, mut gh4) = if NL_MAX == SPARSE_PHANTOM_LONG_VECS {
        let cp2 = [c_phi4[0], c_phi4[1]];
        accum_phantom_grad_vh_simd_2x2(&dv4, &dh4, c_sum, &cp2)
    } else {
        accum_phantom_grad_vh_simd_n::<NL_MAX>(&dv4, &dh4, c_sum, &c_phi4)
    };

    let half_ts = 0.5 * ts;
    let kg = F32x4::splat((half_ts * inv_psi) as f32);
    let kr = F32x4::splat((half_ts * reg) as f32);
    let (mut dot4, mut p2_4) = (F32x4::zero(), F32x4::zero());
    eval_p_vec_slice(
        kg,
        kr,
        &sp.omega_sigma_v4,
        &mut gv4,
        &uv_m,
        &mut dot4,
        &mut p2_4,
    );
    eval_p_vec_slice(
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

#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "wasm32", inline(never))]
pub(crate) fn step_sav_phantom_spec_3x6(
    sp: &SavStringSpec,
    ts: f64,
    nl: usize,
    r_n_ref: &mut f64,
    uv: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vv: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    uh: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vh: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    ul: &mut [F32x4; MAX_PHANTOM_LONG_VECS],
    vl: &[F32x4; MAX_PHANTOM_LONG_VECS],
    lambda_reg: f64,
) {
    let n = nl.min(MAX_PHANTOM_LONG_VECS);
    let (uv_m, dv4, uh_m, dh4, mut u_sq4, d_sq4) =
        prepare_vh_n::<MAX_PHANTOM_TRANSVERSE_VECS>(sp, uv, vv, uh, vh);

    let mut ul_m = [F32x4::zero(); MAX_PHANTOM_LONG_VECS];
    for i in 0..n {
        let um = ul[i] * sp.mask_l4[i];
        ul_m[i] = um;
        u_sq4 = um.mul_add(um, u_sq4);
    }
    let den_u = f64::from(u_sq4.reduce_sum()) + DELTA_U_SAV;
    let inv_u = 1.0 / den_u;
    let phi4 = convolve_transverse_slopes_f32_3x6(&dv4, &dh4);
    let c_phi_scale4 = F32x4::splat((sp.quarter_l * sp.kappa_ph) as f32);
    let mut r2_4 = F32x4::zero();
    let mut c_phi4 = [F32x4::zero(); MAX_PHANTOM_LONG_VECS];
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
    let (mut gv4, mut gh4) = accum_phantom_grad_vh_simd_3x6(&dv4, &dh4, c_sum, &c_phi4);

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

#[inline(always)]
pub(crate) fn prepare_vh(
    sp: &SavStringSpec,
    uv: &PhVecs,
    vv: &PhVecs,
    uh: &PhVecs,
    vh: &PhVecs,
) -> (PhVecs, PhVecs, PhVecs, PhVecs, F32x4, F32x4) {
    prepare_vh_n::<BASE_PHANTOM_TRANSVERSE_VECS>(sp, uv, vv, uh, vh)
}

#[inline(always)]
pub(crate) fn prepare_vh_n<const NT_MAX: usize>(
    sp: &SavStringSpec,
    uv: &[F32x4; NT_MAX],
    vv: &[F32x4; NT_MAX],
    uh: &[F32x4; NT_MAX],
    vh: &[F32x4; NT_MAX],
) -> (
    [F32x4; NT_MAX],
    [F32x4; NT_MAX],
    [F32x4; NT_MAX],
    [F32x4; NT_MAX],
    F32x4,
    F32x4,
) {
    let mut uv_m = [F32x4::zero(); NT_MAX];
    let (mut dv4, mut uh_m, mut dh4) = (uv_m, uv_m, uv_m);
    let (mut u_sq4, mut d_sq4) = (F32x4::zero(), F32x4::zero());
    for i in 0..NT_MAX {
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
