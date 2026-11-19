//! Single-string SAV drift-regulated nonlinear coupling directly over `bp.state` (`no_std`).

use piano_simd::F32x4;

use crate::bridge_port::{BridgePorts, PrefixState};
use crate::phantom::{BASE_PHANTOM_LONG_VECS, MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_VECS};
use crate::phantom_sparse::SPARSE_PHANTOM_LONG_VECS;
use crate::sav_drift::{
    load_nl_state_n, step_sav_no_phantom_spec_n, step_sav_phantom_spec_3x6,
    step_sav_phantom_spec_n, SavBank, LAMBDA_REG_DEFAULT,
};

#[inline(always)]
fn load_pol_bp_n<const NT_MAX: usize>(
    state: &[PrefixState],
    off: usize,
    len: usize,
) -> ([F32x4; NT_MAX], [F32x4; NT_MAX]) {
    let (mut u, mut v) = ([F32x4::zero(); NT_MAX], [F32x4::zero(); NT_MAX]);
    let n = len.min(NT_MAX);
    for i in 0..n {
        let st = state[off + i];
        u[i] = st.u;
        v[i] = st.v;
    }
    (u, v)
}

#[inline(always)]
fn store_pol_u_bp_n<const NT_MAX: usize>(
    state: &mut [PrefixState],
    off: usize,
    len: usize,
    u: &[F32x4; NT_MAX],
) {
    let n = len.min(NT_MAX);
    for (i, &ui) in u[..n].iter().enumerate() {
        state[off + i].u = ui;
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
#[allow(clippy::too_many_arguments)]
fn run_sav_phantom<const NL_MAX: usize>(
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    s_idx: usize,
    lb: usize,
    nl: usize,
    uv: &mut [F32x4; 2],
    vv: &[F32x4; 2],
    uh: &mut [F32x4; 2],
    vh: &[F32x4; 2],
) {
    let n = nl.min(NL_MAX);
    let mut r_n = sav.r_sav[s_idx];
    let (mut ul, vl) = load_nl_state_n::<NL_MAX>(&bp.tl_soa, lb, n);
    step_sav_phantom_spec_n::<NL_MAX>(
        sav.spec(s_idx),
        sav.ts(),
        n,
        &mut r_n,
        uv,
        vv,
        uh,
        vh,
        &mut ul,
        &vl,
        LAMBDA_REG_DEFAULT,
    );
    sav.r_sav[s_idx] = r_n;
    bp.tl_soa.u.as_f32x4_slice_mut()[lb..lb + n].copy_from_slice(&ul[..n]);
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
#[allow(clippy::too_many_arguments)]
fn run_sav_phantom_3x6(
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    s_idx: usize,
    lb: usize,
    nl: usize,
    uv: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vv: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    uh: &mut [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    vh: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) {
    let n = nl.min(MAX_PHANTOM_LONG_VECS);
    let mut r_n = sav.r_sav[s_idx];
    let (mut ul, vl) = load_nl_state_n::<MAX_PHANTOM_LONG_VECS>(&bp.tl_soa, lb, n);
    step_sav_phantom_spec_3x6(
        sav.spec(s_idx),
        sav.ts(),
        n,
        &mut r_n,
        uv,
        vv,
        uh,
        vh,
        &mut ul,
        &vl,
        LAMBDA_REG_DEFAULT,
    );
    sav.r_sav[s_idx] = r_n;
    bp.tl_soa.u.as_f32x4_slice_mut()[lb..lb + n].copy_from_slice(&ul[..n]);
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn run_sav_no_phantom<const NT_MAX: usize>(
    sav: &mut SavBank,
    s_idx: usize,
    uv: &mut [F32x4; NT_MAX],
    vv: &[F32x4; NT_MAX],
    uh: &mut [F32x4; NT_MAX],
    vh: &[F32x4; NT_MAX],
) {
    let mut r_n = sav.r_sav[s_idx];
    step_sav_no_phantom_spec_n::<NT_MAX>(
        sav.spec(s_idx),
        sav.ts(),
        &mut r_n,
        uv,
        vv,
        uh,
        vh,
        LAMBDA_REG_DEFAULT,
    );
    sav.r_sav[s_idx] = r_n;
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
#[allow(clippy::too_many_arguments)]
fn step_sav_scheduled_3x6(
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    s_idx: usize,
    vb: usize,
    hb: usize,
    lb: usize,
    nv: usize,
    nh: usize,
    nl: usize,
    m_ph: usize,
) {
    let (mut uv, vv) = load_pol_bp_n::<MAX_PHANTOM_TRANSVERSE_VECS>(&bp.state, vb, nv);
    let (mut uh, vh) = load_pol_bp_n::<MAX_PHANTOM_TRANSVERSE_VECS>(&bp.state, hb, nh);
    if m_ph == 0 || nl == 0 {
        run_sav_no_phantom::<MAX_PHANTOM_TRANSVERSE_VECS>(sav, s_idx, &mut uv, &vv, &mut uh, &vh);
    } else {
        run_sav_phantom_3x6(bp, sav, s_idx, lb, nl, &mut uv, &vv, &mut uh, &vh);
    }
    store_pol_u_bp_n::<MAX_PHANTOM_TRANSVERSE_VECS>(&mut bp.state, vb, nv, &uv);
    store_pol_u_bp_n::<MAX_PHANTOM_TRANSVERSE_VECS>(&mut bp.state, hb, nh, &uh);
}

/// Steps all active non-dormant SAV strings directly on `bp.state` without stack spill blowup.
pub(crate) fn step_sav_scheduled(
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    sav_strings: &[u8],
    long_on: bool,
) {
    for &s_u8 in sav_strings {
        let s_idx = usize::from(s_u8);
        let (nv, nh, nl, m_ph) = sav.string_prefix_bounds(s_idx, long_on);
        if nv == 0 && nh == 0 {
            continue;
        }
        let (vb, hb, lb) = {
            let m = &bp.meta[s_idx];
            (m.beta_v_off, m.beta_h_off, m.tl_l.start)
        };
        if nv <= 2 && nh <= 2 && nl <= BASE_PHANTOM_LONG_VECS {
            let (mut uv, vv) = load_pol_bp_n::<2>(&bp.state, vb, nv);
            let (mut uh, vh) = load_pol_bp_n::<2>(&bp.state, hb, nh);
            if m_ph == 0 || nl == 0 {
                run_sav_no_phantom::<2>(sav, s_idx, &mut uv, &vv, &mut uh, &vh);
            } else if nl == 1 {
                run_sav_phantom::<1>(bp, sav, s_idx, lb, nl, &mut uv, &vv, &mut uh, &vh);
            } else if nl == SPARSE_PHANTOM_LONG_VECS {
                run_sav_phantom::<SPARSE_PHANTOM_LONG_VECS>(
                    bp, sav, s_idx, lb, nl, &mut uv, &vv, &mut uh, &vh,
                );
            } else {
                run_sav_phantom::<BASE_PHANTOM_LONG_VECS>(
                    bp, sav, s_idx, lb, nl, &mut uv, &vv, &mut uh, &vh,
                );
            }
            store_pol_u_bp_n::<2>(&mut bp.state, vb, nv, &uv);
            store_pol_u_bp_n::<2>(&mut bp.state, hb, nh, &uh);
        } else {
            step_sav_scheduled_3x6(bp, sav, s_idx, vb, hb, lb, nv, nh, nl, m_ph);
        }
    }
}
