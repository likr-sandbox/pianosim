//! Single-string SAV drift-regulated nonlinear coupling directly over `bp.state` (`no_std`).

use piano_simd::F32x4;

use crate::bridge_port::{BridgePorts, PrefixState};
use crate::phantom::MAX_PHANTOM_LONG_VECS;
use crate::sav_drift::{
    load_nl_state_n, step_sav_no_phantom_spec, step_sav_phantom_spec_n, SavBank, LAMBDA_REG_DEFAULT,
};

#[inline(always)]
fn load_pol_bp(state: &[PrefixState], off: usize, len: usize) -> ([F32x4; 2], [F32x4; 2]) {
    let z = F32x4::zero();
    if len >= 2 {
        let (s0, s1) = (state[off], state[off + 1]);
        ([s0.u, s1.u], [s0.v, s1.v])
    } else if len == 1 {
        let s0 = state[off];
        ([s0.u, z], [s0.v, z])
    } else {
        ([z; 2], [z; 2])
    }
}

#[inline(always)]
fn store_pol_u_bp(state: &mut [PrefixState], off: usize, len: usize, u: &[F32x4; 2]) {
    if len >= 2 {
        state[off].u = u[0];
        state[off + 1].u = u[1];
    } else if len == 1 {
        state[off].u = u[0];
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
fn run_sav_no_phantom(
    sav: &mut SavBank,
    s_idx: usize,
    uv: &mut [F32x4; 2],
    vv: &[F32x4; 2],
    uh: &mut [F32x4; 2],
    vh: &[F32x4; 2],
) {
    let mut r_n = sav.r_sav[s_idx];
    step_sav_no_phantom_spec(
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
        let (mut uv, vv) = load_pol_bp(&bp.state, vb, nv);
        let (mut uh, vh) = load_pol_bp(&bp.state, hb, nh);
        if m_ph == 0 || nl == 0 {
            run_sav_no_phantom(sav, s_idx, &mut uv, &vv, &mut uh, &vh);
        } else if nl == 1 {
            run_sav_phantom::<1>(bp, sav, s_idx, lb, nl, &mut uv, &vv, &mut uh, &vh);
        } else {
            run_sav_phantom::<MAX_PHANTOM_LONG_VECS>(
                bp, sav, s_idx, lb, nl, &mut uv, &vv, &mut uh, &vh,
            );
        }
        store_pol_u_bp(&mut bp.state, vb, nv, &uv);
        store_pol_u_bp(&mut bp.state, hb, nh, &uh);
    }
}
