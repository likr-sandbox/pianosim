//! 2-string paired SAV drift-regulated nonlinear coupling over L1d `BridgePorts` (`no_std`).

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_port::{BridgePorts, PrefixState};
use crate::phantom::MAX_PHANTOM_LONG_VECS;
use crate::sav_drift::{load_nl_state_n, SavBank};
use crate::sav_pair::{eval_no_phantom_pair, eval_phantom_pair};

#[inline(always)]
fn load_vh_br(state: &[PrefixState], b: usize) -> ([F32x4; 2], [F32x4; 2]) {
    let (s0, s1) = (state[b], state[b + 1]);
    ([s0.u, s1.u], [s0.v, s1.v])
}

#[inline(always)]
fn store_vh_u_br(state: &mut [PrefixState], b: usize, u: &[F32x4; 2]) {
    state[b].u = u[0];
    state[b + 1].u = u[1];
}

#[derive(Copy, Clone)]
pub(crate) struct SavSchedule {
    pairs: [(u8, u8); TOTAL_STRINGS / 2],
    bucket_bounds: [u8; 6],
    singles: [u8; 8],
    n_singles: usize,
}

impl SavSchedule {
    pub const EMPTY: Self = Self {
        pairs: [(0, 0); TOTAL_STRINGS / 2],
        bucket_bounds: [0; 6],
        singles: [0; 8],
        n_singles: 0,
    };

    pub fn rebuild(&mut self, sav: &SavBank, sav_strings: &[u8], long_on: bool) {
        self.bucket_bounds = [0; 6];
        self.n_singles = 0;
        let mut raw = [(0_u8, 0_u8, 0_u8); TOTAL_STRINGS / 2];
        let mut n_raw = 0_usize;
        let mut counts = [0_usize; 5];
        let mut pending: [Option<u8>; MAX_PHANTOM_LONG_VECS + 1] =
            [None; MAX_PHANTOM_LONG_VECS + 1];
        for &s_u8 in sav_strings {
            let s_idx = usize::from(s_u8);
            let (nv, nh, nl, m_ph) = sav.string_prefix_bounds(s_idx, long_on);
            if nv < 2 || nh < 2 {
                if self.n_singles < self.singles.len() {
                    self.singles[self.n_singles] = s_u8;
                    self.n_singles += 1;
                }
                continue;
            }
            let bucket = if m_ph == 0 {
                0
            } else {
                nl.min(MAX_PHANTOM_LONG_VECS)
            };
            if let Some(s0) = pending[bucket].take() {
                raw[n_raw] = (s0, s_u8, bucket as u8);
                n_raw += 1;
                counts[bucket] += 1;
            } else {
                pending[bucket] = Some(s_u8);
            }
        }
        let mut off = 0_usize;
        for (b, &cnt) in counts.iter().enumerate() {
            self.bucket_bounds[b] = off as u8;
            off += cnt;
        }
        self.bucket_bounds[5] = off as u8;
        let mut write = [
            usize::from(self.bucket_bounds[0]),
            usize::from(self.bucket_bounds[1]),
            usize::from(self.bucket_bounds[2]),
            usize::from(self.bucket_bounds[3]),
            usize::from(self.bucket_bounds[4]),
        ];
        for &(s0, s1, b_u8) in &raw[..n_raw] {
            let b = usize::from(b_u8);
            self.pairs[write[b]] = (s0, s1);
            write[b] += 1;
        }
        for s_u8 in pending.into_iter().flatten() {
            if self.n_singles < self.singles.len() {
                self.singles[self.n_singles] = s_u8;
                self.n_singles += 1;
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn step_sav_bucket_0_slice(bp: &mut BridgePorts, sav: &mut SavBank, pairs: &[(u8, u8)]) {
    for &(s0, s1) in pairs {
        eval_sav_no_phantom_prefix(bp, sav, usize::from(s0), usize::from(s1));
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn step_sav_bucket_1_slice(bp: &mut BridgePorts, sav: &mut SavBank, pairs: &[(u8, u8)]) {
    for &(s0, s1) in pairs {
        eval_sav_phantom_prefix::<1>(bp, sav, usize::from(s0), usize::from(s1), 1);
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn step_sav_bucket_2_slice(bp: &mut BridgePorts, sav: &mut SavBank, pairs: &[(u8, u8)]) {
    for &(s0, s1) in pairs {
        eval_sav_phantom_prefix::<2>(bp, sav, usize::from(s0), usize::from(s1), 2);
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn step_sav_bucket_3_slice(bp: &mut BridgePorts, sav: &mut SavBank, pairs: &[(u8, u8)]) {
    for &(s0, s1) in pairs {
        eval_sav_phantom_prefix::<3>(bp, sav, usize::from(s0), usize::from(s1), 3);
    }
}

#[cfg_attr(target_arch = "wasm32", inline(never))]
fn step_sav_bucket_4_slice(bp: &mut BridgePorts, sav: &mut SavBank, pairs: &[(u8, u8)]) {
    for &(s0, s1) in pairs {
        eval_sav_phantom_prefix::<MAX_PHANTOM_LONG_VECS>(
            bp,
            sav,
            usize::from(s0),
            usize::from(s1),
            4,
        );
    }
}

pub(crate) fn step_sav_scheduled(
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    sched: &SavSchedule,
    long_on: bool,
) {
    let b = &sched.bucket_bounds;
    for bucket in 0..5 {
        let (s, e) = (usize::from(b[bucket]), usize::from(b[bucket + 1]));
        if s == e {
            continue;
        }
        let slice = &sched.pairs[s..e];
        match bucket {
            0 => step_sav_bucket_0_slice(bp, sav, slice),
            1 => step_sav_bucket_1_slice(bp, sav, slice),
            2 => step_sav_bucket_2_slice(bp, sav, slice),
            3 => step_sav_bucket_3_slice(bp, sav, slice),
            _ => step_sav_bucket_4_slice(bp, sav, slice),
        }
    }
    for &s_u8 in &sched.singles[..sched.n_singles] {
        step_single_sav_prefix(bp, sav, usize::from(s_u8), long_on);
    }
}

fn step_single_sav_prefix(bp: &mut BridgePorts, sav: &mut SavBank, s_idx: usize, long_on: bool) {
    let (nv, nh, nl, m_ph) = sav.string_prefix_bounds(s_idx, long_on);
    if nv == 0 && nh == 0 {
        return;
    }
    let m = &bp.meta[s_idx];
    let z = F32x4::zero();
    let load_pol = |off: usize, len: usize| -> ([F32x4; 2], [F32x4; 2]) {
        if len >= 2 {
            load_vh_br(&bp.state, off)
        } else if len == 1 {
            ([bp.state[off].u, z], [bp.state[off].v, z])
        } else {
            ([z; 2], [z; 2])
        }
    };
    let (mut uv, vv) = load_pol(m.beta_v_off, nv);
    let (mut uh, vh) = load_pol(m.beta_h_off, nh);
    let mut r_n = sav.r_sav[s_idx];
    let lb = m.tl_l.start;
    if m_ph == 0 || nl == 0 {
        crate::sav_drift::step_sav_no_phantom_spec(
            sav.spec(s_idx),
            sav.ts(),
            &mut r_n,
            &mut uv,
            &vv,
            &mut uh,
            &vh,
            crate::sav_drift::LAMBDA_REG_DEFAULT,
        );
    } else {
        let n = nl.min(MAX_PHANTOM_LONG_VECS);
        let (mut ul, vl) = load_nl_state_n::<MAX_PHANTOM_LONG_VECS>(&bp.tl_soa, lb, n);
        crate::sav_drift::step_sav_phantom_spec_n::<MAX_PHANTOM_LONG_VECS>(
            sav.spec(s_idx),
            sav.ts(),
            n,
            &mut r_n,
            &mut uv,
            &vv,
            &mut uh,
            &vh,
            &mut ul,
            &vl,
            crate::sav_drift::LAMBDA_REG_DEFAULT,
        );
        bp.tl_soa.u.as_f32x4_slice_mut()[lb..lb + n].copy_from_slice(&ul[..n]);
    }
    sav.r_sav[s_idx] = r_n;
    if nv >= 2 {
        store_vh_u_br(&mut bp.state, m.beta_v_off, &uv);
    } else if nv == 1 {
        bp.state[m.beta_v_off].u = uv[0];
    }
    if nh >= 2 {
        store_vh_u_br(&mut bp.state, m.beta_h_off, &uh);
    } else if nh == 1 {
        bp.state[m.beta_h_off].u = uh[0];
    }
}

#[inline(always)]
fn eval_sav_no_phantom_prefix(bp: &mut BridgePorts, sav: &mut SavBank, s0: usize, s1: usize) {
    let (vb0, hb0) = (bp.meta[s0].beta_v_off, bp.meta[s0].beta_h_off);
    let (vb1, hb1) = (bp.meta[s1].beta_v_off, bp.meta[s1].beta_h_off);
    let (mut uv0, vv0) = load_vh_br(&bp.state, vb0);
    let (mut uh0, vh0) = load_vh_br(&bp.state, hb0);
    let (mut uv1, vv1) = load_vh_br(&bp.state, vb1);
    let (mut uh1, vh1) = load_vh_br(&bp.state, hb1);
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    eval_no_phantom_pair(
        sav.spec(s0),
        sav.spec(s1),
        sav.ts(),
        &mut r0,
        &mut r1,
        &mut uv0,
        &vv0,
        &mut uh0,
        &vh0,
        &mut uv1,
        &vv1,
        &mut uh1,
        &vh1,
    );
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_u_br(&mut bp.state, vb0, &uv0);
    store_vh_u_br(&mut bp.state, hb0, &uh0);
    store_vh_u_br(&mut bp.state, vb1, &uv1);
    store_vh_u_br(&mut bp.state, hb1, &uh1);
}

#[inline(always)]
fn eval_sav_phantom_prefix<const NL_MAX: usize>(
    bp: &mut BridgePorts,
    sav: &mut SavBank,
    s0: usize,
    s1: usize,
    nl: usize,
) {
    let n = nl.min(NL_MAX);
    let (vb0, hb0, lb0) = (
        bp.meta[s0].beta_v_off,
        bp.meta[s0].beta_h_off,
        bp.meta[s0].tl_l.start,
    );
    let (vb1, hb1, lb1) = (
        bp.meta[s1].beta_v_off,
        bp.meta[s1].beta_h_off,
        bp.meta[s1].tl_l.start,
    );
    let (mut uv0, vv0) = load_vh_br(&bp.state, vb0);
    let (mut uh0, vh0) = load_vh_br(&bp.state, hb0);
    let (mut uv1, vv1) = load_vh_br(&bp.state, vb1);
    let (mut uh1, vh1) = load_vh_br(&bp.state, hb1);
    let (mut ul0, vl0) = load_nl_state_n::<NL_MAX>(&bp.tl_soa, lb0, n);
    let (mut ul1, vl1) = load_nl_state_n::<NL_MAX>(&bp.tl_soa, lb1, n);
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    eval_phantom_pair::<NL_MAX>(
        sav.spec(s0),
        sav.spec(s1),
        sav.ts(),
        n,
        &mut r0,
        &mut r1,
        &mut uv0,
        &vv0,
        &mut uh0,
        &vh0,
        &mut ul0,
        &vl0,
        &mut uv1,
        &vv1,
        &mut uh1,
        &vh1,
        &mut ul1,
        &vl1,
    );
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_u_br(&mut bp.state, vb0, &uv0);
    store_vh_u_br(&mut bp.state, hb0, &uh0);
    store_vh_u_br(&mut bp.state, vb1, &uv1);
    store_vh_u_br(&mut bp.state, hb1, &uh1);
    let u_s = bp.tl_soa.u.as_f32x4_slice_mut();
    u_s[lb0..lb0 + n].copy_from_slice(&ul0[..n]);
    u_s[lb1..lb1 + n].copy_from_slice(&ul1[..n]);
}
