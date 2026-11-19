//! Persistent Stage 2 <-> Stage 3 chunk scratch buffers and per-sample prefix WDF orchestrator.

use piano_simd::F32x4;
use piano_spec::{DAMPED_KEYS, NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_kernels::step_uncoupled_prefixes_fast;
use crate::bridge_prefix_step::step_coupled_prefixes;
use crate::bridge_sav::step_sav_scheduled;
use crate::bridge_tails::{
    BridgeRun, ChunkPlan, HammerStepForce, KeyCat, SavLongSlice, TailPortPartial, SUB,
};
use crate::contact::KeyActionDriver;
use crate::contact_ports::{accum_prefix_damp_vel_bp, refresh_prefix_ports_with_tail_bp};

pub(crate) const MAX_PEEL_SLOTS: usize = 24;

#[derive(Clone, Debug)]
pub(crate) struct BridgeScratch {
    pub plan: ChunkPlan,
    pub peel_buf: [[f32; SUB]; MAX_PEEL_SLOTS],
    pub peel_slot: [u8; DAMPED_KEYS],
    pub b_waves: [f32; TOTAL_STRINGS],
    pub zone_out: [f32; 8],
    pub dup_zone: [f32; 8],
    pub acc_th: [F32x4; SUB],
    pub acc_l: [F32x4; SUB],
}

impl BridgeScratch {
    #[must_use]
    pub fn new_boxed() -> alloc::boxed::Box<Self> {
        alloc::boxed::Box::new(Self {
            plan: ChunkPlan {
                key_cat: [KeyCat::Uncoupled; NUM_KEYS],
                key_has_hammer: [false; NUM_KEYS],
                key_has_mid_ev: [false; NUM_KEYS],
                key_alpha_post: [0.0; NUM_KEYS],
                was_nz: [false; TOTAL_STRINGS],
                was_tail_nz: [false; TOTAL_STRINGS],
                str_coupled: [false; TOTAL_STRINGS],
                coupled_strs: [0; TOTAL_STRINGS],
                active_keys: [0; NUM_KEYS],
                sav_long_slices: [SavLongSlice::ZERO; TOTAL_STRINGS],
                sav_strings: [0; TOTAL_STRINGS],
                sav_coupled_damp_strs: [0; TOTAL_STRINGS],
                hammer_forces: [HammerStepForce::ZERO; TOTAL_STRINGS],
                tail_partial_ports: [TailPortPartial::ZERO; TOTAL_STRINGS],
                symp_gv: [0.0; TOTAL_STRINGS],
                symp_gh: [0.0; TOTAL_STRINGS],
                symp_mask: [0.0; TOTAL_STRINGS],
                symp_y_lim: [0.0; TOTAL_STRINGS],
                symp_has_gains: [false; TOTAL_STRINGS],
                symp_dirty_strs: [0; TOTAL_STRINGS],
                zone_tail_n: [[0.0; 8]; SUB],
                n_a: 0,
                n_sl: 0,
                n_sav: 0,
                n_sav_cd: 0,
                n_coupled_strs: 0,
                n_symp_dirty: 0,
            },
            peel_buf: [[0.0_f32; SUB]; MAX_PEEL_SLOTS],
            peel_slot: [u8::MAX; DAMPED_KEYS],
            b_waves: [0.0_f32; TOTAL_STRINGS],
            zone_out: [0.0_f32; 8],
            dup_zone: [0.0_f32; 8],
            acc_th: [F32x4::zero(); SUB],
            acc_l: [F32x4::zero(); SUB],
        })
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_all_strings_prefix_wdf(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    a_waves: &mut [f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
    out_pair: (&mut [f32], &mut [f32]),
    ts: usize,
    peel_slot: &[u8; DAMPED_KEYS],
    peel_buf: &[[f32; SUB]; MAX_PEEL_SLOTS],
) {
    let n_cp = plan.n_coupled_strs;
    let (ul, ur) =
        step_uncoupled_prefixes_fast(r.bridge_ports, r.ports, plan, ts, a_waves, b_waves);
    out_pair.0[ts] += ul;
    out_pair.1[ts] += ur;

    if n_cp > 0 {
        step_coupled_prefixes(
            r, driver, plan, n_cp, a_waves, b_waves, out_pair, ts, peel_slot, peel_buf,
        );
    }

    if r.sav_on && plan.n_sav > 0 {
        step_sav_scheduled(
            r.bridge_ports,
            r.sav,
            &plan.sav_strings[..plan.n_sav],
            r.long_on,
        );
        for &s_u8 in &plan.sav_coupled_damp_strs[..plan.n_sav_cd] {
            let s_idx = usize::from(s_u8);
            if !r.ports.may_be_nonzero[s_idx] {
                continue;
            }
            let k = usize::from(r.strings[s_idx].key_idx);
            if plan.key_has_hammer[k] {
                continue;
            }
            if plan.str_coupled[s_idx] || plan.symp_has_gains[s_idx] {
                let m = &r.bridge_ports.meta[s_idx];
                let vdv_br =
                    accum_prefix_damp_vel_bp(r.bridge_ports, r.ports, m.beta_v_off, m.v_br.clone());
                let vdh_br =
                    accum_prefix_damp_vel_bp(r.bridge_ports, r.ports, m.beta_h_off, m.h_br.clone());
                let (tv, th) = if plan.key_cat[k] == KeyCat::ActiveContact {
                    let tp = plan.tail_partial_ports[s_idx];
                    (tp.vdv_raw, tp.vdh_raw)
                } else {
                    (0.0, 0.0)
                };
                let alpha = r.ports.alpha_vel[s_idx];
                r.ports.v_dv[s_idx] = alpha * (vdv_br + tv);
                r.ports.v_dh[s_idx] = alpha * (vdh_br + th);
            }
        }
    }
    for idx in 0..plan.n_a {
        let k = usize::from(plan.active_keys[idx]);
        if plan.key_has_hammer[k] {
            let alpha_post = plan.key_alpha_post[k];
            for s_idx in r.keys[k].string_range.clone() {
                if r.ports.may_be_nonzero[s_idx] {
                    let hf = plan.hammer_forces[s_idx];
                    let sync_damper = hf.fd_v != 0.0 || hf.fd_h != 0.0;
                    refresh_prefix_ports_with_tail_bp(
                        r.bridge_ports,
                        r.ports,
                        s_idx,
                        r.strings[s_idx].has_damper,
                        sync_damper,
                        alpha_post,
                        &plan.tail_partial_ports[s_idx],
                    );
                }
            }
        }
    }
}
