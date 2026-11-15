//! Stage 2 <-> Stage 3 tail/damper/hammer helpers (`CHUNK = 128`, `1x` per-sample active contact).

use core::ops::Range;

use piano_spec::{NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::BridgePorts;
use crate::choir_map::{KeyMapEntry, StringMapEntry};
use crate::contact::{eval_hammer_choir, sample_damper, KeyActionDriver};
use crate::contact_ports::{refresh_string_ports, ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::{
    step_string_modes, step_string_subblock, step_string_subblock_3, zoh_rotate,
};
use piano_simd::F32x4;

pub(crate) const SUB: usize = 128;

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum KeyCat {
    Uncoupled = 0,
    DampedSymp = 1,
    ActiveContact = 2,
}

#[derive(Copy, Clone)]
pub(crate) struct HammerStepForce {
    pub fh: f32,
    pub fd_v: f32,
    pub fd_h: f32,
}

impl HammerStepForce {
    pub const ZERO: Self = Self {
        fh: 0.0,
        fd_v: 0.0,
        fd_h: 0.0,
    };
}

pub(crate) struct ChunkPlan {
    pub key_cat: [KeyCat; NUM_KEYS],
    pub key_has_hammer: [bool; NUM_KEYS],
    pub was_nz: [bool; TOTAL_STRINGS],
    pub active_keys: [u8; NUM_KEYS],
    pub sav_long_strings: [u8; TOTAL_STRINGS],
    pub hammer_forces: [HammerStepForce; TOTAL_STRINGS],
    pub zone_tail_n: [[f32; 8]; SUB],
    pub n_a: usize,
    pub n_sl: usize,
}

pub(crate) struct BridgeRun<'a> {
    pub soa: &'a mut ModalSoa,
    pub ports: &'a mut ContactPorts,
    pub bridge_ports: &'a BridgePorts,
    pub long_bank: &'a LongitudinalBank,
    pub sav: &'a mut SavBank,
    pub keys: &'a [KeyMapEntry],
    pub strings: &'a [StringMapEntry],
    pub zone_weights: [(u8, f32, u8, f32); TOTAL_STRINGS],
    pub long_on: bool,
    pub sav_on: bool,
}

#[inline]
pub(crate) fn add_zone_n(zone_buf: &mut [f32; 8], zw: (u8, f32, u8, f32), force_n: f32) {
    zone_buf[usize::from(zw.0) & 7] += force_n * zw.1;
    zone_buf[usize::from(zw.2) & 7] += force_n * zw.3;
}

#[inline]
pub(crate) fn sav_long_vecs(r: &BridgeRun<'_>, s_idx: usize) -> usize {
    if !r.ports.struck_active[s_idx] || !r.sav_on || !r.long_on || r.sav.string_n_vh(s_idx) == 0 {
        return 0;
    }
    let (_, _, nl, m_ph) = r.sav.string_prefix_bounds(s_idx, true);
    if m_ph > 0 {
        nl
    } else {
        0
    }
}

#[inline]
fn step_tail_pol(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    f_h: f32,
    f_d: f32,
    has_damper: bool,
) -> f32 {
    if vr.is_empty() {
        return 0.0;
    }
    if f_h == 0.0 && f_d == 0.0 {
        return step_string_modes(
            &mut soa.u.as_f32x4_slice_mut()[vr.clone()],
            &mut soa.v.as_f32x4_slice_mut()[vr.clone()],
            &soa.d_coeff.as_f32x4_slice()[vr.clone()],
            &soa.s_coeff.as_f32x4_slice()[vr.clone()],
            &soa.c_bridge.as_f32x4_slice()[vr],
        );
    }
    let (u, v) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let (u_sl, v_sl) = (&mut u[vr.clone()], &mut v[vr.clone()]);
    let d = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c = &soa.c_bridge.as_f32x4_slice()[vr.clone()];
    let bh = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr];
    let (fh_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(f_d));
    let mut acc = F32x4::zero();
    for i in 0..u_sl.len() {
        let base = fh_v.mul_add(bh[i], u_sl[i]);
        let u_inj = if has_damper {
            fd_v.mul_add(bd[i], base)
        } else {
            base
        };
        let (un, vn) = zoh_rotate(u_inj, v_sl[i], d[i], s[i]);
        (u_sl[i], v_sl[i]) = (un, vn);
        acc = vn.mul_add(c[i], acc);
    }
    acc.reduce_sum()
}

pub(crate) fn step_uncoupled_tails(
    r: &mut BridgeRun<'_>,
    plan: &mut ChunkPlan,
    sub_l: &mut [f32],
    sub_r: &mut [f32],
) {
    let sub = sub_l.len();
    let (mut th_mono, mut l_mono, mut dummy_r) = ([0.0_f32; SUB], [0.0_f32; SUB], [0.0_f32; SUB]);
    for k in 0..NUM_KEYS {
        if !matches!(plan.key_cat[k], KeyCat::Uncoupled | KeyCat::DampedSymp) {
            continue;
        }
        for s_idx in r.keys[k].string_range.clone() {
            let entry = &r.strings[s_idx];
            let m = &r.bridge_ports.meta[s_idx];
            let nl_sav = sav_long_vecs(r, s_idx);
            let l_tail = (entry.long_vecs.start + nl_sav)..entry.long_vecs.end;
            if !plan.was_nz[s_idx] {
                let l_act = if r.long_on { l_tail.clone() } else { 0..0 };
                step_string_subblock_3(
                    r.soa,
                    [m.v_tail.clone(), m.h_tail.clone(), l_act],
                    entry.pan_l,
                    entry.pan_r,
                    sub_l,
                    sub_r,
                );
                if !r.long_on {
                    step_string_subblock(r.soa, l_tail, 0.0, 0.0, sub_l, sub_r);
                }
                continue;
            }
            th_mono[..sub].fill(0.0);
            step_string_subblock_3(
                r.soa,
                [m.v_tail.clone(), m.h_tail.clone(), 0..0],
                1.0,
                0.0,
                &mut th_mono[..sub],
                &mut dummy_r[..sub],
            );
            let has_l = !l_tail.is_empty();
            if has_l {
                l_mono[..sub].fill(0.0);
                let pl = if r.long_on { 1.0 } else { 0.0 };
                step_string_subblock(
                    r.soa,
                    l_tail,
                    pl,
                    0.0,
                    &mut l_mono[..sub],
                    &mut dummy_r[..sub],
                );
            }
            let (zw, nt, nl) = (
                r.zone_weights[s_idx],
                r.bridge_ports.tail_n_trans[s_idx],
                r.bridge_ports.tail_n_long[s_idx],
            );
            for t in 0..sub {
                let fl = if has_l && r.long_on { l_mono[t] } else { 0.0 };
                let audio = th_mono[t] + fl;
                sub_l[t] += audio * entry.pan_l;
                sub_r[t] += audio * entry.pan_r;
                add_zone_n(&mut plan.zone_tail_n[t], zw, th_mono[t] * nt + fl * nl);
            }
        }
    }
}

pub(crate) fn prepare_sample_tails_and_hammers(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    t: usize,
    sub_l: &mut [f32],
    sub_r: &mut [f32],
) {
    for idx in 0..plan.n_sl {
        let s_idx = usize::from(plan.sav_long_strings[idx]);
        let entry = &r.strings[s_idx];
        let nl = r.sav.string_prefix_bounds(s_idx, true).2;
        let lr = entry.long_vecs.start..(entry.long_vecs.start + nl);
        let tl = step_string_modes(
            &mut r.soa.u.as_f32x4_slice_mut()[lr.clone()],
            &mut r.soa.v.as_f32x4_slice_mut()[lr.clone()],
            &r.soa.d_coeff.as_f32x4_slice()[lr.clone()],
            &r.soa.s_coeff.as_f32x4_slice()[lr.clone()],
            &r.soa.c_bridge.as_f32x4_slice()[lr],
        );
        sub_l[t] += tl * entry.pan_l;
        sub_r[t] += tl * entry.pan_r;
        add_zone_n(
            &mut plan.zone_tail_n[t],
            r.zone_weights[s_idx],
            tl * r.bridge_ports.tail_n_long[s_idx],
        );
    }
    for idx in 0..plan.n_a {
        let k = usize::from(plan.active_keys[idx]);
        let key = &r.keys[k];
        let sr = key.string_range.clone();
        for s_idx in sr.clone() {
            if r.ports.may_be_nonzero[s_idx] && !r.ports.ports_synced[s_idx] {
                refresh_string_ports(r.soa, r.ports, s_idx, &r.strings[s_idx]);
            }
        }
        let fh_arr = if plan.key_has_hammer[k] {
            eval_hammer_choir(driver, r.ports, k, sr.clone(), usize::from(key.choir_size))
        } else {
            [0.0; 3]
        };
        let yc = r.ports.y_clear_scratch[k][t];
        let damp_on = key.has_damper && yc < DAMPER_CLEAR_THRESHOLD_M;
        for (c, s_idx) in sr.enumerate() {
            let fh = fh_arr[c];
            let (fd_v, fd_h) = if damp_on && r.ports.may_be_nonzero[s_idx] {
                sample_damper(driver, r.ports, k, s_idx, yc)
            } else {
                (0.0, 0.0)
            };
            plan.hammer_forces[s_idx] = HammerStepForce { fh, fd_v, fd_h };
            let entry = &r.strings[s_idx];
            let m = &r.bridge_ports.meta[s_idx];
            let tv = step_tail_pol(r.soa, r.ports, m.v_tail.clone(), fh, fd_v, key.has_damper);
            let th = step_tail_pol(r.soa, r.ports, m.h_tail.clone(), fh, fd_h, key.has_damper);
            let rl = entry.long_vecs.clone();
            let tl = if rl.is_empty() {
                0.0
            } else {
                if r.long_on && fh > 0.0 {
                    let lv = &mut r.soa.u.as_f32x4_slice_mut()[rl.clone()];
                    r.long_bank.inject_precursor(s_idx, rl.len(), fh, lv);
                }
                let raw_l = step_string_modes(
                    &mut r.soa.u.as_f32x4_slice_mut()[rl.clone()],
                    &mut r.soa.v.as_f32x4_slice_mut()[rl.clone()],
                    &r.soa.d_coeff.as_f32x4_slice()[rl.clone()],
                    &r.soa.s_coeff.as_f32x4_slice()[rl.clone()],
                    &r.soa.c_bridge.as_f32x4_slice()[rl],
                );
                if r.long_on {
                    raw_l
                } else {
                    0.0
                }
            };
            let th_sum = tv + th;
            let audio = th_sum + tl;
            sub_l[t] += audio * entry.pan_l;
            sub_r[t] += audio * entry.pan_r;
            let force_n = th_sum * r.bridge_ports.tail_n_trans[s_idx]
                + tl * r.bridge_ports.tail_n_long[s_idx];
            add_zone_n(&mut plan.zone_tail_n[t], r.zone_weights[s_idx], force_n);
        }
    }
}
