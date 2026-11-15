//! Stage 2 <-> Stage 3 key-batched tail/damper/hammer helpers (`CHUNK = 128`, `1x` active contact).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::{NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::BridgePorts;
use crate::bridge_sav::SavSchedule;
use crate::choir_map::{KeyMapEntry, StringMapEntry};
use crate::contact::{eval_hammer_choir, sample_damper, KeyActionDriver};
use crate::contact_ports::{refresh_string_ports_bp, ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::{accum_soa_range_in, step_string_modes, zoh_rotate};

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

#[derive(Copy, Clone)]
pub(crate) struct SavLongSlice {
    pub s_idx: u8,
    pub start: u16,
    pub end: u16,
}

impl SavLongSlice {
    pub const ZERO: Self = Self {
        s_idx: 0,
        start: 0,
        end: 0,
    };
}

#[derive(Copy, Clone)]
pub(crate) struct TailDampPartial {
    pub ydv: f32,
    pub vdv_raw: f32,
    pub ydh: f32,
    pub vdh_raw: f32,
}

impl TailDampPartial {
    pub const ZERO: Self = Self {
        ydv: 0.0,
        vdv_raw: 0.0,
        ydh: 0.0,
        vdh_raw: 0.0,
    };
}

pub(crate) struct ChunkPlan {
    pub key_cat: [KeyCat; NUM_KEYS],
    pub key_has_hammer: [bool; NUM_KEYS],
    pub key_has_mid_ev: [bool; NUM_KEYS],
    pub was_nz: [bool; TOTAL_STRINGS],
    pub was_tail_nz: [bool; TOTAL_STRINGS],
    pub str_coupled: [bool; TOTAL_STRINGS],
    pub coupled_strs: [u8; TOTAL_STRINGS],
    pub active_keys: [u8; NUM_KEYS],
    pub sav_long_slices: [SavLongSlice; TOTAL_STRINGS],
    pub sav_strings: [u8; TOTAL_STRINGS],
    pub sav_sched: SavSchedule,
    pub sav_sched_dirty: bool,
    pub hammer_forces: [HammerStepForce; TOTAL_STRINGS],
    pub tail_partial_damp: [TailDampPartial; TOTAL_STRINGS],
    pub zone_tail_n: [[f32; 8]; SUB],
    pub n_a: usize,
    pub n_sl: usize,
    pub n_sav: usize,
    pub n_coupled_strs: usize,
}

pub(crate) struct BridgeRun<'a> {
    pub ports: &'a mut ContactPorts,
    pub bridge_ports: &'a mut BridgePorts,
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
fn step_tail_pol_tl(
    tl_soa: &mut ModalSoa,
    ports: &ContactPorts,
    tl_r: Range<usize>,
    port_r: Range<usize>,
    f_h: f32,
    f_d: f32,
    has_damper: bool,
) -> f32 {
    if tl_r.is_empty() {
        return 0.0;
    }
    if f_h == 0.0 && f_d == 0.0 {
        return step_string_modes(
            &mut tl_soa.u.as_f32x4_slice_mut()[tl_r.clone()],
            &mut tl_soa.v.as_f32x4_slice_mut()[tl_r.clone()],
            &tl_soa.d_coeff.as_f32x4_slice()[tl_r.clone()],
            &tl_soa.s_coeff.as_f32x4_slice()[tl_r.clone()],
            &tl_soa.c_bridge.as_f32x4_slice()[tl_r],
        );
    }
    let (u, v) = (tl_soa.u.as_f32x4_slice_mut(), tl_soa.v.as_f32x4_slice_mut());
    let (u_sl, v_sl) = (&mut u[tl_r.clone()], &mut v[tl_r.clone()]);
    let d = &tl_soa.d_coeff.as_f32x4_slice()[tl_r.clone()];
    let s = &tl_soa.s_coeff.as_f32x4_slice()[tl_r.clone()];
    let c = &tl_soa.c_bridge.as_f32x4_slice()[tl_r];
    let bh = &ports.b_strike.as_f32x4_slice()[port_r.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[port_r];
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

#[inline(always)]
fn step_tail_pol_damp_tl(
    tl_soa: &mut ModalSoa,
    ports: &ContactPorts,
    tl_r: Range<usize>,
    port_r: Range<usize>,
    f_d: f32,
) -> (f32, f32, f32) {
    if tl_r.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    if f_d != 0.0 {
        step_tail_pol_damp_tl_inner::<true>(tl_soa, ports, tl_r, port_r, f_d)
    } else {
        step_tail_pol_damp_tl_inner::<false>(tl_soa, ports, tl_r, port_r, 0.0)
    }
}

#[inline(always)]
fn step_tail_pol_damp_tl_inner<const HAS_FD: bool>(
    tl_soa: &mut ModalSoa,
    ports: &ContactPorts,
    tl_r: Range<usize>,
    port_r: Range<usize>,
    f_d: f32,
) -> (f32, f32, f32) {
    let (u_all, v_all) = (tl_soa.u.as_f32x4_slice_mut(), tl_soa.v.as_f32x4_slice_mut());
    let (u_sl, v_sl) = (&mut u_all[tl_r.clone()], &mut v_all[tl_r.clone()]);
    let d = &tl_soa.d_coeff.as_f32x4_slice()[tl_r.clone()];
    let s = &tl_soa.s_coeff.as_f32x4_slice()[tl_r.clone()];
    let c = &tl_soa.c_bridge.as_f32x4_slice()[tl_r];
    let bd = &ports.b_damp.as_f32x4_slice()[port_r.clone()];
    let ad = &ports.a_damp.as_f32x4_slice()[port_r];
    let fd_v = F32x4::splat(f_d);
    let (mut acc_c, mut acc_yd, mut acc_vd) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in 0..u_sl.len() {
        let bdi = bd[i];
        let u_inj = if HAS_FD {
            fd_v.mul_add(bdi, u_sl[i])
        } else {
            u_sl[i]
        };
        let (un, vn) = zoh_rotate(u_inj, v_sl[i], d[i], s[i]);
        (u_sl[i], v_sl[i]) = (un, vn);
        acc_c = vn.mul_add(c[i], acc_c);
        acc_yd = vn.mul_add(ad[i], acc_yd);
        acc_vd = un.mul_add(bdi, acc_vd);
    }
    (acc_c.reduce_sum(), acc_yd.reduce_sum(), acc_vd.reduce_sum())
}

pub(crate) fn step_uncoupled_tails(
    r: &mut BridgeRun<'_>,
    plan: &mut ChunkPlan,
    sub_l: &mut [f32],
    sub_r: &mut [f32],
) {
    let sub = sub_l.len();
    let mut acc_th = [F32x4::zero(); SUB];
    let mut acc_l = [F32x4::zero(); SUB];
    let mut flat_l = [F32x4::zero(); SUB];
    let mut flat_r = [F32x4::zero(); SUB];

    for k in 0..NUM_KEYS {
        if !matches!(plan.key_cat[k], KeyCat::Uncoupled | KeyCat::DampedSymp) {
            continue;
        }
        let kt = r.bridge_ports.key_tails[k];
        let sr = r.keys[k].string_range.clone();
        let any_tail_nz = plan.was_tail_nz[sr.clone()].iter().any(|&b| b);
        let (th_s, th_e, l_e) = (
            usize::from(kt.th_start),
            usize::from(kt.th_end),
            usize::from(kt.l_end),
        );

        if !any_tail_nz {
            let end_v = if r.long_on { l_e } else { th_e };
            acc_th[..sub].fill(F32x4::zero());
            accum_soa_range_in(&mut r.bridge_ports.tl_soa, th_s..end_v, &mut acc_th[..sub]);
            let (pl_v, pr_v) = (F32x4::splat(kt.pan_l), F32x4::splat(kt.pan_r));
            for t in 0..sub {
                let a = acc_th[t];
                flat_l[t] = a.mul_add(pl_v, flat_l[t]);
                flat_r[t] = a.mul_add(pr_v, flat_r[t]);
            }
            if !r.long_on && th_e < l_e {
                acc_l[..sub].fill(F32x4::zero());
                accum_soa_range_in(&mut r.bridge_ports.tl_soa, th_e..l_e, &mut acc_l[..sub]);
            }
            continue;
        }

        acc_th[..sub].fill(F32x4::zero());
        accum_soa_range_in(&mut r.bridge_ports.tl_soa, th_s..th_e, &mut acc_th[..sub]);

        let has_l = th_e < l_e;
        if has_l {
            acc_l[..sub].fill(F32x4::zero());
            for s_idx in sr {
                let nl_sav = sav_long_vecs(r, s_idx);
                let tl_l = r.bridge_ports.meta[s_idx].tl_l.clone();
                accum_soa_range_in(
                    &mut r.bridge_ports.tl_soa,
                    (tl_l.start + nl_sav)..tl_l.end,
                    &mut acc_l[..sub],
                );
            }
        }

        let zw = r.zone_weights[r.keys[k].string_range.start];
        let (nt, nl, use_l) = (kt.tail_n_trans, kt.tail_n_long, has_l && r.long_on);
        for t in 0..sub {
            let th_v = acc_th[t].reduce_sum();
            let fl = if use_l { acc_l[t].reduce_sum() } else { 0.0 };
            let audio = th_v + fl;
            sub_l[t] += audio * kt.pan_l;
            sub_r[t] += audio * kt.pan_r;
            add_zone_n(&mut plan.zone_tail_n[t], zw, th_v * nt + fl * nl);
        }
    }
    for t in 0..sub {
        sub_l[t] += flat_l[t].reduce_sum();
        sub_r[t] += flat_r[t].reduce_sum();
    }
}

pub(crate) fn prepare_sample_tails_and_hammers(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    chunk_base: usize,
    t: usize,
    sub_l: &mut [f32],
    sub_r: &mut [f32],
) {
    let mut note_on_mask = driver.apply_sample_events_at(chunk_base + t);
    while note_on_mask != 0 {
        let k = note_on_mask.trailing_zeros() as usize;
        note_on_mask &= note_on_mask - 1;
        if k < NUM_KEYS {
            for s_idx in r.keys[k].string_range.clone() {
                r.ports.may_be_nonzero[s_idx] = true;
                r.ports.ports_synced[s_idx] = false;
            }
        }
    }
    if plan.n_sl > 0 {
        let tl_soa = &mut r.bridge_ports.tl_soa;
        let (u_all, v_all) = (tl_soa.u.as_f32x4_slice_mut(), tl_soa.v.as_f32x4_slice_mut());
        let d_all = tl_soa.d_coeff.as_f32x4_slice();
        let s_all = tl_soa.s_coeff.as_f32x4_slice();
        let c_all = tl_soa.c_bridge.as_f32x4_slice();
        for idx in 0..plan.n_sl {
            let sl = plan.sav_long_slices[idx];
            let (start, end) = (usize::from(sl.start), usize::from(sl.end));
            let mut acc = F32x4::zero();
            for i in start..end {
                let (un, vn) = zoh_rotate(u_all[i], v_all[i], d_all[i], s_all[i]);
                u_all[i] = un;
                v_all[i] = vn;
                acc = vn.mul_add(c_all[i], acc);
            }
            let tl = acc.reduce_sum();
            let s_idx = usize::from(sl.s_idx);
            let entry = &r.strings[s_idx];
            sub_l[t] += tl * entry.pan_l;
            sub_r[t] += tl * entry.pan_r;
            add_zone_n(
                &mut plan.zone_tail_n[t],
                r.zone_weights[s_idx],
                tl * r.bridge_ports.tail_n_long[s_idx],
            );
        }
    }
    for idx in 0..plan.n_a {
        let k = usize::from(plan.active_keys[idx]);
        let key = &r.keys[k];
        let sr = key.string_range.clone();
        for s_idx in sr.clone() {
            if r.ports.may_be_nonzero[s_idx] && !r.ports.ports_synced[s_idx] {
                refresh_string_ports_bp(
                    r.bridge_ports,
                    r.ports,
                    s_idx,
                    r.strings[s_idx].has_damper,
                );
            }
        }
        if plan.key_has_mid_ev[k] {
            plan.key_has_hammer[k] = driver.is_hammer_active(k);
            r.ports.y_clear_scratch[k][t] = if key.has_damper {
                driver.step_key_clearance(k)
            } else {
                f32::MAX
            };
        }
        let has_hammer = plan.key_has_hammer[k];
        let fh_arr = if has_hammer {
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
            if fh != 0.0 || fd_v != 0.0 || fd_h != 0.0 {
                r.ports.tail_nonzero[s_idx] = true;
            }
            let entry = &r.strings[s_idx];
            let (tl_v, v_tail, tl_h, h_tail, tl_l, nt, nl_f) = {
                let m = &r.bridge_ports.meta[s_idx];
                (
                    m.tl_v.clone(),
                    m.v_tail.clone(),
                    m.tl_h.clone(),
                    m.h_tail.clone(),
                    m.tl_l.clone(),
                    r.bridge_ports.tail_n_trans[s_idx],
                    r.bridge_ports.tail_n_long[s_idx],
                )
            };
            let (tv, th) = if has_hammer {
                (
                    step_tail_pol_tl(
                        &mut r.bridge_ports.tl_soa,
                        r.ports,
                        tl_v,
                        v_tail,
                        fh,
                        fd_v,
                        key.has_damper,
                    ),
                    step_tail_pol_tl(
                        &mut r.bridge_ports.tl_soa,
                        r.ports,
                        tl_h,
                        h_tail,
                        fh,
                        fd_h,
                        key.has_damper,
                    ),
                )
            } else {
                let (tv, ydv, vdv_raw) =
                    step_tail_pol_damp_tl(&mut r.bridge_ports.tl_soa, r.ports, tl_v, v_tail, fd_v);
                let (th, ydh, vdh_raw) =
                    step_tail_pol_damp_tl(&mut r.bridge_ports.tl_soa, r.ports, tl_h, h_tail, fd_h);
                plan.tail_partial_damp[s_idx] = TailDampPartial {
                    ydv,
                    vdv_raw,
                    ydh,
                    vdh_raw,
                };
                (tv, th)
            };
            let tl = if tl_l.is_empty() {
                0.0
            } else {
                if r.long_on && fh > 0.0 {
                    let lv = &mut r.bridge_ports.tl_soa.u.as_f32x4_slice_mut()[tl_l.clone()];
                    r.long_bank.inject_precursor(s_idx, tl_l.len(), fh, lv);
                }
                let tl_soa = &mut r.bridge_ports.tl_soa;
                let raw_l = step_string_modes(
                    &mut tl_soa.u.as_f32x4_slice_mut()[tl_l.clone()],
                    &mut tl_soa.v.as_f32x4_slice_mut()[tl_l.clone()],
                    &tl_soa.d_coeff.as_f32x4_slice()[tl_l.clone()],
                    &tl_soa.s_coeff.as_f32x4_slice()[tl_l.clone()],
                    &tl_soa.c_bridge.as_f32x4_slice()[tl_l],
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
            add_zone_n(
                &mut plan.zone_tail_n[t],
                r.zone_weights[s_idx],
                th_sum * nt + tl * nl_f,
            );
        }
    }
}
