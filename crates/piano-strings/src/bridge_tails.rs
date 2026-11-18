//! Stage 2 <-> Stage 3 key-batched tail/damper/hammer helpers (`CHUNK = 128`, `1x` active contact).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::{NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::BridgePorts;
pub(crate) use crate::bridge_sample_prep::prepare_sample_tails_and_hammers;
use crate::choir_map::{KeyMapEntry, StringMapEntry};
use crate::contact::KeyActionDriver;
use crate::contact_ports::{refresh_prefix_damp_ports_bp, ContactPorts};
use crate::longitudinal::LongitudinalBank;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::accum_soa_range_in;

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

#[rustfmt::skip]
impl HammerStepForce {
    pub const ZERO: Self = Self { fh: 0.0, fd_v: 0.0, fd_h: 0.0 };
}

#[derive(Copy, Clone)]
pub(crate) struct SavLongSlice {
    pub s_idx: u8,
    pub start: u16,
    pub end: u16,
}

#[rustfmt::skip]
impl SavLongSlice {
    pub const ZERO: Self = Self { s_idx: 0, start: 0, end: 0 };
}

#[derive(Copy, Clone)]
pub(crate) struct TailDampPartial {
    pub ydv: f32,
    pub vdv_raw: f32,
    pub ydh: f32,
    pub vdh_raw: f32,
}

#[rustfmt::skip]
impl TailDampPartial {
    pub const ZERO: Self = Self { ydv: 0.0, vdv_raw: 0.0, ydh: 0.0, vdh_raw: 0.0 };
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
    pub hammer_forces: [HammerStepForce; TOTAL_STRINGS],
    pub tail_partial_damp: [TailDampPartial; TOTAL_STRINGS],
    pub symp_gv: [f32; TOTAL_STRINGS],
    pub symp_gh: [f32; TOTAL_STRINGS],
    pub symp_mask: [f32; TOTAL_STRINGS],
    pub symp_y_lim: [f32; TOTAL_STRINGS],
    pub symp_has_gains: [bool; TOTAL_STRINGS],
    pub zone_tail_n: [[f32; 8]; SUB],
    pub n_a: usize,
    pub n_sl: usize,
    pub n_sav: usize,
    pub n_coupled_strs: usize,
}

impl ChunkPlan {
    #[inline(always)]
    pub fn push_coupled(&mut self, s_idx: usize) {
        self.str_coupled[s_idx] = true;
        self.coupled_strs[self.n_coupled_strs] = s_idx as u8;
        self.n_coupled_strs += 1;
    }
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
    pub lossless: bool,
}

#[inline]
pub(crate) fn add_zone_n(zone_buf: &mut [f32; 8], zw: (u8, f32, u8, f32), force_n: f32) {
    zone_buf[usize::from(zw.0) & 7] += force_n * zw.1;
    zone_buf[usize::from(zw.2) & 7] += force_n * zw.3;
}

#[inline]
pub(crate) fn sav_long_vecs(r: &BridgeRun<'_>, s_idx: usize) -> usize {
    if !r.ports.struck_active[s_idx]
        || r.ports.sav_dormant[s_idx]
        || !r.sav_on
        || !r.long_on
        || r.sav.string_n_vh(s_idx) == 0
    {
        return 0;
    }
    let (_, _, nl, m_ph) = r.sav.string_prefix_bounds(s_idx, true);
    if m_ph > 0 {
        nl
    } else {
        0
    }
}

/// Returns `true` when damper clearance `min_c > -1.4e-6 m` exceeds the phase-invariant modal
/// amplitude envelope `sum_m sqrt(u_m^2 + v_m^2) |a_damp,m|` across all strings in `sr`.
#[inline]
pub(crate) fn key_clear_of_open_damper(
    r: &BridgeRun<'_>,
    was_nz: &[bool; TOTAL_STRINGS],
    sr: Range<usize>,
    min_c: f32,
) -> bool {
    let eff_gap = if min_c >= 0.0 {
        min_c + 3.0e-6
    } else if min_c > -1.4e-6 {
        3.0e-6 + 2.0 * min_c
    } else {
        return false;
    };
    let lim = 0.92 * eff_gap - 8.0e-8;
    if lim <= 0.0 {
        return false;
    }
    let bp = &*r.bridge_ports;
    let ad_tail = r.ports.a_damp.as_f32x4_slice();
    let tu = bp.tl_soa.u.as_f32x4_slice();
    let tv = bp.tl_soa.v.as_f32x4_slice();
    for s_idx in sr {
        if !was_nz[s_idx] {
            continue;
        }
        let m = &bp.meta[s_idx];
        let (nv, nh) = (m.v_br.len(), m.h_br.len());
        let has_td = r.ports.tail_nonzero[s_idx] && r.ports.tl_damp_active[s_idx];
        let tv_n = if has_td {
            usize::from(r.ports.tl_v_active[s_idx]).min(m.tl_v.len())
        } else {
            0
        };
        let th_n = if has_td {
            usize::from(r.ports.tl_h_active[s_idx]).min(m.tl_h.len())
        } else {
            0
        };
        // Fast sqrt-free L1 bound: sqrt(u^2 + v^2) <= |u| + |v| <= sqrt(2) * sqrt(u^2 + v^2).
        let (mut l1_v, mut l1_h) = (F32x4::zero(), F32x4::zero());
        for i in 0..nv {
            let st = bp.state[m.beta_v_off + i];
            l1_v = (st.u.abs() + st.v.abs()).mul_add(bp.damp_ad[m.beta_v_off + i].abs(), l1_v);
        }
        for i in 0..nh {
            let st = bp.state[m.beta_h_off + i];
            l1_h = (st.u.abs() + st.v.abs()).mul_add(bp.damp_ad[m.beta_h_off + i].abs(), l1_h);
        }
        for i in 0..tv_n {
            let (u, v) = (tu[m.tl_v.start + i], tv[m.tl_v.start + i]);
            l1_v = (u.abs() + v.abs()).mul_add(ad_tail[m.v_tail.start + i].abs(), l1_v);
        }
        for i in 0..th_n {
            let (u, v) = (tu[m.tl_h.start + i], tv[m.tl_h.start + i]);
            l1_h = (u.abs() + v.abs()).mul_add(ad_tail[m.h_tail.start + i].abs(), l1_h);
        }
        let (sv, sh) = (l1_v.reduce_sum(), l1_h.reduce_sum());
        if sv <= lim && sh <= lim {
            continue;
        }
        if sv > 1.42 * lim || sh > 1.42 * lim {
            return false;
        }
        let mut env_v = F32x4::zero();
        for i in 0..nv {
            let st = bp.state[m.beta_v_off + i];
            let amp = st.u.mul_add(st.u, st.v * st.v).sqrt();
            env_v = amp.mul_add(bp.damp_ad[m.beta_v_off + i].abs(), env_v);
        }
        let mut env_h = F32x4::zero();
        for i in 0..nh {
            let st = bp.state[m.beta_h_off + i];
            let amp = st.u.mul_add(st.u, st.v * st.v).sqrt();
            env_h = amp.mul_add(bp.damp_ad[m.beta_h_off + i].abs(), env_h);
        }
        for i in 0..tv_n {
            let (u, v) = (tu[m.tl_v.start + i], tv[m.tl_v.start + i]);
            env_v = u
                .mul_add(u, v * v)
                .sqrt()
                .mul_add(ad_tail[m.v_tail.start + i].abs(), env_v);
        }
        for i in 0..th_n {
            let (u, v) = (tu[m.tl_h.start + i], tv[m.tl_h.start + i]);
            env_h = u
                .mul_add(u, v * v)
                .sqrt()
                .mul_add(ad_tail[m.h_tail.start + i].abs(), env_h);
        }
        if env_v.reduce_sum() > lim || env_h.reduce_sum() > lim {
            return false;
        }
    }
    true
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn damp_pol_l1_and_env(
    bp_st: &[crate::bridge_port::PrefixState],
    damp_bd: &[F32x4],
    damp_ad: &[F32x4],
    tu: &[F32x4],
    tv: &[F32x4],
    bd_tail: &[F32x4],
    ad_tail: &[F32x4],
    kv: F32x4,
    lim: f32,
) -> Option<bool> {
    let mut l1 = F32x4::zero();
    for ((st, bd), ad) in bp_st.iter().zip(damp_bd).zip(damp_ad) {
        l1 = (st.u.abs() + st.v.abs()).mul_add(kv.mul_add(bd.abs(), ad.abs()), l1);
    }
    for (((&u, &v), bd), ad) in tu.iter().zip(tv).zip(bd_tail).zip(ad_tail) {
        l1 = (u.abs() + v.abs()).mul_add(kv.mul_add(bd.abs(), ad.abs()), l1);
    }
    let s = l1.reduce_sum();
    if s <= lim {
        return Some(false);
    }
    if s > 1.42 * lim {
        return Some(true);
    }
    let mut env = F32x4::zero();
    for ((st, bd), ad) in bp_st.iter().zip(damp_bd).zip(damp_ad) {
        let amp = st.u.mul_add(st.u, st.v * st.v).sqrt();
        env = amp.mul_add(kv.mul_add(bd.abs(), ad.abs()), env);
    }
    for (((&u, &v), bd), ad) in tu.iter().zip(tv).zip(bd_tail).zip(ad_tail) {
        let amp = u.mul_add(u, v * v).sqrt();
        env = amp.mul_add(kv.mul_add(bd.abs(), ad.abs()), env);
    }
    Some(env.reduce_sum() > lim)
}

/// Returns `true` when a key still requires per-sample `ActiveContact` transverse tail damping:
/// under deep preload (`min_c <= -12 um`) while tails remain active (`-120 dBFS` when closed
/// `<= -25 um`), and under shallow half-pedal (`min_c > -12 um`) while strongly grazing.
#[inline]
pub(crate) fn key_needs_active_tail(r: &BridgeRun<'_>, sr: Range<usize>, min_c: f32) -> bool {
    let closed_damp = min_c <= -2.5e-5;
    let shallow = min_c > -1.2e-5;
    let graze_lim = min_c.max(0.0) + 1.8e-5;
    let bp = &*r.bridge_ports;
    let (ad_tail, bd_tail) = (
        r.ports.a_damp.as_f32x4_slice(),
        r.ports.b_damp.as_f32x4_slice(),
    );
    let (tu, tv) = (bp.tl_soa.u.as_f32x4_slice(), bp.tl_soa.v.as_f32x4_slice());
    for s_idx in sr {
        if !r.ports.tail_nonzero[s_idx] || !(closed_damp || r.ports.tl_damp_active[s_idx]) {
            continue;
        }
        let m = &bp.meta[s_idx];
        let tv_n = usize::from(r.ports.tl_v_active[s_idx]).min(m.tl_v.len());
        let th_n = usize::from(r.ports.tl_h_active[s_idx]).min(m.tl_h.len());
        if (tv_n == 0 && th_n == 0) || !shallow {
            if !shallow && (tv_n > 0 || th_n > 0) {
                return true;
            }
            continue;
        }
        let kv = F32x4::splat(3.5e-4 * r.ports.alpha_vel[s_idx]);
        let (nv, nh) = (m.v_br.len(), m.h_br.len());
        let check_pol = |bo: usize, bn: usize, tlr: &Range<usize>, tr: &Range<usize>, tn: usize| {
            damp_pol_l1_and_env(
                &bp.state[bo..bo + bn],
                &bp.damp_bd[bo..bo + bn],
                &bp.damp_ad[bo..bo + bn],
                &tu[tlr.start..tlr.start + tn],
                &tv[tlr.start..tlr.start + tn],
                &bd_tail[tr.start..tr.start + tn],
                &ad_tail[tr.start..tr.start + tn],
                kv,
                graze_lim,
            ) == Some(true)
        };
        if check_pol(m.beta_v_off, nv, &m.tl_v, &m.v_tail, tv_n)
            || check_pol(m.beta_h_off, nh, &m.tl_h, &m.h_tail, th_n)
        {
            return true;
        }
    }
    false
}

#[inline]
fn prefix_damp_l1_below(r: &BridgeRun<'_>, s_idx: usize, lim: f32) -> bool {
    let bp = &*r.bridge_ports;
    let m = &bp.meta[s_idx];
    let (mut l1_v, mut l1_h) = (F32x4::zero(), F32x4::zero());
    for i in 0..m.v_br.len() {
        let st = bp.state[m.beta_v_off + i];
        l1_v = (st.u.abs() + st.v.abs()).mul_add(bp.damp_ad[m.beta_v_off + i].abs(), l1_v);
    }
    for i in 0..m.h_br.len() {
        let st = bp.state[m.beta_h_off + i];
        l1_h = (st.u.abs() + st.v.abs()).mul_add(bp.damp_ad[m.beta_h_off + i].abs(), l1_h);
    }
    l1_v.reduce_sum() <= lim && l1_h.reduce_sum() <= lim
}

/// Routes a `KeyCat::DampedSymp` key's strings between fast 4-string open/preload SIMD
/// and per-sample nonlinear `coupled_strs` with phase-invariant modal amplitude bounds.
pub(crate) fn route_damped_symp_key(
    r: &mut BridgeRun<'_>,
    driver: &mut impl KeyActionDriver,
    plan: &mut ChunkPlan,
    k: usize,
    sr: Range<usize>,
    sub: usize,
    has_peel: bool,
) {
    let yc0 = r.ports.y_clear_scratch[k][0];
    let yc_end = r.ports.y_clear_scratch[k][sub - 1];
    let (yc_min, yc_max) = (yc0.min(yc_end), yc0.max(yc_end));
    let const_closed =
        yc0 <= -2.0e-7 && r.ports.y_clear_scratch[k][..sub].iter().all(|&c| c == yc0);
    let can_linear = yc_min <= -2.0e-7;
    let (y_enter, y_exit) = if yc_max <= -2.0e-7 {
        (
            (-0.85 * yc_max).clamp(1.6e-6, 6.5e-6),
            (-0.98 * yc_max).clamp(2.0e-6, 8.0e-6),
        )
    } else {
        (1.8e-6, 2.5e-6)
    };
    let shallow_symp_lim = (yc_min + 1.1e-5).max(3.5e-6);

    for s_idx in sr {
        if has_peel {
            plan.push_coupled(s_idx);
            continue;
        }
        if !plan.was_nz[s_idx] {
            continue;
        }
        let mut routed_fast = false;
        if !r.ports.struck_active[s_idx]
            && yc_min >= -6.5e-6
            && prefix_damp_l1_below(r, s_idx, shallow_symp_lim)
        {
            driver.relax_open_gap_chunk(s_idx, sub);
            routed_fast = true;
        } else if can_linear && prefix_damp_l1_below(r, s_idx, y_enter) {
            let rv = r.bridge_ports.r_damp_prefix_v[s_idx];
            let rh = r.bridge_ports.r_damp_prefix_h[s_idx];
            let gains_opt = if const_closed {
                driver
                    .symp_linear_gains_if_settled(k, s_idx, rv, rh, yc0)
                    .or_else(|| driver.symp_linear_gains_for_chunk(k, s_idx, rv, rh, yc0, yc0, sub))
            } else {
                driver.symp_linear_gains_for_chunk(k, s_idx, rv, rh, yc0, yc_end, sub)
            };
            if let Some((gv, gh)) = gains_opt {
                if !r.ports.struck_active[s_idx] && gv.abs() < 2.5e-3 && gh.abs() < 2.5e-3 {
                    routed_fast = true;
                } else {
                    if !r.ports.ports_synced[s_idx] {
                        refresh_prefix_damp_ports_bp(r.bridge_ports, r.ports, s_idx);
                    }
                    plan.symp_gv[s_idx] = gv;
                    plan.symp_gh[s_idx] = gh;
                    plan.symp_mask[s_idx] = 1.0;
                    plan.symp_y_lim[s_idx] = y_exit;
                    plan.symp_has_gains[s_idx] = true;
                    routed_fast = true;
                }
            }
        } else if yc_min > -1.4e-6 && prefix_damp_l1_below(r, s_idx, 0.92 * (yc_min + 3.0e-6)) {
            driver.relax_open_gap_chunk(s_idx, sub);
            routed_fast = true;
        }
        if !routed_fast {
            plan.push_coupled(s_idx);
        }
    }
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

    for k in 0..NUM_KEYS {
        let step_transverse = match plan.key_cat[k] {
            KeyCat::Uncoupled | KeyCat::DampedSymp => true,
            KeyCat::ActiveContact if !plan.key_has_hammer[k] && !plan.key_has_mid_ev[k] => false,
            KeyCat::ActiveContact => continue,
        };
        let kt = r.bridge_ports.key_tails[k];
        let sr = r.keys[k].string_range.clone();
        let any_tail_nz = plan.was_tail_nz[sr.clone()].iter().any(|&b| b);
        if !any_tail_nz {
            continue;
        }
        let (th_e, l_e) = (usize::from(kt.th_end), usize::from(kt.l_end));
        acc_th[..sub].fill(F32x4::zero());
        let has_l = th_e < l_e;
        if has_l {
            acc_l[..sub].fill(F32x4::zero());
        }
        if !step_transverse && !has_l {
            continue;
        }
        for s_idx in sr {
            if !plan.was_tail_nz[s_idx] {
                continue;
            }
            let (tl_v, tl_h, tl_l) = {
                let m = &r.bridge_ports.meta[s_idx];
                (m.tl_v.clone(), m.tl_h.clone(), m.tl_l.clone())
            };
            if step_transverse {
                let nv = usize::from(r.ports.tl_v_active[s_idx]).min(tl_v.len());
                let nh = usize::from(r.ports.tl_h_active[s_idx]).min(tl_h.len());
                accum_soa_range_in(
                    &mut r.bridge_ports.tl_soa,
                    tl_v.start..(tl_v.start + nv),
                    &mut acc_th[..sub],
                );
                accum_soa_range_in(
                    &mut r.bridge_ports.tl_soa,
                    tl_h.start..(tl_h.start + nh),
                    &mut acc_th[..sub],
                );
            }
            if has_l {
                let nl_sav = sav_long_vecs(r, s_idx);
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
}
