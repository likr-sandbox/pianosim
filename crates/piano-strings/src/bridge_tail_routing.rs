//! Phase-invariant damper/tail routing helpers for Stage 2 <-> Stage 3 stepping (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::TOTAL_STRINGS;

use crate::bridge_tails::{BridgeRun, ChunkPlan};
use crate::contact::KeyActionDriver;
use crate::contact_ports::refresh_prefix_damp_ports_bp;

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
                    if !plan.symp_has_gains[s_idx] {
                        plan.symp_has_gains[s_idx] = true;
                        plan.symp_dirty_strs[plan.n_symp_dirty] = s_idx as u8;
                        plan.n_symp_dirty += 1;
                    }
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
