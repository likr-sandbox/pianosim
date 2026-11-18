//! Stage 2 <-> Stage 3 per-sample tail stepping and hammer/damper contact prep (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::NUM_KEYS;

use crate::bridge_tails::{add_zone_n, BridgeRun, ChunkPlan, HammerStepForce, TailDampPartial};
use crate::contact::{eval_hammer_choir, sample_damper_with_r, KeyActionDriver};
use crate::contact_ports::{refresh_string_ports_bp, ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::modal_soa::ModalSoa;
use crate::zoh_kernel::{step_string_modes, zoh_rotate};

#[inline]
#[allow(clippy::too_many_arguments)]
fn step_tail_pol_tl(
    tl_soa: &mut ModalSoa,
    ports: &ContactPorts,
    tl_r: Range<usize>,
    port_r: Range<usize>,
    f_h: f32,
    alpha_w: f32,
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
    if alpha_w > 0.0 && f_h != 0.0 {
        step_tail_pol_tl_inner::<true>(tl_soa, ports, tl_r, port_r, f_h, alpha_w, f_d, has_damper)
    } else {
        step_tail_pol_tl_inner::<false>(tl_soa, ports, tl_r, port_r, f_h, 0.0, f_d, has_damper)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_tail_pol_tl_inner<const HAS_ALPHA: bool>(
    tl_soa: &mut ModalSoa,
    ports: &ContactPorts,
    tl_r: Range<usize>,
    port_r: Range<usize>,
    f_h: f32,
    alpha_w: f32,
    f_d: f32,
    has_damper: bool,
) -> f32 {
    let (u, v) = (tl_soa.u.as_f32x4_slice_mut(), tl_soa.v.as_f32x4_slice_mut());
    let (u_sl, v_sl) = (&mut u[tl_r.clone()], &mut v[tl_r.clone()]);
    let d = &tl_soa.d_coeff.as_f32x4_slice()[tl_r.clone()];
    let s = &tl_soa.s_coeff.as_f32x4_slice()[tl_r.clone()];
    let c = &tl_soa.c_bridge.as_f32x4_slice()[tl_r];
    let bh = &ports.b_strike.as_f32x4_slice()[port_r.clone()];
    let bh_diff = &ports.b_strike_diff.as_f32x4_slice()[port_r.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[port_r];
    let (fh_v, alpha_v, fd_v) = (
        F32x4::splat(f_h),
        F32x4::splat(alpha_w),
        F32x4::splat(if has_damper { f_d } else { 0.0 }),
    );
    let mut acc = F32x4::zero();
    let n = u_sl.len();
    let mut i = 0usize;
    while i + 1 < n {
        let (bh0, bh1) = if HAS_ALPHA {
            (
                alpha_v.mul_add(bh_diff[i], bh[i]),
                alpha_v.mul_add(bh_diff[i + 1], bh[i + 1]),
            )
        } else {
            (bh[i], bh[i + 1])
        };
        let inj0 = fd_v.mul_add(bd[i], fh_v.mul_add(bh0, u_sl[i]));
        let inj1 = fd_v.mul_add(bd[i + 1], fh_v.mul_add(bh1, u_sl[i + 1]));
        let (un0, vn0) = zoh_rotate(inj0, v_sl[i], d[i], s[i]);
        let (un1, vn1) = zoh_rotate(inj1, v_sl[i + 1], d[i + 1], s[i + 1]);
        (u_sl[i], v_sl[i]) = (un0, vn0);
        (u_sl[i + 1], v_sl[i + 1]) = (un1, vn1);
        acc = vn0.mul_add(c[i], vn1.mul_add(c[i + 1], acc));
        i += 2;
    }
    if i < n {
        let bhi = if HAS_ALPHA {
            alpha_v.mul_add(bh_diff[i], bh[i])
        } else {
            bh[i]
        };
        let inj = fd_v.mul_add(bd[i], fh_v.mul_add(bhi, u_sl[i]));
        let (un, vn) = zoh_rotate(inj, v_sl[i], d[i], s[i]);
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
    let n = u_sl.len();
    let mut i = 0usize;
    while i + 1 < n {
        let (bd0, bd1) = (bd[i], bd[i + 1]);
        let (inj0, inj1) = if HAS_FD {
            (fd_v.mul_add(bd0, u_sl[i]), fd_v.mul_add(bd1, u_sl[i + 1]))
        } else {
            (u_sl[i], u_sl[i + 1])
        };
        let (un0, vn0) = zoh_rotate(inj0, v_sl[i], d[i], s[i]);
        let (un1, vn1) = zoh_rotate(inj1, v_sl[i + 1], d[i + 1], s[i + 1]);
        (u_sl[i], v_sl[i]) = (un0, vn0);
        (u_sl[i + 1], v_sl[i + 1]) = (un1, vn1);
        acc_c = vn0.mul_add(c[i], vn1.mul_add(c[i + 1], acc_c));
        acc_yd = vn0.mul_add(ad[i], vn1.mul_add(ad[i + 1], acc_yd));
        acc_vd = un0.mul_add(bd0, un1.mul_add(bd1, acc_vd));
        i += 2;
    }
    if i < n {
        let bdi = bd[i];
        let inj = if HAS_FD {
            fd_v.mul_add(bdi, u_sl[i])
        } else {
            u_sl[i]
        };
        let (un, vn) = zoh_rotate(inj, v_sl[i], d[i], s[i]);
        (u_sl[i], v_sl[i]) = (un, vn);
        acc_c = vn.mul_add(c[i], acc_c);
        acc_yd = vn.mul_add(ad[i], acc_yd);
        acc_vd = un.mul_add(bdi, acc_vd);
    }
    (acc_c.reduce_sum(), acc_yd.reduce_sum(), acc_vd.reduce_sum())
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
                r.ports.tail_nonzero[s_idx] = true;
                r.ports.ports_synced[s_idx] = false;
                r.ports.tl_v_active[s_idx] = u16::MAX;
                r.ports.tl_h_active[s_idx] = u16::MAX;
                r.ports.tl_damp_active[s_idx] = true;
                r.ports.sav_dormant[s_idx] = false;
                r.ports.sav_low_nl_chunks[s_idx] = 0;
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
        let step_long_here = plan.key_has_hammer[k] || plan.key_has_mid_ev[k];
        let alpha_sync = if plan.key_has_hammer[k] {
            (driver.hammer_compression_norm(k) / 0.65).clamp(0.0, 1.0)
        } else {
            0.0
        };
        for s_idx in sr.clone() {
            if r.ports.may_be_nonzero[s_idx] && !r.ports.ports_synced[s_idx] {
                refresh_string_ports_bp(
                    r.bridge_ports,
                    r.ports,
                    s_idx,
                    r.strings[s_idx].has_damper,
                    alpha_sync,
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
        let alpha_w = if has_hammer {
            (driver.hammer_compression_norm(k) / 0.65).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let fh_arr = if has_hammer {
            eval_hammer_choir(
                driver,
                r.ports,
                k,
                sr.clone(),
                usize::from(key.choir_size),
                alpha_w,
            )
        } else {
            [0.0; 3]
        };
        let yc = r.ports.y_clear_scratch[k][t];
        let damp_on = key.has_damper && yc < DAMPER_CLEAR_THRESHOLD_M;
        for (c, s_idx) in sr.enumerate() {
            let fh = fh_arr[c];
            let (fd_v, fd_h) = if damp_on && r.ports.may_be_nonzero[s_idx] {
                let (rv, rh) = if r.ports.tail_nonzero[s_idx] || fh != 0.0 {
                    (r.ports.r_port_damp_v[s_idx], r.ports.r_port_damp_h[s_idx])
                } else {
                    (
                        r.bridge_ports.r_damp_prefix_v[s_idx],
                        r.bridge_ports.r_damp_prefix_h[s_idx],
                    )
                };
                sample_damper_with_r(driver, r.ports, k, s_idx, rv, rh, yc)
            } else {
                (0.0, 0.0)
            };
            plan.hammer_forces[s_idx] = HammerStepForce {
                fh,
                fd_v,
                fd_h,
                alpha_w,
            };
            if fh != 0.0 {
                r.ports.tail_nonzero[s_idx] = true;
                r.ports.tl_v_active[s_idx] = u16::MAX;
                r.ports.tl_h_active[s_idx] = u16::MAX;
                r.ports.tl_damp_active[s_idx] = true;
            }
            let entry = &r.strings[s_idx];
            let (tl_v, v_tail, tl_h, h_tail, tl_l, nt, nl_f) = {
                let m = &r.bridge_ports.meta[s_idx];
                let (nv, nh) = if r.ports.tail_nonzero[s_idx] {
                    (
                        usize::from(r.ports.tl_v_active[s_idx]).min(m.tl_v.len()),
                        usize::from(r.ports.tl_h_active[s_idx]).min(m.tl_h.len()),
                    )
                } else {
                    (0, 0)
                };
                let tr = |r: &Range<usize>, n: usize| r.start..(r.start + n);
                (
                    tr(&m.tl_v, nv),
                    tr(&m.v_tail, nv),
                    tr(&m.tl_h, nh),
                    tr(&m.h_tail, nh),
                    m.tl_l.clone(),
                    r.bridge_ports.tail_n_trans[s_idx],
                    r.bridge_ports.tail_n_long[s_idx],
                )
            };
            let (tv, th) = if has_hammer {
                let tl = &mut r.bridge_ports.tl_soa;
                let hd = key.has_damper;
                (
                    step_tail_pol_tl(tl, r.ports, tl_v, v_tail, fh, alpha_w, fd_v, hd),
                    step_tail_pol_tl(tl, r.ports, tl_h, h_tail, fh, alpha_w, fd_h, hd),
                )
            } else {
                let tl = &mut r.bridge_ports.tl_soa;
                let (tv, ydv, vdv_raw) = step_tail_pol_damp_tl(tl, r.ports, tl_v, v_tail, fd_v);
                let (th, ydh, vdh_raw) = step_tail_pol_damp_tl(tl, r.ports, tl_h, h_tail, fd_h);
                plan.tail_partial_damp[s_idx] = TailDampPartial {
                    ydv,
                    vdv_raw,
                    ydh,
                    vdh_raw,
                };
                (tv, th)
            };
            let tl = if !step_long_here || tl_l.is_empty() {
                0.0
            } else {
                let tl_soa = &mut r.bridge_ports.tl_soa;
                if r.long_on && fh > 0.0 {
                    let lv = &mut tl_soa.u.as_f32x4_slice_mut()[tl_l.clone()];
                    r.long_bank.inject_precursor(s_idx, tl_l.len(), fh, lv);
                }
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
