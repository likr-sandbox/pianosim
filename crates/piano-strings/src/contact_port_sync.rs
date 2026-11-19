//! Stage 1 <-> Stage 2 contact port readout refresh helpers (`ModalSoa` and `BridgePorts`).

use core::ops::Range;

use piano_simd::F32x4;

use crate::bridge_port::BridgePorts;
use crate::bridge_tails::TailPortPartial;
use crate::choir_map::StringMapEntry;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::ModalSoa;
use crate::zoh_kernel::zoh_rotate;

#[inline]
fn accum_pol_ports<const HAS_ALPHA: bool>(
    soa: &ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    has_damper: bool,
    alpha_w: f32,
) -> (f32, f32, f32, f32) {
    let (u, v) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bh_diff, ah_diff) = (
        ports.b_strike_diff.as_f32x4_slice(),
        ports.a_strike_diff.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let alpha_v = F32x4::splat(alpha_w);
    let (mut ayh, mut avh, mut ayd, mut avd) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in vr {
        let (ah_eff, bh_eff) = if HAS_ALPHA {
            (
                alpha_v.mul_add(ah_diff[i], ah[i]),
                alpha_v.mul_add(bh_diff[i], bh[i]),
            )
        } else {
            (ah[i], bh[i])
        };
        ayh = v[i].mul_add(ah_eff, ayh);
        avh = u[i].mul_add(bh_eff, avh);
        if has_damper {
            ayd = v[i].mul_add(ad[i], ayd);
            avd = u[i].mul_add(bd[i], avd);
        }
    }
    (
        ayh.reduce_sum(),
        avh.reduce_sum(),
        ayd.reduce_sum(),
        avd.reduce_sum(),
    )
}

/// Recomputes cached `(y_h, v_h, y_dv, v_dv, y_dh, v_dh)` from `soa.(u, v)` after fast-tile steps.
#[inline]
pub(crate) fn refresh_string_ports(
    soa: &ModalSoa,
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
    alpha_w: f32,
) {
    let d = entry.has_damper;
    let (yh_v, vh_v, ydv, vdv, yh_h, vh_h, ydh, vdh) = if alpha_w > 0.0 {
        let v_res = accum_pol_ports::<true>(soa, ports, entry.vert_vecs.clone(), d, alpha_w);
        let h_res = accum_pol_ports::<true>(soa, ports, entry.horiz_vecs.clone(), d, alpha_w);
        (
            v_res.0, v_res.1, v_res.2, v_res.3, h_res.0, h_res.1, h_res.2, h_res.3,
        )
    } else {
        let v_res = accum_pol_ports::<false>(soa, ports, entry.vert_vecs.clone(), d, 0.0);
        let h_res = accum_pol_ports::<false>(soa, ports, entry.horiz_vecs.clone(), d, 0.0);
        (
            v_res.0, v_res.1, v_res.2, v_res.3, h_res.0, h_res.1, h_res.2, h_res.3,
        )
    };
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = yh_v + yh_h;
    ports.v_h[s_idx] = alpha * (vh_v + vh_h);
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = if d { (ydv, alpha * vdv) } else { (0.0, 0.0) };
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = if d { (ydh, alpha * vdh) } else { (0.0, 0.0) };
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn refresh_string_ports_bp(
    bp: &BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
    has_damper: bool,
    alpha_w: f32,
) {
    if alpha_w > 0.0 {
        refresh_string_ports_bp_inner::<true>(bp, ports, s_idx, has_damper, alpha_w);
    } else {
        refresh_string_ports_bp_inner::<false>(bp, ports, s_idx, has_damper, 0.0);
    }
}

#[inline]
fn refresh_string_ports_bp_inner<const HAS_ALPHA: bool>(
    bp: &BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
    has_damper: bool,
    alpha_w: f32,
) {
    let m = &bp.meta[s_idx];
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bh_diff, ah_diff) = (
        ports.b_strike_diff.as_f32x4_slice(),
        ports.a_strike_diff.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (tu, tv) = (bp.tl_soa.u.as_f32x4_slice(), bp.tl_soa.v.as_f32x4_slice());
    let alpha_v = F32x4::splat(alpha_w);

    let accum_pol =
        |p_off: usize, vr_br: Range<usize>, tl_r: Range<usize>, vr_tail: Range<usize>| {
            let (mut ayh, mut avh, mut ayd, mut avd) =
                (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
            for (k, i) in vr_br.enumerate() {
                let st = bp.state[p_off + k];
                let (ah_eff, bh_eff) = if HAS_ALPHA {
                    (
                        alpha_v.mul_add(ah_diff[i], ah[i]),
                        alpha_v.mul_add(bh_diff[i], bh[i]),
                    )
                } else {
                    (ah[i], bh[i])
                };
                ayh = st.v.mul_add(ah_eff, ayh);
                avh = st.u.mul_add(bh_eff, avh);
                if has_damper {
                    ayd = st.v.mul_add(ad[i], ayd);
                    avd = st.u.mul_add(bd[i], avd);
                }
            }
            for (k, i) in vr_tail.enumerate() {
                let (u_v, v_v) = (tu[tl_r.start + k], tv[tl_r.start + k]);
                let (ah_eff, bh_eff) = if HAS_ALPHA {
                    (
                        alpha_v.mul_add(ah_diff[i], ah[i]),
                        alpha_v.mul_add(bh_diff[i], bh[i]),
                    )
                } else {
                    (ah[i], bh[i])
                };
                ayh = v_v.mul_add(ah_eff, ayh);
                avh = u_v.mul_add(bh_eff, avh);
                if has_damper {
                    ayd = v_v.mul_add(ad[i], ayd);
                    avd = u_v.mul_add(bd[i], avd);
                }
            }
            (
                ayh.reduce_sum(),
                avh.reduce_sum(),
                ayd.reduce_sum(),
                avd.reduce_sum(),
            )
        };
    let (nv, nh) = if ports.tail_nonzero[s_idx] {
        (
            usize::from(ports.tl_v_active[s_idx]).min(m.tl_v.len()),
            usize::from(ports.tl_h_active[s_idx]).min(m.tl_h.len()),
        )
    } else {
        (0, 0)
    };
    let (yh_v, vh_v, ydv, vdv) = accum_pol(
        m.beta_v_off,
        m.v_br.clone(),
        m.tl_v.start..(m.tl_v.start + nv),
        m.v_tail.start..(m.v_tail.start + nv),
    );
    let (yh_h, vh_h, ydh, vdh) = accum_pol(
        m.beta_h_off,
        m.h_br.clone(),
        m.tl_h.start..(m.tl_h.start + nh),
        m.h_tail.start..(m.h_tail.start + nh),
    );
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = yh_v + yh_h;
    ports.v_h[s_idx] = alpha * (vh_v + vh_h);
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = if has_damper {
        (ydv, alpha * vdv)
    } else {
        (0.0, 0.0)
    };
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = if has_damper {
        (ydh, alpha * vdh)
    } else {
        (0.0, 0.0)
    };
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn refresh_prefix_ports_with_tail_bp(
    bp: &BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
    has_damper: bool,
    alpha_w: f32,
    tail: &TailPortPartial,
) {
    if alpha_w > 0.0 {
        refresh_prefix_ports_with_tail_bp_inner::<true>(
            bp, ports, s_idx, has_damper, alpha_w, tail,
        );
    } else {
        refresh_prefix_ports_with_tail_bp_inner::<false>(bp, ports, s_idx, has_damper, 0.0, tail);
    }
}

#[inline]
fn refresh_prefix_ports_with_tail_bp_inner<const HAS_ALPHA: bool>(
    bp: &BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
    has_damper: bool,
    alpha_w: f32,
    tail: &TailPortPartial,
) {
    let m = &bp.meta[s_idx];
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bh_diff, ah_diff) = (
        ports.b_strike_diff.as_f32x4_slice(),
        ports.a_strike_diff.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let alpha_v = F32x4::splat(alpha_w);

    let accum_prefix = |p_off: usize, vr_br: Range<usize>| {
        let (mut ayh, mut avh, mut ayd, mut avd) =
            (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
        for (k, i) in vr_br.enumerate() {
            let st = bp.state[p_off + k];
            let (ah_eff, bh_eff) = if HAS_ALPHA {
                (
                    alpha_v.mul_add(ah_diff[i], ah[i]),
                    alpha_v.mul_add(bh_diff[i], bh[i]),
                )
            } else {
                (ah[i], bh[i])
            };
            ayh = st.v.mul_add(ah_eff, ayh);
            avh = st.u.mul_add(bh_eff, avh);
            if has_damper {
                ayd = st.v.mul_add(ad[i], ayd);
                avd = st.u.mul_add(bd[i], avd);
            }
        }
        (
            ayh.reduce_sum(),
            avh.reduce_sum(),
            ayd.reduce_sum(),
            avd.reduce_sum(),
        )
    };
    let (yh_v, vh_v, ydv, vdv) = accum_prefix(m.beta_v_off, m.v_br.clone());
    let (yh_h, vh_h, ydh, vdh) = accum_prefix(m.beta_h_off, m.h_br.clone());
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = (yh_v + tail.yhv) + (yh_h + tail.yhh);
    ports.v_h[s_idx] = alpha * ((vh_v + tail.vhv_raw) + (vh_h + tail.vhh_raw));
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = if has_damper {
        (ydv + tail.ydv, alpha * (vdv + tail.vdv_raw))
    } else {
        (0.0, 0.0)
    };
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = if has_damper {
        (ydh + tail.ydh, alpha * (vdh + tail.vdh_raw))
    } else {
        (0.0, 0.0)
    };
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn refresh_prefix_damp_ports_bp(
    bp: &BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
) {
    let m = &bp.meta[s_idx];
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let accum = |p_off: usize, vr: Range<usize>| {
        let (mut ayd, mut avd) = (F32x4::zero(), F32x4::zero());
        for (k, i) in vr.enumerate() {
            let st = bp.state[p_off + k];
            ayd = st.v.mul_add(ad[i], ayd);
            avd = st.u.mul_add(bd[i], avd);
        }
        (ayd.reduce_sum(), avd.reduce_sum())
    };
    let (ydv, vdv) = accum(m.beta_v_off, m.v_br.clone());
    let (ydh, vdh) = accum(m.beta_h_off, m.h_br.clone());
    let alpha = ports.alpha_vel[s_idx];
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = (ydv, alpha * vdv);
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = (ydh, alpha * vdh);
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn accum_prefix_damp_vel_bp(
    bp: &BridgePorts,
    ports: &ContactPorts,
    p_off: usize,
    vr: Range<usize>,
) -> f32 {
    let bd = ports.b_damp.as_f32x4_slice();
    let mut acc = F32x4::zero();
    for (k, i) in vr.enumerate() {
        acc = bp.state[p_off + k].u.mul_add(bd[i], acc);
    }
    acc.reduce_sum()
}

#[inline]
pub(crate) fn step_transverse_pol<const HAS_DAMPER: bool>(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    f_h: f32,
    alpha_w: f32,
    f_d: f32,
) -> (f32, f32, f32) {
    if alpha_w > 0.0 {
        step_transverse_pol_inner::<HAS_DAMPER, true>(soa, ports, vr, f_h, alpha_w, f_d)
    } else {
        step_transverse_pol_inner::<HAS_DAMPER, false>(soa, ports, vr, f_h, 0.0, f_d)
    }
}

#[inline(always)]
fn step_transverse_pol_inner<const HAS_DAMPER: bool, const HAS_ALPHA: bool>(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    f_h: f32,
    alpha_w: f32,
    f_d: f32,
) -> (f32, f32, f32) {
    let (u_s, v_s) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let (d_s, s_s, c_s) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    let (bh_s, ah_s) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bh_diff, ah_diff) = (
        ports.b_strike_diff.as_f32x4_slice(),
        ports.a_strike_diff.as_f32x4_slice(),
    );
    let (bd_s, ad_s) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (fh_v, alpha_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(alpha_w), F32x4::splat(f_d));
    let (mut acc_b, mut acc_yh, mut acc_yd) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in vr {
        let (bh_eff, ah_eff) = if HAS_ALPHA {
            (
                alpha_v.mul_add(bh_diff[i], bh_s[i]),
                alpha_v.mul_add(ah_diff[i], ah_s[i]),
            )
        } else {
            (bh_s[i], ah_s[i])
        };
        let u_inj = if HAS_DAMPER {
            fd_v.mul_add(bd_s[i], fh_v.mul_add(bh_eff, u_s[i]))
        } else {
            fh_v.mul_add(bh_eff, u_s[i])
        };
        let (un, vn) = zoh_rotate(u_inj, v_s[i], d_s[i], s_s[i]);
        u_s[i] = un;
        v_s[i] = vn;
        acc_b = vn.mul_add(c_s[i], acc_b);
        acc_yh = vn.mul_add(ah_eff, acc_yh);
        if HAS_DAMPER {
            acc_yd = vn.mul_add(ad_s[i], acc_yd);
        }
    }
    (acc_b.reduce_sum(), acc_yh.reduce_sum(), acc_yd.reduce_sum())
}

#[inline]
pub(crate) fn accum_pol_vel<const HAS_DAMPER: bool, const HAS_ALPHA: bool>(
    soa: &ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    alpha_w: f32,
) -> (f32, f32) {
    let (u_s, bh_s, bh_diff, bd_s) = (
        soa.u.as_f32x4_slice(),
        ports.b_strike.as_f32x4_slice(),
        ports.b_strike_diff.as_f32x4_slice(),
        ports.b_damp.as_f32x4_slice(),
    );
    let alpha_v = F32x4::splat(alpha_w);
    let (mut acc_vh, mut acc_vd) = (F32x4::zero(), F32x4::zero());
    for i in vr {
        let bh_eff = if HAS_ALPHA {
            alpha_v.mul_add(bh_diff[i], bh_s[i])
        } else {
            bh_s[i]
        };
        acc_vh = u_s[i].mul_add(bh_eff, acc_vh);
        if HAS_DAMPER {
            acc_vd = u_s[i].mul_add(bd_s[i], acc_vd);
        }
    }
    (acc_vh.reduce_sum(), acc_vd.reduce_sum())
}
