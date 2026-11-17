//! Stage 1 <-> Stage 2 contact port readout refresh helpers (`ModalSoa` and `BridgePorts`).

use core::ops::Range;

use piano_simd::F32x4;

use crate::bridge_port::BridgePorts;
use crate::choir_map::StringMapEntry;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::ModalSoa;

#[inline]
fn accum_pol_ports(
    soa: &ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    has_damper: bool,
) -> (f32, f32, f32, f32) {
    let (u, v) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (mut ayh, mut avh, mut ayd, mut avd) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in vr {
        ayh = v[i].mul_add(ah[i], ayh);
        avh = u[i].mul_add(bh[i], avh);
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
) {
    let d = entry.has_damper;
    let (yh_v, vh_v, ydv, vdv) = accum_pol_ports(soa, ports, entry.vert_vecs.clone(), d);
    let (yh_h, vh_h, ydh, vdh) = accum_pol_ports(soa, ports, entry.horiz_vecs.clone(), d);
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
) {
    let m = &bp.meta[s_idx];
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (tu, tv) = (bp.tl_soa.u.as_f32x4_slice(), bp.tl_soa.v.as_f32x4_slice());

    let accum_pol =
        |p_off: usize, vr_br: Range<usize>, tl_r: Range<usize>, vr_tail: Range<usize>| {
            let (mut ayh, mut avh, mut ayd, mut avd) =
                (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
            for (k, i) in vr_br.enumerate() {
                let st = bp.state[p_off + k];
                ayh = st.v.mul_add(ah[i], ayh);
                avh = st.u.mul_add(bh[i], avh);
                if has_damper {
                    ayd = st.v.mul_add(ad[i], ayd);
                    avd = st.u.mul_add(bd[i], avd);
                }
            }
            for (k, i) in vr_tail.enumerate() {
                let (u_v, v_v) = (tu[tl_r.start + k], tv[tl_r.start + k]);
                ayh = v_v.mul_add(ah[i], ayh);
                avh = u_v.mul_add(bh[i], avh);
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
    let (yh_v, vh_v, ydv, vdv) = accum_pol(
        m.beta_v_off,
        m.v_br.clone(),
        m.tl_v.clone(),
        m.v_tail.clone(),
    );
    let (yh_h, vh_h, ydh, vdh) = accum_pol(
        m.beta_h_off,
        m.h_br.clone(),
        m.tl_h.clone(),
        m.h_tail.clone(),
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
