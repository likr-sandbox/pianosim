//! 4-lane SIMD transpose reduction and single-string prefix stepping helpers (`no_std`).

use piano_simd::F32x4;

use crate::bridge_port::{PrefixCoef, PrefixState};
use crate::zoh_kernel::zoh_rotate;

#[inline(always)]
pub(crate) fn reduce_sum_4(r0: F32x4, r1: F32x4, r2: F32x4, r3: F32x4) -> F32x4 {
    let ac = r0.unpack_lo(r2) + r0.unpack_hi(r2);
    let bd = r1.unpack_lo(r3) + r1.unpack_hi(r3);
    ac.unpack_lo(bd) + ac.unpack_hi(bd)
}

#[inline(always)]
pub(crate) fn pair_sum_2(r0: F32x4, r1: F32x4) -> F32x4 {
    let p = r0.unpack_lo(r1) + r0.unpack_hi(r1);
    p + p.slide2(p)
}

#[inline(always)]
pub(crate) fn pair_alt_hi(r0: F32x4, r1: F32x4) -> F32x4 {
    let p = r0.unpack_lo(r1) + r0.unpack_hi(r1);
    p.slide2(p) - p
}

#[inline(always)]
pub(crate) fn pair_alt_lo(r2: F32x4, r3: F32x4) -> F32x4 {
    let p = r2.unpack_lo(r3) + r2.unpack_hi(r3);
    p - p.slide2(p)
}

#[inline(always)]
pub(crate) fn step_one_prefix<const K: usize>(
    st: &mut [PrefixState],
    cf: &[PrefixCoef],
    neg_vbr: F32x4,
    h_proj: F32x4,
) -> (F32x4, F32x4) {
    let (c0, s0) = (cf[0], st[0]);
    let (un0, mut acc_v) = zoh_rotate(s0.u, c0.beta.mul_add(neg_vbr, s0.v), c0.d, c0.s);
    st[0] = PrefixState { u: un0, v: acc_v };
    let mut acc_br = acc_v * c0.beta;
    for i in 1..K {
        let (ci, si) = (cf[i], st[i]);
        let (uni, vni) = zoh_rotate(si.u, ci.beta.mul_add(neg_vbr, si.v), ci.d, ci.s);
        st[i] = PrefixState { u: uni, v: vni };
        acc_v = acc_v + vni;
        acc_br = vni.mul_add(ci.beta, acc_br);
    }
    let (ch0, sh0) = (cf[K], st[K]);
    let (unh0, mut acc_h) = zoh_rotate(sh0.u, ch0.beta.mul_add(neg_vbr, sh0.v), ch0.d, ch0.s);
    st[K] = PrefixState { u: unh0, v: acc_h };
    acc_br = acc_h.mul_add(ch0.beta, acc_br);
    for i in 1..K {
        let (chi, shi) = (cf[K + i], st[K + i]);
        let (unhi, vnhi) = zoh_rotate(shi.u, chi.beta.mul_add(neg_vbr, shi.v), chi.d, chi.s);
        st[K + i] = PrefixState { u: unhi, v: vnhi };
        acc_h = acc_h + vnhi;
        acc_br = vnhi.mul_add(chi.beta, acc_br);
    }
    (acc_h.mul_add(h_proj, acc_v), acc_br)
}

#[inline(always)]
pub(crate) fn step_one_prefix_vd_only<const K: usize>(
    st: &mut [PrefixState],
    cf: &[PrefixCoef],
    bd: &[F32x4],
    fdv: F32x4,
    fdh: F32x4,
    neg_vbr: F32x4,
    h_proj: F32x4,
) -> (F32x4, F32x4, F32x4, F32x4) {
    let (c0, s0, bd0) = (cf[0], st[0], bd[0]);
    let (un0, mut acc_v) = zoh_rotate(
        fdv.mul_add(bd0, s0.u),
        c0.beta.mul_add(neg_vbr, s0.v),
        c0.d,
        c0.s,
    );
    st[0] = PrefixState { u: un0, v: acc_v };
    let mut acc_br = acc_v * c0.beta;
    let mut acc_vd_v = un0 * bd0;
    for i in 1..K {
        let (ci, si, bdi) = (cf[i], st[i], bd[i]);
        let (uni, vni) = zoh_rotate(
            fdv.mul_add(bdi, si.u),
            ci.beta.mul_add(neg_vbr, si.v),
            ci.d,
            ci.s,
        );
        st[i] = PrefixState { u: uni, v: vni };
        acc_v = acc_v + vni;
        acc_br = vni.mul_add(ci.beta, acc_br);
        acc_vd_v = uni.mul_add(bdi, acc_vd_v);
    }

    let (ch0, sh0, bdh0) = (cf[K], st[K], bd[K]);
    let (unh0, mut acc_h) = zoh_rotate(
        fdh.mul_add(bdh0, sh0.u),
        ch0.beta.mul_add(neg_vbr, sh0.v),
        ch0.d,
        ch0.s,
    );
    st[K] = PrefixState { u: unh0, v: acc_h };
    acc_br = acc_h.mul_add(ch0.beta, acc_br);
    let mut acc_vd_h = unh0 * bdh0;
    for i in 1..K {
        let (chi, shi, bdhi) = (cf[K + i], st[K + i], bd[K + i]);
        let (unhi, vnhi) = zoh_rotate(
            fdh.mul_add(bdhi, shi.u),
            chi.beta.mul_add(neg_vbr, shi.v),
            chi.d,
            chi.s,
        );
        st[K + i] = PrefixState { u: unhi, v: vnhi };
        acc_h = acc_h + vnhi;
        acc_br = vnhi.mul_add(chi.beta, acc_br);
        acc_vd_h = unhi.mul_add(bdhi, acc_vd_h);
    }
    (acc_h.mul_add(h_proj, acc_v), acc_br, acc_vd_v, acc_vd_h)
}
