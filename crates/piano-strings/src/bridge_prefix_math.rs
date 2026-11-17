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
    if K >= 2 {
        let (c1, s1) = (cf[1], st[1]);
        let (un1, vn1) = zoh_rotate(s1.u, c1.beta.mul_add(neg_vbr, s1.v), c1.d, c1.s);
        st[1] = PrefixState { u: un1, v: vn1 };
        acc_v = acc_v + vn1;
        acc_br = vn1.mul_add(c1.beta, acc_br);
    }
    if K >= 3 {
        let (c2, s2) = (cf[2], st[2]);
        let (un2, vn2) = zoh_rotate(s2.u, c2.beta.mul_add(neg_vbr, s2.v), c2.d, c2.s);
        st[2] = PrefixState { u: un2, v: vn2 };
        acc_v = acc_v + vn2;
        acc_br = vn2.mul_add(c2.beta, acc_br);
    }
    let (ch0, sh0) = (cf[K], st[K]);
    let (unh0, mut acc_h) = zoh_rotate(sh0.u, ch0.beta.mul_add(neg_vbr, sh0.v), ch0.d, ch0.s);
    st[K] = PrefixState { u: unh0, v: acc_h };
    acc_br = acc_h.mul_add(ch0.beta, acc_br);
    if K >= 2 {
        let (ch1, sh1) = (cf[K + 1], st[K + 1]);
        let (unh1, vnh1) = zoh_rotate(sh1.u, ch1.beta.mul_add(neg_vbr, sh1.v), ch1.d, ch1.s);
        st[K + 1] = PrefixState { u: unh1, v: vnh1 };
        acc_h = acc_h + vnh1;
        acc_br = vnh1.mul_add(ch1.beta, acc_br);
    }
    if K >= 3 {
        let (ch2, sh2) = (cf[K + 2], st[K + 2]);
        let (unh2, vnh2) = zoh_rotate(sh2.u, ch2.beta.mul_add(neg_vbr, sh2.v), ch2.d, ch2.s);
        st[K + 2] = PrefixState { u: unh2, v: vnh2 };
        acc_h = acc_h + vnh2;
        acc_br = vnh2.mul_add(ch2.beta, acc_br);
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
    if K >= 2 {
        let (c1, s1, bd1) = (cf[1], st[1], bd[1]);
        let (un1, vn1) = zoh_rotate(
            fdv.mul_add(bd1, s1.u),
            c1.beta.mul_add(neg_vbr, s1.v),
            c1.d,
            c1.s,
        );
        st[1] = PrefixState { u: un1, v: vn1 };
        acc_v = acc_v + vn1;
        acc_br = vn1.mul_add(c1.beta, acc_br);
        acc_vd_v = un1.mul_add(bd1, acc_vd_v);
    }
    if K >= 3 {
        let (c2, s2, bd2) = (cf[2], st[2], bd[2]);
        let (un2, vn2) = zoh_rotate(
            fdv.mul_add(bd2, s2.u),
            c2.beta.mul_add(neg_vbr, s2.v),
            c2.d,
            c2.s,
        );
        st[2] = PrefixState { u: un2, v: vn2 };
        acc_v = acc_v + vn2;
        acc_br = vn2.mul_add(c2.beta, acc_br);
        acc_vd_v = un2.mul_add(bd2, acc_vd_v);
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
    if K >= 2 {
        let (ch1, sh1, bdh1) = (cf[K + 1], st[K + 1], bd[K + 1]);
        let (unh1, vnh1) = zoh_rotate(
            fdh.mul_add(bdh1, sh1.u),
            ch1.beta.mul_add(neg_vbr, sh1.v),
            ch1.d,
            ch1.s,
        );
        st[K + 1] = PrefixState { u: unh1, v: vnh1 };
        acc_h = acc_h + vnh1;
        acc_br = vnh1.mul_add(ch1.beta, acc_br);
        acc_vd_h = unh1.mul_add(bdh1, acc_vd_h);
    }
    if K >= 3 {
        let (ch2, sh2, bdh2) = (cf[K + 2], st[K + 2], bd[K + 2]);
        let (unh2, vnh2) = zoh_rotate(
            fdh.mul_add(bdh2, sh2.u),
            ch2.beta.mul_add(neg_vbr, sh2.v),
            ch2.d,
            ch2.s,
        );
        st[K + 2] = PrefixState { u: unh2, v: vnh2 };
        acc_h = acc_h + vnh2;
        acc_br = vnh2.mul_add(ch2.beta, acc_br);
        acc_vd_h = unh2.mul_add(bdh2, acc_vd_h);
    }
    (acc_h.mul_add(h_proj, acc_v), acc_br, acc_vd_v, acc_vd_h)
}
