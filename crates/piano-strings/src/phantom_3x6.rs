//! Specialized `3x6` (`12` transverse x `24` longitudinal) Bank-Sujbert SIMD phantom
//! convolution and Toeplitz+Hankel adjoint gradient kernels for bass keys `0..16` (`no_std`).

use piano_simd::F32x4;

use crate::phantom::{MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_VECS};

#[inline(always)]
fn convolve_pol_3x6(p0: F32x4, p1: F32x4, p2: F32x4) -> [F32x4; MAX_PHANTOM_LONG_VECS] {
    let z = F32x4::zero();
    let (s1_01, s2_01, s3_01) = (p0.slide1(p1), p0.slide2(p1), p0.slide3(p1));
    let (s1_12, s2_12, s3_12) = (p1.slide1(p2), p1.slide2(p2), p1.slide3(p2));
    let (s1_2z, s2_2z, s3_2z) = (p2.slide1(z), p2.slide2(z), p2.slide3(z));

    let (vx0, vx1, vx2, vx3) = (
        p0.splat_lane::<0>(),
        p0.splat_lane::<1>(),
        p0.splat_lane::<2>(),
        p0.splat_lane::<3>(),
    );
    let (vx4, vx5, vx6, vx7) = (
        p1.splat_lane::<0>(),
        p1.splat_lane::<1>(),
        p1.splat_lane::<2>(),
        p1.splat_lane::<3>(),
    );
    let (vx8, vx9, vx10) = (
        p2.splat_lane::<0>(),
        p2.splat_lane::<1>(),
        p2.splat_lane::<2>(),
    );

    let (p0_sq, p1_sq, p2_sq) = (p0 * p0, p1 * p1, p2 * p2);

    let d0 = vx0 * (s1_01 + z.slide2(s1_01))
        + vx1 * s2_01
        + vx2 * s3_01
        + vx3 * p1
        + vx4 * s1_12
        + vx5 * s2_12
        + vx6 * s3_12
        + vx7 * p2
        + vx8 * s1_2z
        + vx9 * s2_2z
        + vx10 * s3_2z;
    let d1 = vx0 * (s1_12 + s3_01)
        + vx1 * (s2_12 + s2_01)
        + vx2 * (s3_12 + z.slide2(s3_01))
        + vx3 * p2
        + vx4 * s1_2z
        + vx5 * s2_2z
        + vx6 * s3_2z;
    let d2 = vx0 * (s1_2z + s3_12)
        + vx1 * (s2_2z + s2_12)
        + vx2 * (s3_2z + s1_12)
        + vx3 * p1
        + vx4 * z.slide2(s1_12);
    let d3 = vx0 * s3_2z
        + vx1 * s2_2z
        + vx2 * s1_2z
        + vx3 * p2
        + vx4 * s3_12
        + vx5 * s2_12
        + vx6 * z.slide2(s3_12);
    let d4 = vx4 * s3_2z + vx5 * s2_2z + vx6 * s1_2z + vx7 * p2 + vx8 * z.slide2(s1_2z);
    let d5 = vx8 * s3_2z + vx9 * s2_2z + vx10 * z.slide2(s3_2z);

    [
        (d0 + d0) + z.unpack_lo(p0_sq),
        (d1 + d1) + z.unpack_hi(p0_sq),
        (d2 + d2) + z.unpack_lo(p1_sq),
        (d3 + d3) + z.unpack_hi(p1_sq),
        (d4 + d4) + z.unpack_lo(p2_sq),
        (d5 + d5) + z.unpack_hi(p2_sq),
    ]
}

/// Evaluates unscaled `phi_sum[1..=24]` across `3` V/H transverse and `6` longitudinal SIMD4 vectors.
#[cfg_attr(target_arch = "wasm32", inline(never))]
pub(crate) fn convolve_transverse_slopes_f32_3x6(
    dv4: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    dh4: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) -> [F32x4; MAX_PHANTOM_LONG_VECS] {
    let pv = convolve_pol_3x6(dv4[0], dv4[1], dv4[2]);
    let ph = convolve_pol_3x6(dh4[0], dh4[1], dh4[2]);
    [
        pv[0] + ph[0],
        pv[1] + ph[1],
        pv[2] + ph[2],
        pv[3] + ph[3],
        pv[4] + ph[4],
        pv[5] + ph[5],
    ]
}

/// Column-wise unrolled `F32x4` Toeplitz+Hankel adjoint gradient across `N = 12` V/H slope modes
/// coupled to `24` longitudinal modes (`6` SIMD4 vectors).
#[cfg_attr(target_arch = "wasm32", inline(never))]
pub(crate) fn accum_phantom_grad_vh_simd_3x6(
    dv4: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    dh4: &[F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    c_sum: f32,
    c_phi4: &[F32x4; MAX_PHANTOM_LONG_VECS],
) -> (
    [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) {
    let z = F32x4::zero();
    let (q0, q1, q2, q3, q4, q5) = (
        c_phi4[0], c_phi4[1], c_phi4[2], c_phi4[3], c_phi4[4], c_phi4[5],
    );
    let (r0, r1, r2) = (q0.reverse(), q1.reverse(), q2.reverse());

    let c_sum4 = F32x4::splat(c_sum);
    let (mut gv0, mut gv1, mut gv2) = (c_sum4 * dv4[0], c_sum4 * dv4[1], c_sum4 * dv4[2]);
    let (mut gh0, mut gh1, mut gh2) = (c_sum4 * dh4[0], c_sum4 * dh4[1], c_sum4 * dh4[2]);

    macro_rules! step_col {
        ($vec_idx:expr, $lane:tt, $k0:expr, $k1:expr, $k2:expr) => {{
            let dvb = dv4[$vec_idx].splat_lane::<$lane>();
            let dhb = dh4[$vec_idx].splat_lane::<$lane>();
            let (k0b, k1b, k2b) = ($k0, $k1, $k2);
            gv0 = k0b.mul_add(dvb, gv0);
            gv1 = k1b.mul_add(dvb, gv1);
            gv2 = k2b.mul_add(dvb, gv2);
            gh0 = k0b.mul_add(dhb, gh0);
            gh1 = k1b.mul_add(dhb, gh1);
            gh2 = k2b.mul_add(dhb, gh2);
        }};
    }

    let (q0_s1, q0_s2, q0_s3) = (q0.slide1(q1), q0.slide2(q1), q0.slide3(q1));
    let (q1_s1, q1_s2, q1_s3) = (q1.slide1(q2), q1.slide2(q2), q1.slide3(q2));
    let (q2_s1, q2_s2, q2_s3) = (q2.slide1(q3), q2.slide2(q3), q2.slide3(q3));
    let (q3_s1, q3_s2, q3_s3) = (q3.slide1(q4), q3.slide2(q4), q3.slide3(q4));
    let (q4_s1, q4_s2, q4_s3) = (q4.slide1(q5), q4.slide2(q5), q4.slide3(q5));

    let p1_0 = r0.slide3(z);
    let t0 = z.slide3(q0);
    let t1 = p1_0 + z.slide2(q0);
    let t2 = r0.slide2(z) + p1_0.reverse();
    let t3 = r0.slide1(z);

    step_col!(0, 0, t0 + q0_s1, q0_s3 + q1_s1, q1_s3 + q2_s1);
    step_col!(0, 1, t1 + q0_s2, q0_s2 + q1_s2, q1_s2 + q2_s2);
    step_col!(0, 2, t2 + q0_s3, q0_s1 + q1_s3, q1_s1 + q2_s3);
    step_col!(0, 3, t3 + q1, q0 + q2, q1 + q3);

    step_col!(1, 0, r0 + q1_s1, t0 + q2_s1, q0_s3 + q3_s1);
    step_col!(1, 1, r1.slide3(r0) + q1_s2, t1 + q2_s2, q0_s2 + q3_s2);
    step_col!(1, 2, r1.slide2(r0) + q1_s3, t2 + q2_s3, q0_s1 + q3_s3);
    step_col!(1, 3, r1.slide1(r0) + q2, t3 + q3, q0 + q4);

    step_col!(2, 0, r1 + q2_s1, r0 + q3_s1, t0 + q4_s1);
    step_col!(
        2,
        1,
        r2.slide3(r1) + q2_s2,
        r1.slide3(r0) + q3_s2,
        t1 + q4_s2
    );
    step_col!(
        2,
        2,
        r2.slide2(r1) + q2_s3,
        r1.slide2(r0) + q3_s3,
        t2 + q4_s3
    );
    step_col!(2, 3, r2.slide1(r1) + q3, r1.slide1(r0) + q4, t3 + q5);

    ([gv0, gv1, gv2], [gh0, gh1, gh2])
}
