//! Sparse `2x2` (`8` transverse x `8` longitudinal) Bank-Sujbert SIMD phantom convolution
//! and exact algebraic transpose `DF(d)^T` adjoint kernels for bass keys `0..16` (`no_std`).

use piano_simd::F32x4;

use crate::phantom::{
    BASE_PHANTOM_TRANSVERSE_VECS, BASS_PHANTOM_MAX_KEY_IDX, MAX_PHANTOM_LONG_VECS,
    MAX_PHANTOM_TRANSVERSE_VECS,
};

/// Sparse transverse SIMD4 vector count per polarization for bass keys `0..16` (`8` modes).
pub(crate) const SPARSE_PHANTOM_TRANSVERSE_VECS: usize = 2;
/// Sparse transverse modes per polarization for bass keys `0..16` (`8`).
#[cfg(test)]
pub(crate) const SPARSE_PHANTOM_TRANSVERSE_MODES: usize = SPARSE_PHANTOM_TRANSVERSE_VECS * 4;
/// Sparse longitudinal SIMD4 vector count for bass keys `0..16` (`8` modes).
pub(crate) const SPARSE_PHANTOM_LONG_VECS: usize = 2;
/// Sparse longitudinal modes for bass keys `0..16` (`8`).
#[cfg(test)]
pub(crate) const SPARSE_PHANTOM_LONG_MODES: usize = SPARSE_PHANTOM_LONG_VECS * 4;

/// Profile-aware maximum coupled transverse SIMD4 vectors per polarization for `key_idx`.
#[inline]
#[must_use]
pub(crate) const fn phantom_transverse_vecs_for_key(key_idx: usize, bass_sparse: bool) -> usize {
    if key_idx < BASS_PHANTOM_MAX_KEY_IDX {
        if bass_sparse {
            SPARSE_PHANTOM_TRANSVERSE_VECS
        } else {
            MAX_PHANTOM_TRANSVERSE_VECS
        }
    } else if key_idx < piano_spec::LONGITUDINAL_MAX_KEY_IDX {
        BASE_PHANTOM_TRANSVERSE_VECS
    } else {
        0
    }
}

/// Profile-aware maximum coupled longitudinal SIMD4 vectors for `key_idx`.
#[inline]
#[must_use]
pub(crate) const fn phantom_long_vecs_for_key(key_idx: usize, bass_sparse: bool) -> usize {
    if key_idx < BASS_PHANTOM_MAX_KEY_IDX {
        if bass_sparse {
            SPARSE_PHANTOM_LONG_VECS
        } else {
            MAX_PHANTOM_LONG_VECS
        }
    } else if key_idx < piano_spec::LONGITUDINAL_MAX_KEY_IDX {
        crate::phantom::BASE_PHANTOM_LONG_VECS
    } else {
        0
    }
}

#[inline(always)]
fn convolve_pol_2x2(p0: F32x4, p1: F32x4) -> [F32x4; SPARSE_PHANTOM_LONG_VECS] {
    let z = F32x4::zero();
    let (s1_01, s2_01, s3_01) = (p0.slide1(p1), p0.slide2(p1), p0.slide3(p1));
    let (s1_1z, s2_1z, s3_1z) = (p1.slide1(z), p1.slide2(z), p1.slide3(z));
    let (vx0, vx1, vx2, vx3) = (
        p0.splat_lane::<0>(),
        p0.splat_lane::<1>(),
        p0.splat_lane::<2>(),
        p0.splat_lane::<3>(),
    );
    let (vx4, vx5, vx6) = (
        p1.splat_lane::<0>(),
        p1.splat_lane::<1>(),
        p1.splat_lane::<2>(),
    );
    let p0_sq = p0 * p0;

    let d0 = vx0 * (s1_01 + z.slide2(s1_01))
        + vx1 * s2_01
        + vx2 * s3_01
        + vx3 * p1
        + vx4 * s1_1z
        + vx5 * s2_1z
        + vx6 * s3_1z;
    let d1 = vx0 * (s1_1z + s3_01) + vx1 * (s2_1z + s2_01) + vx2 * (s3_1z + z.slide2(s3_01));
    [
        (d0 + d0) + z.unpack_lo(p0_sq),
        (d1 + d1) + z.unpack_hi(p0_sq),
    ]
}

/// Evaluates unscaled `phi_sum[1..=8]` across `2` V/H transverse and `2` longitudinal SIMD4 vectors.
#[inline(always)]
pub(crate) fn convolve_transverse_slopes_f32_2x2(
    dv4: &[F32x4; SPARSE_PHANTOM_TRANSVERSE_VECS],
    dh4: &[F32x4; SPARSE_PHANTOM_TRANSVERSE_VECS],
) -> [F32x4; SPARSE_PHANTOM_LONG_VECS] {
    let pv = convolve_pol_2x2(dv4[0], dv4[1]);
    let ph = convolve_pol_2x2(dh4[0], dh4[1]);
    [pv[0] + ph[0], pv[1] + ph[1]]
}

/// Exact algebraic transpose `DF(d)^T` of [`convolve_transverse_slopes_f32_2x2`] across
/// `N = 8` V/H slope modes coupled to `8` longitudinal modes (`2x2`), preserving SAV passivity.
#[inline(always)]
pub(crate) fn accum_phantom_grad_vh_simd_2x2(
    dv4: &[F32x4; SPARSE_PHANTOM_TRANSVERSE_VECS],
    dh4: &[F32x4; SPARSE_PHANTOM_TRANSVERSE_VECS],
    c_sum: f32,
    c_phi4: &[F32x4; SPARSE_PHANTOM_LONG_VECS],
) -> (
    [F32x4; SPARSE_PHANTOM_TRANSVERSE_VECS],
    [F32x4; SPARSE_PHANTOM_TRANSVERSE_VECS],
) {
    let z = F32x4::zero();
    let (q0, q1) = (c_phi4[0], c_phi4[1]);
    let (r0, r1) = (q0.reverse(), q1.reverse());

    let c_sum4 = F32x4::splat(c_sum);
    let (mut gv0, mut gv1) = (c_sum4 * dv4[0], c_sum4 * dv4[1]);
    let (mut gh0, mut gh1) = (c_sum4 * dh4[0], c_sum4 * dh4[1]);

    macro_rules! step_col {
        ($vec_idx:expr, $lane:tt, $k0:expr, $k1:expr) => {{
            let dvb = dv4[$vec_idx].splat_lane::<$lane>();
            let dhb = dh4[$vec_idx].splat_lane::<$lane>();
            let k0b = $k0;
            let k1b = $k1;
            gv0 = k0b.mul_add(dvb, gv0);
            gv1 = k1b.mul_add(dvb, gv1);
            gh0 = k0b.mul_add(dhb, gh0);
            gh1 = k1b.mul_add(dhb, gh1);
        }};
    }

    let (q0_s1, q0_s2, q0_s3) = (q0.slide1(q1), q0.slide2(q1), q0.slide3(q1));
    let (q1_s1, q1_s2, q1_s3) = (q1.slide1(z), q1.slide2(z), q1.slide3(z));
    let p1_0 = r0.slide3(z);
    let t0 = z.slide3(q0);
    let t1 = p1_0 + z.slide2(q0);
    let t2 = r0.slide2(z) + p1_0.reverse();
    let t3 = r0.slide1(z);

    step_col!(0, 0, t0 + q0_s1, q0_s3 + q1_s1);
    step_col!(0, 1, t1 + q0_s2, q0_s2 + q1_s2);
    step_col!(0, 2, t2 + q0_s3, q0_s1 + q1_s3);
    step_col!(0, 3, t3 + q1, q0);
    step_col!(1, 0, r0 + q1_s1, t0);
    step_col!(1, 1, r1.slide3(r0) + q1_s2, t1);
    step_col!(1, 2, r1.slide2(r0) + q1_s3, t2);
    step_col!(1, 3, r1.slide1(r0), t3);

    ([gv0, gv1], [gh0, gh1])
}
