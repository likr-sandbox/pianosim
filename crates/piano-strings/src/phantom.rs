//! Bank-Sujbert exact 78-pair trigonometric modal convolution for nonlinear
//! transverse-to-longitudinal phantom coupling (`no_std`, zero allocations).

use core::f64::consts::PI;

use crate::modal_soa::sqrt_f64;

/// Exclusive upper key index for `3x6` (`12 x 24`) bass phantom harmonic coupling (`keys 0..16`).
pub(crate) const BASS_PHANTOM_MAX_KEY_IDX: usize = 16;
/// Baseline SIMD4 vector count per transverse polarization for tenor/mid keys `16..52` (`8` modes).
pub(crate) const BASE_PHANTOM_TRANSVERSE_VECS: usize = 2;
/// Baseline transverse modes per polarization for keys `16..52` (`8`).
#[allow(dead_code)]
pub(crate) const BASE_PHANTOM_TRANSVERSE_MODES: usize = BASE_PHANTOM_TRANSVERSE_VECS * 4;
/// Baseline SIMD4 vector count of longitudinal modes for keys `16..52` (`16` modes).
pub(crate) const BASE_PHANTOM_LONG_VECS: usize = 4;
/// Baseline longitudinal modes for keys `16..52` (`16`).
#[cfg(test)]
pub(crate) const BASE_PHANTOM_LONG_MODES: usize = BASE_PHANTOM_LONG_VECS * 4;
/// Maximum SIMD4 vector count per transverse polarization coupled via SAV/phantom (`keys 0..16`: `3`).
pub(crate) const MAX_PHANTOM_TRANSVERSE_VECS: usize = 3;
/// Maximum transverse modes per polarization coupled via SAV/phantom (`<= 12`).
pub(crate) const MAX_PHANTOM_TRANSVERSE_MODES: usize = MAX_PHANTOM_TRANSVERSE_VECS * 4;
/// Maximum SIMD4 vector count of longitudinal modes coupled via SAV/phantom (`keys 0..16`: `6`).
pub(crate) const MAX_PHANTOM_LONG_VECS: usize = 6;
/// Maximum longitudinal modes coupled via SAV/phantom (`<= 24`).
pub(crate) const MAX_PHANTOM_LONG_MODES: usize = MAX_PHANTOM_LONG_VECS * 4;
/// 1-based longitudinal projection buffer length (`0` unused, `1..=24` active).
pub(crate) const PHANTOM_ARRAY_LEN: usize = MAX_PHANTOM_LONG_MODES + 1;

/// Maximum coupled transverse SIMD4 vectors per polarization for `key_idx`.
#[inline]
#[must_use]
pub(crate) const fn max_phantom_transverse_vecs_for_key(key_idx: usize) -> usize {
    if key_idx < BASS_PHANTOM_MAX_KEY_IDX {
        MAX_PHANTOM_TRANSVERSE_VECS
    } else if key_idx < piano_spec::LONGITUDINAL_MAX_KEY_IDX {
        BASE_PHANTOM_TRANSVERSE_VECS
    } else {
        0
    }
}

/// Maximum coupled longitudinal SIMD4 vectors for `key_idx`.
#[inline]
#[must_use]
pub(crate) const fn max_phantom_long_vecs_for_key(key_idx: usize) -> usize {
    if key_idx < BASS_PHANTOM_MAX_KEY_IDX {
        MAX_PHANTOM_LONG_VECS
    } else if key_idx < piano_spec::LONGITUDINAL_MAX_KEY_IDX {
        BASE_PHANTOM_LONG_VECS
    } else {
        0
    }
}

/// Maps modal oscillator state `v_{p,n}` to regularized physical modal slope `d_{p,n}`.
#[inline]
#[must_use]
pub(crate) fn slope_factor(partial_n: usize, length_m: f64, gamma_s: f64, omega_rad_s: f64) -> f64 {
    if partial_n == 0 || length_m <= 0.0 || gamma_s <= 0.0 || omega_rad_s <= 0.0 {
        return 0.0;
    }
    let nm1 = (partial_n - 1) as f64;
    let rolloff = sqrt_f64(1.0 + 0.015 * nm1 * nm1);
    ((partial_n as f64) * PI) / (length_m * gamma_s * omega_rad_s * rolloff)
}

/// Evaluates exact tension-stretching integral `S(v)` and modal convolution
/// projections `Phi_tilde_m(v)` for `m = 1..=m_ph`.
#[inline]
#[must_use]
pub(crate) fn convolve_transverse_slopes(
    length_m: f64,
    d_v: &[f64; MAX_PHANTOM_TRANSVERSE_MODES],
    d_h: &[f64; MAX_PHANTOM_TRANSVERSE_MODES],
    m_ph: usize,
) -> (f64, [f64; PHANTOM_ARRAY_LEN]) {
    convolve_transverse_slopes_n(length_m, d_v, d_h, m_ph)
}

#[must_use]
pub(crate) fn convolve_transverse_slopes_n<const N: usize, const ARR: usize>(
    length_m: f64,
    d_v: &[f64; N],
    d_h: &[f64; N],
    m_ph: usize,
) -> (f64, [f64; ARR]) {
    let mut s_v = 0.0_f64;
    let mut s_h = 0.0_f64;
    let mut phi_tilde = [0.0_f64; ARR];

    if m_ph == 0 {
        for a in 0..N {
            s_v += d_v[a] * d_v[a];
            s_h += d_h[a] * d_h[a];
        }
        return (0.25 * length_m * (s_v + s_h), phi_tilde);
    }

    let mut phi_v = [0.0_f64; ARR];
    let mut phi_h = [0.0_f64; ARR];

    for a in 0..N {
        let dv_a = d_v[a];
        let dh_a = d_h[a];
        let dv_sq = dv_a * dv_a;
        let dh_sq = dh_a * dh_a;
        s_v += dv_sq;
        s_h += dh_sq;

        let n = a + 1;
        phi_v[2 * n] += dv_sq;
        phi_h[2 * n] += dh_sq;

        for b in (a + 1)..N {
            let r = b + 1;
            let two_v = 2.0 * dv_a * d_v[b];
            let two_h = 2.0 * dh_a * d_h[b];
            phi_v[n + r] += two_v;
            phi_v[r - n] += two_v;
            phi_h[n + r] += two_h;
            phi_h[r - n] += two_h;
        }
    }

    let s_val = 0.25 * length_m * (s_v + s_h);
    let m_lim = m_ph.min(ARR - 1);
    let scale = 0.125 * length_m;
    for m in 1..=m_lim {
        phi_tilde[m] = scale * (phi_v[m] + phi_h[m]);
    }
    (s_val, phi_tilde)
}

/// Exact analytical transpose accumulating `DF(d)^T (w_s, w_phi)` into `(grad_v, grad_h)`.
/// Caller must ensure `w_phi[m] == 0.0` for `m == 0` and `m > m_ph`.
#[inline]
#[cfg(test)]
pub(crate) fn accum_phantom_grad_d<const N: usize, const ARR: usize>(
    length_m: f64,
    d_v: &[f64; N],
    d_h: &[f64; N],
    w_s: f64,
    w_phi: &[f64; ARR],
    grad_v: &mut [f64; N],
    grad_h: &mut [f64; N],
) {
    let ws_half_l = 0.5 * length_m * w_s;
    let qtr_l = 0.25 * length_m;

    for a in 0..N {
        let n = a + 1;
        let w_diag = ws_half_l + qtr_l * w_phi[2 * n];
        grad_v[a] += w_diag * d_v[a];
        grad_h[a] += w_diag * d_h[a];

        for b in (a + 1)..N {
            let r = b + 1;
            let w_pair = qtr_l * (w_phi[n + r] + w_phi[r - n]);
            grad_v[a] += w_pair * d_v[b];
            grad_v[b] += w_pair * d_v[a];
            grad_h[a] += w_pair * d_h[b];
            grad_h[b] += w_pair * d_h[a];
        }
    }
}

#[inline(always)]
fn convolve_pol_n<const NL: usize>(
    p0: piano_simd::F32x4,
    p1: piano_simd::F32x4,
) -> [piano_simd::F32x4; NL] {
    use piano_simd::F32x4;
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
    let v0 = (d0 + d0) + z.unpack_lo(p0_sq);
    let mut out = [v0; NL];
    if NL >= 2 {
        let t = vx0 * (s1_1z + s3_01) + vx1 * (s2_1z + s2_01) + vx2 * (s3_1z + z.slide2(s3_01));
        out[1] = (t + t) + z.unpack_hi(p0_sq);
    }
    if NL >= 3 {
        let p1_sq = p1 * p1;
        let two_sum2 = vx0 * s3_1z + vx1 * s2_1z + vx2 * s1_1z + vx3 * p1 + vx4 * z.slide2(s1_1z);
        out[2] = (two_sum2 + two_sum2) + z.unpack_lo(p1_sq);
        if NL >= 4 {
            let vx7 = p1.splat_lane::<3>();
            let t3 = vx7 * z.slide3(p1).slide1(z) + vx5 * s2_1z.reverse().slide3(z);
            out[3] = (t3 + t3) + z.unpack_hi(p1_sq);
        }
    }
    out
}

/// Evaluates unscaled `phi_sum[1..=NL*4]` across `NL` longitudinal SIMD4 vectors (`2` transverse vecs).
#[inline(always)]
pub(crate) fn convolve_transverse_slopes_f32_n<const NL: usize>(
    dv4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
) -> [piano_simd::F32x4; NL] {
    let pv = convolve_pol_n::<NL>(dv4[0], dv4[1]);
    let ph = convolve_pol_n::<NL>(dh4[0], dh4[1]);
    core::array::from_fn(|i| pv[i] + ph[i])
}

/// Evaluates unscaled `phi_sum[1..=16]` as 4 SIMD vectors (`[1..=4, 5..=8, 9..=12, 13..=16]`).
#[cfg(test)]
#[inline(always)]
pub(crate) fn convolve_transverse_slopes_f32(
    dv4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
) -> [piano_simd::F32x4; BASE_PHANTOM_LONG_VECS] {
    convolve_transverse_slopes_f32_n::<BASE_PHANTOM_LONG_VECS>(dv4, dh4)
}

/// Column-wise unrolled `F32x4` Toeplitz+Hankel adjoint gradient across `N = 8` V/H slope modes
/// specialized to `NL` active longitudinal SIMD4 vectors (`NL * 4` modes).
#[inline(always)]
pub(crate) fn accum_phantom_grad_vh_simd_n<const NL: usize>(
    dv4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    c_sum: f32,
    c_phi4: &[piano_simd::F32x4; NL],
) -> (
    [piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    [piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
) {
    use piano_simd::F32x4;
    let z = F32x4::zero();
    let q0 = c_phi4[0];
    let q1 = if NL >= 2 { c_phi4[1] } else { z };
    let q2 = if NL >= 3 { c_phi4[2] } else { z };
    let q3 = if NL >= 4 { c_phi4[3] } else { z };
    let r0 = q0.reverse();
    let r1 = q1.reverse();

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
    let q0_s1 = q0.slide1(q1);
    let q0_s2 = q0.slide2(q1);
    let q0_s3 = q0.slide3(q1);
    let q12_s1 = q1.slide1(q2);
    let q12_s2 = q1.slide2(q2);
    let q12_s3 = q1.slide3(q2);
    let p1_0 = r0.slide3(z);
    let t0 = z.slide3(q0);
    let t1 = p1_0 + z.slide2(q0);
    let t2 = r0.slide2(z) + p1_0.reverse();
    let t3 = r0.slide1(z);

    step_col!(0, 0, t0 + q0_s1, q0_s3 + q12_s1);
    step_col!(0, 1, t1 + q0_s2, q0_s2 + q12_s2);
    step_col!(0, 2, t2 + q0_s3, q0_s1 + q12_s3);
    step_col!(0, 3, t3 + q1, q0 + q2);
    step_col!(1, 0, r0 + q12_s1, t0 + q2.slide1(q3));
    step_col!(1, 1, r1.slide3(r0) + q12_s2, t1 + q2.slide2(q3));
    step_col!(1, 2, r1.slide2(r0) + q12_s3, t2 + q2.slide3(q3));
    step_col!(1, 3, r1.slide1(r0) + q2, t3 + q3);

    ([gv0, gv1], [gh0, gh1])
}

/// Column-wise unrolled `F32x4` Toeplitz+Hankel adjoint gradient across `N = 8` V/H slope modes.
#[cfg(test)]
#[inline(always)]
pub(crate) fn accum_phantom_grad_vh_simd(
    dv4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    c_sum: f32,
    c_phi4: &[piano_simd::F32x4; BASE_PHANTOM_LONG_VECS],
) -> (
    [piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
    [piano_simd::F32x4; BASE_PHANTOM_TRANSVERSE_VECS],
) {
    accum_phantom_grad_vh_simd_n(dv4, dh4, c_sum, c_phi4)
}

#[cfg(test)]
#[path = "phantom_tests.rs"]
mod tests;
