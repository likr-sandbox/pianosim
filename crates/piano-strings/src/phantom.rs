//! Bank-Sujbert exact 78-pair trigonometric modal convolution for nonlinear
//! transverse-to-longitudinal phantom coupling (`no_std`, zero allocations).

use core::f64::consts::PI;

use crate::modal_soa::sqrt_f64;

/// Maximum SIMD4 vector count per transverse polarization coupled via SAV/phantom.
pub(crate) const MAX_PHANTOM_TRANSVERSE_VECS: usize = 2;
/// Maximum transverse modes per polarization coupled via SAV/phantom (`<= 8`).
pub(crate) const MAX_PHANTOM_TRANSVERSE_MODES: usize = MAX_PHANTOM_TRANSVERSE_VECS * 4;
/// Maximum SIMD4 vector count of longitudinal modes coupled via SAV/phantom.
pub(crate) const MAX_PHANTOM_LONG_VECS: usize = 4;
/// Maximum longitudinal modes coupled via SAV/phantom (`<= 16`).
pub(crate) const MAX_PHANTOM_LONG_MODES: usize = MAX_PHANTOM_LONG_VECS * 4;
/// 1-based longitudinal projection buffer length (`0` unused, `1..=16` active).
pub(crate) const PHANTOM_ARRAY_LEN: usize = MAX_PHANTOM_LONG_MODES + 1;

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
fn v_rev(x: piano_simd::F32x4) -> piano_simd::F32x4 {
    let [a, b, c, d] = x.to_array();
    piano_simd::F32x4::from_array([d, c, b, a])
}

#[inline(always)]
fn slide1(x: piano_simd::F32x4, y: piano_simd::F32x4) -> piano_simd::F32x4 {
    let [_, x1, x2, x3] = x.to_array();
    let [y0, _, _, _] = y.to_array();
    piano_simd::F32x4::from_array([x1, x2, x3, y0])
}

#[inline(always)]
fn slide2(x: piano_simd::F32x4, y: piano_simd::F32x4) -> piano_simd::F32x4 {
    let [_, _, x2, x3] = x.to_array();
    let [y0, y1, _, _] = y.to_array();
    piano_simd::F32x4::from_array([x2, x3, y0, y1])
}

#[inline(always)]
fn slide3(x: piano_simd::F32x4, y: piano_simd::F32x4) -> piano_simd::F32x4 {
    let [_, _, _, x3] = x.to_array();
    let [y0, y1, y2, _] = y.to_array();
    piano_simd::F32x4::from_array([x3, y0, y1, y2])
}

#[inline(always)]
fn convolve_pol_n<const NL: usize>(
    p0: piano_simd::F32x4,
    p1: piano_simd::F32x4,
) -> [piano_simd::F32x4; NL] {
    use piano_simd::F32x4;
    let [x0, x1, x2, x3] = p0.to_array();
    let [x4, x5, x6, x7] = p1.to_array();
    let z = F32x4::zero();
    let (s1_01, s2_01, s3_01) = (slide1(p0, p1), slide2(p0, p1), slide3(p0, p1));
    let (s1_1z, s2_1z, s3_1z) = (slide1(p1, z), slide2(p1, z), slide3(p1, z));
    let (vx0, vx1, vx2, vx3) = (
        F32x4::splat(x0),
        F32x4::splat(x1),
        F32x4::splat(x2),
        F32x4::splat(x3),
    );
    let (vx4, vx5, vx6) = (F32x4::splat(x4), F32x4::splat(x5), F32x4::splat(x6));

    let d0 = vx0 * s1_01
        + vx1 * s2_01
        + vx2 * s3_01
        + vx3 * p1
        + vx4 * s1_1z
        + vx5 * s2_1z
        + vx6 * s3_1z;
    let v0 = (d0 + d0) + F32x4::from_array([0.0, x0 * x0, 2.0 * x0 * x1, x1 * x1 + 2.0 * x0 * x2]);
    let mut out = [v0; NL];
    if NL >= 2 {
        let d1 = vx0 * s1_1z + vx1 * s2_1z + vx2 * s3_1z;
        let two_sum1 = vx0 * s3_01 + vx1 * s2_01 + vx2 * slide2(z, s3_01);
        let t = d1 + two_sum1;
        out[1] = (t + t) + F32x4::from_array([0.0, x2 * x2, 0.0, x3 * x3]);
    }
    if NL >= 3 {
        let two_sum2 = vx0 * s3_1z + vx1 * s2_1z + vx2 * s1_1z + vx3 * p1 + vx4 * slide2(z, s1_1z);
        out[2] = (two_sum2 + two_sum2) + F32x4::from_array([0.0, x4 * x4, 0.0, x5 * x5]);
    }
    if NL >= 4 {
        out[3] = F32x4::from_array([
            2.0 * (x4 * x7 + x5 * x6),
            2.0 * x5 * x7 + x6 * x6,
            2.0 * x6 * x7,
            x7 * x7,
        ]);
    }
    out
}

/// Evaluates unscaled `phi_sum[1..=NL*4]` across `NL` longitudinal SIMD4 vectors.
#[inline(always)]
pub(crate) fn convolve_transverse_slopes_f32_n<const NL: usize>(
    dv4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) -> [piano_simd::F32x4; NL] {
    let pv = convolve_pol_n::<NL>(dv4[0], dv4[1]);
    let ph = convolve_pol_n::<NL>(dh4[0], dh4[1]);
    core::array::from_fn(|i| pv[i] + ph[i])
}

/// Evaluates unscaled `phi_sum[1..=16]` as 4 SIMD vectors (`[1..=4, 5..=8, 9..=12, 13..=16]`).
#[cfg(test)]
#[inline(always)]
pub(crate) fn convolve_transverse_slopes_f32(
    dv4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) -> [piano_simd::F32x4; MAX_PHANTOM_LONG_VECS] {
    convolve_transverse_slopes_f32_n::<MAX_PHANTOM_LONG_VECS>(dv4, dh4)
}

/// Column-wise unrolled `F32x4` Toeplitz+Hankel adjoint gradient across `N = 8` V/H slope modes
/// specialized to `NL` active longitudinal SIMD4 vectors (`NL * 4` modes).
#[inline(always)]
pub(crate) fn accum_phantom_grad_vh_simd_n<const NL: usize>(
    dv4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    c_sum: f32,
    c_phi4: &[piano_simd::F32x4; NL],
) -> (
    [piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    [piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) {
    use piano_simd::F32x4;
    let z = F32x4::zero();
    let q0 = c_phi4[0];
    let q1 = if NL >= 2 { c_phi4[1] } else { z };
    let q2 = if NL >= 3 { c_phi4[2] } else { z };
    let q3 = if NL >= 4 { c_phi4[3] } else { z };
    let [p1, p2, _, _] = q0.to_array();
    let r0 = v_rev(q0);
    let r1 = v_rev(q1);

    let [v0, v1, v2, v3] = dv4[0].to_array();
    let [v4, v5, v6, v7] = dv4[1].to_array();
    let dv = [v0, v1, v2, v3, v4, v5, v6, v7];
    let [h0, h1, h2, h3] = dh4[0].to_array();
    let [h4, h5, h6, h7] = dh4[1].to_array();
    let dh = [h0, h1, h2, h3, h4, h5, h6, h7];

    let c_sum4 = F32x4::splat(c_sum);
    let (mut gv0, mut gv1) = (c_sum4 * dv4[0], c_sum4 * dv4[1]);
    let (mut gh0, mut gh1) = (c_sum4 * dh4[0], c_sum4 * dh4[1]);

    macro_rules! step_col {
        ($b:expr, $k0:expr, $k1:expr) => {{
            let dvb = F32x4::splat(dv[$b]);
            let dhb = F32x4::splat(dh[$b]);
            let k0b = $k0;
            let k1b = $k1;
            gv0 = k0b.mul_add(dvb, gv0);
            gv1 = k1b.mul_add(dvb, gv1);
            gh0 = k0b.mul_add(dhb, gh0);
            gh1 = k1b.mul_add(dhb, gh1);
        }};
    }
    let q0_s1 = slide1(q0, q1);
    let q0_s2 = slide2(q0, q1);
    let q0_s3 = slide3(q0, q1);
    let q12_s1 = slide1(q1, q2);
    let q12_s2 = slide2(q1, q2);
    let q12_s3 = slide3(q1, q2);
    let t0 = slide3(z, q0);
    let t1 = F32x4::from_array([p1, 0.0, p1, p2]);
    let t2 = F32x4::from_array([p2, p1, 0.0, p1]);
    let t3 = slide1(r0, z);

    step_col!(0, t0 + q0_s1, q0_s3 + q12_s1);
    step_col!(1, t1 + q0_s2, q0_s2 + q12_s2);
    step_col!(2, t2 + q0_s3, q0_s1 + q12_s3);
    step_col!(3, t3 + q1, q0 + q2);
    step_col!(4, r0 + q12_s1, t0 + slide1(q2, q3));
    step_col!(5, slide3(r1, r0) + q12_s2, t1 + slide2(q2, q3));
    step_col!(6, slide2(r1, r0) + q12_s3, t2 + slide3(q2, q3));
    step_col!(7, slide1(r1, r0) + q2, t3 + q3);

    ([gv0, gv1], [gh0, gh1])
}

/// Column-wise unrolled `F32x4` Toeplitz+Hankel adjoint gradient across `N = 8` V/H slope modes.
#[cfg(test)]
#[inline(always)]
pub(crate) fn accum_phantom_grad_vh_simd(
    dv4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    dh4: &[piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    c_sum: f32,
    c_phi4: &[piano_simd::F32x4; MAX_PHANTOM_LONG_VECS],
) -> (
    [piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    [piano_simd::F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
) {
    accum_phantom_grad_vh_simd_n(dv4, dh4, c_sum, c_phi4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_adjoint_under_1e12<const N: usize, const ARR: usize>(m_cases: &[usize]) {
        let length_m = 1.42_f64;
        let mut d_v = [0.0_f64; N];
        let mut d_h = [0.0_f64; N];
        let mut dd_v = [0.0_f64; N];
        let mut dd_h = [0.0_f64; N];
        for i in 0..N {
            let f = (i + 1) as f64;
            d_v[i] = 0.015 * (0.7 * f).sin() / (1.0 + 0.2 * f);
            d_h[i] = -0.011 * (1.1 * f).cos() / (1.0 + 0.15 * f);
            dd_v[i] = 0.008 * (1.3 * f + 0.4).cos();
            dd_h[i] = 0.006 * (0.9 * f - 0.3).sin();
        }

        for &m_ph in m_cases {
            let w_s = 1.37_f64;
            let mut w_phi = [0.0_f64; ARR];
            for (m, w) in w_phi[1..=m_ph].iter_mut().enumerate() {
                *w = 0.85 * ((m as f64 + 1.0) * 0.43).sin() + 0.25;
            }

            // Central difference on homogeneous quadratic S, Phi avoids cancellation at eps=1e-2.
            let eps = 1.0e-2_f64;
            let (mut d_vp, mut d_hp, mut d_vm, mut d_hm) = (d_v, d_h, d_v, d_h);
            for i in 0..N {
                d_vp[i] += eps * dd_v[i];
                d_hp[i] += eps * dd_h[i];
                d_vm[i] -= eps * dd_v[i];
                d_hm[i] -= eps * dd_h[i];
            }
            let (s_p, phi_p) = convolve_transverse_slopes_n::<N, ARR>(length_m, &d_vp, &d_hp, m_ph);
            let (s_m, phi_m) = convolve_transverse_slopes_n::<N, ARR>(length_m, &d_vm, &d_hm, m_ph);
            let mut lhs = w_s * (s_p - s_m) / (2.0 * eps);
            for m in 1..=m_ph {
                lhs += w_phi[m] * (phi_p[m] - phi_m[m]) / (2.0 * eps);
            }

            let (mut gv, mut gh) = ([0.0_f64; N], [0.0_f64; N]);
            accum_phantom_grad_d::<N, ARR>(length_m, &d_v, &d_h, w_s, &w_phi, &mut gv, &mut gh);
            let rhs: f64 = (0..N).map(|i| gv[i] * dd_v[i] + gh[i] * dd_h[i]).sum();
            let diff = (lhs - rhs).abs();
            let rel = diff / lhs.abs().max(1e-20);
            assert!(rel < 1e-12 && diff < 1e-12, "N={N} m={m_ph} rel={rel:e}");
        }
    }

    #[test]
    fn phantom_78_pair_adjoint_matches_directional_derivative_under_1e12() {
        assert_adjoint_under_1e12::<12, 25>(&[6, 15, 24]);
        let (mut d_pair, zero_h) = ([0.0_f64; 12], [0.0_f64; 12]);
        d_pair[1] = 0.5;
        d_pair[4] = 0.3;
        let (s_pair, phi_pair) = convolve_transverse_slopes_n::<12, 25>(1.0, &d_pair, &zero_h, 24);
        for &(a, b) in &[
            (s_pair, 0.25 * (0.25 + 0.09)),
            (phi_pair[4], 0.125 * 0.25),
            (phi_pair[10], 0.125 * 0.09),
            (phi_pair[7], 0.125 * 0.30),
            (phi_pair[3], 0.125 * 0.30),
        ] {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn phantom_runtime_36_pair_adjoint_matches_f32_and_simd() {
        const NVH: usize = MAX_PHANTOM_TRANSVERSE_MODES;
        assert_adjoint_under_1e12::<NVH, PHANTOM_ARRAY_LEN>(&[4, 10, MAX_PHANTOM_LONG_MODES]);
        let (length_m, m_ph) = (1.42_f64, MAX_PHANTOM_LONG_MODES);
        let (mut d_v, mut d_h) = ([0.0_f64; NVH], [0.0_f64; NVH]);
        let (mut dv_s, mut dh_s) = ([0.0_f32; NVH], [0.0_f32; NVH]);
        for i in 0..NVH {
            let f = (i + 1) as f64;
            d_v[i] = 0.015 * (0.7 * f).sin() / (1.0 + 0.2 * f);
            d_h[i] = -0.011 * (1.1 * f).cos() / (1.0 + 0.15 * f);
            dv_s[i] = d_v[i] as f32;
            dh_s[i] = d_h[i] as f32;
        }
        let to4 = |s: &[f32]| piano_simd::F32x4::from_array([s[0], s[1], s[2], s[3]]);
        let (_, phi_f64) = convolve_transverse_slopes(length_m, &d_v, &d_h, m_ph);
        let (dv4, dh4) = ([to4(&dv_s), to4(&dv_s[4..])], [to4(&dh_s), to4(&dh_s[4..])]);
        let phi_vecs = convolve_transverse_slopes_f32(&dv4, &dh4);
        let scale = (0.125 * length_m) as f32;
        for (vi, vec) in phi_vecs.iter().enumerate() {
            for (lane, &x) in vec.to_array().iter().enumerate() {
                let m = 4 * vi + lane + 1;
                assert!((f64::from(x * scale) - phi_f64[m]).abs() < 1e-11, "m={m}");
            }
        }
        // Dense unit-scale check across all NL=4 lanes.
        let tv_s = [0.7_f32, -0.3, 0.5, -0.9, 0.2, -0.6, 0.4, -0.8];
        let th_s = [-0.4_f32, 0.8, -0.1, 0.3, -0.5, 0.2, -0.7, 0.6];
        let tv64: [f64; 8] = core::array::from_fn(|i| f64::from(tv_s[i]));
        let th64: [f64; 8] = core::array::from_fn(|i| f64::from(th_s[i]));
        let (_, phi_d64) = convolve_transverse_slopes(1.0, &tv64, &th64, 16);
        let phi_d = convolve_transverse_slopes_f32(
            &[to4(&tv_s), to4(&tv_s[4..])],
            &[to4(&th_s), to4(&th_s[4..])],
        );
        for (vi, p) in phi_d.iter().enumerate() {
            for (l, &val) in p.to_array().iter().enumerate() {
                assert!((f64::from(val * 0.125) - phi_d64[4 * vi + l + 1]).abs() < 1e-6);
            }
        }
        let (p1, p2, p3) = (
            convolve_transverse_slopes_f32_n::<1>(&dv4, &dh4),
            convolve_transverse_slopes_f32_n::<2>(&dv4, &dh4),
            convolve_transverse_slopes_f32_n::<3>(&dv4, &dh4),
        );
        for &(val, ref_m) in &[
            (p1[0].to_array()[0], phi_f64[1]),
            (p1[0].to_array()[3], phi_f64[4]),
            (p2[1].to_array()[0], phi_f64[5]),
            (p2[1].to_array()[3], phi_f64[8]),
            (p3[2].to_array()[0], phi_f64[9]),
            (p3[2].to_array()[3], phi_f64[12]),
        ] {
            assert!((f64::from(val * scale) - ref_m).abs() < 1e-8);
        }
        let (w_s, mut w_phi, mut c_phi) = (
            1.37_f64,
            [0.0; PHANTOM_ARRAY_LEN],
            [0.0_f32; PHANTOM_ARRAY_LEN],
        );
        for m in 1..=m_ph {
            w_phi[m] = 0.85 * ((m as f64) * 0.43).sin() + 0.25;
            c_phi[m] = (0.25 * length_m * w_phi[m]) as f32;
        }
        let (mut gv_ref, mut gh_ref) = ([0.0_f64; NVH], [0.0_f64; NVH]);
        accum_phantom_grad_d(length_m, &d_v, &d_h, w_s, &w_phi, &mut gv_ref, &mut gh_ref);
        let c_phi4 = [
            to4(&c_phi[1..]),
            to4(&c_phi[5..]),
            to4(&c_phi[9..]),
            to4(&c_phi[13..]),
        ];
        let k_ws = (0.5 * length_m * w_s) as f32;
        let (gv4, gh4) = accum_phantom_grad_vh_simd(&dv4, &dh4, k_ws, &c_phi4);
        for v in 0..MAX_PHANTOM_TRANSVERSE_VECS {
            for l in 0..4 {
                assert!((f64::from(gv4[v].to_array()[l]) - gv_ref[4 * v + l]).abs() < 1e-7);
                assert!((f64::from(gh4[v].to_array()[l]) - gh_ref[4 * v + l]).abs() < 1e-7);
            }
        }
        for nl in 1..=3usize {
            let mut w_sub = [0.0_f64; PHANTOM_ARRAY_LEN];
            w_sub[1..=nl * 4].copy_from_slice(&w_phi[1..=nl * 4]);
            let (mut gv_s, mut gh_s) = ([0.0_f64; 8], [0.0_f64; 8]);
            accum_phantom_grad_d(length_m, &d_v, &d_h, w_s, &w_sub, &mut gv_s, &mut gh_s);
            let (gva, gha) = match nl {
                1 => accum_phantom_grad_vh_simd_n::<1>(&dv4, &dh4, k_ws, &[c_phi4[0]]),
                2 => accum_phantom_grad_vh_simd_n::<2>(&dv4, &dh4, k_ws, &[c_phi4[0], c_phi4[1]]),
                _ => accum_phantom_grad_vh_simd_n::<3>(
                    &dv4,
                    &dh4,
                    k_ws,
                    &[c_phi4[0], c_phi4[1], c_phi4[2]],
                ),
            };
            for v in 0..2 {
                for l in 0..4 {
                    assert!((f64::from(gva[v].to_array()[l]) - gv_s[4 * v + l]).abs() < 1e-7);
                    assert!((f64::from(gha[v].to_array()[l]) - gh_s[4 * v + l]).abs() < 1e-7);
                }
            }
        }
    }
}
