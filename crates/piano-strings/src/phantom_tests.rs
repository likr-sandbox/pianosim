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
