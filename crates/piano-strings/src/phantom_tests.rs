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
    const NVH: usize = BASE_PHANTOM_TRANSVERSE_MODES;
    const BASE_ARR_LEN: usize = BASE_PHANTOM_LONG_MODES + 1;
    assert_adjoint_under_1e12::<NVH, BASE_ARR_LEN>(&[4, 10, BASE_PHANTOM_LONG_MODES]);
    let (length_m, m_ph) = (1.42_f64, BASE_PHANTOM_LONG_MODES);
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
    let (_, phi_f64) =
        convolve_transverse_slopes_n::<NVH, BASE_ARR_LEN>(length_m, &d_v, &d_h, m_ph);
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
    let (_, phi_d64) = convolve_transverse_slopes_n::<NVH, BASE_ARR_LEN>(1.0, &tv64, &th64, 16);
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
    let (w_s, mut w_phi, mut c_phi) = (1.37_f64, [0.0; BASE_ARR_LEN], [0.0_f32; BASE_ARR_LEN]);
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
    for v in 0..BASE_PHANTOM_TRANSVERSE_VECS {
        for l in 0..4 {
            assert!((f64::from(gv4[v].to_array()[l]) - gv_ref[4 * v + l]).abs() < 1e-7);
            assert!((f64::from(gh4[v].to_array()[l]) - gh_ref[4 * v + l]).abs() < 1e-7);
        }
    }
    for nl in 1..=3usize {
        let mut w_sub = [0.0_f64; BASE_ARR_LEN];
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
        for v in 0..BASE_PHANTOM_TRANSVERSE_VECS {
            for l in 0..4 {
                assert!((f64::from(gva[v].to_array()[l]) - gv_s[4 * v + l]).abs() < 1e-7);
                assert!((f64::from(gha[v].to_array()[l]) - gh_s[4 * v + l]).abs() < 1e-7);
            }
        }
    }
}

#[test]
fn phantom_runtime_3x6_12x24_f32_and_simd_adjoint_matches_f64_oracle() {
    use crate::phantom_3x6::{accum_phantom_grad_vh_simd_3x6, convolve_transverse_slopes_f32_3x6};
    use piano_simd::F32x4;

    const NVH: usize = MAX_PHANTOM_TRANSVERSE_MODES;
    const NL: usize = MAX_PHANTOM_LONG_MODES;
    let to4 = |s: &[f32]| F32x4::from_array([s[0], s[1], s[2], s[3]]);

    // Dense unit-scale check across all 12 V/H modes and 24 longitudinal modes.
    let tv_s: [f32; NVH] = [
        0.7, -0.3, 0.5, -0.9, 0.2, -0.6, 0.4, -0.8, 0.35, -0.45, 0.65, -0.25,
    ];
    let th_s: [f32; NVH] = [
        -0.4, 0.8, -0.1, 0.3, -0.5, 0.2, -0.7, 0.6, -0.55, 0.15, -0.85, 0.42,
    ];
    let tv64: [f64; NVH] = core::array::from_fn(|i| f64::from(tv_s[i]));
    let th64: [f64; NVH] = core::array::from_fn(|i| f64::from(th_s[i]));
    let length_m = 1.85_f64;
    let (_, phi_d64) = convolve_transverse_slopes(length_m, &tv64, &th64, NL);

    let dv4 = [to4(&tv_s[0..]), to4(&tv_s[4..]), to4(&tv_s[8..])];
    let dh4 = [to4(&th_s[0..]), to4(&th_s[4..]), to4(&th_s[8..])];
    let phi_d = convolve_transverse_slopes_f32_3x6(&dv4, &dh4);
    let scale = (0.125 * length_m) as f32;
    for (vi, p) in phi_d.iter().enumerate() {
        for (l, &val) in p.to_array().iter().enumerate() {
            let m = 4 * vi + l + 1;
            let diff = (f64::from(val * scale) - phi_d64[m]).abs();
            assert!(diff < 1e-5, "3x6 forward mismatch at m={m}: diff={diff:e}");
        }
    }

    // Adjoint gradient parity across all 24 modes and partial m_ph = 18.
    let w_s = 1.37_f64;
    let k_ws = (0.5 * length_m * w_s) as f32;
    for m_ph in [18usize, NL] {
        let mut w_phi = [0.0_f64; PHANTOM_ARRAY_LEN];
        let mut c_phi = [0.0_f32; PHANTOM_ARRAY_LEN];
        for m in 1..=m_ph {
            w_phi[m] = 0.85 * ((m as f64) * 0.43).sin() + 0.25;
            c_phi[m] = (0.25 * length_m * w_phi[m]) as f32;
        }
        let (mut gv_ref, mut gh_ref) = ([0.0_f64; NVH], [0.0_f64; NVH]);
        accum_phantom_grad_d(
            length_m,
            &tv64,
            &th64,
            w_s,
            &w_phi,
            &mut gv_ref,
            &mut gh_ref,
        );
        let c_phi4: [F32x4; MAX_PHANTOM_LONG_VECS] =
            core::array::from_fn(|vi| to4(&c_phi[1 + vi * 4..]));
        let (gv4, gh4) = accum_phantom_grad_vh_simd_3x6(&dv4, &dh4, k_ws, &c_phi4);
        for v in 0..MAX_PHANTOM_TRANSVERSE_VECS {
            for l in 0..4 {
                let dv_err = (f64::from(gv4[v].to_array()[l]) - gv_ref[4 * v + l]).abs();
                let dh_err = (f64::from(gh4[v].to_array()[l]) - gh_ref[4 * v + l]).abs();
                assert!(
                    dv_err < 2e-5 && dh_err < 2e-5,
                    "m_ph={m_ph} v={v} l={l}: dv_err={dv_err:e} dh_err={dh_err:e}"
                );
            }
        }
    }
}

#[test]
fn bass_k0_to_15_uses_3x6_while_k16_to_51_stays_2x4_and_n_pol_le_3() {
    use piano_spec::{VoicingSpec, LONGITUDINAL_MAX_KEY_IDX, TOTAL_STRINGS};

    let bank = crate::StringBank::new_bridge_coupled(48_000.0, &VoicingSpec::default());
    for s_idx in 0..TOTAL_STRINGS {
        let entry = &bank.choir_map().strings()[s_idx];
        let k = usize::from(entry.key_idx);
        let sp = bank.sav.spec(s_idx);
        let sl = &bank.sav.slopes[s_idx];
        let meta = &bank.bridge_ports.meta[s_idx];
        let n_pol = meta.v_br.len();
        assert_eq!(n_pol, meta.h_br.len());
        if k < BASS_PHANTOM_MAX_KEY_IDX {
            assert_eq!(sp.nv_vecs, 3, "k={k} nv_vecs");
            assert_eq!(sp.nh_vecs, 3, "k={k} nh_vecs");
            assert_eq!(sl.nv_c, 12, "k={k} nv_c");
            assert_eq!(sl.nh_c, 12, "k={k} nh_c");
            assert!((17..=24).contains(&sp.m_ph), "k={k} m_ph={}", sp.m_ph);
            assert!((5..=6).contains(&sp.nl_vecs), "k={k} nl_vecs");
            assert!((5..=6).contains(&n_pol), "k={k} n_pol={n_pol}");
        } else if k < LONGITUDINAL_MAX_KEY_IDX {
            assert!(sp.nv_vecs <= 2 && sp.nh_vecs <= 2, "k={k} stayed 2x4");
            assert!(sl.nv_c <= 8 && sl.nh_c <= 8, "k={k} trans modes <= 8");
            assert!(sp.m_ph <= 16 && sp.nl_vecs <= 4, "k={k} long <= 16");
            assert!((2..=3).contains(&n_pol), "k={k} n_pol={n_pol}");
        } else {
            assert_eq!(sp.n_vh, 0);
            assert_eq!(sp.m_ph, 0);
            assert!(n_pol <= 1, "k={k} n_pol={n_pol}");
        }
    }
}

#[test]
fn bass_3x6_lossless_sav_conservation_and_coupled_bridge_passivity() {
    use piano_spec::{VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

    struct PassiveDriver;
    impl crate::KeyActionDriver for PassiveDriver {
        fn is_hammer_active(&self, _: usize) -> bool {
            false
        }
        fn damper_clearance_m(&self, _: usize) -> f32 {
            0.01
        }
        fn strike_weights(&self, _: usize) -> [f32; 3] {
            [1.0; 3]
        }
        fn step_key_clearance(&mut self, _: usize) -> f32 {
            0.01
        }
        fn step_key_clearance_tile(&mut self, _: usize, t: usize, out: &mut [f32; 8]) -> f32 {
            out[..t].fill(0.01);
            0.01
        }
        fn step_hammer_sample(&mut self, _: usize, _: f32, _: f32, _: f32) -> f32 {
            0.0
        }
        fn step_damper_sample(
            &mut self,
            _: usize,
            _: usize,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> f32 {
            0.0
        }
    }
    struct PassiveBridge {
        r0: [f32; TOTAL_STRINGS],
        zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
    }
    impl crate::BridgePortDriver for PassiveBridge {
        fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
            &self.r0
        }
        fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS] {
            &self.zw
        }
        fn step_bridge_sample(
            &mut self,
            a: &[f32; TOTAL_STRINGS],
            _: &[f32; 8],
            b: &mut [f32; TOTAL_STRINGS],
            zone: &mut [f32; 8],
        ) -> (f32, f32) {
            for (dst, &src) in b.iter_mut().zip(a.iter()) {
                *dst = -0.99 * src;
            }
            zone.fill(0.0);
            (0.0, 0.0)
        }
    }

    // 1. Lossless SAV conservation across render_key_subblock (sav_tile_3x6) & render_subblock.
    let voicing = VoicingSpec::default();
    let mut lossless = crate::StringBank::new_lossless(48_000.0, &voicing);
    let (mut l, mut r) = ([0.0_f32; 64], [0.0_f32; 64]);
    for &key in &[0usize, 15] {
        for use_subblock in [false, true] {
            lossless.reset();
            lossless.apply_excitation_sample(key, [1.0, 1.0, 1.0], 24.0);
            let h0 = lossless.key_sav_energy_joules(key);
            let mut max_rel_err = 0.0_f64;
            for _ in 0..400 {
                if use_subblock {
                    lossless.render_subblock(&mut l, &mut r);
                } else {
                    lossless.render_key_subblock(key, &mut l, &mut r);
                }
                let h_cur = lossless.key_sav_energy_joules(key);
                max_rel_err = max_rel_err.max((h_cur - h0).abs() / h0);
            }
            assert!(
                max_rel_err < 1.0e-5,
                "lossless 3x6 k={key} subblock={use_subblock} drift={max_rel_err:e}"
            );
        }
    }

    // 2. Coupled bridge-port 3x6 SAV stepping (bridge_sav::step_sav_scheduled_3x6) excites
    //    modes 17..=24, preserves passivity, and tracks psi.
    let mut bank_on = crate::StringBank::new_bridge_coupled(48_000.0, &voicing);
    let mut bank_off = crate::StringBank::new_bridge_coupled(48_000.0, &voicing);
    bank_off.set_nonlinear_coupling(true, false);
    for b in [&mut bank_on, &mut bank_off] {
        b.apply_excitation_sample(0, [1.0, 0.0, 0.0], 24.0);
    }
    let mut drv = PassiveDriver;
    let mut br = PassiveBridge {
        r0: *bank_on.bridge_port_resistances(),
        zw: [(0, 1.0, 0, 0.0); TOTAL_STRINGS],
    };
    let (mut bo_l, mut bo_r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let (mut bf_l, mut bf_r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let mut h_prev = bank_on.key_sav_energy_joules(0);
    let h0 = h_prev;
    for _ in 0..40 {
        bank_on.render_subblock_bridge_coupled_no_flush(&mut drv, &mut br, &mut bo_l, &mut bo_r);
        bank_off.render_subblock_bridge_coupled_no_flush(&mut drv, &mut br, &mut bf_l, &mut bf_r);
        assert!(bo_l.iter().all(|x| x.is_finite()) && bo_r.iter().all(|x| x.is_finite()));
        let h_cur = bank_on.key_sav_energy_joules(0);
        assert!(
            h_cur <= h_prev + 1.0e-7 * h0,
            "3x6 coupled passivity violation: h_prev={h_prev:e} h_cur={h_cur:e}"
        );
        h_prev = h_cur;
    }
    bank_on.flush_bridge_subnormals(&[false; NUM_KEYS]);
    let tl_l = bank_on.bridge_ports.meta[0].tl_l.clone();
    assert!(tl_l.len() >= 6);
    let high_long_e: f32 = bank_on.bridge_ports.tl_soa.u.as_f32x4_slice()
        [tl_l.start + 4..tl_l.start + 6]
        .iter()
        .zip(&bank_on.bridge_ports.tl_soa.v.as_f32x4_slice()[tl_l.start + 4..tl_l.start + 6])
        .map(|(u, v)| (*u * *u + *v * *v).reduce_sum())
        .sum();
    let high_long_off: f32 = bank_off.bridge_ports.tl_soa.u.as_f32x4_slice()
        [tl_l.start + 4..tl_l.start + 6]
        .iter()
        .zip(&bank_off.bridge_ports.tl_soa.v.as_f32x4_slice()[tl_l.start + 4..tl_l.start + 6])
        .map(|(u, v)| (*u * *u + *v * *v).reduce_sum())
        .sum();
    assert!(
        high_long_e > high_long_off * 10.0 + 1.0e-12,
        "modes 17..=24 were not excited by 3x6 phantom: on={high_long_e:e} off={high_long_off:e}"
    );
    let e_nl = bank_on.key_nonlinear_energy_joules(0);
    let psi_expected = crate::modal_soa::sqrt_f64(2.0 * (e_nl / bank_on.ports.beta_s[0]) + 1.0);
    assert!((bank_on.sav.r_sav[0] - psi_expected).abs() < 0.02);
}
