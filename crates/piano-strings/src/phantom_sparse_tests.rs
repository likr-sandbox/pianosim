use crate::phantom::{accum_phantom_grad_d, convolve_transverse_slopes_n, BASE_PHANTOM_LONG_MODES};
use crate::phantom_sparse::{
    accum_phantom_grad_vh_simd_2x2, convolve_transverse_slopes_f32_2x2, SPARSE_PHANTOM_LONG_MODES,
    SPARSE_PHANTOM_LONG_VECS, SPARSE_PHANTOM_TRANSVERSE_MODES, SPARSE_PHANTOM_TRANSVERSE_VECS,
};
use crate::{BridgePortDriver, KeyActionDriver, StringBank};
use piano_simd::F32x4;
use piano_spec::{PerfProfile, VoicingSpec, LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

#[test]
fn sparse_2x2_forward_and_adjoint_match_f64_oracle_and_directional_derivative() {
    const NVH: usize = SPARSE_PHANTOM_TRANSVERSE_MODES;
    const NL: usize = SPARSE_PHANTOM_LONG_MODES;
    const ARR: usize = BASE_PHANTOM_LONG_MODES + 1;
    let to4 = |s: &[f32]| F32x4::from_array([s[0], s[1], s[2], s[3]]);

    let tv_s: [f32; NVH] = [0.7, -0.3, 0.5, -0.9, 0.2, -0.6, 0.4, -0.8];
    let th_s: [f32; NVH] = [-0.4, 0.8, -0.1, 0.3, -0.5, 0.2, -0.7, 0.6];
    let tv64: [f64; NVH] = core::array::from_fn(|i| f64::from(tv_s[i]));
    let th64: [f64; NVH] = core::array::from_fn(|i| f64::from(th_s[i]));
    let dd_v: [f64; NVH] = core::array::from_fn(|i| 0.008 * ((i as f64 + 1.0) * 1.3 + 0.4).cos());
    let dd_h: [f64; NVH] = core::array::from_fn(|i| 0.006 * ((i as f64 + 1.0) * 0.9 - 0.3).sin());

    let length_m = 1.85_f64;
    let (_, phi_d64) = convolve_transverse_slopes_n::<NVH, ARR>(length_m, &tv64, &th64, NL);
    let dv4 = [to4(&tv_s[0..]), to4(&tv_s[4..])];
    let dh4 = [to4(&th_s[0..]), to4(&th_s[4..])];
    let phi_d = convolve_transverse_slopes_f32_2x2(&dv4, &dh4);
    let scale = (0.125 * length_m) as f32;
    for (vi, p) in phi_d.iter().enumerate() {
        for (l, &val) in p.to_array().iter().enumerate() {
            let m = 4 * vi + l + 1;
            let diff = (f64::from(val * scale) - phi_d64[m]).abs();
            assert!(diff < 1e-5, "2x2 forward mismatch m={m}: diff={diff:e}");
        }
    }

    let w_s = 1.37_f64;
    let mut w_phi = [0.0_f64; ARR];
    let mut c_phi = [0.0_f32; ARR];
    for m in 1..=NL {
        w_phi[m] = 0.85 * ((m as f64) * 0.43).sin() + 0.25;
        c_phi[m] = (0.25 * length_m * w_phi[m]) as f32;
    }
    let eps = 1.0e-2_f64;
    let mut tvp = tv64;
    let mut thp = th64;
    let mut tvm = tv64;
    let mut thm = th64;
    for i in 0..NVH {
        tvp[i] += eps * dd_v[i];
        thp[i] += eps * dd_h[i];
        tvm[i] -= eps * dd_v[i];
        thm[i] -= eps * dd_h[i];
    }
    let (sp, pp) = convolve_transverse_slopes_n::<NVH, ARR>(length_m, &tvp, &thp, NL);
    let (sm, pm) = convolve_transverse_slopes_n::<NVH, ARR>(length_m, &tvm, &thm, NL);
    let mut lhs = w_s * (sp - sm) / (2.0 * eps);
    for m in 1..=NL {
        lhs += w_phi[m] * (pp[m] - pm[m]) / (2.0 * eps);
    }

    let (mut gv_ref, mut gh_ref) = ([0.0_f64; NVH], [0.0_f64; NVH]);
    accum_phantom_grad_d::<NVH, ARR>(
        length_m,
        &tv64,
        &th64,
        w_s,
        &w_phi,
        &mut gv_ref,
        &mut gh_ref,
    );
    let rhs_ref: f64 = (0..NVH)
        .map(|i| gv_ref[i] * dd_v[i] + gh_ref[i] * dd_h[i])
        .sum();
    assert!((lhs - rhs_ref).abs() / lhs.abs().max(1e-20) < 1e-12);

    let k_ws = (0.5 * length_m * w_s) as f32;
    let c_phi4 = [to4(&c_phi[1..]), to4(&c_phi[5..])];
    let (gv4, gh4) = accum_phantom_grad_vh_simd_2x2(&dv4, &dh4, k_ws, &c_phi4);
    let mut rhs_simd = 0.0_f64;
    for v in 0..SPARSE_PHANTOM_TRANSVERSE_VECS {
        for l in 0..4 {
            let gv = f64::from(gv4[v].to_array()[l]);
            let gh = f64::from(gh4[v].to_array()[l]);
            assert!((gv - gv_ref[4 * v + l]).abs() < 1e-5);
            assert!((gh - gh_ref[4 * v + l]).abs() < 1e-5);
            rhs_simd += gv * dd_v[4 * v + l] + gh * dd_h[4 * v + l];
        }
    }
    assert!((lhs - rhs_simd).abs() / lhs.abs().max(1e-20) < 1e-5);
}

#[test]
fn bass_sparse_profile_caps_sav_bounds_and_preserves_sav_conservation_and_passivity() {
    struct PassiveDriver;
    impl KeyActionDriver for PassiveDriver {
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
    impl BridgePortDriver for PassiveBridge {
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

    let perf = PerfProfile {
        bass_phantom_sparse: true,
        ..PerfProfile::full()
    };
    let voicing = VoicingSpec::default();
    let mut lossless = StringBank::new_lossless_with_profile(48_000.0, &voicing, &perf);
    for s_idx in 0..TOTAL_STRINGS {
        let k = usize::from(lossless.choir_map().strings()[s_idx].key_idx);
        let sp = lossless.sav.spec(s_idx);
        let meta = &lossless.bridge_ports.meta[s_idx];
        assert!(meta.v_br.len() >= sp.nv_vecs);
        assert!(meta.h_br.len() >= sp.nh_vecs);
        if k < 16 {
            assert_eq!(sp.nv_vecs, SPARSE_PHANTOM_TRANSVERSE_VECS);
            assert_eq!(sp.nh_vecs, SPARSE_PHANTOM_TRANSVERSE_VECS);
            assert_eq!(sp.nl_vecs, SPARSE_PHANTOM_LONG_VECS);
            assert_eq!(sp.m_ph, SPARSE_PHANTOM_LONG_MODES);
        } else if k < LONGITUDINAL_MAX_KEY_IDX {
            assert!(sp.nv_vecs <= 2 && sp.nh_vecs <= 2 && sp.nl_vecs <= 4);
        }
    }

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
                "lossless 2x2 k={key} subblock={use_subblock} drift={max_rel_err:e}"
            );
        }
    }

    let mut bank_coupled = StringBank::new_bridge_coupled_with_profile(48_000.0, &voicing, &perf);
    bank_coupled.apply_excitation_sample(0, [1.0, 0.0, 0.0], 24.0);
    let mut drv = PassiveDriver;
    let mut br = PassiveBridge {
        r0: *bank_coupled.bridge_port_resistances(),
        zw: [(0, 1.0, 0, 0.0); TOTAL_STRINGS],
    };
    let (mut bo_l, mut bo_r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let mut h_prev = bank_coupled.key_sav_energy_joules(0);
    let h0 = h_prev;
    for _ in 0..40 {
        bank_coupled
            .render_subblock_bridge_coupled_no_flush(&mut drv, &mut br, &mut bo_l, &mut bo_r);
        let h_cur = bank_coupled.key_sav_energy_joules(0);
        assert!(
            h_cur <= h_prev + 1.0e-7 * h0,
            "passivity {h_cur} > {h_prev}"
        );
        h_prev = h_cur;
    }
    bank_coupled.flush_bridge_subnormals(&[false; NUM_KEYS]);
}
