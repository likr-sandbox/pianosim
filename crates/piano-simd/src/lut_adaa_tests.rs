use crate::{adaa1_lerp, AdaaPowerLut, AdaaState};

#[test]
fn lerp_endpoints_and_t_clamping_are_exact_and_panic_free() {
    let lo = AdaaPowerLut::new(2.10, 1.0, 1024);
    let hi = AdaaPowerLut::new(2.16, 1.0, 1024);

    for &x in &[-0.25_f32, 0.0, 0.05, 0.37, 0.92, 1.0, 1.35] {
        for &t_lo in &[0.0_f32, -0.5, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(lo.eval_f_lerp(&hi, t_lo, x), lo.eval_f(x));
            assert_eq!(lo.eval_f1_lerp(&hi, t_lo, x), lo.eval_f1(x));
        }
        for &t_hi in &[1.0_f32, 1.75, f32::INFINITY] {
            assert_eq!(lo.eval_f_lerp(&hi, t_hi, x), hi.eval_f(x));
            assert_eq!(lo.eval_f1_lerp(&hi, t_hi, x), hi.eval_f1(x));
        }
    }

    // ADAA1 endpoint parity including ill-conditioned small steps and negative regions.
    let pairs = [
        (-0.1_f32, -0.2_f32),
        (0.0, 0.0),
        (0.4, 0.4 + 2.0e-7),
        (0.25, 0.65),
        (1.3, 1.1),
    ];
    for &(x, x_prev) in &pairs {
        assert_eq!(lo.adaa1_lerp(&hi, 0.0, x, x_prev), lo.adaa1(x, x_prev));
        assert_eq!(adaa1_lerp(&lo, &hi, 1.0, x, x_prev), hi.adaa1(x, x_prev));
    }

    // Non-finite x values remain panic-free.
    assert_eq!(lo.eval_f_lerp(&hi, 0.5, f32::NAN), 0.0);
    assert_eq!(lo.eval_f1_lerp(&hi, 0.5, f32::NAN), 0.0);
    assert_eq!(lo.adaa1_lerp(&hi, 0.5, f32::NAN, 0.2), 0.0);
    assert!(lo.adaa1_lerp(&hi, 0.5, f32::INFINITY, 0.2).is_finite());
}

#[test]
fn lerp_24pt_p_grid_matches_exact_midpoint_lut() {
    const GRID_N: usize = 24;
    const P_MIN: f32 = 2.10;
    const P_MAX: f32 = 3.50;
    let dp = (P_MAX - P_MIN) / ((GRID_N - 1) as f32);
    let grid: Vec<AdaaPowerLut> = (0..GRID_N)
        .map(|i| AdaaPowerLut::new(P_MIN + (i as f32) * dp, 1.0, 1024))
        .collect();

    let mut max_f_err = 0.0_f32;
    let mut max_f1_err = 0.0_f32;
    let mut max_adaa_err = 0.0_f32;

    for i in 0..(GRID_N - 1) {
        let lo = &grid[i];
        let hi = &grid[i + 1];
        for &t in &[0.25_f32, 0.5, 0.75] {
            let p_mid = lo.exponent() + t * (hi.exponent() - lo.exponent());
            let exact = AdaaPowerLut::new(p_mid, 1.0, 1024);

            for step in 0..=24 {
                let x = -0.05 + (step as f32) * 0.05;
                let f_err = (lo.eval_f_lerp(hi, t, x) - exact.eval_f(x)).abs();
                let f1_err = (lo.eval_f1_lerp(hi, t, x) - exact.eval_f1(x)).abs();
                max_f_err = max_f_err.max(f_err);
                max_f1_err = max_f1_err.max(f1_err);
            }

            // Compare stateful ADAA1 trajectory across strike compression & rebound.
            let mut state_lerp = AdaaState::new();
            let mut state_exact = AdaaState::new();
            for n in 0..64 {
                let phase = (n as f32) * 0.1;
                let x = if n < 4 {
                    -0.02 + 0.005 * (n as f32)
                } else if n == 30 {
                    // Force ill-conditioned tiny step near peak.
                    0.85_f32 + 2.0e-7_f32
                } else if n == 29 {
                    0.85_f32
                } else {
                    (0.9 * (phase * 0.6).sin()).max(-0.05)
                };
                let y_lerp = state_lerp.process_lerp(lo, hi, t, x);
                let y_exact = state_exact.process(&exact, x);
                assert!(y_lerp.is_finite());
                max_adaa_err = max_adaa_err.max((y_lerp - y_exact).abs());
            }
        }
    }

    assert!(max_f_err < 1.0e-4, "max_f_err={max_f_err}");
    assert!(max_f1_err < 1.0e-4, "max_f1_err={max_f1_err}");
    assert!(max_adaa_err < 1.0e-4, "max_adaa_err={max_adaa_err}");
}

#[test]
fn lerp_adaa1_derivative_consistency_and_ill_conditioned_fallback() {
    let lo = AdaaPowerLut::new(2.45, 1.0, 4096);
    let hi = AdaaPowerLut::new(2.55, 1.0, 4096);

    for &t in &[0.0_f32, 0.35, 0.5, 0.8, 1.0] {
        // Finite difference of eval_f1_lerp matches eval_f_lerp across domain and > x_max.
        let h = 5.0e-4_f32;
        for &x in &[0.05_f32, 0.25, 0.60, 0.90, 1.30] {
            let num_deriv =
                (lo.eval_f1_lerp(&hi, t, x + h) - lo.eval_f1_lerp(&hi, t, x - h)) / (2.0 * h);
            let f_val = lo.eval_f_lerp(&hi, t, x);
            assert!(
                (num_deriv - f_val).abs() < 1.0e-4,
                "t={t}, x={x}: num_deriv={num_deriv} vs f_val={f_val}"
            );
        }

        // Ill-conditioned tiny step (< ADAA_DIFF_EPS) matches midpoint eval_f_lerp,
        // and just above ADAA_DIFF_EPS agrees closely with midpoint.
        for &x0 in &[0.0_f32, 0.2, 0.72, 1.15] {
            let mid_eps = 3.0e-7_f32;
            let guarded = lo.adaa1_lerp(&hi, t, x0 + mid_eps, x0);
            let expected_mid = lo.eval_f_lerp(&hi, t, x0 + 0.5 * mid_eps);
            assert!(
                (guarded - expected_mid).abs() < 1.0e-7,
                "guard mismatch at t={t}, x0={x0}: {guarded} vs {expected_mid}"
            );

            let small_step = 2.0e-6_f32;
            let secant = lo.adaa1_lerp(&hi, t, x0 + small_step, x0);
            let mid_small = lo.eval_f_lerp(&hi, t, x0 + 0.5 * small_step);
            let linear_combo =
                (1.0 - t) * lo.adaa1(x0 + small_step, x0) + t * hi.adaa1(x0 + small_step, x0);
            assert!(
                (secant - linear_combo).abs() < 1.0e-6,
                "secant linear-combo mismatch at t={t}, x0={x0}: {secant} vs {linear_combo}"
            );
            assert!(
                (secant - mid_small).abs() < 1.0e-2,
                "f32 ULP bound exceeded at t={t}, x0={x0}: {secant} vs {mid_small}"
            );

            let med_step = 1.0e-4_f32;
            let secant_med = lo.adaa1_lerp(&hi, t, x0 + med_step, x0);
            let mid_med = lo.eval_f_lerp(&hi, t, x0 + 0.5 * med_step);
            assert!(
                (secant_med - mid_med).abs() < 3.0e-4,
                "continuity mismatch at t={t}, x0={x0}: {secant_med} vs {mid_med}"
            );
        }
    }
}

#[test]
fn lerp_handles_mismatched_table_geometries() {
    let lo = AdaaPowerLut::new(2.2, 1.0, 512);
    let hi = AdaaPowerLut::new(2.3, 1.5, 1024);
    let x = 0.6_f32;
    let t = 0.4_f32;
    let expected_f = 0.6 * lo.eval_f(x) + 0.4 * hi.eval_f(x);
    let expected_f1 = 0.6 * lo.eval_f1(x) + 0.4 * hi.eval_f1(x);
    assert!((lo.eval_f_lerp(&hi, t, x) - expected_f).abs() < 1e-7);
    assert!((lo.eval_f1_lerp(&hi, t, x) - expected_f1).abs() < 1e-7);
    let expected_adaa = 0.6 * lo.adaa1(0.8, 0.3) + 0.4 * hi.adaa1(0.8, 0.3);
    assert!((lo.adaa1_lerp(&hi, t, 0.8, 0.3) - expected_adaa).abs() < 1e-6);
}
