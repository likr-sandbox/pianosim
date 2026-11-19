use crate::{AdaaState, PolyPowerApprox, POLY_ADAA_DIFF_EPS};

#[test]
fn poly_power_approx_is_nonnegative_monotone_convex_and_accurate() {
    for step in 0..=28 {
        let p = 2.1 + (step as f32) * 0.05;
        let poly = PolyPowerApprox::new(p);

        assert_eq!(poly.eval_f(-0.25), 0.0);
        assert_eq!(poly.eval_f(0.0), 0.0);
        assert_eq!(poly.eval_f1(-0.25), 0.0);
        assert_eq!(poly.eval_f1(0.0), 0.0);
        assert_eq!(poly.eval_f_and_df(-0.1), (0.0, 0.0));
        assert_eq!(poly.eval_f_and_df(0.0), (0.0, 0.0));
        assert!((poly.f_at_1() - 1.0).abs() < 1e-6);

        let mut max_err = 0.0_f32;
        let mut prev_f = 0.0_f32;
        let mut prev_df = 0.0_f32;
        let c = poly.coeffs();

        for i in 0..=200 {
            let u = (i as f32) / 200.0;
            let (f, df) = poly.eval_f_and_df(u);
            assert!(f >= 0.0, "negative f at p={p}, u={u}: {f}");
            assert!(df >= 0.0, "negative df at p={p}, u={u}: {df}");
            assert!(
                f + 1e-7 >= prev_f,
                "non-monotone f at p={p}, u={u}: {f} < {prev_f}"
            );
            assert!(
                df + 1e-6 >= prev_df,
                "non-monotone df at p={p}, u={u}: {df} < {prev_df}"
            );
            let d2f = 2.0 * c[0]
                + u * (6.0 * c[1] + u * (12.0 * c[2] + u * (20.0 * c[3] + u * (30.0 * c[4]))));
            assert!(d2f >= -1e-6, "non-convex d2f at p={p}, u={u}: {d2f}");

            let exact = u.powf(p);
            max_err = max_err.max((f - exact).abs());
            prev_f = f;
            prev_df = df;
        }
        assert!(
            max_err < 1.0e-3,
            "approximation error too high at p={p}: max_err={max_err}"
        );
    }
}

#[test]
fn poly_power_approx_derivative_and_antiderivative_consistency() {
    let h = 5.0e-4_f32;
    for &p in &[2.1_f32, 2.45, 2.8, 3.0, 3.25, 3.5] {
        let poly = PolyPowerApprox::new(p);

        for i in 1..25 {
            let u = (i as f32) * 0.05; // spans [0.05, 1.20] across u=1.0 boundary
            if (u - 1.0).abs() < 2.0 * h {
                // C^1 piecewise boundary at u = 1.0: verify both one-sided limits match f(1.0).
                let left_d = (poly.eval_f1(1.0) - poly.eval_f1(1.0 - h)) / h;
                let right_d = (poly.eval_f1(1.0 + h) - poly.eval_f1(1.0)) / h;
                assert!((left_d - poly.f_at_1()).abs() < 1.0e-3);
                assert!((right_d - poly.f_at_1()).abs() < 1.0e-4);
                continue;
            }
            let f_val = poly.eval_f(u);
            let num_deriv = (poly.eval_f1(u + h) - poly.eval_f1(u - h)) / (2.0 * h);
            assert!(
                (num_deriv - f_val).abs() < 1.0e-4,
                "F1 derivative mismatch at p={p}, u={u}: {num_deriv} vs {f_val}"
            );

            if u < 1.0 - 2.0 * h {
                let (f_pair, df_val) = poly.eval_f_and_df(u);
                assert!((f_pair - f_val).abs() < 1e-7);
                let num_df = (poly.eval_f(u + h) - poly.eval_f(u - h)) / (2.0 * h);
                let tol_df = 1.5e-4 * df_val.abs().max(1.0);
                assert!(
                    (num_df - df_val).abs() < tol_df,
                    "f derivative mismatch at p={p}, u={u}: {num_df} vs {df_val}"
                );
            }
        }

        // Saturation region u > 1.0.
        assert_eq!(poly.eval_f(1.5), poly.f_at_1());
        assert_eq!(poly.eval_f_and_df(1.5), (poly.f_at_1(), 0.0));
        assert!((poly.adaa1(1.8, 1.2) - poly.f_at_1()).abs() < 1e-5);
    }
}

#[test]
fn poly_power_approx_adaa1_guard_and_stateful_process() {
    let poly = PolyPowerApprox::new(2.85);
    assert_eq!(poly.adaa1(0.0, 0.0), 0.0);
    assert_eq!(poly.adaa1(-0.2, -0.5), 0.0);
    assert_eq!(poly.adaa1(f32::NAN, 0.1), 0.0);

    for &u0 in &[0.0_f32, 0.2, 0.65, 0.99, 1.15] {
        let eps = 0.25 * POLY_ADAA_DIFF_EPS;
        let guarded = poly.adaa1(u0 + eps, u0);
        let expected_mid = poly.eval_f(u0 + 0.5 * eps);
        assert!(
            (guarded - expected_mid).abs() < 1e-7,
            "guard mismatch at u0={u0}: {guarded} vs {expected_mid}"
        );
    }

    let mut state = AdaaState::new();
    assert!(state.is_zero());
    let trajectory = [0.0_f32, 0.15, 0.45, 0.45 + 1.0e-6, 0.85, 1.1, 0.3, -0.05];
    let mut prev = 0.0_f32;
    for &u in &trajectory {
        let out = poly.process(u, &mut state);
        let ref_out = poly.adaa1(u, prev);
        assert!(out.is_finite());
        assert_eq!(out, ref_out);
        prev = u;
    }
    state.reset();
    assert!(state.is_zero());
    assert_eq!(poly.process(0.0, &mut state), 0.0);
}
