use alloc::vec;
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::limiter::{soft_ceiling_curve, StereoLimiter, LINEAR_THRESHOLD};
use crate::nupols_tests::test_preset;
use crate::RadiatorBank;

#[test]
fn soft_ceiling_curve_is_c2_continuous_and_bounded() {
    assert_eq!(soft_ceiling_curve(0.0), 0.0);
    assert_eq!(soft_ceiling_curve(0.5), 0.5);
    assert_eq!(soft_ceiling_curve(LINEAR_THRESHOLD), LINEAR_THRESHOLD);

    let eps = 1.0e-4_f32;
    let f_plus = soft_ceiling_curve(LINEAR_THRESHOLD + eps);
    let f_0 = soft_ceiling_curve(LINEAR_THRESHOLD);
    let f_minus = soft_ceiling_curve(LINEAR_THRESHOLD - eps);
    let slope_right = (f_plus - f_0) / eps;
    let slope_left = (f_0 - f_minus) / eps;
    let second_deriv = (f_plus - 2.0 * f_0 + f_minus) / (eps * eps);

    assert!(
        (slope_right - 1.0).abs() < 1.0e-3 && (slope_left - 1.0).abs() < 1.0e-3,
        "C^1 slope mismatch at threshold: left={slope_left}, right={slope_right}"
    );
    assert!(
        second_deriv.abs() < 5.0e-2,
        "C^2 second derivative jump at threshold: {second_deriv}"
    );

    let mut prev = LINEAR_THRESHOLD;
    for &u in &[0.95_f32, 1.0, 1.2, 2.0, 10.0] {
        let y = soft_ceiling_curve(u);
        assert!(y.is_finite() && y > prev && y <= 1.0, "u={u} -> y={y}");
        prev = y;
    }
    for &u in &[500.0_f32, 1.0e6] {
        let y = soft_ceiling_curve(u);
        assert!(y.is_finite() && y >= prev && y <= 1.0, "u={u} -> y={y}");
        prev = y;
    }
}

#[test]
fn stereo_limiter_unity_below_threshold_and_shared_lr_gain_above() {
    let mut lim = StereoLimiter::new(48_000.0);
    let in_l = [0.0_f32, 0.25, -0.5, 0.89, -LINEAR_THRESHOLD, 0.12];
    let in_r = [0.1_f32, -0.35, 0.4, -0.7, 0.85, -0.12];
    let mut out_l = in_l;
    let mut out_r = in_r;
    lim.process_block(&mut out_l, &mut out_r);
    assert_eq!(out_l, in_l);
    assert_eq!(out_r, in_r);

    // Asymmetric overload: L=1.8, R=0.45 (ratio 4:1) must preserve exact 4:1 stereo balance.
    lim.reset();
    let mut ol = [1.8_f32, 0.4, 0.2];
    let mut or = [0.45_f32, 0.1, 0.05];
    lim.process_block(&mut ol, &mut or);
    for i in 0..3 {
        assert!(ol[i].abs() <= 1.0 && or[i].abs() <= 1.0);
        assert!(
            ((ol[i] / or[i]) - 4.0).abs() < 1.0e-5,
            "stereo image shifted at {i}: L={} R={}",
            ol[i],
            or[i]
        );
    }
}

#[test]
fn stereo_limiter_two_stage_release_fast_transient_vs_sustained_overload() {
    // 2-sample transient spike -> fast 35 ms release dominates.
    let mut lim_trans = StereoLimiter::new(48_000.0);
    let mut l1 = vec![0.0_f32; 2_400];
    let mut r1 = vec![0.0_f32; 2_400];
    l1[0] = 2.0;
    l1[1] = 2.0;
    lim_trans.process_block(&mut l1, &mut r1);
    let env_after_transient = lim_trans.envelope();

    // 200 ms sustained overload followed by 50 ms silence -> slow 180 ms release dominates.
    let mut lim_sus = StereoLimiter::new(48_000.0);
    let mut l2 = vec![2.0_f32; 9_600];
    l2.extend(vec![0.0_f32; 2_400]);
    let mut r2 = vec![0.0_f32; 12_000];
    lim_sus.process_block(&mut l2, &mut r2);
    let env_after_sustained = lim_sus.envelope();

    assert!(
        env_after_transient < 0.25,
        "expected fast 35 ms recovery after 2-sample spike, got env={env_after_transient}"
    );
    assert!(
        env_after_sustained > 1.35,
        "expected slow 180 ms recovery after sustained overload, got env={env_after_sustained}"
    );
}

#[test]
fn stereo_limiter_sanitizes_non_finite_and_resets_cleanly() {
    let mut lim = StereoLimiter::new(48_000.0);
    let mut l = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 500.0, 0.5];
    let mut r = [0.2, f32::NAN, 0.1, -500.0, -0.5];
    lim.process_block(&mut l, &mut r);
    assert_eq!(l[0], 0.0);
    assert_eq!(l[1], 0.0);
    assert_eq!(l[2], 0.0);
    assert_eq!(r[1], 0.0);
    assert!(l[3].abs() <= 1.0 && r[3].abs() <= 1.0);
    assert!(l[4].abs() > 0.1);
    assert!(lim.envelope().is_finite() && lim.envelope() > 0.0);

    lim.reset();
    assert_eq!(lim.envelope(), 0.0);
    let mut check_l = [0.75_f32];
    let mut check_r = [-0.75_f32];
    lim.process_block(&mut check_l, &mut check_r);
    assert_eq!(check_l[0], 0.75);
    assert_eq!(check_r[0], -0.75);
}

#[test]
fn radiator_bank_limiter_resets_and_is_block_size_invariant() {
    let preset = test_preset(32);
    let mut bank_a = RadiatorBank::from_preset(&preset, 48_000.0);
    let mut bank_b = bank_a.clone();

    let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 256];
    frames[0][2] = 6.0;
    frames[1][3] = -6.0;
    frames[2][2] = 0.35;

    let (mut la, mut ra) = (vec![0.0; 256], vec![0.0; 256]);
    let (mut lb, mut rb) = (vec![0.0; 256], vec![0.0; 256]);
    for i in 0..256 {
        bank_a.process_block(&frames[i..i + 1], &mut la[i..i + 1], &mut ra[i..i + 1]);
    }
    bank_b.process_block(&frames, &mut lb, &mut rb);
    for i in 0..256 {
        assert!(la[i].abs() <= 1.0 && ra[i].abs() <= 1.0);
        assert!((la[i] - lb[i]).abs() < 1.0e-5, "L mismatch at {i}");
        assert!((ra[i] - rb[i]).abs() < 1.0e-5, "R mismatch at {i}");
    }

    bank_a.reset();
    let (mut la2, mut ra2) = (vec![0.0; 256], vec![0.0; 256]);
    bank_a.process_block(&frames, &mut la2, &mut ra2);
    assert_eq!(la2, lb);
    assert_eq!(ra2, rb);
}
