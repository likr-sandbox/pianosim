use super::*;
use piano_spec::{build_scale_design, partial_frequency_hz, VoicingSpec};

#[test]
fn effective_outer_radius_and_winding_friction_boundary_at_k26() {
    let specs = build_scale_design(&VoicingSpec::default());

    for (k, spec) in specs.iter().enumerate().take(WOUND_KEY_BREAK) {
        let r_eff = effective_outer_radius_m(spec);
        assert!(
            r_eff > spec.core_radius_m,
            "wound string k={k} must have r_eff ({r_eff}) > r_core ({})",
            spec.core_radius_m
        );
    }
    for (k, spec) in specs.iter().enumerate().skip(WOUND_KEY_BREAK) {
        let r_eff = effective_outer_radius_m(spec);
        assert!(
            (r_eff - spec.core_radius_m).abs() < 1e-12,
            "plain steel k={k} r_eff ({r_eff}) != r_core ({})",
            spec.core_radius_m
        );
    }

    // Winding friction strictly vanishes for plain steel (k >= 26) and grows with copper ratio.
    let f_test = 200.0;
    let a_25 = transverse_partial_alpha(25, &specs[25], f_test, false);
    let a_26 = transverse_partial_alpha(26, &specs[26], f_test, false);
    assert!(a_25.is_finite() && a_26.is_finite() && a_25 > 0.0 && a_26 > 0.0);
    // k=25 has small copper fraction; k=0 has >95% copper mass fraction.
    let a_0_at_27_5 = transverse_partial_alpha(0, &specs[0], 27.5, false);
    assert!(
        a_0_at_27_5 > 0.10,
        "A0 winding friction should produce realistic ~20-35s T60, got alpha={a_0_at_27_5}"
    );
}

#[test]
fn three_mechanism_t60_removes_clamps_and_scales_physically_with_frequency() {
    let specs = build_scale_design(&VoicingSpec::default());

    // Verify .max(1.85) clamp removal on C8 (k=87) for both bridge_coupled = false and true.
    for &coupled in &[false, true] {
        let f1_c8 = partial_frequency_hz(specs[87].f0_et_hz, specs[87].inharmonicity_b, 1);
        let t60_c8 = t60_transverse_partial(87, &specs[87], f1_c8, 1.0, coupled);
        assert!(
            t60_c8 > 0.5 && t60_c8 < 2.8,
            "coupled={coupled}: C8 intrinsic T60={t60_c8:.3}s out of relaxed range"
        );

        // Verify .max(0.45) / .max(0.25) clamp removal at 18 kHz via omega^2 thermoelasticity.
        let t60_18k = t60_transverse_partial(39, &specs[39], 18_000.0, 1.0, coupled);
        assert!(
            t60_18k > 0.02 && t60_18k < 0.70,
            "coupled={coupled}: 18 kHz T60={t60_18k:.4}s out of relaxed range"
        );

        // Strict monotonic T60 decrease with increasing partial frequency on fixed key.
        let mut prev_t60 = f64::INFINITY;
        for j in 1..=30 {
            let f_j = partial_frequency_hz(specs[39].f0_et_hz, specs[39].inharmonicity_b, j);
            let t60 = t60_transverse_partial(39, &specs[39], f_j, 1.0, coupled);
            assert!(t60.is_finite() && t60 > 0.0 && t60 < prev_t60);
            prev_t60 = t60;
        }
    }
}
