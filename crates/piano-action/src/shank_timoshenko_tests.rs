use super::*;
use piano_spec::u7_to_u16;

#[test]
fn velocity_mapping_is_strictly_monotonic_and_hits_targets() {
    let mut prev = 0.0_f32;
    for v in 1..=u16::MAX {
        let speed = velocity_from_u16(v);
        assert!(speed > prev, "non-monotonic at {v}: {speed} <= {prev}");
        prev = speed;
    }
    let v_esc = (2.0 * GRAVITY_MPS2 * (-HAMMER_LET_OFF_Y0_M)).sqrt();
    for v in 1..=SILENT_KEY_MAX_U16 {
        let speed = velocity_from_u16(v);
        assert!(
            speed < v_esc,
            "silent velocity {v} exceeded escape {speed} >= {v_esc}"
        );
    }
    assert!((velocity_from_u16(1) - SILENT_KEY_V_MIN_MPS).abs() < 1e-6);
    assert!((velocity_from_u16(SILENT_KEY_MAX_U16) - SILENT_KEY_V_MAX_MPS).abs() < 1e-6);

    let v_pp = velocity_from_u16(u7_to_u16(1));
    let v_mf = velocity_from_u16(51_602);
    let v_fff = velocity_from_u16(u16::MAX);
    assert!((0.09..=0.11).contains(&v_pp), "pp={v_pp}");
    assert!((1.70..=1.95).contains(&v_mf), "mf={v_mf}");
    assert!((4.55..=4.65).contains(&v_fff), "fff={v_fff}");
}

#[test]
fn gravity_returns_sub_escapement_hammer_to_rest_rail() {
    for &sr in &[44_100.0_f64, 48_000.0, 96_000.0] {
        let params = ShankParams::for_key(39, sr);
        for &vel_u16 in &[1_u16, 32, 64] {
            let mut s = ShankState::default();
            s.launch(velocity_from_u16(vel_u16));
            let mut max_y = s.y_h();
            let mut saw_reversal = false;

            for _ in 0..params.timeout_host_samples {
                for _ in 0..8 {
                    let pred = s.predict_free(&params);
                    s.apply_force(&params, pred, 0.0);
                    max_y = max_y.max(s.y_h());
                    if s.v0 < 0.0 {
                        saw_reversal = true;
                    }
                }
                if !s.finish_host_sample(&params) {
                    break;
                }
            }
            assert!(max_y < 0.0, "sr={sr} vel={vel_u16} reached string: {max_y}");
            assert!(saw_reversal);
            assert!(!s.active && !s.has_contacted);
            assert!(s.elapsed_host_samples < params.timeout_host_samples);
            assert_eq!(s.y_h(), HAMMER_LET_OFF_Y0_M);
        }

        // Audible pp (u7_to_u16(1)) overcomes gravity and reaches y >= 0.
        let mut s_pp = ShankState::default();
        s_pp.launch(velocity_from_u16(u7_to_u16(1)));
        let mut reached = false;
        for _ in 0..params.timeout_host_samples {
            for _ in 0..8 {
                let pred = s_pp.predict_free(&params);
                s_pp.apply_force(&params, pred, 0.0);
                if s_pp.y_h() >= 0.0 {
                    reached = true;
                }
            }
            if !s_pp.finish_host_sample(&params) {
                break;
            }
        }
        assert!(reached, "pp did not cross let-off gap at sr={sr}");
    }
}

#[test]
fn shank_params_and_trapezoidal_passivity_hold() {
    let p0 = ShankParams::for_key(0, 48_000.0);
    let p48 = ShankParams::for_key(48, 48_000.0);
    let p87 = ShankParams::for_key(87, 48_000.0);
    assert!((p0.hammer_mass_kg - 0.0112).abs() < 1e-5);
    assert!((p87.hammer_mass_kg - 0.0035).abs() < 1e-5);
    assert!(p0.hammer_mass_kg > p48.hammer_mass_kg && p48.hammer_mass_kg > p87.hammer_mass_kg);
    assert!((1_500.0..=3_500.0).contains(&p0.shank_freq_hz));
    assert!((1_500.0..=3_500.0).contains(&p48.shank_freq_hz));
    assert!((1_500.0..=3_500.0).contains(&p87.shank_freq_hz));

    // Unforced ringing of flexible mode strictly dissipates energy.
    let mut s = ShankState {
        eta1: 1.0e-4,
        v1: 0.25,
        active: true,
        ..ShankState::default()
    };
    let omega1 = TAU * f64::from(p48.shank_freq_hz);
    let k1 = f64::from(p48.modal_mass_kg) * omega1 * omega1;
    let energy = |st: &ShankState| {
        0.5 * f64::from(p48.modal_mass_kg) * f64::from(st.v1) * f64::from(st.v1)
            + 0.5 * k1 * f64::from(st.eta1) * f64::from(st.eta1)
    };
    let mut e_prev = energy(&s);
    for _ in 0..400 {
        let pred = s.predict_free(&p48);
        s.apply_force(&p48, pred, 0.0);
        let e_cur = energy(&s);
        assert!(e_cur <= e_prev + 1e-15);
        e_prev = e_cur;
    }
    assert!(e_prev < 0.9 * (0.5 * f64::from(p48.modal_mass_kg) * 0.0625 + 0.5 * k1 * 1e-8));
}

#[test]
fn backcheck_and_timeout_deactivate_hammer() {
    let params = ShankParams::for_key(48, 48_000.0);
    let mut s = ShankState::default();
    s.launch(1.8);
    let pred = s.predict_free(&params);
    s.apply_force(&params, pred, 100.0);
    assert!(s.has_contacted);
    s.y0 = -7.0e-5;
    s.v0 = -1.0;
    assert!(!s.finish_host_sample(&params));

    // Timeout without contact.
    s.launch(0.10);
    for _ in 0..params.timeout_host_samples - 1 {
        assert!(s.finish_host_sample(&params));
    }
    assert!(!s.finish_host_sample(&params));
}
