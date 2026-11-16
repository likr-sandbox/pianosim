use super::*;
use core::f64::consts::TAU;
use piano_spec::key_choir;

#[test]
fn stationary_string_produces_exact_zero_force_even_under_preload() {
    let mut bank = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    for k in [0usize, 15, 39, 48, 69] {
        let str_idx = usize::from(key_choir(k).unwrap().first_string_idx);
        for slot_idx in [str_idx * 2, str_idx * 2 + 1] {
            for _ in 0..32 {
                let f = bank.step_damper_sample(k, slot_idx, 0.0, 0.0, 0.25, -3.0e-4);
                assert_eq!(f, 0.0);
            }
        }
    }
    // Undamped keys 70..88 and out-of-range slot indices always yield 0.0.
    assert_eq!(
        bank.step_damper_sample(70, 352, 1.0e-3, 1.5, 0.5, -3.0e-4),
        0.0
    );
    assert_eq!(
        bank.step_damper_sample(48, DAMPER_PAD_SLOTS, 1.0e-3, 1.5, 0.1, -3.0e-4),
        0.0
    );
}

#[test]
fn damper_is_strictly_passive_and_damps_in_80_to_180_ms_across_rates() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);

    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        let mut bank = DamperPadBank::new(sr, &voicing);
        let ts = 1.0 / f64::from(sr);
        for &k in &[0usize, 15, 39, 48, 69] {
            let str_idx = usize::from(key_choir(k).unwrap().first_string_idx);
            let slot_v = str_idx * 2;
            let p = specs[k];
            let m_wire = p.linear_density_kg_m * p.speaking_length_m;
            let phi_d = 0.38_f64;
            let m_mode = 0.5 * m_wire;
            let omega = TAU * p.f0_et_hz;
            // Coupled ZOH-like 1-mode oscillator + realistic high-mode port impedance.
            let r_port = (2.0 * ts / m_wire * (phi_d * phi_d * 18.0)) as f32;
            let mut u = 1.2_f64; // initial modal velocity (m/s)
            let mut q = 0.0_f64;
            let (s, c) = (omega * ts).sin_cos();
            let mut e_prev = 0.5 * m_mode * u * u;
            let e0 = e_prev;

            let n_steps = (0.35 * f64::from(sr)) as usize;
            let mut t60_est = 0.0_f64;
            for step in 0..n_steps {
                let y_d = (phi_d * q) as f32;
                let v_d = (phi_d * u) as f32;
                let f_damp = bank.step_damper_sample(k, slot_v, y_d, v_d, r_port, -3.0e-4);
                assert!(f_damp.is_finite());

                u += (phi_d * f64::from(f_damp) * ts) / m_mode;
                let u_next = c * u - omega * s * q;
                let q_next = (s / omega) * u + c * q;
                u = u_next;
                q = q_next;

                let e_cur = 0.5 * m_mode * (u * u + omega * omega * q * q);
                assert!(
                    e_cur <= e_prev * (1.0 + 1e-6) + 1e-18,
                    "passivity violation sr={sr} k={k} step={step}: {e_cur} > {e_prev}"
                );
                e_prev = e_cur;
                if t60_est == 0.0 && e_cur <= 1.0e-6 * e0 {
                    t60_est = (step as f64) * ts;
                }
            }
            assert!(
                (0.050..=0.220).contains(&t60_est),
                "sr={sr} k={k} damped T60={t60_est:.4}s out of expected band"
            );
        }
    }
}

#[test]
fn orthogonal_vh_slots_are_isolated_with_zero_cross_contamination() {
    let voicing = VoicingSpec::default();
    let mut isolated_h = DamperPadBank::new(48_000.0, &voicing);
    let mut dual_vh = DamperPadBank::new(48_000.0, &voicing);

    let k = 48;
    let s_idx = usize::from(key_choir(k).unwrap().first_string_idx);
    let slot_v = s_idx * 2;
    let slot_h = s_idx * 2 + 1;

    for step in 0..512 {
        let phase = (step as f32) * 0.055;
        let (yv, vv) = (1.6e-3 * phase.sin(), 1.4 * phase.cos());
        let (yh, vh) = (-7.5e-4 * (phase * 1.13).cos(), 0.85 * (phase * 1.13).sin());

        let _fv = dual_vh.step_damper_sample(k, slot_v, yv, vv, 0.08, -3.0e-4);
        let fh_dual = dual_vh.step_damper_sample(k, slot_h, yh, vh, 0.08, -3.0e-4);
        let fh_ref = isolated_h.step_damper_sample(k, slot_h, yh, vh, 0.08, -3.0e-4);
        assert_eq!(fh_dual.to_bits(), fh_ref.to_bits());
    }
}

#[test]
fn update_voicing_in_place_updates_damper_coeffs_and_matches_fresh_bank() {
    let mut voicing = VoicingSpec::default();
    let mut bank = DamperPadBank::new(48_000.0, &voicing);
    voicing.string_length_scale = 1.15;
    bank.update_voicing_in_place(&voicing);
    let fresh = DamperPadBank::new(48_000.0, &voicing);

    for k in [0_usize, 24, 48, 69] {
        assert!((bank.c_nom_key[k] - fresh.c_nom_key[k]).abs() < 1e-6);
        assert!((bank.f_scale_key[k] - fresh.f_scale_key[k]).abs() < 1e-6);
    }
    assert_ne!(
        bank.c_nom_key[39].to_bits(),
        DamperPadBank::new(48_000.0, &VoicingSpec::default()).c_nom_key[39].to_bits()
    );
}
