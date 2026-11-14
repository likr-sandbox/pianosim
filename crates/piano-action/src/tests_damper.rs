//! Unilateral damper passivity, half-pedal spectral shaping, and pedal control tests.

use piano_spec::{PedalKind, VoicingSpec};

use crate::tests_common::TestModalString;
use crate::ActionBank;

#[test]
fn unilateral_damper_passive_across_rates_and_half_pedal_shapes_high_partials() {
    let voicing = VoicingSpec::default();

    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        for &k in &[0_usize, 24, 48, 69] {
            let mut bank = ActionBank::new(sr, &voicing);
            // Rest invariant under -0.30 mm preload.
            let f_rest = bank.step_damper_sample(k, k * 2, 0.0, 0.0, 0.05, -3.0e-4);
            assert_eq!(f_rest, 0.0);

            let mut s = TestModalString::new(k, f64::from(sr), &voicing);
            s.excite_fundamental_velocity(0.45);
            let e0 = s.total_energy();
            let mut prev_e = e0;
            let frames = (0.20 * f64::from(sr)) as usize;
            let mut t60_ms = None;
            for step in 0..frames {
                let (y_d, v_d) = s.damp_port();
                let f_d = bank.step_damper_sample(k, k, y_d, v_d, s.r_port_damp, -3.0e-4);
                s.step_forces(0.0, f_d);
                let e_now = s.total_energy();
                assert!(
                    e_now <= prev_e * (1.0 + 5.0e-6) + e0 * 1.0e-8,
                    "passivity fail sr={sr} k={k} step={step}: {e_now} > {prev_e}"
                );
                if t60_ms.is_none() && e_now <= e0 * 1.0e-6 {
                    t60_ms = Some((step as f64) * 1000.0 / f64::from(sr));
                }
                prev_e = e_now;
            }
            if k > 0 {
                assert!(
                    prev_e < e0 * 1.0e-4,
                    "insufficient damping sr={sr} k={k}: ratio={}",
                    prev_e / e0
                );
            } else {
                assert!(
                    prev_e < e0 * 0.025,
                    "insufficient A0 damping sr={sr} k={k}: ratio={}",
                    prev_e / e0
                );
            }
            if let Some(ms) = t60_ms {
                assert!(ms < 220.0, "T60 too slow sr={sr} k={k}: {ms} ms");
            }
        }
    }

    // Half-pedal selective high-partial damping vs open sustain & dry release.
    let measure_partial_decay = |clear_m: f32| -> (f64, f64) {
        let mut bank = ActionBank::new(48_000.0, &voicing);
        let mut s = TestModalString::new(36, 48_000.0, &voicing);
        s.excite_modes(0.45, 0.35);
        let e1_0 = s.mode_energy(0);
        let e5_0 = s.mode_energy(4);
        for _ in 0..4_800 {
            let (y_d, v_d) = s.damp_port();
            let f_d = bank.step_damper_sample(36, 100, y_d, v_d, s.r_port_damp, clear_m);
            s.step_forces(0.0, f_d);
        }
        (s.mode_energy(0) / e1_0, s.mode_energy(4) / e5_0)
    };

    let (_open_f1, open_f5) = measure_partial_decay(5.0e-3);
    let mut bank_hp = ActionBank::new(48_000.0, &voicing);
    bank_hp.set_pedal(PedalKind::Sustain, 32_768);
    bank_hp.note_off(36, 32_768);
    for _ in 0..4_000 {
        bank_hp.step_key_clearance(36);
    }
    let hp_clear = bank_hp.damper_clearance_m(36);
    assert!(
        hp_clear > 1.5e-5 && hp_clear < 6.0e-5,
        "hp_clear={hp_clear}"
    );
    let (hp_f1, hp_f5) = measure_partial_decay(hp_clear);
    let (dry_f1, _) = measure_partial_decay(-3.0e-4);

    assert!(
        hp_f1 > dry_f1 * 3.0,
        "half pedal loses fundamental: {hp_f1} vs {dry_f1}"
    );
    assert!(
        hp_f5 < open_f5 * 0.45,
        "half pedal misses upper partials: {hp_f5} vs {open_f5}"
    );
    assert!(
        hp_f1 > hp_f5 * 1.5,
        "half pedal should favor f1={hp_f1} over f5={hp_f5}"
    );
}

#[test]
fn sostenuto_una_corda_and_bounds_safety() {
    let voicing = VoicingSpec::default();
    let mut bank = ActionBank::new(48_000.0, &voicing);

    // Key 24 held open, key 36 dry released -> press Sostenuto -> only 24 latches.
    bank.note_on(24, 40_000);
    bank.set_pedal(PedalKind::Sostenuto, 65_535);
    bank.note_off(24, 65_535);
    bank.note_on(36, 40_000);
    bank.note_off(36, 65_535);
    for _ in 0..4_800 {
        bank.step_key_clearance(24);
        bank.step_key_clearance(36);
    }
    assert!(bank.damper_clearance_m(24) > 4.5e-3);
    assert!(bank.damper_clearance_m(36) < 0.0);

    // Release Sostenuto -> latched key 24 drops to preload.
    bank.set_pedal(PedalKind::Sostenuto, 0);
    for _ in 0..9_600 {
        bank.step_key_clearance(24);
    }
    assert!(bank.damper_clearance_m(24) < 0.0);

    // Una Corda choir strike weights.
    bank.set_pedal(PedalKind::UnaCorda, 65_535);
    assert!(bank.strike_weights(0)[0] < 1.0 && bank.strike_weights(0)[1] == 0.0);
    assert_eq!(bank.strike_weights(15)[0], 0.0);
    assert!(bank.strike_weights(48)[0] < 0.05 && bank.strike_weights(48)[1] == 1.0);

    // velocity_u16 == 0 acts as note_off; out-of-range indices never panic.
    bank.note_on(48, 40_000);
    bank.note_on(48, 0);
    let mut clears = [0.0_f32; 8];
    let min_c = bank.step_key_clearance_tile(48, 8, &mut clears);
    assert_eq!(min_c, clears[7]);
    assert!(bank.step_damper_sample(48, 459, 1.0e-3, 0.5, 0.1, -3e-4) < 0.0);
    assert_eq!(
        bank.step_damper_sample(48, 460, 1.0e-3, 0.5, 0.1, -3e-4),
        0.0
    );
    bank.note_on(999, 40_000);
    bank.note_off(999, 40_000);
    assert!(!bank.is_hammer_active(999));
    assert_eq!(bank.step_hammer_sample(999, 0.0, 0.0, 0.0), 0.0);
    assert_eq!(bank.step_damper_sample(999, 999, 0.1, 0.1, 0.1, -3e-4), 0.0);
}
