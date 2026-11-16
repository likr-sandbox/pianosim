use alloc::vec;

use piano_spec::{u7_to_u16, PedalKind, VoicingSpec};

use crate::tests_common::{spectral_centroid_and_band, TestModalString};
use crate::ActionBank;

#[test]
fn silent_key_lifts_damper_without_string_contact_and_3_pedals_hold() {
    let voicing = VoicingSpec::default();
    let mut bank = ActionBank::new(48_000.0, &voicing);
    let c4 = 39_usize;
    let e4 = 43_usize;

    for vel in [1_u16, 16, 32, 64] {
        bank.reset();
        bank.note_on(c4, vel);
        let mut peak_force = 0.0_f32;
        for _ in 0..2_400 {
            let f = bank.step_hammer_sample(c4, 0.0, 0.0, 0.01);
            peak_force = peak_force.max(f);
            let c = bank.step_key_clearance(c4);
            assert!(c >= 4.5e-3, "vel={vel} clearance dropped: {c}");
        }
        assert_eq!(peak_force, 0.0, "silent vel={vel} struck string");
        assert!(!bank.is_hammer_active(c4));
        assert!(bank.damper_clearance_m(c4) >= 4.5e-3);
    }

    // Audible MIDI velocity 1 (`u7_to_u16(1)`) DOES strike the string.
    bank.reset();
    bank.note_on(c4, u7_to_u16(1));
    let mut pp_peak = 0.0_f32;
    for _ in 0..480 {
        pp_peak = pp_peak.max(bank.step_hammer_sample(c4, 0.0, 0.0, 0.01));
    }
    assert!(pp_peak > 0.0, "pp strike produced zero force");

    // 3-pedal invariants: Una Corda, Sostenuto, and Sustain.
    bank.reset();
    bank.set_pedal(PedalKind::UnaCorda, u16::MAX);
    assert_eq!(bank.strike_weights(c4), [0.0, 1.0, 1.0]);

    bank.note_on(c4, 1); // Silent C4 held open -> latched by Sostenuto.
    bank.set_pedal(PedalKind::Sostenuto, u16::MAX);
    bank.note_off(c4, 32_768);
    bank.note_on(e4, 40_000);
    bank.note_off(e4, 32_768);
    let mut tile = [0.0_f32; 8];
    for _ in 0..1_200 {
        bank.step_key_clearance_tile(c4, 8, &mut tile);
        bank.step_key_clearance_tile(e4, 8, &mut tile);
    }
    assert!(bank.damper_clearance_m(c4) >= 4.5e-3);
    assert!(bank.damper_clearance_m(e4) < 0.0);

    bank.set_pedal(PedalKind::Sostenuto, 0);
    bank.set_pedal(PedalKind::Sustain, u16::MAX);
    for _ in 0..1_200 {
        bank.step_key_clearance_tile(e4, 8, &mut tile);
    }
    assert!(bank.damper_clearance_m(e4) >= 4.5e-3);
}

#[test]
fn una_corda_felt_softening_extends_contact_and_lowers_centroid() {
    let voicing = VoicingSpec::default();
    let measure_pulse = |key_idx: usize, vel_u16: u16, una_corda: bool| -> (f32, f32, Vec<f32>) {
        let mut bank = ActionBank::new(48_000.0, &voicing);
        if una_corda {
            bank.set_pedal(PedalKind::UnaCorda, u16::MAX);
            assert!(bank.pedals.una_corda_felt_mod(key_idx).0 < 0.30);
        } else {
            assert_eq!(bank.pedals.una_corda_felt_mod(key_idx), (1.0, 0.0));
        }
        let mut s = TestModalString::new(key_idx, 48_000.0, &voicing);
        let port_scale = if una_corda { 1.5_f32 } else { 1.0 };
        bank.note_on(key_idx, vel_u16);
        let mut pulse = vec![0.0_f32; 512];
        for out in &mut pulse {
            let (y, v) = s.strike_port();
            let f = bank.step_hammer_sample(key_idx, y, v, s.r_port_strike * port_scale);
            *out = f;
            s.step_forces(f * port_scale, 0.0);
        }
        let peak = pulse.iter().copied().fold(0.0_f32, f32::max);
        let dur_ms =
            (pulse.iter().filter(|&&f| f > 0.05 * peak).count() as f32) * 1000.0 / 48_000.0;
        let (centroid, _) = spectral_centroid_and_band(&pulse, 48_000.0, 100.0, 20_000.0);
        (dur_ms, centroid, pulse)
    };

    for &key_idx in &[39_usize, 48] {
        for &vel_u16 in &[48_000_u16, 60_000] {
            let (dur_off_a, cent_off_a, pulse_a) = measure_pulse(key_idx, vel_u16, false);
            let (dur_off_b, _, pulse_b) = measure_pulse(key_idx, vel_u16, false);
            assert_eq!(pulse_a, pulse_b);

            let (dur_uc, cent_uc, _) = measure_pulse(key_idx, vel_u16, true);
            assert!(
                dur_uc > dur_off_a,
                "k={key_idx} v={vel_u16}: dur_uc={dur_uc} <= dur_off={dur_off_a}"
            );
            assert!(
                cent_uc < cent_off_a,
                "k={key_idx} v={vel_u16}: cent_uc={cent_uc} >= cent_off={cent_off_a}"
            );
            let _ = dur_off_b;
        }
    }
}

#[test]
fn keybed_and_voicing_update_work_in_place_on_action_bank() {
    let nominal = VoicingSpec::default();
    let mut bank = ActionBank::new(48_000.0, &nominal);
    assert_eq!(bank.step_keybed_force_sample(), 0.0);
    bank.note_on(48, u7_to_u16(1) - 1);
    assert_eq!(bank.step_keybed_force_sample(), 0.0);

    bank.note_on(48, u7_to_u16(96));
    for _ in 0..300 {
        assert_eq!(bank.step_keybed_force_sample(), 0.0);
    }
    let peak_kb = (0..300).fold(0.0_f32, |m, _| m.max(bank.step_keybed_force_sample()));
    assert!(peak_kb > 0.0);
    bank.note_on(48, u7_to_u16(96));
    bank.reset();
    assert_eq!(bank.step_keybed_force_sample(), 0.0);

    let orig_p = bank.keys[48].felt_spec.exponent_p;
    let strike_stats = |b: &mut ActionBank| -> (f32, f32) {
        b.reset();
        let mut s = TestModalString::new(48, 48_000.0, &nominal);
        b.note_on(48, 51_602);
        let mut pulse = vec![0.0_f32; 256];
        for out in &mut pulse {
            let (y, v) = s.strike_port();
            let f = b.step_hammer_sample(48, y, v, s.r_port_strike);
            *out = f;
            s.step_forces(f, 0.0);
        }
        let pk = pulse.iter().copied().fold(0.0_f32, f32::max);
        let dur = pulse.iter().filter(|&&f| f > 0.05 * pk).count() as f32;
        let (cent, _) = spectral_centroid_and_band(&pulse, 48_000.0, 100.0, 20_000.0);
        (dur, cent)
    };

    let (dur_nom, cent_nom) = strike_stats(&mut bank);
    let vspec = |h: f32| VoicingSpec {
        hammer_hardness: h,
        ..VoicingSpec::default()
    };
    bank.update_voicing_in_place(&vspec(0.1));
    assert_eq!(bank.keys[48].felt_spec.exponent_p, orig_p);
    let (dur_soft, cent_soft) = strike_stats(&mut bank);
    bank.update_voicing_in_place(&vspec(0.9));
    assert_eq!(bank.keys[48].felt_spec.exponent_p, orig_p);
    let (dur_hard, cent_hard) = strike_stats(&mut bank);
    assert!(dur_hard < dur_soft && cent_hard > cent_soft);

    bank.update_voicing_in_place(&nominal);
    let (dur_restored, cent_restored) = strike_stats(&mut bank);
    assert!((dur_restored - dur_nom).abs() <= 1.0 && (cent_restored - cent_nom).abs() < 1e-2);
}
