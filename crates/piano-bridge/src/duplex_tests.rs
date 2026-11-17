use core::f64::consts::TAU;

use piano_spec::{build_scale_design, key_tuned_f0_hz, VoicingSpec, CHOIR_LAYOUTS, TOTAL_STRINGS};

use crate::duplex::NUM_DUPLEX_MODES;
use crate::tests::wdf_absorbed_power_w;
use crate::zone_proj::zone_stereo_pan;
use crate::{BridgeBank, DuplexBank, NUM_BRIDGE_ZONES, TREBLE_DUPLEX_ZONE_START};

#[test]
fn duplex_retune_a4_scales_mode_frequencies_and_clamps_bounds() {
    let mut bank = DuplexBank::new(48_000.0);
    assert_eq!(bank.a4_freq_hz(), 440.0);
    let base_f0 = bank.mode_freq_hz(0);
    assert!(base_f0 > 1_300.0);
    let specs_440 = build_scale_design(&VoicingSpec::default());
    let tuned_27_440 = key_tuned_f0_hz(27, &specs_440, &VoicingSpec::default());

    bank.retune_a4(415.0);
    assert_eq!(bank.a4_freq_hz(), 415.0);
    let f_415 = bank.mode_freq_hz(0);
    let v_415 = VoicingSpec::from_globals_with_a4(0.5, 1.0, 1.0, 1.0, 415.0);
    let specs_415 = build_scale_design(&v_415);
    let tuned_27_415 = key_tuned_f0_hz(27, &specs_415, &v_415);
    assert!((f64::from(f_415 / base_f0) - tuned_27_415 / tuned_27_440).abs() < 1e-5);
    assert!((f_415 / base_f0 - 415.0 / 440.0).abs() < 5e-4);

    bank.retune_a4(442.0);
    assert_eq!(bank.a4_freq_hz(), 442.0);
    let f_442 = bank.mode_freq_hz(0);
    let v_442 = VoicingSpec::from_globals_with_a4(0.5, 1.0, 1.0, 1.0, 442.0);
    let specs_442 = build_scale_design(&v_442);
    let tuned_27_442 = key_tuned_f0_hz(27, &specs_442, &v_442);
    assert!((f64::from(f_442 / base_f0) - tuned_27_442 / tuned_27_440).abs() < 1e-5);
    assert!((f_442 / base_f0 - 442.0 / 440.0).abs() < 5e-4);

    bank.retune_a4(300.0);
    assert_eq!(bank.a4_freq_hz(), 415.0);
    bank.retune_a4(f32::NAN);
    assert_eq!(bank.a4_freq_hz(), 440.0);
}

#[test]
fn duplex_retune_voicing_tracks_stretch_and_note_pitch_within_bounds() {
    let mut bank = DuplexBank::new(48_000.0);
    let nominal_c8_mode = bank.mode_freq_hz(57); // Zone 7 mode anchored to C8 (key 87) * 2
    let mut stretched = VoicingSpec::from_globals(0.5, 1.0, 2.0, 1.0);
    stretched.note_offsets[87].pitch_cents = 25.0;
    bank.retune_voicing(&stretched);

    let stretched_c8_mode = bank.mode_freq_hz(57);
    let ratio_cents = 1200.0 * f64::from(stretched_c8_mode / nominal_c8_mode).log2();
    assert!(
        ratio_cents > 45.0 && ratio_cents < 70.0,
        "expected stretch+25c shift on C8 duplex mode, got {ratio_cents:.2} cents"
    );
    for m in 0..NUM_DUPLEX_MODES {
        let f = bank.mode_freq_hz(m);
        assert!(
            (1_350.0..=12_500.0).contains(&f),
            "stretched duplex mode {m} out of range: {f} Hz"
        );
    }
}

#[test]
fn duplex_subnormal_flush_resets_active_and_reengages_idle_bypass() {
    let mut bank = DuplexBank::new(48_000.0);
    assert!(bank.is_quiescent());

    // Drive a tiny impulse so active becomes true, then let it decay/flush.
    let v_pulse = [0.0, 0.0, 1.0e-18_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let _ = bank.step_with_zone_velocities(&v_pulse);
    assert!(!bank.is_quiescent());

    // After flushing < 1e-20 states, active must return to false and stereo out to (0, 0).
    bank.flush_subnormals();
    assert!(bank.is_quiescent());
    assert_eq!(bank.energy_joules(), 0.0);
    assert_eq!(bank.stereo_out(), (0.0, 0.0));

    // Normal audible state >= 1e-20 must NOT be cleared prematurely.
    let v_audible = [0.0, 0.0, 0.05_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let _ = bank.step_with_zone_velocities(&v_audible);
    bank.flush_subnormals();
    assert!(!bank.is_quiescent());
    assert!(bank.energy_joules() > 0.0);
}

#[test]
fn duplex_treble_excitation_and_bass_isolation() {
    let mut standalone = DuplexBank::new(48_000.0);
    assert_eq!(standalone.port_resistances()[0], 0.0);
    assert_eq!(standalone.port_resistances()[1], 0.0);
    for z in TREBLE_DUPLEX_ZONE_START..NUM_BRIDGE_ZONES {
        assert!(standalone.port_resistances()[z] > 0.0);
    }
    for m in 0..NUM_DUPLEX_MODES {
        let f = standalone.mode_freq_hz(m);
        assert!(
            (1_350.0..=12_500.0).contains(&f),
            "duplex mode {m} out of range: {f} Hz"
        );
    }

    // Oscillating only bass zones (0, 1) must not excite duplex bank at all.
    for n in 0..512 {
        let v = 0.02 * ((TAU * 200.0 * (n as f64 / 48_000.0)).sin() as f32);
        let _ = standalone.step_with_zone_velocities(&[v, v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    assert_eq!(standalone.energy_joules(), 0.0);
    assert_eq!(standalone.stereo_out(), (0.0, 0.0));

    // Full BridgeBank driven in treble C5..C7 excites DuplexBank passively with stereo output.
    let mut bridge = BridgeBank::new(48_000.0);
    let r0 = *bridge.port_resistances();
    let e0 = bridge.total_energy_joules();
    let f_dup_a = f64::from(standalone.mode_freq_hz(20));
    let f_dup_b = f64::from(standalone.mode_freq_hz(32));
    let treble_keys = [55_usize, 63, 72];
    let mut max_dup_e = 0.0_f64;
    let mut max_stereo = 0.0_f32;
    let mut e_in_wdf = 0.0_f64;
    let ts = 1.0 / 48_000.0_f64;
    for n in 0..2_048 {
        let t = (n as f64) * ts;
        let mut a = [0.0_f32; TOTAL_STRINGS];
        let strike_env = if n < 48 { 1.2 } else { 0.8 };
        let val = (strike_env * (0.6 * (TAU * f_dup_a * t).sin() + 0.6 * (TAU * f_dup_b * t).sin()))
            as f32;
        for &key_idx in &treble_keys {
            let s0 = usize::from(CHOIR_LAYOUTS[key_idx].first_string_idx);
            let count = usize::from(CHOIR_LAYOUTS[key_idx].strings_per_key);
            for slot in &mut a[s0..s0 + count] {
                *slot = val;
            }
        }
        let mut b = [0.0_f32; TOTAL_STRINGS];
        let mut z = [0.0_f32; NUM_BRIDGE_ZONES];
        let (l, r) = bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
        e_in_wdf += ts * wdf_absorbed_power_w(&r0, &a, &b);
        max_dup_e = max_dup_e.max(bridge.duplex_energy_joules());
        max_stereo = max_stereo.max(l.abs().max(r.abs()));
    }
    assert!(max_dup_e > 1.0e-10, "duplex failed to excite: {max_dup_e}");
    assert!(max_stereo > 0.0, "duplex stereo output remained zero");
    assert!(bridge.total_energy_joules() - e0 <= e_in_wdf + 1.0e-10);
}

#[test]
fn duplex_zones_match_stereo_pan_and_overwrite_input() {
    const FREQS: [f64; 8] = [
        120.0, 210.0, 1_800.0, 2_400.0, 3_500.0, 5_200.0, 6_800.0, 8_900.0,
    ];
    let (mut dup_a, mut dup_b) = (DuplexBank::new(48_000.0), DuplexBank::new(48_000.0));
    let mut dup_zones = [999.0_f32; NUM_BRIDGE_ZONES];
    let mut max_treble = 0.0_f32;

    for n in 0..256 {
        let t = (n as f64) / 48_000.0;
        let mut v = [0.0_f32; NUM_BRIDGE_ZONES];
        for (dst, &f) in v.iter_mut().zip(&FREQS) {
            *dst = 0.04 * (TAU * f * t).sin() as f32;
        }
        let (la, ra) = dup_a.step_with_zone_velocities(&v);
        let (lb, rb) = dup_b.step_with_zone_velocities_and_zones(&v, &mut dup_zones);
        assert_eq!((la, ra), (lb, rb));
        assert_eq!(&dup_zones[..TREBLE_DUPLEX_ZONE_START], &[0.0, 0.0]);
        let (mut sum_l, mut sum_r) = (0.0_f32, 0.0_f32);
        for (z, &val) in dup_zones.iter().enumerate().skip(TREBLE_DUPLEX_ZONE_START) {
            let (pl, pr) = zone_stereo_pan(z);
            sum_l += val * pl;
            sum_r += val * pr;
            max_treble = max_treble.max(val.abs());
        }
        assert!((sum_l - lb).abs() <= 1.0e-7 && (sum_r - rb).abs() <= 1.0e-7);
    }
    assert!(max_treble > 1.0e-6);

    let mut bad_v = [0.0_f32; NUM_BRIDGE_ZONES];
    (bad_v[0], bad_v[1], bad_v[2]) = (f32::NAN, f32::INFINITY, f32::NEG_INFINITY);
    let (l, r) = dup_b.step_with_zone_velocities_and_zones(&bad_v, &mut dup_zones);
    assert!(l.is_finite() && r.is_finite() && dup_zones.iter().all(|v| v.is_finite()));
}

#[test]
fn bridge_step_with_duplex_zones_and_per_run_sanitization() {
    let (mut b1, mut b2) = (BridgeBank::new(48_000.0), BridgeBank::new(48_000.0));
    let mut a = [0.0_f32; TOTAL_STRINGS];
    (a[120], a[180]) = (0.35, -0.25);
    let audio = [0.1, -0.2, 0.05, 0.12, -0.08, 0.03, 0.01, -0.02];

    let (mut bw1, mut bw2) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; TOTAL_STRINGS]);
    let (mut z1, mut z2) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
    let mut dz2 = [77.0_f32; NUM_BRIDGE_ZONES];
    for _ in 0..8 {
        assert_eq!(
            b1.step_bridge_sample(&a, &audio, &mut bw1, &mut z1),
            b2.step_bridge_sample_with_duplex_zones(&a, &audio, &mut bw2, &mut z2, &mut dz2)
        );
        assert!(bw1 == bw2 && z1 == z2);
    }
    assert_eq!(&dz2[..TREBLE_DUPLEX_ZONE_START], &[0.0, 0.0]);
    assert!(dz2[TREBLE_DUPLEX_ZONE_START..].iter().any(|&v| v != 0.0));

    let mut bad_r0 = *b2.port_resistances();
    bad_r0[0] = 0.0;
    b2.set_port_resistances(&bad_r0);
    (a[0], a[5], a[150], a[TOTAL_STRINGS - 1]) =
        (f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::INFINITY);
    let (l, r) = b2.step_bridge_sample_with_duplex_zones(&a, &audio, &mut bw2, &mut z2, &mut dz2);
    assert!(l.is_finite() && r.is_finite() && bw2.iter().all(|v| v.is_finite()));
    assert!(z2.iter().all(|v| v.is_finite()) && dz2.iter().all(|v| v.is_finite()));
}
