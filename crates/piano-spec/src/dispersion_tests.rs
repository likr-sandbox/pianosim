use super::*;
use crate::math::exp2_f64;
use crate::scale_design::{A4_KEY_IDX, NUM_KEYS};

#[test]
fn analytical_dispersion_law_matches_direct_formula() {
    let (f0, b) = (110.0, 0.00045);
    assert_eq!(partial_frequency_hz(f0, b, 0), 0.0);
    for j in 1..=32 {
        let expected = (j as f64) * f0 * (1.0 + b * (j as f64) * (j as f64)).sqrt();
        assert!((partial_frequency_hz(f0, b, j) - expected).abs() < 1e-12);
    }
}

#[test]
fn railsback_stretch_is_zero_at_a4_and_monotone_across_88_keys() {
    let specs = build_scale_design(&VoicingSpec::default());
    let curve = railsback_stretch_cents_table(&specs);

    assert_eq!(curve[A4_KEY_IDX], 0.0);
    assert_eq!(railsback_stretch_cents(A4_KEY_IDX, &specs), 0.0);
    assert!((-32.0..=-20.0).contains(&curve[0]), "A0: {}", curve[0]);
    assert!(
        (25.0..=40.0).contains(&curve[NUM_KEYS - 1]),
        "C8: {}",
        curve[NUM_KEYS - 1]
    );
    for k in 1..NUM_KEYS {
        assert!(
            curve[k] > curve[k - 1],
            "k={k}: {} vs {}",
            curve[k - 1],
            curve[k]
        );
    }

    let no_stretch = VoicingSpec {
        stretch_amount: 0.0,
        ..VoicingSpec::default()
    };
    for k in 0..NUM_KEYS {
        assert_eq!(scaled_railsback_stretch_cents(k, &specs, &no_stretch), 0.0);
    }
}

#[test]
fn unison_detune_and_polarization_respect_scale_and_determinism() {
    let zero_detune = VoicingSpec {
        unison_detune_scale: 0.0,
        ..VoicingSpec::default()
    };
    let nominal = VoicingSpec::default();
    let specs = build_scale_design(&nominal);

    for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
        for s in 0..(layout.strings_per_key as usize) {
            assert_eq!(
                unison_detune_cents(k, s, layout.strings_per_key, &zero_detune),
                0.0
            );
            assert_eq!(
                polarization_offset_cents(k, s, Polarization::Vertical, &zero_detune),
                0.0
            );
            assert_eq!(
                polarization_offset_cents(k, s, Polarization::Horizontal, &zero_detune),
                0.0
            );
        }
    }
    assert_eq!(
        effective_string_f0_hz(A4_KEY_IDX, 1, Polarization::Vertical, &specs, &zero_detune),
        440.0
    );
    assert_eq!(
        effective_string_f0_hz(
            A4_KEY_IDX,
            1,
            Polarization::Horizontal,
            &specs,
            &zero_detune
        ),
        440.0
    );

    assert_eq!(unison_detune_cents(0, 0, 1, &nominal), 0.0);
    let d0 = unison_detune_cents(A4_KEY_IDX, 0, 3, &nominal);
    let d1 = unison_detune_cents(A4_KEY_IDX, 1, 3, &nominal);
    let d2 = unison_detune_cents(A4_KEY_IDX, 2, 3, &nominal);
    assert!(d0 < -0.22 && d1.abs() < 0.20 && d2 > 0.22);
    assert_eq!(d0, unison_detune_cents(A4_KEY_IDX, 0, 3, &nominal));

    let pv = polarization_offset_cents(A4_KEY_IDX, 1, Polarization::Vertical, &nominal);
    let ph = polarization_offset_cents(A4_KEY_IDX, 1, Polarization::Horizontal, &nominal);
    assert_eq!(pv, 0.0);
    assert!((0.15..=0.30).contains(&(ph - pv)));

    // C4 (key_idx=39) and A4 (key_idx=48) retain nominal 0.2..0.5 cents choir spread.
    let c4_0 = unison_detune_cents(39, 0, 3, &nominal);
    let c4_2 = unison_detune_cents(39, 2, 3, &nominal);
    assert!(c4_0 < -0.22 && c4_2 > 0.22);
}

#[test]
fn bridge_pin_couplings_provide_basal_asymmetry_and_unit_norm() {
    for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
        let n = layout.strings_per_key;
        let mut couplings = [(0.0_f64, 0.0_f64); 3];
        for (s, slot) in couplings.iter_mut().enumerate().take(n as usize) {
            let theta = bridge_pin_angle_rad(k, s, n);
            assert!((0.10..=0.28).contains(&theta), "k={k} s={s} theta={theta}");
            let (cv, ch) = bridge_pin_couplings(k, s, n);
            assert!(((cv * cv + ch * ch) - 1.0).abs() < 1e-12);
            assert!(cv > 0.9 && ch > 0.09);
            *slot = (cv, ch);
        }
        if n == 2 {
            assert!((couplings[0].1 - couplings[1].1).abs() > 0.02);
        } else if n == 3 {
            assert!((couplings[0].1 - couplings[1].1).abs() > 0.01);
            assert!((couplings[1].1 - couplings[2].1).abs() > 0.004);
            assert!((couplings[0].1 - couplings[2].1).abs() > 0.04);
        }
    }
}

#[test]
fn high_treble_unison_detune_tapers_above_c5_to_keep_c8_beating_gentle() {
    let nominal = VoicingSpec::default();
    let specs = build_scale_design(&nominal);

    // C8 (key_idx = 87) outer detune tapers to 0.18..0.32 cents, yielding < 0.9 Hz adjacent beating.
    let c8_0 = unison_detune_cents(87, 0, 3, &nominal);
    let c8_1 = unison_detune_cents(87, 1, 3, &nominal);
    let c8_2 = unison_detune_cents(87, 2, 3, &nominal);
    assert!(
        (0.18..=0.32).contains(&c8_0.abs()),
        "C8 string 0 detune out of 0.18..0.32 cents: {c8_0}"
    );
    assert!(
        (0.18..=0.32).contains(&c8_2.abs()),
        "C8 string 2 detune out of 0.18..0.32 cents: {c8_2}"
    );
    let f0_c8 = key_tuned_f0_hz(87, &specs, &nominal);
    let f1_c8 = partial_frequency_hz(f0_c8, specs[87].inharmonicity_b, 1);
    let beat_01 = f1_c8 * (exp2_f64((c8_1 - c8_0).abs() / 1200.0) - 1.0);
    let beat_12 = f1_c8 * (exp2_f64((c8_2 - c8_1).abs() / 1200.0) - 1.0);
    assert!(
        beat_01 < 0.9 && beat_12 < 0.9,
        "C8 adjacent beating too fast: {beat_01:.3} Hz, {beat_12:.3} Hz"
    );

    // All treble keys above C5 keep adjacent string beating below 0.9 Hz.
    for k in 51..NUM_KEYS {
        let o0 = unison_detune_cents(k, 0, 3, &nominal);
        let o1 = unison_detune_cents(k, 1, 3, &nominal);
        let o2 = unison_detune_cents(k, 2, 3, &nominal);
        let fk = key_tuned_f0_hz(k, &specs, &nominal);
        let b01 = fk * (exp2_f64((o1 - o0).abs() / 1200.0) - 1.0);
        let b12 = fk * (exp2_f64((o2 - o1).abs() / 1200.0) - 1.0);
        assert!(
            b01 < 0.9 && b12 < 0.9,
            "key {k} adjacent beating >= 0.9 Hz: {b01:.3}, {b12:.3}"
        );
    }
}

#[test]
fn bank_mode_counts_hit_concert_grand_25k_to_26k5_budget() {
    let counts = analyze_bank_mode_counts(&VoicingSpec::default());
    assert!(
        (1_400..=1_900).contains(&counts.longitudinal_unpadded),
        "{counts:?}"
    );
    assert!(
        (25_000..=26_500).contains(&counts.total_padded),
        "{counts:?}"
    );
}

#[test]
fn temperament_and_concert_pitch_shift_f0_prior_to_analytical_dispersion() {
    use crate::Temperament;

    let kirn = VoicingSpec {
        stretch_amount: 0.0,
        unison_detune_scale: 0.0,
        a4_freq_hz: 415.0,
        temperament: Temperament::KirnbergerIII,
        ..VoicingSpec::default()
    };
    let specs = build_scale_design(&kirn);

    // A4 remains exact concert pitch 415.0 Hz across temperaments.
    let f0_a4 = key_tuned_f0_hz(A4_KEY_IDX, &specs, &kirn);
    assert_eq!(f0_a4, 415.0);

    // C4 (key 39) -> E4 (key 43) must form a pure 5:4 major third before dispersion.
    let f0_c4 = key_tuned_f0_hz(39, &specs, &kirn);
    let f0_e4 = key_tuned_f0_hz(43, &specs, &kirn);
    assert!(
        (f0_e4 / f0_c4 - 1.25).abs() < 1e-12,
        "Kirnberger III C4-E4 ratio={}",
        f0_e4 / f0_c4
    );

    // Verify analytical dispersion f_j = j * F0 * sqrt(1 + B * j^2) uses the temperament F0.
    let f0_str = effective_string_f0_hz(39, 0, Polarization::Vertical, &specs, &kirn);
    assert_eq!(f0_str, f0_c4);
    for j in 1..=12 {
        let f_j = partial_frequency_hz(f0_str, specs[39].inharmonicity_b, j);
        let expected =
            (j as f64) * f0_str * (1.0 + specs[39].inharmonicity_b * (j * j) as f64).sqrt();
        assert!((f_j - expected).abs() < 1e-12);
    }
}

#[test]
fn equal_440_condition_zero_is_bit_exact_and_condition_adds_deterministic_detune() {
    use crate::Temperament;

    let a = VoicingSpec::default();
    let b = VoicingSpec::from_globals_full(0.5, 1.0, 1.0, 1.0, 440.0, Temperament::Equal, 0.0);
    let specs_a = build_scale_design(&a);
    let specs_b = build_scale_design(&b);
    for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
        assert_eq!(
            key_tuned_f0_hz(k, &specs_a, &a).to_bits(),
            key_tuned_f0_hz(k, &specs_b, &b).to_bits()
        );
        for s in 0..usize::from(layout.strings_per_key) {
            for pol in [Polarization::Vertical, Polarization::Horizontal] {
                assert_eq!(
                    effective_string_f0_hz(k, s, pol, &specs_a, &a).to_bits(),
                    effective_string_f0_hz(k, s, pol, &specs_b, &b).to_bits()
                );
            }
        }
    }

    let aged = VoicingSpec::default().with_condition(0.8);
    let specs_aged = build_scale_design(&aged);
    let f0_mint = effective_string_f0_hz(48, 0, Polarization::Vertical, &specs_a, &a);
    let f0_aged = effective_string_f0_hz(48, 0, Polarization::Vertical, &specs_aged, &aged);
    assert_ne!(f0_mint.to_bits(), f0_aged.to_bits());
    assert_eq!(
        f0_aged.to_bits(),
        effective_string_f0_hz(48, 0, Polarization::Vertical, &specs_aged, &aged).to_bits()
    );
}
