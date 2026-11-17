use super::*;

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
    assert!(d0 < -0.35 && d1.abs() < 0.20 && d2 > 0.35);
    assert_eq!(d0, unison_detune_cents(A4_KEY_IDX, 0, 3, &nominal));

    let pv = polarization_offset_cents(A4_KEY_IDX, 1, Polarization::Vertical, &nominal);
    let ph = polarization_offset_cents(A4_KEY_IDX, 1, Polarization::Horizontal, &nominal);
    assert_eq!(pv, 0.0);
    assert!((0.15..=0.30).contains(&(ph - pv)));
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
