use super::*;

#[test]
fn defaults_match_nominal_concert_spec() {
    let spec = VoicingSpec::default();
    assert_eq!(spec.hammer_hardness, 0.5);
    assert_eq!(spec.unison_detune_scale, 1.0);
    assert_eq!(spec.stretch_amount, 1.0);
    assert_eq!(spec.string_length_scale, 1.0);
    assert_eq!(spec.a4_freq_hz, 440.0);
    assert_eq!(spec.effective_a4_freq_hz(), 440.0);
    assert_eq!(spec.effective_concert_pitch_hz(), 440.0);
    assert_eq!(spec.temperament, Temperament::Equal);
    assert_eq!(spec.condition, 0.0);
    assert_eq!(spec.effective_condition(), 0.0);
    assert_eq!(spec.note_offsets.len(), 88);

    for k in 0..NUM_KEYS {
        assert_eq!(spec.note_offsets[k], NoteVoicingOffset::ZERO);
        assert_eq!(spec.effective_note_offset(k), NoteVoicingOffset::ZERO);
        assert_eq!(spec.effective_hammer_hardness(k), 0.5);
        assert_eq!(spec.effective_pitch_cents(k), 0.0);
        assert_eq!(spec.effective_inharmonicity_scale(k), 1.0);
        assert_eq!(spec.effective_level_db_offset(k), 0.0);
        assert_eq!(spec.effective_damping_scale(k), 1.0);
    }
}

#[test]
fn clamping_and_nan_safety_and_per_note_overrides() {
    let mut spec = VoicingSpec {
        hammer_hardness: 0.85,
        unison_detune_scale: -1.5,
        stretch_amount: 5.0,
        string_length_scale: f32::NAN,
        a4_freq_hz: 380.0,
        temperament: Temperament::Vallotti,
        condition: 1.5,
        note_offsets: [NoteVoicingOffset::ZERO; NUM_KEYS],
    };
    spec.note_offsets[48] = NoteVoicingOffset {
        pitch_cents: 12.5,
        inharmonicity_scale: 0.25,
        hammer_hardness_offset: 0.4,
        level_db_offset: -40.0,
        damping_scale: 0.25,
    };
    spec.note_offsets[0] = NoteVoicingOffset {
        pitch_cents: f32::NAN,
        inharmonicity_scale: -2.0,
        hammer_hardness_offset: f32::NEG_INFINITY,
        level_db_offset: 25.0,
        damping_scale: f32::NAN,
    };

    assert_eq!(spec.effective_unison_detune_scale(), 0.0);
    assert_eq!(spec.effective_stretch_amount(), STRETCH_AMOUNT_MAX);
    assert_eq!(
        spec.effective_string_length_scale(),
        STRING_LENGTH_SCALE_DEFAULT
    );
    assert_eq!(spec.effective_a4_freq_hz(), 392.0);
    spec.a4_freq_hz = 500.0;
    assert_eq!(spec.effective_a4_freq_hz(), 466.2);
    spec.a4_freq_hz = f32::NAN;
    assert_eq!(spec.effective_a4_freq_hz(), 440.0);
    assert_eq!(
        VoicingSpec::from_globals_with_a4(0.5, 1.0, 1.0, 1.0, 392.0).a4_freq_hz,
        392.0
    );
    assert_eq!(
        VoicingSpec::from_globals_with_a4(0.5, 1.0, 1.0, 1.0, 466.2).a4_freq_hz,
        466.2
    );

    assert_eq!(spec.effective_condition(), 1.0);
    spec.condition = -0.5;
    assert_eq!(spec.effective_condition(), 0.0);
    spec.condition = f32::NAN;
    assert_eq!(spec.effective_condition(), 0.0);

    // With condition == 0.0, per-note overrides preserve exact legacy clamping.
    assert_eq!(spec.effective_pitch_cents(48), 12.5);
    assert_eq!(spec.effective_inharmonicity_scale(48), 1.25);
    assert_eq!(spec.effective_hammer_hardness(48), 1.0);
    assert_eq!(spec.effective_level_db_offset(48), NOTE_LEVEL_DB_MIN);
    assert_eq!(spec.effective_damping_scale(48), 1.25);

    assert_eq!(spec.effective_pitch_cents(0), 0.0);
    assert_eq!(
        spec.effective_inharmonicity_scale(0),
        NOTE_INHARMONICITY_SCALE_MIN
    );
    assert!((spec.effective_hammer_hardness(0) - 0.35).abs() < 1e-6);
    assert_eq!(spec.effective_level_db_offset(0), NOTE_LEVEL_DB_MAX);
    assert_eq!(spec.effective_damping_scale(0), 1.0);

    // Out-of-range key index safely falls back without panicking.
    assert_eq!(spec.effective_pitch_cents(999), 0.0);
    assert_eq!(spec.effective_inharmonicity_scale(999), 1.0);
    assert_eq!(spec.effective_damping_scale(999), 1.0);
}

#[test]
fn condition_macro_uses_deterministic_splitmix64_and_is_zero_when_mint() {
    for &c in &[0.0_f32, -0.25, f32::NAN] {
        for k in 0..NUM_KEYS {
            assert_eq!(condition_note_offset(c, k), NoteVoicingOffset::ZERO);
        }
    }
    assert_eq!(condition_note_offset(1.0, 999), NoteVoicingOffset::ZERO);

    let aged_half = VoicingSpec::default()
        .with_temperament(Temperament::KirnbergerIII)
        .with_concert_pitch_hz(415.0)
        .with_condition(0.5);
    let aged_full =
        VoicingSpec::from_globals_full(0.5, 1.0, 1.0, 1.0, 415.0, Temperament::KirnbergerIII, 1.0);
    assert_eq!(
        aged_half.effective_temperament(),
        Temperament::KirnbergerIII
    );
    assert_eq!(aged_half.effective_concert_pitch_hz(), 415.0);

    let mut non_zero_pitch_keys = 0usize;
    let mut non_zero_hammer_keys = 0usize;
    let mut non_zero_damp_keys = 0usize;
    for k in 0..NUM_KEYS {
        let o_half = condition_note_offset(0.5, k);
        let o_full = condition_note_offset(1.0, k);
        // Exact determinism across repeated calls.
        assert_eq!(o_half, condition_note_offset(0.5, k));
        assert_eq!(o_half, aged_half.effective_note_offset(k));
        assert_eq!(o_full, aged_full.effective_note_offset(k));
        // Exact linearity in condition scalar.
        assert!((o_full.pitch_cents - 2.0 * o_half.pitch_cents).abs() < 1e-6);
        assert!((o_full.hammer_hardness_offset - 2.0 * o_half.hammer_hardness_offset).abs() < 1e-6);
        assert!((o_full.damping_scale - 2.0 * o_half.damping_scale).abs() < 1e-6);
        assert!((o_full.inharmonicity_scale - 2.0 * o_half.inharmonicity_scale).abs() < 1e-6);
        assert!((o_full.level_db_offset - 2.0 * o_half.level_db_offset).abs() < 1e-6);

        if o_full.pitch_cents.abs() > 0.1 {
            non_zero_pitch_keys += 1;
        }
        if o_full.hammer_hardness_offset.abs() > 0.005 {
            non_zero_hammer_keys += 1;
        }
        if o_full.damping_scale.abs() > 0.005 {
            non_zero_damp_keys += 1;
        }
        assert!((-10.0..=10.0).contains(&aged_full.effective_pitch_cents(k)));
        assert!((0.40..=0.70).contains(&aged_full.effective_hammer_hardness(k)));
        assert!((0.90..=1.15).contains(&aged_full.effective_damping_scale(k)));
    }
    assert!(non_zero_pitch_keys > 70);
    assert!(non_zero_hammer_keys > 70);
    assert!(non_zero_damp_keys > 70);
}
