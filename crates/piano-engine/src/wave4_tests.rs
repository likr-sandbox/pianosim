use core::f64::consts::TAU;

use piano_spec::{
    build_scale_design, key_tuned_f0_hz, partial_frequency_hz, u7_to_u16, EventKind, SampleEvent,
    Temperament, VoicingSpec,
};
use piano_strings::VoicingPrepContext;

use crate::m3_tests::mono_mix;
use crate::m4_tests::note_on;
use crate::PianoEngine;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

fn goertzel_mag_sq(samples: &[f32], freq_hz: f64) -> f64 {
    let w = TAU * freq_hz / SR64;
    let coeff = 2.0 * w.cos();
    let n = samples.len().max(1) as f64;
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for (i, &x) in samples.iter().enumerate() {
        let win = 0.5 * (1.0 - (TAU * (i as f64) / n).cos());
        let s0 = f64::from(x) * win + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn estimate_peak_hz(samples: &[f32], f_min: f64, f_max: f64) -> f64 {
    let mut best_f = f_min;
    let mut best_m = -1.0_f64;
    let mut f = f_min;
    while f <= f_max {
        let m = goertzel_mag_sq(samples, f);
        if m > best_m {
            best_m = m;
            best_f = f;
        }
        f += 0.25;
    }
    let lo = (best_f - 0.5).max(f_min);
    let hi = (best_f + 0.5).min(f_max);
    let mut fine = lo;
    while fine <= hi {
        let m = goertzel_mag_sq(samples, fine);
        if m > best_m {
            best_m = m;
            best_f = fine;
        }
        fine += 0.005;
    }
    best_f
}

fn render_bridge_blocks(
    engine: &mut PianoEngine,
    total_frames: usize,
    first_events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut off = 0;
    let mut first = true;
    while off < total_frames {
        let n = (total_frames - off).min(128);
        let evs = if first { first_events } else { &[] };
        first = false;
        engine.render_block_bridge(evs, &mut out_l[off..off + n], &mut out_r[off..off + n]);
        off += n;
    }
    (out_l, out_r)
}

fn render_stage4_blocks(
    engine: &mut PianoEngine,
    total_frames: usize,
    first_events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut off = 0;
    let mut first = true;
    while off < total_frames {
        let n = (total_frames - off).min(128);
        let evs = if first { first_events } else { &[] };
        first = false;
        engine.render_block(evs, &mut out_l[off..off + n], &mut out_r[off..off + n]);
        off += n;
    }
    (out_l, out_r)
}

#[test]
fn wave4_iv1_concert_pitch_bounds_392_and_466_2_hz_accuracy() {
    let on_a4 = [note_on(69, u7_to_u16(100), 0)];
    let mut prep = VoicingPrepContext::new(SR);

    for &(a4_hz, lo_hz, hi_hz) in &[(392.0_f32, 375.0, 410.0), (466.2_f32, 448.0, 484.0)] {
        let spec = VoicingSpec::default().with_concert_pitch_hz(a4_hz);
        let scales = build_scale_design(&spec);
        let expected_p1 = partial_frequency_hz(f64::from(a4_hz), scales[48].inharmonicity_b, 1);

        // 1. Direct constructor
        let mut eng_direct = PianoEngine::new_with_voicing(SR, &spec);
        assert_eq!(
            eng_direct.voicing_smoother().current_concert_pitch_hz(),
            a4_hz
        );
        assert_eq!(eng_direct.core.bridge.duplex().a4_freq_hz(), a4_hz);
        let (ld, rd) = render_bridge_blocks(&mut eng_direct, 24_576, &on_a4);
        let md = mono_mix(&ld, &rd);
        let pk_direct = estimate_peak_hz(&md[1_024..17_408], lo_hz, hi_hz);
        assert!(
            (pk_direct - expected_p1).abs() < 0.6,
            "direct A4={a4_hz}: expected {expected_p1:.2} Hz, got {pk_direct:.2}"
        );

        // 2. Live smoother transition from 440.0 Hz
        let mut eng_live = PianoEngine::new(SR);
        eng_live.set_concert_pitch_hz(a4_hz);
        let _ = render_bridge_blocks(&mut eng_live, 12_288, &[]);
        assert!((eng_live.voicing_smoother().current_concert_pitch_hz() - a4_hz).abs() < 1e-4);
        assert!((eng_live.core.bridge.duplex().a4_freq_hz() - a4_hz).abs() < 1e-4);
        let (ll, rl) = render_bridge_blocks(&mut eng_live, 24_576, &on_a4);
        let ml = mono_mix(&ll, &rl);
        let pk_live = estimate_peak_hz(&ml[1_024..17_408], lo_hz, hi_hz);
        assert!(
            (pk_live - expected_p1).abs() < 0.6,
            "live A4={a4_hz}: expected {expected_p1:.2} Hz, got {pk_live:.2}"
        );

        // 3. Off-thread VoicingPrepContext blob swap
        let mut eng_prep = PianoEngine::new(SR);
        let blob = prep.prepare_voicing(&spec);
        assert!(eng_prep.apply_precomputed_voicing(spec.clone(), blob));
        assert_eq!(eng_prep.core.bridge.duplex().a4_freq_hz(), a4_hz);
        let (lp, rp) = render_bridge_blocks(&mut eng_prep, 24_576, &on_a4);
        let mp = mono_mix(&lp, &rp);
        let pk_prep = estimate_peak_hz(&mp[1_024..17_408], lo_hz, hi_hz);
        assert!(
            (pk_prep - expected_p1).abs() < 0.6,
            "prep A4={a4_hz}: expected {expected_p1:.2} Hz, got {pk_prep:.2}"
        );
    }
}

#[test]
fn wave4_iv1_all_four_temperaments_spectral_pitch_accuracy() {
    let mut prep = VoicingPrepContext::new(SR);
    let temperaments = [
        Temperament::Equal,
        Temperament::WerckmeisterIII,
        Temperament::KirnbergerIII,
        Temperament::Vallotti,
    ];

    let mut equal_c4_hz = 0.0_f64;
    let mut equal_e4_hz = 0.0_f64;

    for temp in temperaments {
        let spec = VoicingSpec::default().with_temperament(temp);
        let scales = build_scale_design(&spec);

        // Live set_temperament updates engine & DuplexBank cleanly.
        let mut eng_live = PianoEngine::new(SR);
        eng_live.set_temperament(temp);
        let _ = render_bridge_blocks(&mut eng_live, 256, &[]);
        assert_eq!(eng_live.voicing_smoother().current_temperament(), temp);
        assert_eq!(eng_live.core.bridge.duplex().voicing().temperament, temp);

        // Off-thread VoicingPrepContext blob swap matches temperament.
        let mut eng_prep = PianoEngine::new(SR);
        let blob = prep.prepare_voicing(&spec);
        assert!(eng_prep.apply_precomputed_voicing(spec.clone(), blob));
        assert_eq!(eng_prep.voicing_smoother().current_temperament(), temp);
        assert_eq!(eng_prep.core.bridge.duplex().voicing().temperament, temp);

        // Verify A4 (anchor = 0 cents) stays at 440.0 Hz and C4/E4 track exact temperament cents.
        for &(midi, key_idx, lo_hz, hi_hz) in &[
            (69_u8, 48_usize, 425.0, 455.0),
            (60, 39, 250.0, 275.0),
            (64, 43, 315.0, 345.0),
        ] {
            let expected_f0 = key_tuned_f0_hz(key_idx, &scales, &spec);
            let expected_p1 = partial_frequency_hz(expected_f0, scales[key_idx].inharmonicity_b, 1);

            eng_live.reset();
            let (ll, rl) =
                render_bridge_blocks(&mut eng_live, 32_768, &[note_on(midi, u7_to_u16(100), 0)]);
            let pk_live = estimate_peak_hz(&mono_mix(&ll, &rl)[1_024..28_672], lo_hz, hi_hz);
            let err_cents_live = 1200.0 * (pk_live / expected_p1).log2().abs();
            assert!(
                err_cents_live < 5.0,
                "{:?} midi={midi} live peak {pk_live:.3} Hz vs {expected_p1:.3} Hz ({err_cents_live:.2} cents)",
                temp
            );

            eng_prep.reset();
            let (lp, rp) =
                render_bridge_blocks(&mut eng_prep, 32_768, &[note_on(midi, u7_to_u16(100), 0)]);
            let pk_prep = estimate_peak_hz(&mono_mix(&lp, &rp)[1_024..28_672], lo_hz, hi_hz);
            assert!(
                (pk_live - pk_prep).abs() < 0.02,
                "{:?} midi={midi} live ({pk_live:.3}) != prep ({pk_prep:.3})",
                temp
            );

            if temp == Temperament::Equal {
                if midi == 60 {
                    equal_c4_hz = pk_live;
                } else if midi == 64 {
                    equal_e4_hz = pk_live;
                }
            } else if midi == 60 || midi == 64 {
                let ref_hz = if midi == 60 { equal_c4_hz } else { equal_e4_hz };
                let delta_cents = 1200.0 * (pk_live / ref_hz).log2();
                let expected_cents = temp.cent_offset_for_key(key_idx);
                assert!(
                    (delta_cents - expected_cents).abs() < 1.0,
                    "{:?} midi={midi} relative shift {delta_cents:.2}c vs expected {expected_cents:.2}c",
                    temp
                );
            }
        }
    }
}

#[test]
fn wave4_iv1_condition_macro_deterministic_splitmix64_and_voicing_sync() {
    let chord = [
        note_on(48, u7_to_u16(96), 0),
        note_on(60, u7_to_u16(104), 0),
        note_on(67, u7_to_u16(100), 0),
    ];

    let spec_mint = VoicingSpec::default();
    let spec_aged = VoicingSpec::default().with_condition(0.75);

    let mut eng_mint = PianoEngine::new_with_voicing(SR, &spec_mint);
    let mut eng_aged_a = PianoEngine::new_with_voicing(SR, &spec_aged);
    let mut eng_aged_b = PianoEngine::new_with_voicing(SR, &spec_aged);

    let (lm, _rm) = render_stage4_blocks(&mut eng_mint, 4_096, &chord);
    let (la, ra) = render_stage4_blocks(&mut eng_aged_a, 4_096, &chord);
    let (lb, rb) = render_stage4_blocks(&mut eng_aged_b, 4_096, &chord);

    // Deterministic SplitMix64 behavior: identical condition -> bit-identical output.
    assert_eq!(
        la.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        lb.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        ra.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        rb.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );

    // Aged instrument differs audibly from mint instrument.
    let diff = lm
        .iter()
        .zip(la.iter())
        .fold(0.0_f32, |m, (&x, &y)| m.max((x - y).abs()));
    assert!(
        diff > 1e-3,
        "condition=0.75 did not alter audio (diff={diff})"
    );

    // Live smoothing and prep blob swap both propagate condition to smoother & DuplexBank.
    let mut eng_live = PianoEngine::new(SR);
    eng_live.set_condition(0.85);
    let _ = render_stage4_blocks(&mut eng_live, 12_288, &[]);
    assert!((eng_live.voicing_smoother().current_condition() - 0.85).abs() < 1e-4);
    assert!((eng_live.core.bridge.duplex().voicing().condition - 0.85).abs() < 1e-4);

    let mut prep = VoicingPrepContext::new(SR);
    let spec_full =
        VoicingSpec::from_globals_full(0.55, 1.1, 1.05, 1.0, 415.0, Temperament::Vallotti, 0.6);
    let blob = prep.prepare_voicing(&spec_full);
    let mut eng_prep = PianoEngine::new(SR);
    assert!(eng_prep.apply_precomputed_voicing(spec_full, blob));
    assert_eq!(
        eng_prep.voicing_smoother().current_concert_pitch_hz(),
        415.0
    );
    assert_eq!(
        eng_prep.voicing_smoother().current_temperament(),
        Temperament::Vallotti
    );
    assert_eq!(eng_prep.voicing_smoother().current_condition(), 0.6);
    assert_eq!(
        eng_prep.core.bridge.duplex().voicing().temperament,
        Temperament::Vallotti
    );
    assert_eq!(eng_prep.core.bridge.duplex().voicing().condition, 0.6);
}

#[test]
fn wave4_iv1_default_440_equal_condition_zero_bit_exact_parity() {
    let events = [
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: piano_spec::PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        },
        note_on(36, u7_to_u16(90), 0),
        note_on(60, u7_to_u16(100), 16),
        note_on(84, u7_to_u16(110), 48),
    ];

    let mut eng_default = PianoEngine::new(SR);
    let explicit_spec =
        VoicingSpec::from_globals_full(0.5, 1.0, 1.0, 1.0, 440.0, Temperament::Equal, 0.0);
    let mut eng_explicit = PianoEngine::new_with_voicing(SR, &explicit_spec);

    let (ld, rd) = render_stage4_blocks(&mut eng_default, 4_096, &events);
    let (le, re) = render_stage4_blocks(&mut eng_explicit, 4_096, &events);

    for (i, ((&a, &b), (&c, &d))) in ld
        .iter()
        .zip(le.iter())
        .zip(rd.iter().zip(re.iter()))
        .enumerate()
    {
        assert_eq!(a.to_bits(), b.to_bits(), "L mismatch at sample {i}");
        assert_eq!(c.to_bits(), d.to_bits(), "R mismatch at sample {i}");
    }

    // Mutating to non-default and restoring via set_voicing_full restores duplex & smoother parity.
    let mut eng_restored = PianoEngine::new(SR);
    eng_restored.set_voicing_full(0.8, 1.5, 1.3, 0.95, 392.0, Temperament::KirnbergerIII, 0.9);
    let _ = render_stage4_blocks(&mut eng_restored, 12_288, &[]);
    eng_restored.set_voicing_full(0.5, 1.0, 1.0, 1.0, 440.0, Temperament::Equal, 0.0);
    let _ = render_stage4_blocks(&mut eng_restored, 12_288, &[]);

    assert_eq!(
        eng_restored.voicing_smoother().current_concert_pitch_hz(),
        440.0
    );
    assert_eq!(
        eng_restored.voicing_smoother().current_temperament(),
        Temperament::Equal
    );
    assert_eq!(eng_restored.voicing_smoother().current_condition(), 0.0);
    for m in 0..piano_bridge::NUM_DUPLEX_MODES {
        assert_eq!(
            eng_restored.core.bridge.duplex().mode_freq_hz(m),
            eng_default.core.bridge.duplex().mode_freq_hz(m)
        );
    }
}
