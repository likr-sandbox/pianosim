//! Wave 3 verification tests (`III-1` hall FDN reverb & binaural IACC, `III-2` multi-body Stage 3/4 sync).

use piano_radiator::{embedded_body_bytes, BodyArchetype, SpacePrepContext};
use piano_spec::{u7_to_u16, BodyPreset, EventKind, LidState, ListenerPerspective, SampleEvent};

use crate::{PianoEngine, DEFAULT_REVERB_DECAY_S, DEFAULT_REVERB_MIX};

const SR: f32 = 48_000.0;

fn note_on(midi_key: u8, vel: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::NoteOn {
            midi_key,
            velocity_u16: vel,
        },
    }
}

fn note_off(midi_key: u8, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::NoteOff {
            midi_key,
            release_vel_u16: 32_768,
        },
    }
}

fn render_chunks(
    engine: &mut PianoEngine,
    total_frames: usize,
    first_events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut pos = 0_usize;
    let mut first = true;
    while pos < total_frames {
        let n = (total_frames - pos).min(128);
        let evs = if first { first_events } else { &[] };
        first = false;
        engine.render_block(evs, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    (out_l, out_r)
}

fn rms(slice: &[f32]) -> f64 {
    let sum_sq: f64 = slice.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / (slice.len().max(1) as f64)).sqrt()
}

fn assert_bridge_matches_preset(engine: &PianoEngine, archetype: BodyArchetype) {
    let preset = BodyPreset::decode(embedded_body_bytes(archetype)).expect("valid .pbdy");
    let ssm = engine.core.bridge.ssm();
    assert_eq!(ssm.y_inf(), &preset.y_inf);
    for (pole, expected) in ssm.poles().iter().zip(preset.reduced_poles.iter()) {
        assert!((pole.freq_hz - expected.freq_hz).abs() < 1e-4);
        assert!((pole.zeta - expected.zeta).abs() < 1e-5);
        assert!((pole.modal_mass_kg - expected.modal_mass_kg).abs() < 1e-4);
        assert_eq!(pole.zone_gains, expected.zone_gains);
    }
}

#[test]
fn wave3_iii2_multibody_sync_stage3_and_stage4_across_all_archetypes() {
    let mut engine = PianoEngine::new(SR);
    assert_eq!(engine.body_archetype(), BodyArchetype::ConcertGrand);
    assert_bridge_matches_preset(&engine, BodyArchetype::ConcertGrand);

    let archetypes = [
        BodyArchetype::ConcertGrand,
        BodyArchetype::UprightStudio,
        BodyArchetype::VintageSaloon,
    ];
    let mut waveforms = Vec::with_capacity(3);
    for arch in archetypes {
        engine.reset();
        engine.set_body_archetype(arch);
        assert_eq!(engine.body_archetype(), arch);
        assert_bridge_matches_preset(&engine, arch);

        let active = engine.space_guard().active_radiator_idx();
        let expected_preset = BodyPreset::decode(embedded_body_bytes(arch)).expect("valid .pbdy");
        assert_eq!(
            engine.core.radiators[active].preset().direct_zone_gain,
            expected_preset.direct_zone_gain
        );

        let (ol, or) = render_chunks(&mut engine, 8_192, &[note_on(60, u7_to_u16(100), 0)]);
        assert!(ol
            .iter()
            .chain(or.iter())
            .all(|x| x.is_finite() && x.abs() <= 1.0));
        let mono: Vec<f32> = ol
            .iter()
            .zip(or.iter())
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        assert!(rms(&mono) > 1e-3, "{arch:?} produced silent audio");
        waveforms.push(mono);
    }

    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let diff_rms = waveforms[i]
            .iter()
            .zip(waveforms[j].iter())
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / (waveforms[i].len() as f64);
        assert!(
            diff_rms.sqrt() > 1e-4,
            "{:?} and {:?} waveforms unexpectedly identical",
            archetypes[i],
            archetypes[j]
        );
    }
}

#[test]
fn wave3_iii2_precomputed_body_blob_staging_commit_crossfade_and_sync() {
    let mut engine = PianoEngine::new(SR);
    engine.set_reverb(2.1, 0.25);

    // Start ringing A4 on ConcertGrand (Player, Open).
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    engine.render_block(&[note_on(69, u7_to_u16(100), 0)], &mut l, &mut r);
    for _ in 0..8 {
        engine.render_block(&[], &mut l, &mut r);
    }

    // Prepare UprightStudio blob for the SAME (Player, Open) perspective & lid.
    let mut prep = SpacePrepContext::new_from_preset_bytes(
        embedded_body_bytes(BodyArchetype::UprightStudio),
        f64::from(SR),
    )
    .expect("upright prep");
    let blob = prep.prepare_space(ListenerPerspective::Player, LidState::Open);
    let total = blob.len();

    // Stream in chunks into standby.
    let mut off = 0_usize;
    while off < total {
        let n = 4_096.min(total - off);
        assert!(engine.stage_precomputed_body_space_chunk(
            BodyArchetype::UprightStudio,
            ListenerPerspective::Player,
            LidState::Open,
            off,
            &blob[off..off + n],
            total,
        ));
        off += n;
    }

    // Mismatched body commit must fail cleanly.
    assert!(!engine.commit_precomputed_body_space(
        BodyArchetype::ConcertGrand,
        ListenerPerspective::Player,
        LidState::Open,
    ));

    // Correct commit triggers crossfade (audio is ringing) and syncs Stage 3 even though persp/lid stayed Player/Open!
    assert!(engine.commit_precomputed_body_space(
        BodyArchetype::UprightStudio,
        ListenerPerspective::Player,
        LidState::Open,
    ));
    assert!(engine.space_guard().is_crossfading());
    assert_eq!(engine.body_archetype(), BodyArchetype::UprightStudio);
    assert_bridge_matches_preset(&engine, BodyArchetype::UprightStudio);
    assert_eq!(engine.reverb_params(), (2.1, 0.25));

    // Render past 8 ms crossfade without clicks.
    let mut prev = l[127];
    let mut max_delta = 0.0_f32;
    for _ in 0..16 {
        engine.render_block(&[], &mut l, &mut r);
        for &s in &l {
            assert!(s.is_finite() && s.abs() <= 1.0);
            max_delta = max_delta.max((s - prev).abs());
            prev = s;
        }
    }
    assert!(!engine.space_guard().is_crossfading());
    assert!(max_delta < 0.15, "crossfade click max_delta={max_delta}");

    // Synchronous set_space after precomputed body commit must stay on UprightStudio!
    engine.set_space(ListenerPerspective::ConcertHall, LidState::Half);
    for _ in 0..16 {
        engine.render_block(&[], &mut l, &mut r);
    }
    assert_eq!(engine.body_archetype(), BodyArchetype::UprightStudio);
    let active = engine.space_guard().active_radiator_idx();
    let upright_preset =
        BodyPreset::decode(embedded_body_bytes(BodyArchetype::UprightStudio)).expect("upright");
    assert_eq!(
        engine.core.radiators[active].preset().direct_zone_gain,
        upright_preset.direct_zone_gain
    );
    assert_eq!(engine.reverb_params(), (2.1, 0.25));

    // One-shot apply_precomputed_body_space switches to VintageSaloon cleanly.
    prep.set_body_preset_bytes(embedded_body_bytes(BodyArchetype::VintageSaloon))
        .expect("saloon bytes");
    let saloon_blob = prep.prepare_space(ListenerPerspective::BinauralHrtf, LidState::Open);
    assert!(engine.apply_precomputed_body_space(
        BodyArchetype::VintageSaloon,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
        saloon_blob,
    ));
    assert_eq!(engine.body_archetype(), BodyArchetype::VintageSaloon);
    assert_bridge_matches_preset(&engine, BodyArchetype::VintageSaloon);
}

#[test]
fn wave3_iii1_reverb_controls_preserve_across_space_and_clear_on_all_notes_off() {
    let mut eng = PianoEngine::new(SR);
    assert_eq!(
        eng.reverb_params(),
        (DEFAULT_REVERB_DECAY_S, DEFAULT_REVERB_MIX)
    );

    // Clamping and non-finite sanitization.
    eng.set_reverb(9.9, 2.5);
    assert_eq!(eng.reverb_params(), (3.2, 1.0));
    eng.set_reverb(f32::NAN, f32::INFINITY);
    assert_eq!(eng.reverb_params(), (0.0, 0.0));

    // Short staccato C4: wet tail [0.55..1.05 s] must exceed dry mix=0 tail substantially.
    let render_staccato_tail = |decay: f32, mix: f32| -> f64 {
        let mut e = PianoEngine::new(SR);
        e.set_reverb(decay, mix);
        let (_, _) = render_chunks(&mut e, 3_840, &[note_on(60, u7_to_u16(110), 0)]);
        let (_, _) = render_chunks(&mut e, 22_000, &[note_off(60, 0)]);
        let (tail_l, tail_r) = render_chunks(&mut e, 24_000, &[]);
        let mono: Vec<f32> = tail_l
            .iter()
            .zip(tail_r.iter())
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        rms(&mono)
    };
    let dry_tail = render_staccato_tail(2.4, 0.0);
    let wet_tail = render_staccato_tail(2.4, 0.45);
    assert!(
        wet_tail > dry_tail * 2.5,
        "expected strong reverb tail: wet={wet_tail:.6}, dry={dry_tail:.6}"
    );

    // Space reconfiguration preserves reverb settings on both radiators.
    eng.set_reverb(1.9, 0.35);
    eng.set_space(ListenerPerspective::ConcertHall, LidState::Half);
    assert_eq!(eng.reverb_params(), (1.9, 0.35));
    assert_eq!(eng.core.radiators[0].reverb_params(), (1.9, 0.35));
    assert_eq!(eng.core.radiators[1].reverb_params(), (1.9, 0.35));

    // AllNotesOff resets reverb FDN/ER and limiter immediately to exact zero output.
    let _ = render_chunks(&mut eng, 4_096, &[note_on(60, u7_to_u16(115), 0)]);
    let (post_l, post_r) = render_chunks(
        &mut eng,
        512,
        &[SampleEvent {
            sample_offset: 0,
            kind: EventKind::AllNotesOff,
        }],
    );
    assert!(
        post_l.iter().chain(post_r.iter()).all(|&x| x == 0.0),
        "AllNotesOff must clear FDN reverb tail immediately"
    );
    assert_eq!(eng.reverb_params(), (1.9, 0.35));
}

fn one_pole_lp(sig: &[f32], cutoff_hz: f64, sr: f64) -> Vec<f64> {
    let a = (-2.0 * core::f64::consts::PI * cutoff_hz / sr).exp();
    let mut y = 0.0_f64;
    sig.iter()
        .map(|&x| {
            y = (1.0 - a) * f64::from(x) + a * y;
            y
        })
        .collect()
}

fn one_pole_hp(sig: &[f32], cutoff_hz: f64, sr: f64) -> Vec<f64> {
    let lp = one_pole_lp(sig, cutoff_hz, sr);
    sig.iter()
        .zip(lp.iter())
        .map(|(&x, &l)| f64::from(x) - l)
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let (mut num, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += x * y;
        aa += x * x;
        bb += y * y;
    }
    num / (aa * bb).sqrt().max(1e-30)
}

#[test]
fn wave3_iii1_binaural_hrtf_low_freq_iacc_coherent_in_engine() {
    let mut eng = PianoEngine::new(SR);
    eng.set_body_space(
        BodyArchetype::ConcertGrand,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
    );
    eng.set_reverb(2.2, 0.65);

    // Strike C4 + release quickly so late window captures diffuse reverb tail.
    let _ = render_chunks(&mut eng, 2_400, &[note_on(60, u7_to_u16(115), 0)]);
    let _ = render_chunks(&mut eng, 16_000, &[note_off(60, 0)]);
    let (tail_l, tail_r) = render_chunks(&mut eng, 24_000, &[]);

    let lf_l = one_pole_lp(&tail_l, 350.0, f64::from(SR));
    let lf_r = one_pole_lp(&tail_r, 350.0, f64::from(SR));
    let hf_l = one_pole_hp(&tail_l, 1_500.0, f64::from(SR));
    let hf_r = one_pole_hp(&tail_r, 1_500.0, f64::from(SR));

    let iacc_lf = correlation(&lf_l[2_000..], &lf_r[2_000..]);
    let iacc_hf = correlation(&hf_l[2_000..], &hf_r[2_000..]);
    assert!(
        iacc_lf > 0.60 && iacc_lf > iacc_hf + 0.15,
        "BinauralHrtf <400 Hz IACC expected coherent >0.60 and >HF: lf={iacc_lf:.3}, hf={iacc_hf:.3}"
    );
}
