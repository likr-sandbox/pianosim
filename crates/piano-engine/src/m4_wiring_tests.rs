//! Integration tests for M4 Wave 3 wiring: time-sliced `VoicingCommitState`,
//! `SpaceBlob` sliced standby clearing, `Realtime` Sostenuto retention, and `PerfProfile` runtime sync.

use piano_radiator::{BodyArchetype, SpacePrepContext};
use piano_spec::{
    u7_to_u16, EventKind, LidState, ListenerPerspective, PedalKind, PerfProfile, SampleEvent,
    Temperament, VoicingSpec,
};
use piano_strings::VoicingPrepContext;

use crate::PianoEngine;

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

fn pedal(kind: PedalKind, value_u16: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::Pedal { kind, value_u16 },
    }
}

#[test]
fn voicing_commit_state_11_block_sliced_matches_synchronous_apply() {
    let sr = 48_000.0;
    let spec = VoicingSpec::from_globals_full(
        0.72,
        1.6,
        1.25,
        0.96,
        415.0,
        Temperament::WerckmeisterIII,
        0.35,
    );
    let mut vc_prep = VoicingPrepContext::new_with_profile(sr, &PerfProfile::balanced());
    let blob = vc_prep.prepare_voicing(&spec).to_vec();

    let mut eng_sync =
        PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::balanced());
    let mut eng_sliced =
        PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::balanced());

    assert!(eng_sync.apply_precomputed_voicing(spec.clone(), &blob));
    assert!(!eng_sync.is_voicing_applying());

    assert!(eng_sliced.begin_voicing_commit(spec.clone(), &blob));
    assert!(eng_sliced.is_voicing_applying());
    // Concurrent commit while active must be rejected cleanly.
    assert!(!eng_sliced.begin_voicing_commit(spec.clone(), &blob));

    let (mut l_sl, mut r_sl) = ([0.0_f32; 128], [0.0_f32; 128]);
    let (mut l_sy, mut r_sy) = ([0.0_f32; 128], [0.0_f32; 128]);
    let mut blocks = 0_usize;
    while eng_sliced.is_voicing_applying() {
        eng_sliced.render_block(&[], &mut l_sl, &mut r_sl);
        eng_sync.render_block(&[], &mut l_sy, &mut r_sy);
        blocks += 1;
        assert!(blocks <= 16, "voicing commit exceeded 11 blocks");
    }
    assert_eq!(blocks, 11);
    assert!(!eng_sliced.is_voicing_applying());

    // After block 11, both engines must produce identical audio for struck notes.
    let evs = [
        note_on(36, u7_to_u16(96), 0),
        note_on(69, u7_to_u16(100), 16),
        note_on(84, u7_to_u16(90), 32),
    ];
    for idx in 0..8 {
        let block_evs = if idx == 0 { &evs[..] } else { &[] };
        eng_sliced.render_block(block_evs, &mut l_sl, &mut r_sl);
        eng_sync.render_block(block_evs, &mut l_sy, &mut r_sy);
        let diff_l = l_sl
            .iter()
            .zip(l_sy.iter())
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        let diff_r = r_sl
            .iter()
            .zip(r_sy.iter())
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(
            diff_l < 1e-6 && diff_r < 1e-6,
            "block {idx} diverged after 11-block commit: diff_l={diff_l}, diff_r={diff_r}"
        );
    }
}

#[test]
fn realtime_sostenuto_sustains_above_0_001_rms_across_low_and_treble_keys() {
    let sr = 48_000.0;
    for midi_key in [60_u8, 76_u8] {
        let measure = |profile: PerfProfile| -> f64 {
            let mut eng = PianoEngine::new_with_profile(sr, VoicingSpec::default(), profile);
            let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);

            // Match pedals-voicing.spec.ts timing: hold 128 blocks, latch Sostenuto, release 128 blocks,
            // staccato G4 (80 blocks + 160 blocks decay) -> ~496 blocks (~1.32 s).
            eng.render_block(&[note_on(midi_key, u7_to_u16(100), 0)], &mut l, &mut r);
            for _ in 0..127 {
                eng.render_block(&[], &mut l, &mut r);
            }
            eng.render_block(
                &[
                    pedal(PedalKind::Sostenuto, u16::MAX, 0),
                    note_off(midi_key, 0),
                ],
                &mut l,
                &mut r,
            );
            for _ in 0..127 {
                eng.render_block(&[], &mut l, &mut r);
            }
            eng.render_block(&[note_on(67, u7_to_u16(100), 0)], &mut l, &mut r);
            for _ in 0..79 {
                eng.render_block(&[], &mut l, &mut r);
            }
            eng.render_block(&[note_off(67, 0)], &mut l, &mut r);
            for _ in 0..159 {
                eng.render_block(&[], &mut l, &mut r);
            }
            let mut sq_sum = 0.0_f64;
            let mut n = 0_usize;
            for _ in 0..16 {
                eng.render_block(&[], &mut l, &mut r);
                for (&sl, &sr_s) in l.iter().zip(r.iter()) {
                    let m = 0.5 * f64::from(sl + sr_s);
                    sq_sum += m * m;
                    n += 1;
                }
            }
            (sq_sum / n as f64).sqrt()
        };
        let rms_full = measure(PerfProfile::full());
        let rms_rt = measure(PerfProfile::realtime());
        if midi_key == 60 {
            assert!(
                rms_rt > 0.001,
                "Realtime Sostenuto MIDI {midi_key} RMS {rms_rt:.6} <= 0.001 (full={rms_full:.6})"
            );
        } else {
            let mut flat_cutoff = PerfProfile::realtime();
            flat_cutoff.low_order_energy_horizon_db = flat_cutoff.energy_horizon_db;
            let rms_flat = measure(flat_cutoff);
            assert!(
                rms_rt >= rms_flat,
                "Realtime low-order horizon should retain >= flat cutoff on MIDI {midi_key}: rt={rms_rt:.6}, flat={rms_flat:.6}"
            );
        }
        assert!(
            rms_rt >= rms_full * 0.65,
            "Realtime Sostenuto MIDI {midi_key} lost low-order sustain: rt={rms_rt:.6} vs full={rms_full:.6}"
        );
    }
}

#[test]
fn space_blob_sliced_standby_clear_handles_partial_and_full_chunk_counts() {
    let sr = 48_000.0;
    let mut eng =
        PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::balanced());
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    eng.render_block(&[note_on(60, 52_000, 0)], &mut l, &mut r);
    for _ in 0..12 {
        eng.render_block(&[], &mut l, &mut r);
    }

    let mut prep = SpacePrepContext::new_with_profile(f64::from(sr), &PerfProfile::balanced());
    prep.set_body_preset_bytes(crate::embedded_body_bytes(BodyArchetype::UprightStudio))
        .unwrap();
    let blob1 = prep
        .prepare_space(ListenerPerspective::ConcertHall, LidState::Half)
        .to_vec();

    // First swap (>= 16 chunks of 2048 floats), sets standby_delay_dirty = true on commit.
    let mut off = 0_usize;
    while off < blob1.len() {
        let n = 2_048.min(blob1.len() - off);
        assert!(eng.stage_precomputed_body_space_chunk(
            BodyArchetype::UprightStudio,
            ListenerPerspective::ConcertHall,
            LidState::Half,
            off,
            &blob1[off..off + n],
            blob1.len(),
        ));
        off += n;
    }
    assert!(eng.commit_precomputed_body_space(
        BodyArchetype::UprightStudio,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    ));
    for _ in 0..8 {
        eng.render_block(&[], &mut l, &mut r);
    }
    assert!(!eng.space_guard().is_crossfading());
    assert!(eng.space_guard().standby_delay_dirty());

    // Second swap with >= 16 chunks clears slices 0..16 across chunks, leaving dirty=false before commit.
    prep.set_body_preset_bytes(crate::embedded_body_bytes(BodyArchetype::VintageSaloon))
        .unwrap();
    let blob2 = prep
        .prepare_space(ListenerPerspective::BinauralHrtf, LidState::Closed)
        .to_vec();
    let mut off2 = 0_usize;
    while off2 < blob2.len() {
        let n = 2_048.min(blob2.len() - off2);
        assert!(eng.stage_precomputed_body_space_chunk(
            BodyArchetype::VintageSaloon,
            ListenerPerspective::BinauralHrtf,
            LidState::Closed,
            off2,
            &blob2[off2..off2 + n],
            blob2.len(),
        ));
        off2 += n;
    }
    assert!(!eng.space_guard().standby_delay_dirty());
    assert!(eng.commit_precomputed_body_space(
        BodyArchetype::VintageSaloon,
        ListenerPerspective::BinauralHrtf,
        LidState::Closed,
    ));
    for _ in 0..8 {
        eng.render_block(&[], &mut l, &mut r);
    }
    assert!(!eng.space_guard().is_crossfading());
    assert!(eng.space_guard().standby_delay_dirty());

    // Third swap with < 16 chunks (4 chunks) flushes remaining slices on commit cleanly.
    prep.set_body_preset_bytes(crate::embedded_body_bytes(BodyArchetype::ConcertGrand))
        .unwrap();
    let blob3 = prep
        .prepare_space(ListenerPerspective::Player, LidState::Open)
        .to_vec();
    let chunk_sz = blob3.len().div_ceil(4);
    let mut off3 = 0_usize;
    while off3 < blob3.len() {
        let n = chunk_sz.min(blob3.len() - off3);
        assert!(eng.stage_precomputed_body_space_chunk(
            BodyArchetype::ConcertGrand,
            ListenerPerspective::Player,
            LidState::Open,
            off3,
            &blob3[off3..off3 + n],
            blob3.len(),
        ));
        off3 += n;
    }
    assert!(eng.space_guard().standby_delay_dirty());
    assert!(eng.commit_precomputed_body_space(
        BodyArchetype::ConcertGrand,
        ListenerPerspective::Player,
        LidState::Open,
    ));
    assert!(!eng.space_guard().standby_delay_dirty());
    for _ in 0..8 {
        eng.render_block(&[], &mut l, &mut r);
        assert!(l
            .iter()
            .chain(r.iter())
            .all(|x| x.is_finite() && x.abs() <= 1.0));
    }
}

#[test]
fn runtime_perf_profile_syncs_low_order_horizon_and_decim_16() {
    let mut eng = PianoEngine::new(48_000.0);
    let mut custom = PerfProfile::full();
    custom.low_order_energy_horizon_db = -108.0;
    custom.damper_control_decim = 16;
    assert!(eng.set_perf_profile_runtime(custom));
    assert_eq!(eng.perf_profile().low_order_energy_horizon_db, -108.0);
    assert_eq!(eng.perf_profile().damper_control_decim, 16);
    assert_eq!(eng.core.action.perf_profile().damper_control_decim, 16);
}

fn assert_audio_parity(l_a: &[f32], r_a: &[f32], l_b: &[f32], r_b: &[f32], label: &str) {
    let diff = l_a
        .iter()
        .chain(r_a.iter())
        .zip(l_b.iter().chain(r_b.iter()))
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(diff < 1e-6, "{label} diverged: {diff}");
}

#[test]
fn voicing_commit_full_and_realtime_parity_with_active_strikes_and_hardness() {
    let sr = 48_000.0;
    for profile in [PerfProfile::full(), PerfProfile::realtime()] {
        let spec = VoicingSpec::from_globals_full(
            0.55,
            1.4,
            1.2,
            0.97,
            432.0,
            Temperament::KirnbergerIII,
            0.25,
        );
        let blob = VoicingPrepContext::new_with_profile(sr, &profile)
            .prepare_voicing(&spec)
            .to_vec();
        let mut eng_sync = PianoEngine::new_with_profile(sr, VoicingSpec::default(), profile);
        let mut eng_sliced = PianoEngine::new_with_profile(sr, VoicingSpec::default(), profile);
        let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);

        // Ring active notes before commit begins, then re-strike during blocks 1..10.
        eng_sliced.render_block(&[note_on(48, u7_to_u16(96), 0)], &mut l, &mut r);
        assert!(eng_sliced.begin_voicing_commit(spec.clone(), &blob));
        assert!(eng_sync.apply_precomputed_voicing(spec.clone(), &blob));
        eng_sliced.set_hammer_hardness(0.84);
        eng_sync.set_hammer_hardness(0.84);

        let mut blocks = 0_usize;
        while eng_sliced.is_voicing_applying() {
            let ev_2 = [note_on(48, u7_to_u16(88), 16)];
            let ev_5 = [note_on(72, u7_to_u16(102), 8)];
            let slice: &[SampleEvent] = match blocks {
                2 => &ev_2,
                5 => &ev_5,
                _ => &[],
            };
            eng_sliced.render_block(slice, &mut l, &mut r);
            assert!(l
                .iter()
                .chain(r.iter())
                .all(|x| x.is_finite() && x.abs() <= 1.0));
            blocks += 1;
            assert!(blocks <= 11);
        }
        assert_eq!(blocks, 11);
        while eng_sliced.voicing_smoother().is_active() {
            eng_sliced.render_block(&[], &mut l, &mut r);
        }
        while eng_sync.voicing_smoother().is_active() {
            eng_sync.render_block(&[], &mut l, &mut r);
        }
        assert!((eng_sliced.core.current_voicing.hammer_hardness - 0.84).abs() < 1e-5);
        eng_sliced.reset();
        eng_sync.reset();

        let (mut l_sl, mut r_sl) = ([0.0_f32; 128], [0.0_f32; 128]);
        let (mut l_sy, mut r_sy) = ([0.0_f32; 128], [0.0_f32; 128]);
        let post_evs = [
            note_on(36, u7_to_u16(96), 0),
            note_on(76, u7_to_u16(98), 24),
        ];
        for idx in 0..4 {
            let slice = if idx == 0 { &post_evs[..] } else { &[] };
            eng_sliced.render_block(slice, &mut l_sl, &mut r_sl);
            eng_sync.render_block(slice, &mut l_sy, &mut r_sy);
            assert_audio_parity(&l_sl, &r_sl, &l_sy, &r_sy, "post-commit parity");
        }
    }
}

#[test]
fn voicing_commit_preserves_concurrent_dirty_tuning_and_resyncs_on_abort() {
    let sr = 48_000.0;
    let spec_a = VoicingSpec::from_globals_full(
        0.65,
        1.5,
        1.2,
        0.98,
        415.0,
        Temperament::WerckmeisterIII,
        0.30,
    );
    let blob_a = VoicingPrepContext::new_with_profile(sr, &PerfProfile::full())
        .prepare_voicing(&spec_a)
        .to_vec();

    // 1. Concurrent DIRTY_TUNING queued during blocks 1..10 must survive block 11 and settle.
    let mut eng = PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::full());
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    assert!(eng.begin_voicing_commit(spec_a.clone(), &blob_a));
    for _ in 0..3 {
        eng.render_block(&[], &mut l, &mut r);
    }
    assert!(eng.is_voicing_applying());
    eng.set_a4_freq_hz(432.0);
    eng.set_temperament(Temperament::KirnbergerIII);
    while eng.is_voicing_applying() {
        eng.render_block(&[], &mut l, &mut r);
    }
    for _ in 0..64 {
        eng.render_block(&[], &mut l, &mut r);
    }
    assert!((eng.core.current_voicing.a4_freq_hz - 432.0).abs() < 0.05);
    assert_eq!(
        eng.core.current_voicing.temperament,
        Temperament::KirnbergerIII
    );
    assert!(!eng.voicing_smoother().is_active());

    // 2. Structural set_perf_profile abort mid-commit must resync Stage 1/2/3/smoother.
    let mut abort_eng =
        PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::full());
    assert!(abort_eng.begin_voicing_commit(spec_a, &blob_a));
    for _ in 0..4 {
        abort_eng.render_block(&[], &mut l, &mut r);
    }
    abort_eng.set_perf_profile(PerfProfile::realtime());
    assert!(!abort_eng.is_voicing_applying());
    assert_eq!(abort_eng.core.current_voicing, VoicingSpec::default());
    assert_eq!(abort_eng.voicing_smoother().current_a4_freq_hz(), 440.0);
    assert!(!abort_eng.voicing_smoother().is_active());

    let mut ref_eng =
        PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::realtime());
    let (mut l_ref, mut r_ref) = ([0.0_f32; 128], [0.0_f32; 128]);
    let evs = [
        note_on(24, u7_to_u16(95), 0),
        note_on(76, u7_to_u16(95), 16),
    ];
    for idx in 0..4 {
        let slice = if idx == 0 { &evs[..] } else { &[] };
        abort_eng.render_block(slice, &mut l, &mut r);
        ref_eng.render_block(slice, &mut l_ref, &mut r_ref);
        assert_audio_parity(&l, &r, &l_ref, &r_ref, "aborted commit resync");
    }
}

#[test]
fn interrupted_space_blob_stream_and_perf_profile_clear_staged_body() {
    let sr = 48_000.0;
    let (up, hall, half) = (
        BodyArchetype::UprightStudio,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    );
    let mut eng =
        PianoEngine::new_with_profile(sr, VoicingSpec::default(), PerfProfile::balanced());
    let mut prep = SpacePrepContext::new_with_profile(f64::from(sr), &PerfProfile::balanced());
    prep.set_body_preset_bytes(crate::embedded_body_bytes(up))
        .unwrap();
    let blob = prep.prepare_space(hall, half).to_vec();

    // Start streaming UprightStudio, then reject mismatched body chunk & commit.
    assert!(eng.stage_precomputed_body_space_chunk(up, hall, half, 0, &blob[..2_048], blob.len()));
    assert!(!eng.stage_precomputed_body_space_chunk(
        BodyArchetype::VintageSaloon,
        hall,
        half,
        2_048,
        &blob[2_048..4_096],
        blob.len(),
    ));
    assert!(!eng.commit_precomputed_body_space(up, hall, half));

    // Fully stage UprightStudio, then mutate radiators via set_perf_profile -> stale stage cleared.
    let mut off = 0_usize;
    while off < blob.len() {
        let n = 2_048.min(blob.len() - off);
        assert!(eng.stage_precomputed_body_space_chunk(
            up,
            hall,
            half,
            off,
            &blob[off..off + n],
            blob.len()
        ));
        off += n;
    }
    eng.set_perf_profile(PerfProfile::realtime());
    assert!(!eng.commit_precomputed_body_space(up, hall, half));
}
