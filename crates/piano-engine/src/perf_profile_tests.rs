//! Comprehensive tests for `PerfProfile`, `P0-2`, `P1-4`, and `SpaceBlob` v2 wiring in `piano-engine`.

use piano_radiator::{BodyArchetype, SpacePrepContext};
use piano_spec::{
    u7_to_u16, EventKind, LidState, ListenerPerspective, PedalKind, PerfPreset, PerfProfile,
    SampleEvent, VoicingSpec,
};

use crate::{space_blob_len_for_profile, PianoEngine, SPACE_BLOB_MAGIC_V2};

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

fn render_blocks(
    engine: &mut PianoEngine,
    total_frames: usize,
    block_size: usize,
    first_events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut pos = 0;
    let mut first = true;
    while pos < total_frames {
        let n = (total_frames - pos).min(block_size);
        let evs = if first { first_events } else { &[] };
        first = false;
        engine.render_block(evs, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    (out_l, out_r)
}

#[inline(never)]
fn render_with_profile(
    sr: f32,
    profile: PerfProfile,
    total_frames: usize,
    evs: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut eng = PianoEngine::new_with_profile(sr, VoicingSpec::default(), profile);
    render_blocks(&mut eng, total_frames, 128, evs)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f32, |m, (&x, &y)| m.max((x - y).abs()))
}

#[test]
fn full_profile_has_bit_exact_parity_with_engine_new() {
    let mut ref_eng = PianoEngine::new(48_000.0);
    let mut prof_eng =
        PianoEngine::new_with_profile(48_000.0, VoicingSpec::default(), PerfProfile::full());
    assert_eq!(prof_eng.perf_profile(), PerfProfile::full());
    assert_eq!(ref_eng.total_modes(), prof_eng.total_modes());

    let evs = [
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        },
        note_on(36, u7_to_u16(100), 0),
        note_on(60, u7_to_u16(90), 16),
        note_on(84, u7_to_u16(110), 48),
    ];
    let (l_ref, r_ref) = render_blocks(&mut ref_eng, 4096, 128, &evs);
    let (l_prof, r_prof) = render_blocks(&mut prof_eng, 4096, 128, &evs);
    assert_eq!(l_ref, l_prof);
    assert_eq!(r_ref, r_prof);
}

#[test]
fn balanced_and_realtime_render_finite_audio_across_sample_rates() {
    let evs = [
        note_on(28, u16::MAX, 0),
        note_on(60, u7_to_u16(96), 8),
        note_on(96, u7_to_u16(110), 24),
    ];
    for sr in [44_100.0_f32, 48_000.0, 96_000.0] {
        for preset in [PerfPreset::Balanced, PerfPreset::Realtime] {
            let profile = PerfProfile::from_preset(preset);
            let (out_l, out_r) = render_with_profile(sr, profile, 4096, &evs);
            assert!(out_l.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
            assert!(out_r.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
            assert!(out_l.iter().any(|&x| x.abs() > 1e-4));
        }
    }
}

#[test]
fn stage1_and_stage3_knobs_take_effect_genuinely() {
    let sr = 48_000.0_f32;
    let base_ev = [note_on(60, u16::MAX, 0)];
    let (base_l, _) = render_with_profile(sr, PerfProfile::full(), 2048, &base_ev);

    // 1. action_oversample
    let p_os = PerfProfile {
        action_oversample: 4,
        ..PerfProfile::full()
    };
    let (l_os, _) = render_with_profile(sr, p_os, 2048, &base_ev);
    assert!(max_abs_diff(&base_l, &l_os) > 1e-6);

    // 2. polynomial_adaa
    let p_poly = PerfProfile {
        polynomial_adaa: true,
        ..PerfProfile::full()
    };
    let (l_poly, _) = render_with_profile(sr, p_poly, 2048, &base_ev);
    assert!(max_abs_diff(&base_l, &l_poly) > 1e-6);

    // 3. shank_flex_enabled
    let p_sh = PerfProfile {
        shank_flex_enabled: false,
        ..PerfProfile::full()
    };
    let (l_sh, _) = render_with_profile(sr, p_sh, 2048, &base_ev);
    assert!(max_abs_diff(&base_l, &l_sh) > 1e-6);

    // 4. damper_control_decim
    let run_damper_release = |decim: u8| -> Vec<f32> {
        let p = PerfProfile {
            damper_control_decim: decim,
            ..PerfProfile::full()
        };
        let mut eng = PianoEngine::new_with_profile(sr, VoicingSpec::default(), p);
        assert_eq!(eng.core.action.perf_profile().damper_control_decim, decim);
        let _ = render_blocks(&mut eng, 512, 128, &[note_on(48, 60_000, 0)]);
        render_blocks(&mut eng, 4096, 128, &[note_off(48, 0)]).0
    };
    assert!(max_abs_diff(&run_damper_release(1), &run_damper_release(4)) > 0.0);

    // 5. hertz_nonlinear
    let p_hz = PerfProfile {
        hertz_nonlinear: false,
        ..PerfProfile::full()
    };
    let (l_hz, _) = render_with_profile(sr, p_hz, 2048, &base_ev);
    assert!(max_abs_diff(&base_l, &l_hz) > 1e-7);

    // 6. active_bridge_poles
    let p_po = PerfProfile {
        active_bridge_poles: 4,
        ..PerfProfile::full()
    };
    let mut eng_po = PianoEngine::new_with_profile(sr, VoicingSpec::default(), p_po);
    assert_eq!(eng_po.core.bridge.active_poles(), 4);
    let (l_po, _) = render_blocks(&mut eng_po, 2048, 128, &base_ev);
    assert!(max_abs_diff(&base_l, &l_po) > 1e-6);
}

#[test]
fn stage2_and_stage4_knobs_take_effect_genuinely() {
    let sr = 48_000.0_f32;
    let base_modes = PianoEngine::new(sr).total_modes();

    // horiz_max_hz & dead_mode_prune_db reduce mode counts
    let eng_h = PianoEngine::new_with_profile(
        sr,
        VoicingSpec::default(),
        PerfProfile {
            horiz_max_hz: 2_800.0,
            ..PerfProfile::full()
        },
    );
    assert!(eng_h.total_modes() < base_modes);
    drop(eng_h);

    let eng_d = PianoEngine::new_with_profile(
        sr,
        VoicingSpec::default(),
        PerfProfile {
            dead_mode_prune_db: -50.0,
            ..PerfProfile::full()
        },
    );
    assert!(eng_d.total_modes() < base_modes);
    drop(eng_d);

    // energy_horizon_db truncates earlier
    let run_tail = |db: f32| -> Vec<f32> {
        let mut eng = PianoEngine::new_with_profile(
            sr,
            VoicingSpec::default(),
            PerfProfile {
                energy_horizon_db: db,
                ..PerfProfile::full()
            },
        );
        let _ = render_blocks(&mut eng, 512, 128, &[note_on(84, u7_to_u16(50), 0)]);
        render_blocks(&mut eng, 4096, 128, &[note_off(84, 0)]).0
    };
    assert!(max_abs_diff(&run_tail(-40.0)[2048..], &run_tail(-120.0)[2048..]) > 1e-6);

    // sav_dormancy_db & bass_phantom_sparse
    let c3_mf = [note_on(48, u7_to_u16(50), 0)];
    let (lc3_full, _) = render_with_profile(sr, PerfProfile::full(), 4096, &c3_mf);
    let (lc3_sd, _) = render_with_profile(
        sr,
        PerfProfile {
            sav_dormancy_db: -20.0,
            ..PerfProfile::full()
        },
        4096,
        &c3_mf,
    );
    assert!(max_abs_diff(&lc3_full, &lc3_sd) > 0.0);

    let bass_ev = [note_on(21, u16::MAX, 0)];
    let (la0_full, _) = render_with_profile(sr, PerfProfile::full(), 2048, &bass_ev);
    let (la0_bp, _) = render_with_profile(
        sr,
        PerfProfile {
            bass_phantom_sparse: true,
            ..PerfProfile::full()
        },
        2048,
        &bass_ev,
    );
    assert!(max_abs_diff(&la0_full, &la0_bp) > 1e-6);

    // sav_max_key_idx (disables SAV for key 39 >= 10, altering C4 ff waveform)
    let (lc4_sm10, _) = render_with_profile(
        sr,
        PerfProfile {
            sav_max_key_idx: 10,
            ..PerfProfile::full()
        },
        2048,
        &[note_on(60, u16::MAX, 0)],
    );
    let (lc4_sm52, _) =
        render_with_profile(sr, PerfProfile::full(), 2048, &[note_on(60, u16::MAX, 0)]);
    assert!(max_abs_diff(&lc4_sm10, &lc4_sm52) > 1e-6);

    // passive_res_max_hz limits passive open-string resonance
    let sus_c6 = [
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        },
        note_on(84, u16::MAX, 0),
    ];
    let (l_pr_full, _) = render_with_profile(sr, PerfProfile::full(), 2048, &sus_c6);
    let (l_pr_lim, _) = render_with_profile(
        sr,
        PerfProfile {
            passive_res_max_hz: 1_200.0,
            ..PerfProfile::full()
        },
        2048,
        &sus_c6,
    );
    assert!(max_abs_diff(&l_pr_full, &l_pr_lim) > 1e-7);

    // Stage 4 knobs
    let p_s4 = PerfProfile {
        ir_duration_secs: 0.125,
        spatial_tail_split_tap: 1_024,
        fir_head_taps: 16,
        fir_block_fft: true,
        hall_fdn_lines: 16,
        ..PerfProfile::full()
    };
    let mut eng_s4 = PianoEngine::new_with_profile(sr, VoicingSpec::default(), p_s4);
    assert_eq!(eng_s4.core.radiators[0].ir_len_samples(), 6_000);
    assert_eq!(eng_s4.core.radiators[0].head_taps(), 16);
    assert_eq!(eng_s4.core.radiators[0].algorithmic_latency_samples(), 32);
    assert_eq!(eng_s4.core.radiators[0].fdn_lines(), 16);
    assert_eq!(eng_s4.perf_profile().spatial_tail_split_tap, 1_024);
    let (l_s4, _) = render_blocks(&mut eng_s4, 2048, 128, &[note_on(60, u16::MAX, 0)]);
    assert!(l_s4[..32].iter().all(|&x| x == 0.0));
    assert!(l_s4[32..].iter().any(|&x| x.abs() > 1e-4));
}

#[test]
fn set_perf_profile_and_runtime_hot_swap_preserve_state_and_report_rebuild_status() {
    let mut eng = PianoEngine::new(48_000.0);
    eng.set_body_space(
        BodyArchetype::UprightStudio,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    );
    eng.set_reverb(2.1, 0.25);

    let rt_only = PerfProfile {
        energy_horizon_db: -90.0,
        sav_dormancy_db: -40.0,
        sav_max_key_idx: 30,
        passive_res_max_hz: 3_800.0,
        damper_control_decim: 4,
        hall_fdn_lines: 4,
        hertz_nonlinear: false,
        active_bridge_poles: 6,
        shank_flex_enabled: false,
        polynomial_adaa: true,
        ..PerfProfile::full()
    };
    assert!(eng.set_perf_profile_runtime(rt_only));
    assert_eq!(eng.perf_profile(), rt_only);
    assert_eq!(eng.core.radiators[0].fdn_lines(), 4);
    assert_eq!(eng.core.radiators[1].fdn_lines(), 4);
    assert_eq!(eng.core.bridge.active_poles(), 6);
    assert!(!eng.core.bridge.is_hertz_nonlinear());

    assert!(!eng.set_perf_profile_runtime(PerfProfile::realtime()));
    assert_eq!(eng.core.radiators[0].fdn_lines(), 4);
    assert_eq!(eng.core.bridge.active_poles(), 4);

    eng.set_perf_profile(PerfProfile::balanced());
    assert_eq!(eng.perf_profile(), PerfProfile::balanced());
    assert_eq!(eng.body_archetype(), BodyArchetype::UprightStudio);
    assert_eq!(
        eng.space_guard().current_perspective(),
        ListenerPerspective::ConcertHall
    );
    assert_eq!(eng.space_guard().current_lid(), LidState::Half);
    assert_eq!(eng.reverb_params(), (2.1, 0.25));
    assert_eq!(eng.core.radiators[0].ir_len_samples(), 12_000);
    assert_eq!(eng.core.radiators[0].head_taps(), 32);

    let (l, r) = render_blocks(&mut eng, 1024, 128, &[note_on(60, 50_000, 0)]);
    assert!(l.iter().chain(r.iter()).all(|x| x.is_finite()));
    assert!(l.iter().any(|&x| x.abs() > 1e-4));
    drop(eng);

    let mut poly_eng =
        PianoEngine::new_with_profile(48_000.0, VoicingSpec::default(), PerfProfile::realtime());
    assert!(!poly_eng.set_perf_profile_runtime(PerfProfile {
        polynomial_adaa: false,
        ..PerfProfile::realtime()
    }));
    assert!(poly_eng.perf_profile().polynomial_adaa);
}

#[test]
fn p1_4_flush_decimation_and_space_blob_v2_work_cleanly() {
    let mut eng64 = PianoEngine::new(48_000.0);
    let (l64, r64) = render_blocks(&mut eng64, 512, 64, &[note_on(60, 50_000, 0)]);
    assert_eq!(eng64.core.flush_sample_accum, 0);
    let mut tmp_l = [0.0_f32; 64];
    let mut tmp_r = [0.0_f32; 64];
    eng64.render_block(&[], &mut tmp_l, &mut tmp_r);
    assert_eq!(eng64.core.flush_sample_accum, 64);
    eng64.reset();
    assert_eq!(eng64.core.flush_sample_accum, 0);
    assert_eq!(eng64.active_voices(), 0);
    drop(eng64);

    let (l128, r128) = render_with_profile(
        48_000.0,
        PerfProfile::full(),
        512,
        &[note_on(60, 50_000, 0)],
    );
    assert!(max_abs_diff(&l64, &l128) < 1e-4);
    assert!(max_abs_diff(&r64, &r128) < 1e-4);

    let mut prep = SpacePrepContext::new_with_profile(48_000.0, &PerfProfile::balanced());
    let bal_blob = prep.prepare_space(ListenerPerspective::Player, LidState::Open);
    assert_eq!(bal_blob[0].to_bits(), SPACE_BLOB_MAGIC_V2);
    assert_eq!(
        bal_blob.len(),
        space_blob_len_for_profile(48_000.0, &PerfProfile::balanced())
    );

    let mut eng = PianoEngine::new(48_000.0);
    assert!(eng.apply_precomputed_space(ListenerPerspective::Player, LidState::Open, bal_blob));
    assert_eq!(eng.core.radiators[0].ir_len_samples(), 12_000);
    assert_eq!(eng.perf_profile().ir_duration_secs, 0.25);
    assert!(eng.apply_precomputed_space(ListenerPerspective::Player, LidState::Open, bal_blob));
}
