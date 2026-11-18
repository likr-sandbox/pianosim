use piano_action::oversample::HammerOversampleState;
use piano_action::pedals::PedalBank;
use piano_spec::{
    build_scale_design, effective_string_f0_hz, key_tuned_f0_hz, midi_to_key, partial_frequency_hz,
    u7_to_u16, EventKind, PedalKind, Polarization, SampleEvent, VoicingSpec, DAMPED_KEYS,
};

use crate::m3_tests::{band_power, hann_windowed_spectrum, mono_mix, slice_rms};
use crate::m4_tests::{note_off, note_on};
use crate::PianoEngine;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

fn pedal_ev(kind: PedalKind, value_u16: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::Pedal { kind, value_u16 },
    }
}

fn render_stage4_chunks(
    engine: &mut PianoEngine,
    total_samples: usize,
    chunk: usize,
    events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_samples];
    let mut out_r = vec![0.0_f32; total_samples];
    let mut offset = 0;
    let mut pending: Vec<SampleEvent> = events.to_vec();
    while offset < total_samples {
        let len = (total_samples - offset).min(chunk);
        let len_u32 = len as u32;
        let mut block_evs = Vec::new();
        let mut rest = Vec::new();
        for mut ev in pending {
            if ev.sample_offset < len_u32 {
                block_evs.push(ev);
            } else {
                ev.sample_offset -= len_u32;
                rest.push(ev);
            }
        }
        pending = rest;
        engine.render_block(
            &block_evs,
            &mut out_l[offset..offset + len],
            &mut out_r[offset..offset + len],
        );
        offset += len;
    }
    (out_l, out_r)
}

fn render_bridge_chunks(
    engine: &mut PianoEngine,
    total_samples: usize,
    chunk: usize,
    events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_samples];
    let mut out_r = vec![0.0_f32; total_samples];
    let mut offset = 0;
    let mut pending: Vec<SampleEvent> = events.to_vec();
    while offset < total_samples {
        let len = (total_samples - offset).min(chunk);
        let len_u32 = len as u32;
        let mut block_evs = Vec::new();
        let mut rest = Vec::new();
        for mut ev in pending {
            if ev.sample_offset < len_u32 {
                block_evs.push(ev);
            } else {
                ev.sample_offset -= len_u32;
                rest.push(ev);
            }
        }
        pending = rest;
        engine.render_block_bridge(
            &block_evs,
            &mut out_l[offset..offset + len],
            &mut out_r[offset..offset + len],
        );
        offset += len;
    }
    (out_l, out_r)
}

fn peak_hz(samples: &[f32], lo_hz: f64, hi_hz: f64) -> f64 {
    let spec = hann_windowed_spectrum(samples);
    let n_fft = samples.len();
    let bin_hz = SR64 / (n_fft as f64);
    let lo_bin = ((lo_hz / bin_hz).floor() as usize).max(1);
    let hi_bin = ((hi_hz / bin_hz).ceil() as usize).min(spec.len() - 1);
    let mut best_k = lo_bin;
    let mut best_p = -1.0_f64;
    for (idx, c) in spec[lo_bin..=hi_bin].iter().enumerate() {
        let p = f64::from(c.norm_sq());
        if p > best_p {
            best_p = p;
            best_k = lo_bin + idx;
        }
    }
    (best_k as f64) * bin_hz
}

#[test]
fn wave2_ii1_bass_stage4_balance_and_fff_polyphony_headroom() {
    let mut engine = PianoEngine::new(SR);
    let mf = u7_to_u16(64);
    let s50 = (0.05 * SR64).round() as usize;
    let s300 = (0.30 * SR64).round() as usize;
    let total = (0.35 * SR64).round() as usize;

    let mut level_db = |midi: u8| -> f64 {
        engine.reset();
        let (l, r) = render_stage4_chunks(&mut engine, total, 128, &[note_on(midi, mf, 0)]);
        let mono = mono_mix(&l, &r);
        20.0 * slice_rms(&mono[s50..s300]).max(1e-12).log10()
    };

    let c4_db = level_db(60);
    for (name, midi) in [("A0", 21_u8), ("C1", 24), ("C2", 36)] {
        let db = level_db(midi);
        assert!(
            db >= -36.0,
            "{name} Stage 4 mf sustained RMS {db:+.2} dBFS < -36 dBFS"
        );
        assert!(
            (db - c4_db).abs() <= 10.0,
            "{name} ({db:+.2} dBFS) vs C4 ({c4_db:+.2} dBFS) spread > 10 dB"
        );
    }

    engine.reset();
    let chord: Vec<SampleEvent> = [21_u8, 28, 36, 43, 48, 55, 60, 64, 67, 72, 79, 84]
        .iter()
        .map(|&m| note_on(m, u16::MAX, 0))
        .collect();
    let (l, r) = render_stage4_chunks(&mut engine, (0.40 * SR64).round() as usize, 128, &chord);
    let peak = l
        .iter()
        .chain(r.iter())
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert!(peak.is_finite() && peak <= 1.0, "12-note fff peak={peak}");
}

#[test]
fn wave2_ii2_b7_c8_stage3_wdf_coupling_and_c8_unison_beating() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let k_c8 = midi_to_key(108).expect("C8");
    let f0_c8 = key_tuned_f0_hz(k_c8, &specs, &voicing);
    let expected_c8_p1 = partial_frequency_hz(f0_c8, specs[k_c8].inharmonicity_b, 1);

    let f_s0 = effective_string_f0_hz(k_c8, 0, Polarization::Vertical, &specs, &voicing);
    let f_s1 = effective_string_f0_hz(k_c8, 1, Polarization::Vertical, &specs, &voicing);
    let f_s2 = effective_string_f0_hz(k_c8, 2, Polarization::Vertical, &specs, &voicing);
    let beat_01 = (f_s0 - f_s1).abs();
    let beat_12 = (f_s1 - f_s2).abs();
    assert!(
        beat_01 < 0.9 && beat_12 < 0.9,
        "C8 adjacent unison beating {beat_01:.4}/{beat_12:.4} Hz >= 0.9 Hz"
    );

    let mut engine = PianoEngine::new(SR);
    let total = 24_576;
    for (label, midi, lo_hz, hi_hz, ref_hz) in [
        ("B7", 107_u8, 3_600.0, 4_600.0, 4_115.0),
        ("C8", 108_u8, 3_900.0, 5_000.0, expected_c8_p1),
    ] {
        engine.reset();
        let (l, r) = render_bridge_chunks(&mut engine, total, 128, &[note_on(midi, 48_000, 0)]);
        let mono = mono_mix(&l, &r);
        let rms = slice_rms(&mono[..8_192]);
        assert!(
            rms > 5.0e-4,
            "{label} Stage 3 WDF coupled RMS too low: {rms}"
        );
        let pk_hz = peak_hz(&mono[1_024..17_408], lo_hz, hi_hz);
        let cents = 1_200.0 * (pk_hz / ref_hz).log2().abs();
        assert!(
            cents <= 35.0,
            "{label} Stage 3 peak {pk_hz:.1} Hz drifted {cents:.1} cents from {ref_hz:.1} Hz"
        );
        if midi == 108 {
            // < 0.9 Hz beating implies smooth decay over early 30..250 ms without rapid beat nulling.
            let win_a = slice_rms(&mono[1_440..3_840]);
            let win_b = slice_rms(&mono[3_840..6_240]);
            let win_c = slice_rms(&mono[6_240..8_640]);
            assert!(win_a > win_b && win_b > win_c && win_c > 1e-4);
        }
    }
}

#[test]
fn wave2_ii3_three_mechanism_t60_decay_and_register_ordering() {
    let mut engine = PianoEngine::new(SR);
    let total = (2.0 * SR64).round() as usize;
    let s100 = (0.10 * SR64).round() as usize;
    let s300 = (0.30 * SR64).round() as usize;
    let s1000 = (1.00 * SR64).round() as usize;
    let s1500 = (1.50 * SR64).round() as usize;

    let mut measure_ratio = |midi: u8| -> f64 {
        engine.reset();
        let (l, r) = render_bridge_chunks(&mut engine, total, 128, &[note_on(midi, 33_025, 0)]);
        let mono = mono_mix(&l, &r);
        let early = slice_rms(&mono[s100..s300]);
        let late = slice_rms(&mono[s1000..s1500]);
        20.0 * (late / early.max(1e-12)).log10()
    };

    let a0_drop_db = measure_ratio(21);
    let c4_drop_db = measure_ratio(60);
    let c8_drop_db = measure_ratio(108);
    assert!(
        a0_drop_db > -15.0 && c8_drop_db < c4_drop_db && c8_drop_db < -18.0,
        "register decay mismatch: A0={a0_drop_db:+.2} dB, C4={c4_drop_db:+.2} dB, C8={c8_drop_db:+.2} dB"
    );

    // High partials in C4 must decay substantially faster than C4 fundamental (thermoelastic + air visc).
    engine.reset();
    let (l4, r4) = render_bridge_chunks(&mut engine, total, 128, &[note_on(60, 56_000, 0)]);
    let mono4 = mono_mix(&l4, &r4);
    let s50 = (0.05 * SR64).round() as usize;
    let s500 = (0.50 * SR64).round() as usize;
    let early_spec = hann_windowed_spectrum(&mono4[s50..s50 + 8_192]);
    let late_spec = hann_windowed_spectrum(&mono4[s500..s500 + 8_192]);
    let f0_drop = band_power(&late_spec, 8_192, SR64, 240.0, 290.0)
        / band_power(&early_spec, 8_192, SR64, 240.0, 290.0).max(1e-24);
    let hf_drop = band_power(&late_spec, 8_192, SR64, 4_000.0, 10_000.0)
        / band_power(&early_spec, 8_192, SR64, 4_000.0, 10_000.0).max(1e-24);
    assert!(
        hf_drop < f0_drop * 0.75,
        "HF partials did not roll off faster than f0: hf_drop={hf_drop:.3e}, f0_drop={f0_drop:.3e}"
    );
}

#[test]
fn wave2_ii4_duplex_tuning_link_and_subnormal_quiescent_flush() {
    let len = (0.20 * SR64).round() as usize;

    let mut eng_on = PianoEngine::new(SR);
    let (l_on, r_on) =
        render_stage4_chunks(&mut eng_on, len, 128, &[note_on(84, u7_to_u16(100), 0)]);

    let mut eng_off = PianoEngine::new(SR);
    eng_off.set_duplex_gain(0.0);
    let (l_off, r_off) =
        render_stage4_chunks(&mut eng_off, len, 128, &[note_on(84, u7_to_u16(100), 0)]);

    let m_on = mono_mix(&l_on, &r_on);
    let m_off = mono_mix(&l_off, &r_off);
    let dup_diff_rms = (m_on
        .iter()
        .zip(m_off.iter())
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / (len as f64))
        .sqrt();
    assert!(
        dup_diff_rms > 2.0e-5,
        "C6 Stage 4 duplex diff_rms too low: {dup_diff_rms:.6e}"
    );

    // Retuning A4 -> 442 Hz updates DuplexBank pitch parity with a freshly constructed engine.
    let v442 = VoicingSpec::from_globals_with_a4(0.5, 1.0, 1.0, 1.0, 442.0);
    let mut eng_retuned = PianoEngine::new(SR);
    eng_retuned.set_a4_freq_hz(442.0);
    let settle = (0.25 * SR64).round() as usize;
    let _ = render_stage4_chunks(&mut eng_retuned, settle, 128, &[]);
    let (lr, rr) = render_stage4_chunks(
        &mut eng_retuned,
        len,
        128,
        &[note_on(84, u7_to_u16(100), 0)],
    );

    let mut eng_fresh = PianoEngine::new_with_voicing(SR, &v442);
    let (lf, rf) =
        render_stage4_chunks(&mut eng_fresh, len, 128, &[note_on(84, u7_to_u16(100), 0)]);
    let err = lr
        .iter()
        .chain(rr.iter())
        .zip(lf.iter().chain(rf.iter()))
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    assert!(
        err < 1.0e-3,
        "retuned A4=442 duplex output diverged from fresh engine: max_err={err}"
    );

    // Stage 3 < 1e-20 subnormal flush restores BridgeBank::is_quiescent() idle bypass after decay.
    let mut eng_q = PianoEngine::new(SR);
    assert!(eng_q.is_bridge_quiescent());
    let mut bl = [0.0_f32; 128];
    let mut br = [0.0_f32; 128];
    eng_q.excite_bridge_subnormal_for_test();
    assert!(!eng_q.is_bridge_quiescent());
    for _ in 0..400 {
        eng_q.render_block_bridge(&[], &mut bl, &mut br);
        if eng_q.is_bridge_quiescent() {
            break;
        }
    }
    assert!(
        eng_q.is_bridge_quiescent(),
        "BridgeBank failed to return to quiescent idle bypass after subnormal flush"
    );
}

#[test]
fn wave2_ii5_peel_off_keybed_release_and_oversample_launch_continuity() {
    // 1. Untouched virgin keys emit zero on Sustain pump; played damped key emits 0.5..2.0 mN peel pulse.
    let mut eng = PianoEngine::new(SR);
    let (vl, vr) = render_stage4_chunks(
        &mut eng,
        2_048,
        128,
        &[pedal_ev(PedalKind::Sustain, u16::MAX, 0)],
    );
    let virgin_peak = vl
        .iter()
        .chain(vr.iter())
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert_eq!(virgin_peak, 0.0, "virgin sustain lift emitted audio");

    let mut pedals = PedalBank::new(SR);
    let mut clears = [0.0_f32; 8];
    let mut peels = [0.0_f32; 8];
    pedals.set_pedal(PedalKind::Sustain, u16::MAX);
    for k in 0..DAMPED_KEYS {
        for _ in 0..400 {
            let (_, any_p) =
                pedals.step_key_clearance_tile_with_peel(k, 8, &mut clears, &mut peels);
            assert!(!any_p, "virgin key {k} emitted spurious peel pulse");
        }
    }
    pedals.set_pedal(PedalKind::Sustain, 0);
    for _ in 0..600 {
        let _ = pedals.step_key_clearance_tile(39, 8, &mut clears);
    }
    pedals.note_on(39);
    pedals.note_off(39, 40_000);
    for _ in 0..800 {
        let _ = pedals.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
    }
    pedals.set_pedal(PedalKind::Sustain, u16::MAX);
    let mut peel_peak_n = 0.0_f32;
    for _ in 0..400 {
        let _ = pedals.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
        for &p in &peels {
            peel_peak_n = peel_peak_n.max(p);
        }
    }
    assert!(
        (5.0e-4..=2.0e-3).contains(&peel_peak_n),
        "C4 peel peak {peel_peak_n} N outside [0.5, 2.0] mN"
    );

    // End-to-end Stage 4 audible peel response on a played & released damped key vs virgin 0.0.
    eng.reset();
    let mut settle_l = [0.0_f32; 128];
    let mut settle_r = [0.0_f32; 128];
    eng.render_block(&[note_on(60, 40_000, 0)], &mut settle_l, &mut settle_r);
    eng.render_block(&[note_off(60, 32_768, 0)], &mut settle_l, &mut settle_r);
    for _ in 0..600 {
        eng.render_block(&[], &mut settle_l, &mut settle_r);
    }
    eng.render_block(
        &[pedal_ev(PedalKind::Sustain, u16::MAX, 0)],
        &mut settle_l,
        &mut settle_r,
    );
    let mut audio_peak = 0.0_f32;
    for _ in 0..64 {
        eng.render_block(&[], &mut settle_l, &mut settle_r);
        audio_peak = audio_peak.max(
            settle_l
                .iter()
                .chain(settle_r.iter())
                .fold(0.0_f32, |m, &x| m.max(x.abs())),
        );
    }
    assert!(audio_peak > 1.0e-5, "C4 peel pulse inaudible: {audio_peak}");

    // 2. NoteOff Zone 8 release shock scales monotonically with release velocity.
    let release_peak = |vel: u16| -> f32 {
        let mut e = PianoEngine::new(SR);
        let (l, r) = render_stage4_chunks(&mut e, 1_536, 128, &[note_off(60, vel, 0)]);
        l.iter()
            .chain(r.iter())
            .fold(0.0_f32, |m, &x| m.max(x.abs()))
    };
    let pk_soft = release_peak(u7_to_u16(32));
    let pk_fast = release_peak(u7_to_u16(127));
    assert!(
        pk_soft > 1.0e-6 && pk_fast > pk_soft * 1.5,
        "NoteOff release shock scaling invalid: soft={pk_soft}, fast={pk_fast}"
    );

    // 3. HammerOversampleState::launch preserves prev_sub_forces FIR history across rapid re-strike.
    let mut st = HammerOversampleState {
        prev_sub_forces: [0.25; 8],
        ..Default::default()
    };
    st.launch(2.0);
    assert_eq!(st.prev_sub_forces, [0.25; 8]);
    assert!(st.is_active());
}
