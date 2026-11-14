//! Milestone 4 closed-loop nonlinear string SAV, pitch drift, longitudinal precursor,
//! and Bank-Sujbert phantom partial acoustic gate verification suite.

use core::f64::consts::TAU;

use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, EventKind, SampleEvent,
    VoicingSpec, NUM_KEYS,
};
use piano_strings::StringBank;

use crate::m3_tests::{mono_mix, render_chunks, slice_rms};
use crate::PianoEngine;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

fn note_on(midi_key: u8, velocity_u16: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::NoteOn {
            midi_key,
            velocity_u16,
        },
    }
}

fn note_off(midi_key: u8, release_vel_u16: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::NoteOff {
            midi_key,
            release_vel_u16,
        },
    }
}

fn blackman_harris_windowed(samples: &[f32]) -> Vec<f64> {
    let n = samples.len();
    let denom = (n - 1).max(1) as f64;
    samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = TAU * (i as f64) / denom;
            let w =
                0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
            f64::from(x) * w
        })
        .collect()
}

fn goertzel_power_f64(win: &[f64], freq_hz: f64, sr: f64) -> f64 {
    let coeff = 2.0 * (TAU * freq_hz / sr).cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in win {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(1.0e-30)
}

fn measure_f0_hz(mono: &[f32], f_nominal_hz: f64, sr: f64) -> f64 {
    let win = blackman_harris_windowed(mono);
    let mut best_f = f_nominal_hz;
    let mut best_p = 0.0_f64;
    for step in -200..=200 {
        let f_try = f_nominal_hz * 2.0_f64.powf((step as f64 * 0.5) / 1200.0);
        let p = goertzel_power_f64(&win, f_try, sr);
        if p > best_p {
            best_p = p;
            best_f = f_try;
        }
    }
    let dc = 0.5_f64;
    let r = 2.0_f64.powf(dc / 1200.0);
    let p_lo = goertzel_power_f64(&win, best_f / r, sr);
    let p_mid = goertzel_power_f64(&win, best_f, sr);
    let p_hi = goertzel_power_f64(&win, best_f * r, sr);
    let denom_q = p_lo - 2.0 * p_mid + p_hi;
    let shift = if denom_q < 0.0 {
        (0.5 * (p_lo - p_hi) / denom_q).clamp(-1.0, 1.0) * dc
    } else {
        0.0
    };
    best_f * 2.0_f64.powf(shift / 1200.0)
}

#[test]
fn m4_gate1a_lossless_sav_conservation() {
    let voicing = VoicingSpec::default();
    let mut bank = StringBank::new_lossless(SR, &voicing);
    bank.set_nonlinear_coupling(true, true);

    let keys_and_forces = [
        (0_usize, 24.0_f32),
        (15_usize, 28.0_f32),
        (27_usize, 48.0_f32),
    ];
    let mut h0_keys = [0.0_f64; 3];
    for (idx, &(key, force)) in keys_and_forces.iter().enumerate() {
        bank.apply_excitation_sample(key, [1.0, 0.0, 0.0], force);
        h0_keys[idx] = bank.key_sav_energy_joules(key);
        assert!(h0_keys[idx] > 0.0, "key={key}: expected H0 > 0");
    }
    let h0_total = bank.total_sav_energy_joules();
    assert!(h0_total > 0.0);

    let mut max_rel_err_keys = [0.0_f64; 3];
    let mut max_enl_ratio_keys = [0.0_f64; 3];
    let mut max_rel_err_total = 0.0_f64;
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let blocks = 188_usize; // 188 * 128 = 24,064 frames >= 24,000 (0.5 s @ 48 kHz)
    for _ in 0..blocks {
        bank.render_subblock(&mut l, &mut r);
        for (&sl, &sr) in l.iter().zip(r.iter()) {
            assert!(sl.is_finite() && sr.is_finite());
        }
        let h_tot = bank.total_sav_energy_joules();
        max_rel_err_total = max_rel_err_total.max((h_tot - h0_total).abs() / h0_total);
        for (idx, &(key, _)) in keys_and_forces.iter().enumerate() {
            let hk = bank.key_sav_energy_joules(key);
            let enl = bank.key_nonlinear_energy_joules(key);
            max_rel_err_keys[idx] =
                max_rel_err_keys[idx].max((hk - h0_keys[idx]).abs() / h0_keys[idx]);
            max_enl_ratio_keys[idx] = max_enl_ratio_keys[idx].max(enl / h0_keys[idx]);
        }
    }

    eprintln!(
        "Gate1A lossless 24064f: total_err={max_rel_err_total:e}, A0=({:e}, E_nl/H0={:e}), C2=({:e}, E_nl/H0={:e}), C3=({:e}, E_nl/H0={:e})",
        max_rel_err_keys[0],
        max_enl_ratio_keys[0],
        max_rel_err_keys[1],
        max_enl_ratio_keys[1],
        max_rel_err_keys[2],
        max_enl_ratio_keys[2],
    );
    assert!(
        max_rel_err_total < 1.0e-5,
        "total SAV conservation failed: {max_rel_err_total:e}"
    );
    for (idx, &(key, _)) in keys_and_forces.iter().enumerate() {
        assert!(
            max_enl_ratio_keys[idx] > 1.0e-3,
            "key={key} nonlinear regime too weak: {:e}",
            max_enl_ratio_keys[idx]
        );
        assert!(
            max_rel_err_keys[idx] < 1.0e-5,
            "key={key} SAV drift exceeded 1e-5: {:e}",
            max_rel_err_keys[idx]
        );
    }
}

#[test]
fn m4_gate1b_closed_loop_damped_passivity() {
    let mut engine = PianoEngine::new(SR);
    let chord_on = [
        note_on(21, 60_000, 0),
        note_on(36, 60_000, 0),
        note_on(48, 60_000, 0),
    ];
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    let mut contact_frames = 0_usize;
    for b in 0..24 {
        let evs = if b == 0 { &chord_on[..] } else { &[] };
        engine.render_block(evs, &mut l, &mut r);
        for (&sl, &sr) in l.iter().zip(r.iter()) {
            assert!(sl.is_finite() && sr.is_finite());
        }
        contact_frames += 128;
        if contact_frames >= 512 && !engine.is_any_hammer_active() {
            break;
        }
    }
    assert!(
        !engine.is_any_hammer_active(),
        "hammer still active after {contact_frames} frames"
    );

    let h_post_contact = engine.total_sav_energy_joules();
    assert!(h_post_contact > 0.0);
    let mut h_prev = h_post_contact;
    let mut max_increase_free = 0.0_f64;
    for b in 0..32 {
        engine.render_block(&[], &mut l, &mut r);
        for (&sl, &sr) in l.iter().zip(r.iter()) {
            assert!(sl.is_finite() && sr.is_finite());
        }
        let h_cur = engine.total_sav_energy_joules();
        let diff = h_cur - h_prev;
        max_increase_free = max_increase_free.max(diff);
        assert!(
            h_cur <= h_prev + 1.0e-12,
            "free block {b} passivity violation: h_prev={h_prev:e} h_cur={h_cur:e} diff={diff:e}"
        );
        h_prev = h_cur;
    }
    let h_after_free = h_prev;
    assert!(h_after_free < h_post_contact);

    let chord_off = [
        note_off(21, 48_000, 0),
        note_off(36, 48_000, 0),
        note_off(48, 48_000, 0),
    ];
    let mut max_increase_damp = 0.0_f64;
    for b in 0..48 {
        let evs = if b == 0 { &chord_off[..] } else { &[] };
        engine.render_block(evs, &mut l, &mut r);
        for (&sl, &sr) in l.iter().zip(r.iter()) {
            assert!(sl.is_finite() && sr.is_finite());
        }
        let h_cur = engine.total_sav_energy_joules();
        let diff = h_cur - h_prev;
        max_increase_damp = max_increase_damp.max(diff);
        assert!(
            h_cur <= h_prev + 1.0e-12,
            "damped block {b} passivity violation: h_prev={h_prev:e} h_cur={h_cur:e} diff={diff:e}"
        );
        h_prev = h_cur;
    }
    eprintln!(
        "Gate1B passivity: H_post={h_post_contact:.6e} -> H_free={h_after_free:.6e} -> H_damp={h_prev:.6e}, max_inc_free={max_increase_free:e}, max_inc_damp={max_increase_damp:e}"
    );
    assert!(h_prev < h_after_free * 0.8);
}

#[test]
fn m4_gate2a_c2_ff_pitch_glide_chirp() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let key = midi_to_key(36).unwrap();
    let f_nom = partial_frequency_hz(
        key_tuned_f0_hz(key, &specs, &voicing),
        specs[key].inharmonicity_b,
        1,
    );

    let mut engine_ff = PianoEngine::new(SR);
    let (l_ff, r_ff) = render_chunks(&mut engine_ff, 49_152, 128, &[note_on(36, 60_000, 0)]);
    let mono_ff = mono_mix(&l_ff, &r_ff);
    let f_ff_early = measure_f0_hz(&mono_ff[768..17_152], f_nom, SR64);
    let f_ff_late = measure_f0_hz(&mono_ff[32_768..49_152], f_nom, SR64);

    let mut engine_pp = PianoEngine::new(SR);
    let (l_pp, r_pp) = render_chunks(&mut engine_pp, 17_152, 128, &[note_on(36, 6_000, 0)]);
    let mono_pp = mono_mix(&l_pp, &r_pp);
    let f_pp_early = measure_f0_hz(&mono_pp[768..17_152], f_nom, SR64);

    let glide_cents = 1200.0 * (f_ff_early / f_ff_late).log2();
    let dyn_glide_cents = 1200.0 * (f_ff_early / f_pp_early).log2();
    eprintln!(
        "Gate2A C2 chirp: f_nom={f_nom:.4}, ff_early={f_ff_early:.4}, ff_late={f_ff_late:.4}, pp_early={f_pp_early:.4}, glide={glide_cents:+.3}c, ff_vs_pp={dyn_glide_cents:+.3}c"
    );
    assert!(
        f_ff_early > f_ff_late && glide_cents > 0.25,
        "expected positive C2 ff pitch glide (early={f_ff_early:.4}, late={f_ff_late:.4}, {glide_cents:+.3}c)"
    );
    assert!(
        f_ff_early > f_pp_early && dyn_glide_cents > 0.25,
        "expected ff initial pitch > pp initial pitch ({f_ff_early:.4} vs {f_pp_early:.4}, {dyn_glide_cents:+.3}c)"
    );
}

fn verify_hundred_strike_drift(midi_key: u8) {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let key = midi_to_key(midi_key).unwrap();
    let f_nom = partial_frequency_hz(
        key_tuned_f0_hz(key, &specs, &voicing),
        specs[key].inharmonicity_b,
        1,
    );

    let mut engine = PianoEngine::new(SR);
    let (l1, r1) = render_chunks(&mut engine, 36_000, 128, &[note_on(midi_key, 60_000, 0)]);
    let mono1 = mono_mix(&l1, &r1);
    assert!(mono1.iter().all(|x| x.is_finite()));
    let f_first = measure_f0_hz(&mono1[512..], f_nom, SR64);

    // Compact NoteOff + 99 ff strike/release cycles, then damped settling before clean strike 101.
    let _ = render_chunks(&mut engine, 512, 128, &[note_off(midi_key, 65_535, 0)]);
    let (mut l_blk, mut r_blk) = ([0.0_f32; 384], [0.0_f32; 384]);
    let cycle_evs = [
        note_on(midi_key, 60_000, 0),
        note_off(midi_key, 65_535, 256),
    ];
    for _ in 1..100 {
        engine.render_block(&cycle_evs, &mut l_blk, &mut r_blk);
        assert!(l_blk.iter().chain(r_blk.iter()).all(|x| x.is_finite()));
    }
    let e_before_settle = engine.key_sav_energy_joules(key);
    assert!(
        e_before_settle > 1.0e-3,
        "midi={midi_key}: expected sustained ff energy before settle, got {e_before_settle:e}"
    );
    let _ = render_chunks(&mut engine, 36_000, 128, &[]);
    let e_after_settle = engine.key_sav_energy_joules(key);
    let enl_after_settle = engine.key_nonlinear_energy_joules(key);
    assert!(
        e_after_settle < e_before_settle * 1.0e-4,
        "midi={midi_key}: insufficient damped settling before #101 ({e_after_settle:e} vs {e_before_settle:e})"
    );

    let (l101, r101) = render_chunks(&mut engine, 36_000, 128, &[note_on(midi_key, 60_000, 0)]);
    let mono101 = mono_mix(&l101, &r101);
    assert!(mono101.iter().all(|x| x.is_finite()));
    let f_101 = measure_f0_hz(&mono101[512..], f_nom, SR64);
    let drift_cents = (1200.0 * (f_101 / f_first).log2()).abs();
    eprintln!(
        "Gate2B midi={midi_key} key={key}: f_nom={f_nom:.4}, f1={f_first:.4}, f101={f_101:.4}, drift={drift_cents:.4}c, E_pre={e_before_settle:e}, E_post={e_after_settle:e}, Enl_post={enl_after_settle:e}"
    );
    assert!(
        drift_cents < 0.5,
        "midi={midi_key}: 100-strike ff pitch drift {drift_cents:.4}c >= 0.5c (f1={f_first:.4}, f101={f_101:.4})"
    );
}

#[test]
fn m4_gate2b_100_strike_drift_a0() {
    verify_hundred_strike_drift(21);
}

#[test]
fn m4_gate2b_100_strike_drift_e1() {
    verify_hundred_strike_drift(28);
}

#[test]
fn m4_gate2b_100_strike_drift_c2() {
    verify_hundred_strike_drift(36);
}

#[test]
fn m4_gate2b_100_strike_drift_c3() {
    verify_hundred_strike_drift(48);
}

#[test]
fn m4_gate3a_longitudinal_precursor_a0_c1() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);

    for &midi_key in &[21_u8, 24_u8] {
        let key = midi_to_key(midi_key).unwrap();
        let x = (key as f64) / ((NUM_KEYS - 1) as f64);
        let beta_k = 0.137 - 0.042 * x;
        let tau_l_s = (1.0 - beta_k) / (2.0 * specs[key].f_long_hz);
        let tau_t_s = (1.0 - beta_k) / (2.0 * specs[key].f0_et_hz);

        let run = |long_on: bool| -> Vec<f32> {
            let mut engine = PianoEngine::new(SR);
            engine.set_nonlinear_coupling(long_on, true);
            let (l, r) = render_chunks(&mut engine, 1_200, 128, &[note_on(midi_key, 60_000, 0)]);
            mono_mix(&l, &r)
        };
        let mono_on = run(true);
        let mono_off = run(false);

        let pre_end = ((0.60 * tau_l_s * SR64).floor() as usize).clamp(4, 36);
        let win_lo = (0.0008 * SR64).round() as usize; // 0.8 ms
        let win_hi = (0.0055 * SR64).round() as usize; // 5.5 ms
        let main_lo = (0.0110 * SR64).round() as usize;

        let rms_pre = slice_rms(&mono_on[..pre_end]);
        let rms_early_on = slice_rms(&mono_on[win_lo..win_hi]);
        let rms_early_off = slice_rms(&mono_off[win_lo..win_hi]);
        let rms_main_on = slice_rms(&mono_on[main_lo..]);

        eprintln!(
            "Gate3A midi={midi_key} tau_L={:.2}ms tau_T={:.2}ms: pre={rms_pre:.5}, early_on={rms_early_on:.5}, early_off={rms_early_off:.5}, main={rms_main_on:.5}",
            tau_l_s * 1e3,
            tau_t_s * 1e3
        );
        assert!(rms_main_on > rms_early_on);
        assert!(
            rms_early_on > rms_pre * 4.0 && rms_early_on > rms_early_off * 4.0 && rms_early_on > 1.0e-4,
            "midi={midi_key}: weak longitudinal precursor early_on={rms_early_on:e} pre={rms_pre:e} early_off={rms_early_off:e}"
        );
    }
}

fn verify_phantom_partials(midi_key: u8) {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let key = midi_to_key(midi_key).unwrap();
    let f0 = key_tuned_f0_hz(key, &specs, &voicing);
    let b = specs[key].inharmonicity_b;
    let f1 = partial_frequency_hz(f0, b, 1);
    let f5 = partial_frequency_hz(f0, b, 5);
    let f6 = partial_frequency_hz(f0, b, 6);
    let targets = [("2*f5", 2.0 * f5), ("2*f6", 2.0 * f6), ("f5+f6", f5 + f6)];

    const N: usize = 32_768;
    let measure = |vel: u16| -> (f64, Vec<f64>) {
        let mut engine = PianoEngine::new(SR);
        let (l, r) = render_chunks(&mut engine, N + 512, 128, &[note_on(midi_key, vel, 0)]);
        let mono = mono_mix(&l[512..], &r[512..]);
        let win = blackman_harris_windowed(&mono);
        let p_f1 = goertzel_power_f64(&win, f1, SR64);
        let ratios_db = targets
            .iter()
            .map(|&(_, freq)| 10.0 * (goertzel_power_f64(&win, freq, SR64) / p_f1).log10())
            .collect();
        (p_f1, ratios_db)
    };

    let (p1_ff, db_ff) = measure(60_000);
    let (p1_pp, db_pp) = measure(5_000);
    assert!(p1_ff > p1_pp);
    for ((label, freq), (&r_ff, &r_pp)) in targets.iter().zip(db_ff.iter().zip(db_pp.iter())) {
        let delta_db = r_ff - r_pp;
        eprintln!(
            "Gate3B midi={midi_key} {label} ({freq:.2} Hz): ff_ratio={r_ff:.1} dB, pp_ratio={r_pp:.1} dB, gain={delta_db:+.1} dB"
        );
        assert!(
            delta_db >= 8.0,
            "midi={midi_key} {label}: normalized phantom ratio gain {delta_db:+.2} dB < +8.0 dB"
        );
    }
}

#[test]
fn m4_gate3b_phantom_partials_c2() {
    verify_phantom_partials(36);
}

#[test]
fn m4_gate3b_phantom_partials_g1() {
    verify_phantom_partials(31);
}
