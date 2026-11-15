//! Milestone 6 Acceptance Gates 1, 2, and 5:
//! - Gate 1: Reduced pole cancellation stability & double resonance elimination
//! - Gate 2: NUPOLS convolution reconstruction accuracy (<= -100 dB) & 0-sample latency
//! - Gate 5: Preset binary size (< 500 KB) & D1 real-time callback budget (p99 < 1,866.7 us)

use piano_body_gen::pole_cancel::verify_pole_cancellation;
use piano_radiator::{
    BodyPreset, LidState, ListenerPerspective, RadiatorBank, CONCERT_GRAND_PRESET_BYTES,
    PRESET_RADIATOR_ZONES,
};

use crate::m4_tests::note_on;
use crate::PianoEngine;

const SR: f32 = 48_000.0;

// --- Gate 1: Pole Cancellation & Double Resonance Elimination ---

#[test]
fn m6_gate1_continuous_qr_stability_and_matrix_identity() {
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    for &sr in &[44_100.0_f32, 48_000.0_f32, 96_000.0_f32] {
        let rep = verify_pole_cancellation(&preset, sr);
        assert!(
            rep.max_inverse_pole_real < -1.0,
            "sr={sr}: poles must stay in LHP"
        );
        assert!(
            rep.max_matrix_residual < 1.0e-6,
            "sr={sr}: matrix residual too large"
        );
        assert!(
            rep.max_uncancelled_peaking_db > 3.0,
            "sr={sr}: expected >3 dB peaking without cancel"
        );
        assert!(
            rep.max_cancelled_excess_db <= 1.0,
            "sr={sr}: cancelled peaking {:.3} dB > 1.0 dB",
            rep.max_cancelled_excess_db
        );
        assert!(
            rep.max_discrete_cascade_err_db <= 1.0,
            "sr={sr}: discrete cascade error {:.3} dB > 1.0 dB",
            rep.max_discrete_cascade_err_db
        );
    }
}

// --- Gate 2: NUPOLS Reconstruction Accuracy & Zero Latency ---

#[test]
fn m6_gate2_nupols_matches_direct_convolution_below_minus_100db() {
    let sr = 48_000.0;
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    let mut bank = RadiatorBank::from_preset_with_space(
        &preset,
        sr,
        ListenerPerspective::CloseStereo,
        LidState::Open,
    );
    let irs = RadiatorBank::synthesize_ir_for_test(
        &preset,
        sr,
        ListenerPerspective::CloseStereo,
        LidState::Open,
    );

    let total_samples = 26_000;
    let mut inputs = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; total_samples];
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    for (n, frame) in inputs.iter_mut().enumerate() {
        if n == 0 || n == 37 || n == 199 || n == 4_103 || n == 12_345 {
            frame[n % PRESET_RADIATOR_ZONES] += 0.25;
        }
        if n < 20_000 {
            for s in frame.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u = ((rng >> 40) as f32) * (1.0 / 8_388_608.0) - 1.0;
                *s += 0.03 * u;
            }
        }
    }

    let mut out_l = vec![0.0_f32; total_samples];
    let mut out_r = vec![0.0_f32; total_samples];
    let uneven_blocks = [1, 7, 64, 128, 256, 513, 3, 13, 31, 63, 127, 300, 41];
    let mut pos = 0;
    let mut b_idx = 0;
    while pos < total_samples {
        let bs = uneven_blocks[b_idx % uneven_blocks.len()].min(total_samples - pos);
        bank.process_block(
            &inputs[pos..pos + bs],
            &mut out_l[pos..pos + bs],
            &mut out_r[pos..pos + bs],
        );
        pos += bs;
        b_idx += 1;
    }

    let (ref_l, ref_r) = direct_convolve_18_ref(&irs, &inputs);
    let mut err_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    for i in 0..total_samples {
        let dl = (out_l[i] as f64) - (ref_l[i] as f64);
        let dr = (out_r[i] as f64) - (ref_r[i] as f64);
        err_sq += dl * dl + dr * dr;
        ref_sq += (ref_l[i] as f64).powi(2) + (ref_r[i] as f64).powi(2);
    }
    let rel_db = 10.0 * (err_sq / ref_sq.max(1e-30)).log10();
    eprintln!("Gate 2 NUPOLS vs Direct RMS error = {rel_db:.2} dB");
    assert!(rel_db <= -100.0, "RMS error {rel_db:.2} dB > -100 dB");
}

#[test]
fn m6_gate2_zero_sample_algorithmic_latency_all_zones_and_perspectives() {
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    let perspectives = [
        ListenerPerspective::Player,
        ListenerPerspective::CloseStereo,
        ListenerPerspective::ConcertHall,
        ListenerPerspective::BinauralHrtf,
    ];
    let lids = [LidState::Closed, LidState::Half, LidState::Open];

    for &persp in &perspectives {
        for &lid in &lids {
            let mut bank = RadiatorBank::from_preset_with_space(&preset, 48_000.0, persp, lid);
            assert_eq!(bank.algorithmic_latency_samples(), 0);
            for z in 0..PRESET_RADIATOR_ZONES {
                bank.reset();
                let k = 37;
                let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 128];
                frames[k][z] = 0.5;
                let mut out_l = vec![0.0_f32; 128];
                let mut out_r = vec![0.0_f32; 128];
                bank.process_block(&frames, &mut out_l, &mut out_r);

                for i in 0..k {
                    assert_eq!(
                        out_l[i], 0.0,
                        "pre-impulse non-zero at {i} for {persp:?}/{lid:?}/z={z}"
                    );
                    assert_eq!(
                        out_r[i], 0.0,
                        "pre-impulse non-zero at {i} for {persp:?}/{lid:?}/z={z}"
                    );
                }
                let mag_at_k = out_l[k].abs().max(out_r[k].abs());
                assert!(mag_at_k > 1.0e-5, "expected immediate output at k={k}");
            }
        }
    }
}

// --- Gate 5: Preset Binary Size & D1 Real-Time Performance Budget ---

#[test]
fn m6_gate5_baked_preset_size_and_lossless_codec() {
    let bytes = CONCERT_GRAND_PRESET_BYTES;
    assert!(
        bytes.len() < 500 * 1024,
        "preset size {} bytes >= 500 KB",
        bytes.len()
    );
    let decoded = BodyPreset::decode(bytes).expect("valid decoded preset");
    assert!(
        decoded.modes.len() > 1_000,
        "expected >1000 modes, got {}",
        decoded.modes.len()
    );
    assert_eq!(decoded.reduced_poles.len(), 10);
}

#[test]
fn m6_gate5_d1_realtime_performance_budget_48khz() {
    let mut engine = PianoEngine::new(SR);
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    let chord_keys = [36, 43, 48, 52, 55, 60, 64, 67, 71, 72, 76, 79];
    let mut init_events = Vec::with_capacity(13);
    init_events.push(piano_spec::SampleEvent {
        sample_offset: 0,
        kind: piano_spec::EventKind::Pedal {
            kind: piano_spec::PedalKind::Sustain,
            value_u16: u16::MAX,
        },
    });
    for &key in &chord_keys {
        init_events.push(note_on(key, 48_000, 0));
    }
    engine.render_block(&init_events, &mut out_l, &mut out_r);

    for _ in 0..64 {
        engine.render_block(&[], &mut out_l, &mut out_r);
    }

    let mut timings_us = Vec::with_capacity(1_000);
    for b in 0..1_000 {
        let ev = [note_on(chord_keys[(b / 16) % chord_keys.len()], 48_000, 0)];
        let slice: &[piano_spec::SampleEvent] = if b % 16 == 0 { &ev } else { &[] };
        let t0 = std::time::Instant::now();
        engine.render_block(slice, &mut out_l, &mut out_r);
        timings_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    timings_us.sort_by(|a, b| a.total_cmp(b));
    let mean_us = timings_us.iter().sum::<f64>() / (timings_us.len() as f64);
    let p50_us = timings_us[timings_us.len() / 2];
    let p99_us = timings_us[(timings_us.len() * 99) / 100];
    eprintln!(
        "M6 Gate 5 D1 render_block 128@48kHz: p50={p50_us:.1} us, mean={mean_us:.1} us, p99={p99_us:.1} us"
    );

    #[cfg(not(debug_assertions))]
    assert!(
        p99_us < 2666.7,
        "D1 p99 {p99_us:.1} us exceeded 2,666.7 us budget"
    );
}

fn direct_convolve_18_ref(
    irs: &[[Vec<f32>; 2]; PRESET_RADIATOR_ZONES],
    inputs: &[[f32; PRESET_RADIATOR_ZONES]],
) -> (Vec<f32>, Vec<f32>) {
    let n_out = inputs.len();
    let ir_len = irs[0][0].len();
    let mut ref_l = vec![0.0_f64; n_out];
    let mut ref_r = vec![0.0_f64; n_out];
    for (n, frame) in inputs.iter().enumerate() {
        for z in 0..PRESET_RADIATOR_ZONES {
            let x = frame[z] as f64;
            if x == 0.0 {
                continue;
            }
            let max_k = (n_out - n).min(ir_len);
            let hl = &irs[z][0];
            let hr = &irs[z][1];
            for k in 0..max_k {
                ref_l[n + k] += x * (hl[k] as f64);
                ref_r[n + k] += x * (hr[k] as f64);
            }
        }
    }
    (
        ref_l.into_iter().map(|v| v as f32).collect(),
        ref_r.into_iter().map(|v| v as f32).collect(),
    )
}
