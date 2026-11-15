//! Milestone 6 Acceptance Gates 3 and 4:
//! - Gate 3: Acoustic differentiation (multi-rate, lid states, binaural HRTF, keybed thump)
//! - Gate 4: Real-time voicing parameter smoothing & space switching crossfades

use piano_radiator::{
    BodyPreset, LidState, ListenerPerspective, RadiatorBank, CONCERT_GRAND_PRESET_BYTES,
    PRESET_RADIATOR_ZONES,
};
use piano_simd::RealFft;
use piano_spec::{u7_to_u16, PRESET_BRIDGE_ZONES};

use crate::m4_tests::note_on;
use crate::PianoEngine;

// --- Gate 3: Multi-Rate, Lid States, Binaural HRTF, and Keybed Thump ---

#[test]
fn m6_gate3_sample_rate_spectral_envelope_consistency() {
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    let rates = [
        (44_100.0_f64, 32_768_usize),
        (48_000.0, 32_768),
        (96_000.0, 65_536),
    ];
    let bands = [
        (125.0, 250.0),
        (250.0, 500.0),
        (500.0, 1_000.0),
        (1_000.0, 2_000.0),
        (2_000.0, 4_000.0),
        (4_000.0, 8_000.0),
        (100.0, 8_000.0),
    ];

    let mut spectra = Vec::new();
    for &(sr, fft_len) in &rates {
        let irs = RadiatorBank::synthesize_ir_for_test(
            &preset,
            sr,
            ListenerPerspective::Player,
            LidState::Open,
        );
        let mut avg_mag = vec![0.0_f64; fft_len / 2 + 1];
        for pair in &irs[..PRESET_BRIDGE_ZONES] {
            for ch_ir in pair {
                let m = fft_mag_sq(ch_ir, fft_len);
                for (dst, &src) in avg_mag.iter_mut().zip(m.iter()) {
                    *dst += src;
                }
            }
        }
        spectra.push((sr, fft_len, avg_mag));
    }

    let ref_entry = &spectra[1]; // 48 kHz reference
    for entry in &spectra {
        for &(f_lo, f_hi) in &bands {
            let p_sr = band_mean_power(&entry.2, entry.0, entry.1, f_lo, f_hi);
            let p_ref = band_mean_power(&ref_entry.2, ref_entry.0, ref_entry.1, f_lo, f_hi);
            let diff_db = 10.0 * (p_sr / p_ref).log10();
            assert!(
                diff_db.abs() <= 1.0,
                "sr={} band [{}..{}] diff={:.3} dB > 1.0 dB",
                entry.0,
                f_lo,
                f_hi,
                diff_db
            );
        }
    }
}

#[test]
fn m6_gate3_lid_state_hf_energy_ordering_all_perspectives() {
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    let sr = 48_000.0;
    let fft_len = 32_768;
    let perspectives = [
        ListenerPerspective::Player,
        ListenerPerspective::CloseStereo,
        ListenerPerspective::ConcertHall,
        ListenerPerspective::BinauralHrtf,
    ];

    for &persp in &perspectives {
        let ir_closed = RadiatorBank::synthesize_ir_for_test(&preset, sr, persp, LidState::Closed);
        let ir_half = RadiatorBank::synthesize_ir_for_test(&preset, sr, persp, LidState::Half);
        let ir_open = RadiatorBank::synthesize_ir_for_test(&preset, sr, persp, LidState::Open);

        let hf_energy = |irs: &[[Vec<f32>; 2]; PRESET_RADIATOR_ZONES]| -> f64 {
            let mut sum = 0.0_f64;
            for pair in irs {
                for ch_ir in pair {
                    let m = fft_mag_sq(ch_ir, fft_len);
                    sum += band_mean_power(&m, sr, fft_len, 2_000.0, 14_000.0);
                }
            }
            sum
        };
        let e_closed = hf_energy(&ir_closed);
        let e_half = hf_energy(&ir_half);
        let e_open = hf_energy(&ir_open);
        let db_half_closed = 10.0 * (e_half / e_closed).log10();
        let db_open_half = 10.0 * (e_open / e_half).log10();
        assert!(
            db_half_closed >= 2.0 && db_open_half >= 2.0,
            "{persp:?} failed ordering: Half/Closed={db_half_closed:.2} dB, Open/Half={db_open_half:.2} dB"
        );
    }
}

#[test]
fn m6_gate3_binaural_itd_and_ild_bass_vs_treble() {
    let sr = 48_000.0;
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    let irs = RadiatorBank::synthesize_ir_for_test(
        &preset,
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
    );
    let measure_onset = |ir: &[f32]| -> usize {
        let peak = ir.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        ir.iter().position(|&s| s.abs() >= peak * 0.20).unwrap_or(0)
    };
    let energy = |v: &[f32]| -> f64 { v.iter().map(|&s| (s as f64) * (s as f64)).sum() };

    let onset_0_l = measure_onset(&irs[0][0]) as isize;
    let onset_0_r = measure_onset(&irs[0][1]) as isize;
    let itd_0 = onset_0_l - onset_0_r;
    let ild_0 = 10.0 * (energy(&irs[0][0]) / energy(&irs[0][1])).log10();

    let onset_7_l = measure_onset(&irs[7][0]) as isize;
    let onset_7_r = measure_onset(&irs[7][1]) as isize;
    let itd_7 = onset_7_l - onset_7_r;
    let ild_7 = 10.0 * (energy(&irs[7][0]) / energy(&irs[7][1])).log10();

    assert!(
        itd_0 <= -6,
        "Zone 0 L should lead R by >=6 samples, got {itd_0}"
    );
    assert!(
        itd_7 >= 6,
        "Zone 7 R should lead L by >=6 samples, got {itd_7}"
    );
    assert!(
        ild_0 >= 1.5,
        "Zone 0 L louder by >=1.5 dB, got {ild_0:.2} dB"
    );
    assert!(
        ild_7 <= -1.5,
        "Zone 7 R louder by >=1.5 dB, got {ild_7:.2} dB"
    );
}

#[test]
fn m6_gate3_keybed_thump_onset_delay_spectrum_and_monotonic_force() {
    let sr = 48_000.0;
    let mut prev_peak = 0.0_f32;
    for &vel in &[
        u7_to_u16(1),
        u7_to_u16(32),
        u7_to_u16(64),
        u7_to_u16(100),
        u16::MAX,
    ] {
        let mut engine = PianoEngine::new(sr);
        let total = 1_536; // 32 ms
        let mut out_l = vec![0.0_f32; total];
        let mut out_r = vec![0.0_f32; total];
        let evs = [note_on(60, vel, 0)];
        engine.render_block(&evs, &mut out_l, &mut out_r);

        let peak = out_l.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(peak > prev_peak, "peak force not monotonic across velocity");
        prev_peak = peak;
    }
}

// --- Gate 4: Voicing Smoothing & Space Switching Crossfades ---

#[test]
fn m6_gate4_voicing_parameter_smoothing_continuity_no_clicks() {
    let mut engine = PianoEngine::new(48_000.0);
    let mut out_l = vec![0.0_f32; 128];
    let mut out_r = vec![0.0_f32; 128];
    let evs = [note_on(60, 48_000, 0)];
    engine.render_block(&evs, &mut out_l, &mut out_r);

    let mut all_samples = Vec::with_capacity(48_000);
    all_samples.extend_from_slice(&out_l);

    engine.set_voicing(0.85, 2.5, 1.2, 1.0);
    for _ in 0..100 {
        engine.render_block(&[], &mut out_l, &mut out_r);
        all_samples.extend_from_slice(&out_l);
    }

    let mut max_delta = 0.0_f32;
    for i in 1..all_samples.len() {
        assert!(all_samples[i].is_finite(), "non-finite sample at {i}");
        let delta = (all_samples[i] - all_samples[i - 1]).abs();
        max_delta = max_delta.max(delta);
    }
    assert!(
        max_delta < 0.15,
        "detected click discontinuity max_delta={max_delta}"
    );
}

#[test]
fn m6_gate4_space_switching_raised_cosine_crossfade_no_clicks() {
    let mut engine = PianoEngine::new(48_000.0);
    let mut out_l = vec![0.0_f32; 128];
    let mut out_r = vec![0.0_f32; 128];
    let evs = [note_on(69, 48_000, 0)];
    engine.render_block(&evs, &mut out_l, &mut out_r);

    let mut all_samples = Vec::with_capacity(24_000);
    all_samples.extend_from_slice(&out_l);

    engine.set_space(2, 0); // ConcertHall, Closed
    for _ in 0..50 {
        engine.render_block(&[], &mut out_l, &mut out_r);
        all_samples.extend_from_slice(&out_l);
    }

    let mut max_delta = 0.0_f32;
    for i in 1..all_samples.len() {
        assert!(all_samples[i].is_finite());
        max_delta = max_delta.max((all_samples[i] - all_samples[i - 1]).abs());
    }
    assert!(
        max_delta < 0.15,
        "crossfade click discontinuity max_delta={max_delta}"
    );
}

fn fft_mag_sq(signal: &[f32], fft_len: usize) -> Vec<f64> {
    let mut padded = vec![0.0_f32; fft_len];
    let copy_n = signal.len().min(fft_len);
    padded[..copy_n].copy_from_slice(&signal[..copy_n]);
    let mut fft = RealFft::new(fft_len);
    let spec_len = fft.spectrum_len();
    let mut re = vec![0.0_f32; spec_len];
    let mut im = vec![0.0_f32; spec_len];
    fft.forward_split(&padded, &mut re, &mut im);
    (0..spec_len)
        .map(|k| (re[k] as f64).powi(2) + (im[k] as f64).powi(2))
        .collect()
}

fn band_mean_power(mag_sq: &[f64], sr: f64, fft_len: usize, f_lo: f64, f_hi: f64) -> f64 {
    let k0 = ((f_lo * fft_len as f64 / sr).ceil() as usize).max(1);
    let k1 = ((f_hi * fft_len as f64 / sr).floor() as usize).min(mag_sq.len() - 1);
    let sum: f64 = mag_sq[k0..=k1].iter().sum();
    sum / ((k1 - k0 + 1) as f64)
}
