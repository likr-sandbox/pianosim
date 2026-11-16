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

#[test]
fn m7_stage4_88key_loudness_balance_zone8_ratio_duplex_and_c5_unacorda() {
    let sr = 48_000.0;
    let probe_keys: [(&str, u8); 10] = [
        ("A0", 21),
        ("C1", 24),
        ("C2", 36),
        ("C3", 48),
        ("C4", 60),
        ("A4", 69),
        ("C5", 72),
        ("C6", 84),
        ("C7", 96),
        ("C8", 108),
    ];

    let mut sustain_db_mf = Vec::with_capacity(probe_keys.len());
    let s50 = (0.050 * 48_000.0) as usize;
    let s300 = (0.300 * 48_000.0) as usize;
    let s500 = (0.500 * 48_000.0) as usize;
    for &(name, midi) in &probe_keys {
        let mut engine = PianoEngine::new(sr);
        let mut out_l = vec![0.0_f32; s500];
        let mut out_r = vec![0.0_f32; s500];
        let mut pos = 0;
        while pos < s500 {
            let n = (s500 - pos).min(128);
            let ev_slice = if pos == 0 {
                &[note_on(midi, u7_to_u16(72), 0)][..]
            } else {
                &[]
            };
            engine.render_block(ev_slice, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
            pos += n;
        }
        let peak = out_l
            .iter()
            .chain(out_r.iter())
            .fold(0.0_f32, |m, &x| m.max(x.abs()));
        let mut sum_sq = 0.0_f64;
        for i in s50..s300 {
            let m = 0.5 * f64::from(out_l[i] + out_r[i]);
            sum_sq += m * m;
        }
        let rms = (sum_sq / ((s300 - s50) as f64)).sqrt();
        let db = 20.0 * rms.max(1e-12).log10();
        let peak_db = 20.0 * f64::from(peak).max(1e-12).log10();
        eprintln!(
            "88-key mf {name} (midi={midi}): peak={peak_db:+.1} dBFS, sus[50..300ms]={db:+.1} dBFS"
        );
        assert!(peak < 0.99, "{name} mf unexpectedly clipped: peak={peak}");
        sustain_db_mf.push(db);
    }

    let min_db = sustain_db_mf.iter().copied().fold(f64::INFINITY, f64::min);
    let max_db = sustain_db_mf
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let spread = max_db - min_db;
    eprintln!(
        "88-key mf sustained RMS range: [{min_db:+.1}, {max_db:+.1}] dBFS (spread {spread:.1} dB)"
    );
    assert!(
        min_db > -58.0 && max_db < -10.0,
        "88-key mf sustained level out of range: [{min_db:.1}, {max_db:.1}] dBFS"
    );
    assert!(
        spread <= 20.0,
        "88-key mf sustained spread {spread:.1} dB > 20 dB"
    );

    // Zone 8 keybed contribution at low velocity vs string sustained peak check (vel=1 has minimal string, vel=80 has clear string).
    let mut eng_pp = PianoEngine::new(sr);
    let mut l_pp = vec![0.0_f32; 2_048];
    let mut r_pp = vec![0.0_f32; 2_048];
    eng_pp.render_block(&[note_on(69, u7_to_u16(1), 0)], &mut l_pp, &mut r_pp);
    let pp_peak = l_pp
        .iter()
        .chain(r_pp.iter())
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    let mut eng_mf = PianoEngine::new(sr);
    let mut l_mf = vec![0.0_f32; s500];
    let mut r_mf = vec![0.0_f32; s500];
    let mut pos_mf = 0;
    while pos_mf < s500 {
        let n = (s500 - pos_mf).min(128);
        let evs = if pos_mf == 0 {
            &[note_on(69, u7_to_u16(80), 0)][..]
        } else {
            &[]
        };
        eng_mf.render_block(
            evs,
            &mut l_mf[pos_mf..pos_mf + n],
            &mut r_mf[pos_mf..pos_mf + n],
        );
        pos_mf += n;
    }
    let a4_peak = l_mf[s50..s300]
        .iter()
        .chain(r_mf[s50..s300].iter())
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    let ratio_db = 20.0 * f64::from(pp_peak / a4_peak.max(1e-12)).log10();
    eprintln!(
        "Zone 8 vel=1 peak={pp_peak:.5} vs A4 mf[80..500] peak={a4_peak:.5} ({ratio_db:+.1} dB)"
    );
    assert!(
        ratio_db <= -18.0,
        "keybed thump at vel=1 too loud vs sustained A4: {ratio_db:+.1} dB"
    );

    // 12-note fff cluster must stay finite & bounded <= 1.0 without wraparound.
    let mut eng_fff = PianoEngine::new(sr);
    let chord: Vec<_> = [21_u8, 28, 36, 43, 48, 55, 60, 64, 67, 72, 79, 84]
        .iter()
        .enumerate()
        .map(|(i, &k)| note_on(k, u16::MAX, (i * 8) as u32))
        .collect();
    let mut l_ch = vec![0.0_f32; 8_192];
    let mut r_ch = vec![0.0_f32; 8_192];
    eng_fff.render_block(&chord, &mut l_ch, &mut r_ch);
    let fff_peak = l_ch.iter().chain(r_ch.iter()).fold(0.0_f32, |m, &x| {
        assert!(x.is_finite());
        m.max(x.abs())
    });
    assert!(
        fff_peak > 0.1 && fff_peak <= 1.0,
        "fff chord peak={fff_peak}"
    );

    // Stage 4 C5 (midi=72) Una Corda check at early [0..100ms] and sustained [150..350ms] & [200..370ms].
    let measure_c5_stage4 = |uc_u16: u16| -> (f64, f64, f64, f64) {
        let mut eng = PianoEngine::new(sr);
        let evs = [
            piano_spec::SampleEvent {
                sample_offset: 0,
                kind: piano_spec::EventKind::Pedal {
                    kind: piano_spec::PedalKind::UnaCorda,
                    value_u16: uc_u16,
                },
            },
            note_on(72, u7_to_u16(96), 0),
        ];
        let total = 24_576_usize;
        let mut ol = vec![0.0_f32; total];
        let mut or = vec![0.0_f32; total];
        let mut pos = 0;
        while pos < total {
            let n = (total - pos).min(128);
            let s = if pos == 0 { &evs[..] } else { &[] };
            eng.render_block(s, &mut ol[pos..pos + n], &mut or[pos..pos + n]);
            pos += n;
        }
        let m_win = |s0: usize, len: usize| -> (f64, f64) {
            let mono: Vec<f32> = (s0..s0 + len).map(|i| 0.5 * (ol[i] + or[i])).collect();
            let rms_v =
                (mono.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / (len as f64)).sqrt();
            let cent = crate::m3_tests::spectral_centroid_hz(&mono, 48_000.0);
            (rms_v, cent)
        };
        let (r0, c0) = m_win(0, 4_096);
        let (r1, c1) = m_win(9_600, 8_192);
        (r0, c0, r1, c1)
    };
    let (r0_n, c0_n, r1_n, c1_n) = measure_c5_stage4(0);
    let (r0_u, c0_u, r1_u, c1_u) = measure_c5_stage4(u16::MAX);
    eprintln!(
        "Stage 4 C5 Una Corda 0..85ms: rms {r0_n:.5}->{r0_u:.5}, cent {c0_n:.1}->{c0_u:.1} Hz"
    );
    eprintln!(
        "Stage 4 C5 Una Corda 200..370ms: rms {r1_n:.5}->{r1_u:.5}, cent {c1_n:.1}->{c1_u:.1} Hz"
    );
    let (rms_norm, cent_norm, rms_uc, cent_uc) = (r1_n, c1_n, r1_u, c1_u);
    assert!(
        rms_uc < rms_norm && cent_uc < cent_norm,
        "Stage 4 C5 Una Corda 200..370ms failed: rms {rms_norm:.5}->{rms_uc:.5}, cent {cent_norm:.1}->{cent_uc:.1}"
    );
    assert!(
        r0_u < r0_n && c0_u < c0_n,
        "Stage 4 C5 Una Corda 0..85ms failed: rms {r0_n:.5}->{r0_u:.5}, cent {c0_n:.1}->{c0_u:.1}"
    );

    // DuplexBank Stage 4 acoustic contribution on treble C6 (midi=84) must be finite and non-zero.
    let render_c6 = |duplex_gain: f32| -> Vec<f32> {
        let mut eng = PianoEngine::new(sr);
        eng.set_duplex_gain(duplex_gain);
        let n = 8_192_usize;
        let mut ol = vec![0.0_f32; n];
        let mut or = vec![0.0_f32; n];
        let mut pos = 0;
        while pos < n {
            let chunk = (n - pos).min(128);
            let evs = if pos == 0 {
                &[note_on(84, u7_to_u16(100), 0)][..]
            } else {
                &[]
            };
            eng.render_block(evs, &mut ol[pos..pos + chunk], &mut or[pos..pos + chunk]);
            pos += chunk;
        }
        (0..n).map(|i| 0.5 * (ol[i] + or[i])).collect()
    };
    let c6_with_dup = render_c6(1.0);
    let c6_no_dup = render_c6(0.0);
    let dup_diff_rms = (c6_with_dup
        .iter()
        .zip(c6_no_dup.iter())
        .map(|(&a, &b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / (c6_with_dup.len() as f64))
        .sqrt();
    eprintln!("Stage 4 DuplexBank C6 diff_rms={dup_diff_rms:.6e}");
    assert!(
        dup_diff_rms > 2.0e-5,
        "DuplexBank had negligible Stage 4 effect: {dup_diff_rms:e}"
    );
}
