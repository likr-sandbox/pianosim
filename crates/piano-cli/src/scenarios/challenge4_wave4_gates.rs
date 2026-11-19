//! Wave 4 Unified Stage 4 Acoustic Gates (`IV-3`):
//! 1. 88-key ISO-226 loudness continuity (`A0..C8`)
//! 2. Binaural IACC (`ListenerPerspective::BinauralHrtf`: `< 350 Hz` coherent vs `> 1.2 kHz` decorrelated)
//! 3. Fortissimo (`ff`) multi-tone / chord IMD & spurious aliasing floor

use piano_engine::{PerfProfile, PianoEngine};
use piano_radiator::BodyArchetype;
use piano_simd::RealFft;
use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, u7_to_u16, EventKind,
    LidState, ListenerPerspective, SampleEvent, VoicingSpec, NUM_KEYS,
};

use super::spectral_utils::{hann_windowed_spectrum, mono_mix, peak_abs};

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

/// ISO 226:2003 Table 1 preferred 1/3-octave center frequencies [Hz].
const ISO226_FREQS: [f64; 29] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0, 12500.0,
];
/// ISO 226:2003 loudness exponent `alpha_f`.
const ISO226_AF: [f64; 29] = [
    0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301, 0.288,
    0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245,
    0.254, 0.271, 0.301,
];
/// ISO 226:2003 linear transfer magnitude `L_U` [dB].
const ISO226_LU: [f64; 29] = [
    -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1, -0.4, 0.0,
    0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1,
];
/// ISO 226:2003 hearing threshold `T_f` [dB SPL].
const ISO226_TF: [f64; 29] = [
    78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0,
    2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
];

#[derive(Clone, Debug)]
pub struct Challenge4Report {
    pub pass: bool,
    pub iso226_spread_db: f64,
    pub iso226_max_adj_step_db: f64,
    pub iso226_max_zone_step_db: f64,
    pub binaural_iacc_lf: f64,
    pub binaural_iacc_hf: f64,
    pub ff_imd_spurious_db: f64,
    pub ff_aliasing_hf_db: f64,
    pub details: Vec<String>,
}

/// Computes ISO 226:2003 equal-loudness SPL contour at `phon` for band index `idx`.
fn iso226_spl_at_band(idx: usize, phon: f64) -> f64 {
    let af = ISO226_AF[idx];
    let lu = ISO226_LU[idx];
    let tf = ISO226_TF[idx];
    let thresh_term = (0.4 * 10.0_f64.powf((tf + lu) / 10.0 - 9.0)).powf(af);
    let a_f = 4.47e-3 * (10.0_f64.powf(0.025 * phon) - 1.15) + thresh_term;
    (10.0 / af) * a_f.log10() - lu + 94.0
}

/// Returns perceptual ISO-226 power weight relative to `1000 Hz` at `70 phon`.
#[must_use]
pub fn iso226_power_weight(freq_hz: f64, phon: f64) -> f64 {
    let spl_1k = iso226_spl_at_band(17, phon); // index 17 == 1000 Hz
    let f = freq_hz.clamp(20.0, 16_000.0);
    let spl_f = if f >= 12_500.0 {
        let s_hi = iso226_spl_at_band(28, phon);
        let s_prev = iso226_spl_at_band(27, phon);
        let t = (f / 12_500.0).log2() / (12_500.0_f64 / 10_000.0).log2();
        s_hi + (s_hi - s_prev) * t.min(1.0)
    } else {
        let mut idx = 0_usize;
        while idx + 1 < ISO226_FREQS.len() && ISO226_FREQS[idx + 1] < f {
            idx += 1;
        }
        let f0 = ISO226_FREQS[idx];
        let f1 = ISO226_FREQS[idx + 1];
        let t = (f / f0).log2() / (f1 / f0).log2();
        let s0 = iso226_spl_at_band(idx, phon);
        let s1 = iso226_spl_at_band(idx + 1, phon);
        s0 + t * (s1 - s0)
    };
    10.0_f64.powf((spl_1k - spl_f) / 10.0)
}

fn note_on(midi_key: u8, velocity_u16: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::NoteOn {
            midi_key,
            velocity_u16,
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
    first_events: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut pos = 0_usize;
    while pos < total_frames {
        let n = (total_frames - pos).min(128);
        let cur = if pos == 0 { first_events } else { &[] };
        engine.render_block(cur, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    (out_l, out_r)
}

/// Computes ISO-226-weighted loudness [dB] for a note window `[s_start..s_start + n_fft]`.
fn measure_iso226_loudness_db(mono: &[f32], s_start: usize, n_fft: usize) -> f64 {
    let spec = hann_windowed_spectrum(&mono[s_start..s_start + n_fft]);
    let bin_hz = SR64 / (n_fft as f64);
    let mut weighted_pow = 0.0_f64;
    for (bin, c) in spec.iter().enumerate().skip(1) {
        let f_hz = (bin as f64) * bin_hz;
        if !(20.0..=16_000.0).contains(&f_hz) {
            continue;
        }
        let w = iso226_power_weight(f_hz, 70.0);
        weighted_pow += f64::from(c.norm_sq()) * w;
    }
    // Normalize by N^2 * Hann coherent/incoherent power factor for clean dB scale.
    let norm = weighted_pow / ((n_fft as f64) * (n_fft as f64) * 0.375);
    10.0 * norm.max(1.0e-18).log10()
}

fn band_iacc(l: &[f32], r: &[f32], sr: f64, f_lo: f64, f_hi: f64) -> f64 {
    let n_fft = 16_384_usize;
    let mut pl = vec![0.0_f32; n_fft];
    let mut pr = vec![0.0_f32; n_fft];
    let copy_n = l.len().min(n_fft);
    pl[..copy_n].copy_from_slice(&l[..copy_n]);
    pr[..copy_n].copy_from_slice(&r[..copy_n]);

    let mut fft = RealFft::new(n_fft);
    let spec_len = fft.spectrum_len();
    let (mut re_l, mut im_l) = (vec![0.0_f32; spec_len], vec![0.0_f32; spec_len]);
    let (mut re_r, mut im_r) = (vec![0.0_f32; spec_len], vec![0.0_f32; spec_len]);
    fft.forward_split(&pl, &mut re_l, &mut im_l);
    fft.forward_split(&pr, &mut re_r, &mut im_r);

    let k0 = ((f_lo * (n_fft as f64) / sr).ceil() as usize).max(1);
    let k1 = ((f_hi * (n_fft as f64) / sr).floor() as usize).min(spec_len - 1);
    let (mut cross, mut el, mut er) = (0.0_f64, 0.0_f64, 0.0_f64);
    for k in k0..=k1 {
        let al_re = f64::from(re_l[k]);
        let al_im = f64::from(im_l[k]);
        let ar_re = f64::from(re_r[k]);
        let ar_im = f64::from(im_r[k]);
        cross += al_re * ar_re + al_im * ar_im;
        el += al_re * al_re + al_im * al_im;
        er += ar_re * ar_re + ar_im * ar_im;
    }
    cross / (el * er).sqrt().max(1.0e-30)
}

/// Executes Wave 4 Stage 4 unified acoustic gates (`IV-3`) under `profile`.
pub fn verify_challenge4_with_profile(profile: PerfProfile) -> Result<Challenge4Report, String> {
    let mut details = Vec::new();
    let mut all_pass = true;

    // --- Gate 4A: 88-key ISO-226 loudness continuity ---
    let mut loudness_88 = Vec::with_capacity(NUM_KEYS);
    let s_start = (0.015 * SR64).round() as usize; // 15 ms (post-keybed attack)
    let n_fft = 8_192_usize; // ~170.7 ms auditory temporal integration window
    let total_frames = s_start + n_fft;
    let vel_mf = u7_to_u16(72);

    let mut eng_88 = PianoEngine::new_with_profile(SR, VoicingSpec::default(), profile);
    for k in 0..NUM_KEYS {
        let midi = 21 + (k as u8);
        eng_88.reset();
        let (ol, or) = render_blocks(&mut eng_88, total_frames, &[note_on(midi, vel_mf, 0)]);
        let mono = mono_mix(&ol, &or);
        let db = measure_iso226_loudness_db(&mono, s_start, n_fft);
        loudness_88.push(db);
    }

    let min_db = loudness_88.iter().copied().fold(f64::INFINITY, f64::min);
    let max_db = loudness_88
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let spread_db = max_db - min_db;
    let mut adj_steps: Vec<f64> = loudness_88
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect();
    let max_adj_step_db = adj_steps.iter().copied().fold(0.0_f64, f64::max);
    adj_steps.sort_by(f64::total_cmp);
    let p95_adj_step_db = adj_steps[((adj_steps.len() * 95) / 100).min(adj_steps.len() - 1)];

    // 8 register zones of 11 keys each
    let mut zone_means = [0.0_f64; 8];
    for (z, zm) in zone_means.iter_mut().enumerate() {
        let lo = z * 11;
        let hi = lo + 11;
        *zm = loudness_88[lo..hi].iter().sum::<f64>() / 11.0;
    }
    let max_zone_step_db = zone_means
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max);

    if spread_db > 18.0 || max_adj_step_db > 9.0 || p95_adj_step_db > 4.5 || max_zone_step_db > 8.0
    {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 4A: ISO-226 88-key spread={spread_db:.2} dB, adj_max={max_adj_step_db:.2} dB (p95={p95_adj_step_db:.2}), zone_step={max_zone_step_db:.2} dB"
        ));
    } else {
        details.push(format!(
            "PASS Gate 4A: ISO-226 88-key spread={spread_db:.2} dB, adj_max={max_adj_step_db:.2} dB (p95={p95_adj_step_db:.2}), zone_step={max_zone_step_db:.2} dB"
        ));
    }

    // --- Gate 4B: Binaural IACC (<350 Hz coherent vs >1.2 kHz decorrelated) ---
    let mut eng_bin = PianoEngine::new_with_profile(SR, VoicingSpec::default(), profile);
    eng_bin.set_body_space(
        BodyArchetype::ConcertGrand,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
    );
    eng_bin.set_reverb(1.8, 0.35);
    let _ = render_blocks(&mut eng_bin, 2_400, &[note_on(60, u7_to_u16(115), 0)]);
    let _ = render_blocks(&mut eng_bin, 16_000, &[note_off(60, 0)]);
    let (tail_l, tail_r) = render_blocks(&mut eng_bin, 16_384, &[]);
    let iacc_lf = band_iacc(&tail_l, &tail_r, SR64, 60.0, 350.0);
    let iacc_hf = band_iacc(&tail_l, &tail_r, SR64, 1_200.0, 6_000.0);

    if iacc_lf < 0.70 || iacc_hf.abs() > 0.40 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 4B: Binaural IACC lf={iacc_lf:.3} (<0.70) or |hf|={:.3} (>0.40)",
            iacc_hf.abs()
        ));
    } else {
        details.push(format!(
            "PASS Gate 4B: Binaural IACC lf={iacc_lf:.3} >= 0.70, hf={iacc_hf:+.3} (|hf| <= 0.40)"
        ));
    }

    // --- Gate 4C: Fortissimo (ff) IMD & spurious aliasing ratio ---
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let dyad_keys = [84_u8, 91_u8]; // C6 (1046.5 Hz) + G6 (1568.0 Hz)
    let mut eng_ff = PianoEngine::new_with_profile(SR, VoicingSpec::default(), profile);
    let dyad_evs = [
        note_on(dyad_keys[0], u16::MAX, 0),
        note_on(dyad_keys[1], u16::MAX, 0),
    ];
    let (ff_l, ff_r) = render_blocks(&mut eng_ff, 24_576, &dyad_evs);
    let ff_peak = peak_abs(&ff_l).max(peak_abs(&ff_r));
    let ff_mono = mono_mix(&ff_l, &ff_r);

    let fft_imd = 16_384_usize;
    let imd_start = 4_096_usize; // post-attack (~85 ms..426 ms)
    let spec_imd = hann_windowed_spectrum(&ff_mono[imd_start..imd_start + fft_imd]);
    let bin_hz = SR64 / (fft_imd as f64);

    let mut partial_freqs = Vec::new();
    for &mk in &dyad_keys {
        let k = midi_to_key(mk).expect("valid key");
        let f0 = key_tuned_f0_hz(k, &specs, &voicing);
        let b = specs[k].inharmonicity_b;
        for j in 1..32_usize {
            let fj = partial_frequency_hz(f0, b, j);
            if fj > 20_000.0 {
                break;
            }
            partial_freqs.push(fj);
        }
    }

    let mut sig_pow = 0.0_f64;
    let mut spurious_pow = 0.0_f64;
    let mut hf_alias_pow = 0.0_f64;
    for (bin, c) in spec_imd.iter().enumerate().skip(1) {
        let f_hz = (bin as f64) * bin_hz;
        let p = f64::from(c.norm_sq());
        if (20.0..=18_000.0).contains(&f_hz) {
            sig_pow += p;
            let near_partial = partial_freqs
                .iter()
                .any(|&fj| (f_hz - fj).abs() <= (0.02 * fj).max(30.0));
            if !near_partial {
                spurious_pow += p;
            }
        } else if (18_000.0..=23_500.0).contains(&f_hz) {
            hf_alias_pow += p;
        }
    }
    let spurious_db = 10.0 * (spurious_pow / sig_pow.max(1e-24)).max(1e-18).log10();
    let alias_hf_db = 10.0 * (hf_alias_pow / sig_pow.max(1e-24)).max(1e-18).log10();

    // Also verify 4-note ff chord remains bounded <= 1.0 and finite.
    let mut eng_ch = PianoEngine::new_with_profile(SR, VoicingSpec::default(), profile);
    let chord_evs = [
        note_on(60, u16::MAX, 0),
        note_on(64, u16::MAX, 8),
        note_on(67, u16::MAX, 16),
        note_on(72, u16::MAX, 24),
    ];
    let (ch_l, ch_r) = render_blocks(&mut eng_ch, 8_192, &chord_evs);
    let ch_finite = ch_l
        .iter()
        .chain(ch_r.iter())
        .all(|x| x.is_finite() && x.abs() <= 1.0);

    if !ch_finite || ff_peak > 1.0 || spurious_db > -35.0 || alias_hf_db > -60.0 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 4C: ff IMD spurious={spurious_db:.2} dBc, HF alias={alias_hf_db:.2} dBc, ff_peak={ff_peak:.4}, chord_ok={ch_finite}"
        ));
    } else {
        details.push(format!(
            "PASS Gate 4C: ff IMD spurious={spurious_db:.2} dBc, HF alias={alias_hf_db:.2} dBc, ff_peak={ff_peak:.4}"
        ));
    }

    let report = Challenge4Report {
        pass: all_pass,
        iso226_spread_db: spread_db,
        iso226_max_adj_step_db: max_adj_step_db,
        iso226_max_zone_step_db: max_zone_step_db,
        binaural_iacc_lf: iacc_lf,
        binaural_iacc_hf: iacc_hf,
        ff_imd_spurious_db: spurious_db,
        ff_aliasing_hf_db: alias_hf_db,
        details,
    };

    if all_pass {
        Ok(report)
    } else {
        Err(format!(
            "Challenge 4 FAILED:\n{}",
            report.details.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn iso226_contour_weighting_matches_1khz_reference_and_band_trends() {
        let w_1k = iso226_power_weight(1_000.0, 70.0);
        let w_30 = iso226_power_weight(30.0, 70.0);
        let w_3150 = iso226_power_weight(3_150.0, 70.0);
        assert!((w_1k - 1.0).abs() < 1.0e-12);
        assert!(
            w_30 < 0.05,
            "30 Hz should be strongly attenuated, got {w_30}"
        );
        assert!(w_3150 > 2.0, "3.15 kHz should be boosted, got {w_3150}");
    }

    #[test]
    fn band_iacc_distinguishes_coherent_lf_from_orthogonal_hf() {
        let n = 4_096_usize;
        let mut l = vec![0.0_f32; n];
        let mut r = vec![0.0_f32; n];
        for i in 0..n {
            let t = (i as f64) / SR64;
            let lf = (TAU * 150.0 * t).sin();
            let hf_l = (TAU * 2_400.0 * t).sin();
            let hf_r = (TAU * 2_400.0 * t).cos();
            l[i] = (lf + hf_l) as f32;
            r[i] = (lf + hf_r) as f32;
        }
        let lf_iacc = band_iacc(&l, &r, SR64, 60.0, 350.0);
        let hf_iacc = band_iacc(&l, &r, SR64, 1_200.0, 6_000.0);
        assert!(lf_iacc > 0.99, "expected coherent LF IACC, got {lf_iacc}");
        assert!(
            hf_iacc.abs() < 0.05,
            "expected orthogonal HF IACC near 0, got {hf_iacc}"
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn verify_challenge4_wave4_gates_pass() {
        let rep =
            verify_challenge4_with_profile(PerfProfile::full()).unwrap_or_else(|e| panic!("{e}"));
        assert!(rep.pass);
    }
}
