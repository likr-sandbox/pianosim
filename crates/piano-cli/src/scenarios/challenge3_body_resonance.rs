//! Challenge 3: Low-Mid Sustained Body Resonance Density & Keybed Thump.

use piano_engine::PianoEngine;
use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, u7_to_u16, EventKind,
    PedalKind, SampleEvent, VoicingSpec,
};

use super::spectral_utils::{decay_slope_db_per_s, hann_windowed_spectrum, mono_mix, peak_abs};

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

#[derive(Clone, Debug)]
pub struct Challenge3Report {
    pub pass: bool,
    pub c3_two_stage_ratio: f64,
    pub a4_two_stage_ratio: f64,
    pub lehtonen_sustain_gain_db: f64,
    pub keybed_pp_vs_mf_ratio_db: f64,
    pub details: Vec<String>,
}

fn note_on(midi_key: u8, velocity_u16: u16) -> SampleEvent {
    SampleEvent {
        sample_offset: 0,
        kind: EventKind::NoteOn {
            midi_key,
            velocity_u16,
        },
    }
}

fn render_note_hold(midi_key: u8, vel_u16: u16, sustain_u16: u16, secs: f64) -> Vec<f32> {
    let mut engine = PianoEngine::new(SR);
    let total = (secs * SR64).round() as usize;
    let mut out_l = vec![0.0_f32; total];
    let mut out_r = vec![0.0_f32; total];

    let evs = [
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: sustain_u16,
            },
        },
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key,
                velocity_u16: vel_u16,
            },
        },
    ];

    let mut pos = 0;
    while pos < total {
        let n = (total - pos).min(128);
        let cur = if pos == 0 { &evs[..] } else { &[] };
        engine.render_block_bridge(cur, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    mono_mix(&out_l, &out_r)
}

/// Executes Challenge 3 body resonance, two-stage decay, and keybed thump verification.
pub fn verify_challenge3() -> Result<Challenge3Report, String> {
    let mut details = Vec::new();
    let mut all_pass = true;

    // --- Gate 3A: Two-stage decay slope ratio >= 1.5x on C3 and A4 ---
    let c3_mono = render_note_hold(48, 48_000, 0, 3.1);
    let a4_mono = render_note_hold(69, 48_000, 0, 3.1);

    let c3_prompt = decay_slope_db_per_s(&c3_mono, SR64, 0.05, 0.45, 0.05);
    let c3_late = decay_slope_db_per_s(&c3_mono, SR64, 1.40, 3.00, 0.10);
    let c3_ratio = c3_prompt / c3_late.max(1e-9);

    let a4_prompt = decay_slope_db_per_s(&a4_mono, SR64, 0.05, 0.45, 0.05);
    let a4_late = decay_slope_db_per_s(&a4_mono, SR64, 1.40, 3.00, 0.10);
    let a4_ratio = a4_prompt / a4_late.max(1e-9);

    if c3_ratio < 1.5 || a4_ratio < 1.5 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 3A: Two-stage decay ratio < 1.5x (C3={c3_ratio:.2}x, A4={a4_ratio:.2}x)"
        ));
    } else {
        details.push(format!(
            "PASS Gate 3A: Two-stage decay ratio >= 1.5x (C3 prompt={c3_prompt:.1} / late={c3_late:.1} dB/s = {c3_ratio:.2}x, A4 prompt={a4_prompt:.1} / late={a4_late:.1} dB/s = {a4_ratio:.2}x)"
        ));
    }

    // --- Gate 3B: Sustain pedal background resonance gain (+6 to +14 dB) ---
    let c4_dry = render_note_hold(60, 48_000, 0, 3.8);
    let c4_sus = render_note_hold(60, 48_000, u16::MAX, 3.8);

    let n_fft = 16_384_usize;
    let hop = 8_192_usize;
    let spec_len = n_fft / 2 + 1;
    let bin_hz = SR64 / (n_fft as f64);
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let k_c4 = midi_to_key(60).expect("C4");
    let f0 = key_tuned_f0_hz(k_c4, &specs, &voicing);
    let b = specs[k_c4].inharmonicity_b;

    let mut partials = Vec::new();
    for j in 1..128_usize {
        let fj = partial_frequency_hz(f0, b, j);
        if fj > 16_500.0 {
            break;
        }
        partials.push((fj, (0.012 * fj).max(15.0)));
    }
    let mut in_band = vec![false; spec_len];
    let mut keep_res = vec![false; spec_len];
    for bin in 1..spec_len {
        let f_bin = (bin as f64) * bin_hz;
        if (20.0..=16_000.0).contains(&f_bin) {
            in_band[bin] = true;
            keep_res[bin] = partials.iter().all(|&(fj, hw)| (f_bin - fj).abs() > hw);
        }
    }

    let s1 = (1.0 * SR64).round() as usize;
    let s_end = (3.7 * SR64).round() as usize;
    let sum_bins = |mono: &[f32]| -> (f64, f64) {
        let (mut tot, mut res) = (0.0_f64, 0.0_f64);
        let mut pos = s1;
        while pos + n_fft <= s_end {
            let spec = hann_windowed_spectrum(&mono[pos..pos + n_fft]);
            for (bin, c) in spec.iter().enumerate() {
                if in_band[bin] {
                    let p = f64::from(c.norm_sq());
                    tot += p;
                    if keep_res[bin] {
                        res += p;
                    }
                }
            }
            pos += hop;
        }
        (tot, res)
    };

    let (e_tot_dry, e_res_dry) = sum_bins(&c4_dry);
    let (_e_tot_sus, e_res_sus) = sum_bins(&c4_sus);
    let e_floor = 1.0e-3 * e_tot_dry;
    let floor_db = 10.0 * ((e_res_sus + e_floor) / (e_res_dry + e_floor)).log10();

    if !(6.0..=14.0).contains(&floor_db) {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 3B: Lehtonen sustain residual {floor_db:+.2} dB outside +6 to +14 dB range"
        ));
    } else {
        details.push(format!(
            "PASS Gate 3B: Lehtonen sustain background resonance gain {floor_db:+.2} dB (in +6..+14 dB)"
        ));
    }

    // --- Gate 3C: Keybed thump velocity monotonicity and pp subdued level <= -18 dB ---
    let mut prev_peak = 0.0_f32;
    let mut monotonic = true;
    for &vel in &[
        u7_to_u16(1),
        u7_to_u16(32),
        u7_to_u16(64),
        u7_to_u16(100),
        u16::MAX,
    ] {
        let mut eng = PianoEngine::new(SR);
        let mut out_l = vec![0.0_f32; 1536];
        let mut out_r = vec![0.0_f32; 1536];
        eng.render_block(&[note_on(60, vel)], &mut out_l, &mut out_r);
        let pk = peak_abs(&out_l).max(peak_abs(&out_r));
        if pk <= prev_peak {
            monotonic = false;
        }
        prev_peak = pk;
    }

    if !monotonic {
        all_pass = false;
        details
            .push("FAIL Gate 3C: Keybed thump peak force not monotonic across velocities".into());
    } else {
        details.push("PASS Gate 3C: Keybed thump peak force monotonic across velocities".into());
    }

    // Subdued check: vel=1 keybed thump vs A4 sustained mf tone
    let pp_peak = {
        let mut eng = PianoEngine::new(SR);
        let mut out_l = vec![0.0_f32; 2048];
        let mut out_r = vec![0.0_f32; 2048];
        eng.render_block(&[note_on(69, u7_to_u16(1))], &mut out_l, &mut out_r);
        peak_abs(&out_l).max(peak_abs(&out_r))
    };

    let a4_mf_peak = {
        let a4_mf = render_note_hold(69, u7_to_u16(80), 0, 0.5);
        let s50 = (0.05 * SR64).round() as usize;
        let s300 = (0.30 * SR64).round() as usize;
        peak_abs(&a4_mf[s50..s300])
    };

    let pp_ratio_db = 20.0 * f64::from(pp_peak / a4_mf_peak.max(1e-12)).log10();
    if pp_ratio_db > -18.0 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 3C: Keybed thump at pp too loud vs sustained mf A4: {pp_ratio_db:+.1} dB > -18.0 dB"
        ));
    } else {
        details.push(format!(
            "PASS Gate 3C: Keybed thump at pp subdued: {pp_ratio_db:+.1} dB <= -18.0 dB vs mf sustained A4"
        ));
    }

    let report = Challenge3Report {
        pass: all_pass,
        c3_two_stage_ratio: c3_ratio,
        a4_two_stage_ratio: a4_ratio,
        lehtonen_sustain_gain_db: floor_db,
        keybed_pp_vs_mf_ratio_db: pp_ratio_db,
        details,
    };

    if all_pass {
        Ok(report)
    } else {
        Err(format!(
            "Challenge 3 FAILED:\n{}",
            report.details.join("\n")
        ))
    }
}
