//! Challenge 1: Strike Chirp, Longitudinal Precursor & Zero-Delay Radiation Integration.

use piano_engine::PianoEngine;
use piano_radiator::{RadiatorBank, PRESET_RADIATOR_ZONES};
use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, EventKind, SampleEvent,
    VoicingSpec, NUM_KEYS,
};

use super::spectral_utils::{measure_f0_hz, mono_mix, slice_rms};

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

#[derive(Clone, Debug)]
pub struct Challenge1Report {
    pub pass: bool,
    pub tau_l_ms: f64,
    pub tau_t_ms: f64,
    pub precursor_snr_db: f64,
    pub chirp_cents: f64,
    pub fir_latency_samples: usize,
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

fn render_timeline(engine: &mut PianoEngine, total_frames: usize, evs: &[SampleEvent]) -> Vec<f32> {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut pos = 0;
    while pos < total_frames {
        let n = (total_frames - pos).min(128);
        let cur_evs = if pos == 0 { evs } else { &[] };
        engine.render_block_bridge(cur_evs, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    mono_mix(&out_l, &out_r)
}

/// Executes all Challenge 1 verification gates.
pub fn verify_challenge1() -> Result<Challenge1Report, String> {
    let mut details = Vec::new();
    let mut all_pass = true;

    // --- Gate 1A: Bass strike longitudinal precursor arrival (tau_L < tau_T, SNR >= 12 dB) ---
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let key_a0 = midi_to_key(21).expect("A0");
    let x_a0 = (key_a0 as f64) / ((NUM_KEYS - 1) as f64);
    let beta_k = 0.137 - 0.042 * x_a0;
    let tau_l_s = (1.0 - beta_k) / (2.0 * specs[key_a0].f_long_hz);
    let tau_t_s = (1.0 - beta_k) / (2.0 * specs[key_a0].f0_et_hz);
    let tau_l_ms = tau_l_s * 1000.0;
    let tau_t_ms = tau_t_s * 1000.0;

    if tau_l_ms >= tau_t_ms {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 1A: tau_L ({tau_l_ms:.2} ms) >= tau_T ({tau_t_ms:.2} ms)"
        ));
    } else {
        details.push(format!(
            "PASS Gate 1A: tau_L ({tau_l_ms:.2} ms) < tau_T ({tau_t_ms:.2} ms)"
        ));
    }

    let mono_a0 = {
        let mut engine = PianoEngine::new(SR);
        engine.set_nonlinear_coupling(true, true);
        render_timeline(&mut engine, 2048, &[note_on(21, 60_000)])
    };

    let pre_end = ((0.60 * tau_l_s * SR64).floor() as usize).clamp(4, 36);
    let win_lo = (0.0008 * SR64).round() as usize; // 0.8 ms
    let win_hi = (0.0055 * SR64).round() as usize; // 5.5 ms
    let rms_pre = slice_rms(&mono_a0[..pre_end]).max(1e-12);
    let rms_early = slice_rms(&mono_a0[win_lo..win_hi]);
    let snr_db = 20.0 * (rms_early / rms_pre).log10();

    if snr_db < 12.0 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 1A: Precursor SNR {snr_db:.2} dB < 12.0 dB threshold"
        ));
    } else {
        details.push(format!(
            "PASS Gate 1A: Precursor SNR {snr_db:.2} dB >= 12.0 dB"
        ));
    }

    // --- Gate 1B: Strike chirp (> 2.0 cents dynamic pitch glide in early 40 ms) ---
    let mut best_chirp = 0.0_f64;
    let mut best_glide = 0.0_f64;
    let mut best_dyn = 0.0_f64;
    let mut best_key = 36_u8;

    for &test_key in &[21_u8, 24, 28, 33, 36, 40, 45, 48] {
        let key_idx = midi_to_key(test_key).expect("valid key");
        let f_nom = partial_frequency_hz(
            key_tuned_f0_hz(key_idx, &specs, &voicing),
            specs[key_idx].inharmonicity_b,
            1,
        );

        let mono_ff = {
            let mut eng = PianoEngine::new(SR);
            render_timeline(&mut eng, 49_152, &[note_on(test_key, u16::MAX)])
        };
        let mono_pp = {
            let mut eng = PianoEngine::new(SR);
            render_timeline(&mut eng, 17_152, &[note_on(test_key, 6_000)])
        };

        // Early window: >= 40 ms and >= 2.2 fundamental cycles so bass strikes span multiple
        // bridge reflections rather than truncating to a single pre-period pulse.
        let early_start = (0.005 * SR64).round() as usize; // 5 ms post-strike
        let early_len = (((2.2 / f_nom) * SR64).max(0.040 * SR64).round() as usize)
            .min(mono_pp.len().saturating_sub(early_start));
        let f_early = measure_f0_hz(&mono_ff[early_start..early_start + early_len], f_nom, SR64);
        let f_late = measure_f0_hz(&mono_ff[32_768..49_152], f_nom, SR64);
        let f_pp = measure_f0_hz(&mono_pp[early_start..early_start + early_len], f_nom, SR64);

        let glide = 1200.0 * (f_early / f_late).log2();
        let dyn_glide = 1200.0 * (f_early / f_pp).log2();
        let max_g = glide.max(dyn_glide);
        if max_g > best_chirp {
            best_chirp = max_g;
            best_glide = glide;
            best_dyn = dyn_glide;
            best_key = test_key;
        }
        if best_chirp > 2.0 {
            break;
        }
    }

    if best_chirp < 2.0 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 1B: Key {best_key} strike chirp {best_chirp:.2}c < 2.0c threshold"
        ));
    } else {
        details.push(format!(
            "PASS Gate 1B: Key {best_key} strike chirp {best_chirp:.2}c >= 2.0c (decay={best_glide:+.2}c, dyn={best_dyn:+.2}c)"
        ));
    }

    // --- Gate 1C: Zero-delay body initial integration in Stage 4 direct FIR head ---
    let bank = RadiatorBank::new_concert_grand(SR64);
    let latency = bank.algorithmic_latency_samples();
    if latency != 0 {
        all_pass = false;
        details.push(format!(
            "FAIL Gate 1C: Stage 4 FIR head reported latency {latency} > 0"
        ));
    } else {
        details.push("PASS Gate 1C: Stage 4 FIR head strictly 0-sample algorithmic delay".into());
    }

    // Impulse verification: non-zero starting at sample 0 without prepending delay
    let mut rad_test = RadiatorBank::new_concert_grand(SR64);
    let mut in_frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 64];
    in_frames[0][0] = 1.0; // impulse at sample 0
    let mut out_l = vec![0.0_f32; 64];
    let mut out_r = vec![0.0_f32; 64];
    rad_test.process_block(&in_frames, &mut out_l, &mut out_r);

    if out_l[0] == 0.0 && out_r[0] == 0.0 {
        all_pass = false;
        details.push("FAIL Gate 1C: Impulse response at t=0 was zero".into());
    } else {
        details.push("PASS Gate 1C: Direct radiation impulse response active at sample 0".into());
    }

    let report = Challenge1Report {
        pass: all_pass,
        tau_l_ms,
        tau_t_ms,
        precursor_snr_db: snr_db,
        chirp_cents: best_chirp,
        fir_latency_samples: latency,
        details,
    };

    if all_pass {
        Ok(report)
    } else {
        Err(format!(
            "Challenge 1 FAILED:\n{}",
            report.details.join("\n")
        ))
    }
}
