//! Challenge 2: 12 Hz Ribattuto Stulov Wool Memory Evolution.

use piano_engine::{PerfProfile, PianoEngine};
use piano_spec::{EventKind, SampleEvent, VoicingSpec};

use super::spectral_utils::{mono_mix, slice_rms, spectral_centroid_hz};

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;
const STRIKES: usize = 12;
const STRIKE_PERIOD: usize = 4_000; // 12 Hz @ 48 kHz (83.33 ms)

#[derive(Clone, Debug)]
pub struct Challenge2Report {
    pub pass: bool,
    pub a4_centroid_drift_pct: f64,
    pub a4_memory_divergence_pct: f64,
    pub c5_centroid_drift_pct: f64,
    pub c5_memory_divergence_pct: f64,
    pub details: Vec<String>,
}

fn render_chunks(
    engine: &mut PianoEngine,
    total_frames: usize,
    chunk_size: usize,
    evs: &[SampleEvent],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut pos = 0;
    while pos < total_frames {
        let n = (total_frames - pos).min(chunk_size);
        let cur_evs = if pos == 0 { evs } else { &[] };
        engine.render_block_bridge(cur_evs, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    (out_l, out_r)
}

fn run_series(midi_key: u8, memory_enabled: bool, profile: PerfProfile) -> (Vec<f64>, Vec<f64>) {
    let mut engine = PianoEngine::new_with_profile(SR, VoicingSpec::default(), profile);
    engine.set_felt_memory_enabled(memory_enabled);
    let mut rms_list = Vec::with_capacity(STRIKES);
    let mut cent_list = Vec::with_capacity(STRIKES);

    for _ in 0..STRIKES {
        let on = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key,
                velocity_u16: 52_000,
            },
        }];
        let (l_atk, r_atk) = render_chunks(&mut engine, 2_048, 128, &on);
        let mono = mono_mix(&l_atk, &r_atk);
        rms_list.push(slice_rms(&mono));
        cent_list.push(spectral_centroid_hz(&mono, SR64));

        let off = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOff {
                midi_key,
                release_vel_u16: 48_000,
            },
        }];
        let _ = render_chunks(&mut engine, STRIKE_PERIOD - 2_048, 128, &off);
    }
    (rms_list, cent_list)
}

/// Executes Challenge 2 ribattuto hysteresis memory verification on A4 and C5 under `profile`.
pub fn verify_challenge2_with_profile(profile: PerfProfile) -> Result<Challenge2Report, String> {
    let mut details = Vec::new();
    let mut all_pass = true;

    let mut drifts = Vec::new();
    let mut divs = Vec::new();

    for &midi_key in &[69_u8, 72_u8] {
        let note_name = if midi_key == 69 { "A4" } else { "C5" };
        let (rms_mem, cent_mem) = run_series(midi_key, true, profile);
        let (rms_nomem, cent_nomem) = run_series(midi_key, false, profile);

        // Gate 2A: Spectral centroid drift across 12 strikes > 0.5%
        let cent_drift = (cent_mem[STRIKES - 1] - cent_mem[0]).abs() / cent_mem[0].max(1.0);
        let drift_pct = cent_drift * 100.0;
        drifts.push(drift_pct);

        if cent_drift <= 0.005 {
            all_pass = false;
            details.push(format!(
                "FAIL Gate 2A ({note_name}): Spectral centroid drift {drift_pct:.2}% <= 0.5% threshold"
            ));
        } else {
            details.push(format!(
                "PASS Gate 2A ({note_name}): Spectral centroid drift {drift_pct:.2}% > 0.5% (initial={:.1} Hz -> strike12={:.1} Hz)",
                cent_mem[0], cent_mem[STRIKES - 1]
            ));
        }

        // Gate 2B: Stulov felt memory divergence from memoryless baseline > 3.0%
        let rms_diff =
            (rms_mem[STRIKES - 1] - rms_nomem[STRIKES - 1]).abs() / rms_nomem[STRIKES - 1];
        let cent_diff =
            (cent_mem[STRIKES - 1] - cent_nomem[STRIKES - 1]).abs() / cent_nomem[STRIKES - 1];
        let max_div = rms_diff.max(cent_diff);
        let max_div_pct = max_div * 100.0;
        divs.push(max_div_pct);

        if max_div <= 0.03 {
            all_pass = false;
            details.push(format!(
                "FAIL Gate 2B ({note_name}): Memory divergence {max_div_pct:.2}% <= 3.0% threshold (rms_diff={:.2}%, cent_diff={:.2}%)",
                rms_diff * 100.0, cent_diff * 100.0
            ));
        } else {
            details.push(format!(
                "PASS Gate 2B ({note_name}): Stulov memory divergence {max_div_pct:.2}% > 3.0% (rms_diff={:.2}%, cent_diff={:.2}%)",
                rms_diff * 100.0, cent_diff * 100.0
            ));
        }
    }

    let report = Challenge2Report {
        pass: all_pass,
        a4_centroid_drift_pct: drifts[0],
        a4_memory_divergence_pct: divs[0],
        c5_centroid_drift_pct: drifts[1],
        c5_memory_divergence_pct: divs[1],
        details,
    };

    if all_pass {
        Ok(report)
    } else {
        Err(format!(
            "Challenge 2 FAILED:\n{}",
            report.details.join("\n")
        ))
    }
}
