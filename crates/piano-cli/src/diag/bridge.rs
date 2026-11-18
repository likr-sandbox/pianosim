//! `diag-bridge`: Fixed-End vs WDF bridge two-stage decay & Lehtonen resonance comparison.
//!
//! Note: `PianoEngine::set_bridge_fixed_end(bool)` lands during Wave 2 integration.
//! All two-stage break-level, Lehtonen off-partial sympathetic gain, WAV export, and
//! Fixed-vs-WDF comparison logic is complete here, with explicit Wave 2 hook comments.

use std::fs;
use std::path::PathBuf;

use piano_engine::PianoEngine;
use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, u7_to_u16, EventKind,
    PedalKind, SampleEvent, VoicingSpec,
};

use crate::scenarios::spectral_utils::{
    decay_slope_db_per_s, hann_windowed_spectrum, mono_mix, slice_rms,
};
use crate::wav_writer::write_stereo_f32_wav;

use super::parse_flag_value;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;
const SR_U32: u32 = 48_000;

#[derive(Clone, Debug, PartialEq)]
pub struct TwoStageMetrics {
    pub prompt_slope_db_per_s: f64,
    pub late_slope_db_per_s: f64,
    pub ratio: f64,
    pub break_level_db: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyBridgeMetrics {
    pub label: String,
    pub wdf_two_stage: TwoStageMetrics,
    pub fixed_two_stage: Option<TwoStageMetrics>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BridgeComparisonReport {
    pub c2: KeyBridgeMetrics,
    pub c4: KeyBridgeMetrics,
    pub wdf_sympathetic_gain_db: f64,
    pub fixed_sympathetic_gain_db: Option<f64>,
}

struct BridgeDiagOptions {
    out_dir: Option<PathBuf>,
    keys: Vec<(String, u8)>,
}

fn parse_key_token(tok: &str) -> Result<(String, u8), String> {
    match tok.trim().to_ascii_uppercase().as_str() {
        "C2" => Ok((String::from("c2"), 36)),
        "C3" => Ok((String::from("c3"), 48)),
        "C4" => Ok((String::from("c4"), 60)),
        "A4" => Ok((String::from("a4"), 69)),
        other => Err(format!(
            "unsupported key '{other}' in --keys (expected C2, C3, C4, or A4)"
        )),
    }
}

fn parse_options(args: &[String]) -> Result<BridgeDiagOptions, String> {
    let usage = super::diag_usage();
    let mut out_dir: Option<PathBuf> = None;
    let mut keys = vec![(String::from("c2"), 36_u8), (String::from("c4"), 60_u8)];

    let mut i = 0;
    while i < args.len() {
        if let Some(v) = parse_flag_value(args, &mut i, "--keys", usage)? {
            let mut parsed = Vec::new();
            for tok in v.split(',') {
                if !tok.trim().is_empty() {
                    parsed.push(parse_key_token(tok)?);
                }
            }
            if parsed.is_empty() {
                return Err(String::from("--keys requires at least one key"));
            }
            keys = parsed;
        } else if args[i].starts_with('-') {
            return Err(String::from(usage));
        } else if out_dir.is_none() {
            out_dir = Some(PathBuf::from(&args[i]));
            i += 1;
        } else {
            return Err(String::from(usage));
        }
    }
    Ok(BridgeDiagOptions { out_dir, keys })
}

/// Measures prompt (`0.05..0.45 s`) vs late (`1.40..3.60 s`) decay slopes and initial break level.
#[must_use]
pub fn analyze_two_stage_decay(mono: &[f32], sr: f64) -> TwoStageMetrics {
    let prompt = decay_slope_db_per_s(mono, sr, 0.05, 0.45, 0.05);
    let late = decay_slope_db_per_s(mono, sr, 1.40, 3.60, 0.10);
    let ratio = prompt / late.max(1.0e-9);

    let s_early0 = (0.05 * sr).round() as usize;
    let s_early1 = ((0.10 * sr).round() as usize).min(mono.len());
    let s_knee0 = (0.45 * sr).round() as usize;
    let s_knee1 = ((0.65 * sr).round() as usize).min(mono.len());
    let early_rms = if s_early1 > s_early0 {
        slice_rms(&mono[s_early0..s_early1])
    } else {
        1.0e-12
    };
    let knee_rms = if s_knee1 > s_knee0 {
        slice_rms(&mono[s_knee0..s_knee1])
    } else {
        1.0e-12
    };
    let break_level_db = 20.0 * (early_rms.max(1.0e-12) / knee_rms.max(1.0e-12)).log10();

    TwoStageMetrics {
        prompt_slope_db_per_s: prompt,
        late_slope_db_per_s: late,
        ratio,
        break_level_db,
    }
}

/// Computes Lehtonen harmonic-notched off-partial sympathetic resonance gain (`1.0..3.7 s`).
#[must_use]
pub fn analyze_lehtonen_sympathetic_gain_db(
    dry_mono: &[f32],
    sus_mono: &[f32],
    midi_key: u8,
    sr: f64,
) -> f64 {
    let n_fft = 16_384_usize;
    let hop = 8_192_usize;
    let spec_len = n_fft / 2 + 1;
    let bin_hz = sr / (n_fft as f64);

    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let k = midi_to_key(midi_key).unwrap_or(39);
    let f0 = key_tuned_f0_hz(k, &specs, &voicing);
    let b = specs[k].inharmonicity_b;

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

    let s1 = (1.0 * sr).round() as usize;
    let s_end = ((3.7 * sr).round() as usize)
        .min(dry_mono.len())
        .min(sus_mono.len());
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

    let (e_tot_dry, e_res_dry) = sum_bins(dry_mono);
    let (_e_tot_sus, e_res_sus) = sum_bins(sus_mono);
    let e_floor = 1.0e-3 * e_tot_dry;
    10.0 * ((e_res_sus + e_floor) / (e_res_dry + e_floor)).log10()
}

/// Builds a comparative report from rendered WDF and optional Fixed-End audio buffers.
#[must_use]
pub fn compare_fixed_vs_wdf(
    c2_wdf: &[f32],
    c2_fixed: Option<&[f32]>,
    c4_dry_wdf: &[f32],
    c4_sus_wdf: &[f32],
    c4_dry_fixed: Option<&[f32]>,
    c4_sus_fixed: Option<&[f32]>,
    sr: f64,
) -> BridgeComparisonReport {
    let wdf_symp = analyze_lehtonen_sympathetic_gain_db(c4_dry_wdf, c4_sus_wdf, 60, sr);
    let fixed_symp = match (c4_dry_fixed, c4_sus_fixed) {
        (Some(d), Some(s)) => Some(analyze_lehtonen_sympathetic_gain_db(d, s, 60, sr)),
        _ => None,
    };
    BridgeComparisonReport {
        c2: KeyBridgeMetrics {
            label: String::from("C2"),
            wdf_two_stage: analyze_two_stage_decay(c2_wdf, sr),
            fixed_two_stage: c2_fixed.map(|buf| analyze_two_stage_decay(buf, sr)),
        },
        c4: KeyBridgeMetrics {
            label: String::from("C4"),
            wdf_two_stage: analyze_two_stage_decay(c4_dry_wdf, sr),
            fixed_two_stage: c4_dry_fixed.map(|buf| analyze_two_stage_decay(buf, sr)),
        },
        wdf_sympathetic_gain_db: wdf_symp,
        fixed_sympathetic_gain_db: fixed_symp,
    }
}

fn render_note_stereo(
    midi_key: u8,
    vel_u16: u16,
    sustain_u16: u16,
    secs: f64,
    fixed_end: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut engine = PianoEngine::new(SR);
    engine.set_bridge_fixed_end(fixed_end);

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
        engine.render_block(cur, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
        pos += n;
    }
    (out_l, out_r)
}

/// Executes `diag-bridge` WDF vs Fixed-End analysis and optional WAV dump.
pub fn cmd_diag_bridge(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    if let Some(ref dir) = opts.out_dir {
        fs::create_dir_all(dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }

    let vel_mf = u7_to_u16(64);
    println!("diag-bridge [sr={SR_U32}, vel_mf={vel_mf}, duration=3.8s]");
    let mut c2_wdf_buf: Option<Vec<f32>> = None;
    let mut c2_fixed_buf: Option<Vec<f32>> = None;
    let mut c4_dry_wdf_buf: Option<Vec<f32>> = None;
    let mut c4_sus_wdf_buf: Option<Vec<f32>> = None;
    let mut c4_dry_fixed_buf: Option<Vec<f32>> = None;
    let mut c4_sus_fixed_buf: Option<Vec<f32>> = None;

    for (slug, midi) in &opts.keys {
        let (wl, wr) = render_note_stereo(*midi, vel_mf, 0, 3.8, false);
        let (fl, fr) = render_note_stereo(*midi, vel_mf, 0, 3.8, true);
        if let Some(ref dir) = opts.out_dir {
            write_stereo_f32_wav(&dir.join(format!("{slug}_wdf.wav")), SR_U32, &wl, &wr)?;
            write_stereo_f32_wav(&dir.join(format!("{slug}_fixed.wav")), SR_U32, &fl, &fr)?;
        }
        let wdf_mono = mono_mix(&wl, &wr);
        let fixed_mono = mono_mix(&fl, &fr);
        let ts = analyze_two_stage_decay(&wdf_mono, SR64);
        let ts_fix = analyze_two_stage_decay(&fixed_mono, SR64);
        println!(
            "  {} (midi={midi}) WDF: prompt={:.2} dB/s, late={:.2} dB/s, ratio={:.2}x (>= 2.0x), break_level={:.2} dB (target 10..15 dB) | Fixed: ratio={:.2}x",
            slug.to_ascii_uppercase(),
            ts.prompt_slope_db_per_s,
            ts.late_slope_db_per_s,
            ts.ratio,
            ts.break_level_db,
            ts_fix.ratio
        );

        if *midi == 36 {
            c2_wdf_buf = Some(wdf_mono);
            c2_fixed_buf = Some(fixed_mono);
        } else if *midi == 60 {
            let (sl, sr_buf) = render_note_stereo(60, vel_mf, u16::MAX, 3.8, false);
            let (sfl, sfr_buf) = render_note_stereo(60, vel_mf, u16::MAX, 3.8, true);
            if let Some(ref dir) = opts.out_dir {
                write_stereo_f32_wav(&dir.join("c4_sus_wdf.wav"), SR_U32, &sl, &sr_buf)?;
                write_stereo_f32_wav(&dir.join("c4_sus_fixed.wav"), SR_U32, &sfl, &sfr_buf)?;
            }
            let sus_mono = mono_mix(&sl, &sr_buf);
            let sus_fixed_mono = mono_mix(&sfl, &sfr_buf);
            c4_dry_wdf_buf = Some(wdf_mono);
            c4_sus_wdf_buf = Some(sus_mono);
            c4_dry_fixed_buf = Some(fixed_mono);
            c4_sus_fixed_buf = Some(sus_fixed_mono);
        }
    }

    if let (Some(c2_w), Some(c4_d), Some(c4_s)) = (
        c2_wdf_buf.as_deref(),
        c4_dry_wdf_buf.as_deref(),
        c4_sus_wdf_buf.as_deref(),
    ) {
        let report = compare_fixed_vs_wdf(
            c2_w,
            c2_fixed_buf.as_deref(),
            c4_d,
            c4_s,
            c4_dry_fixed_buf.as_deref(),
            c4_sus_fixed_buf.as_deref(),
            SR64,
        );
        println!(
            "  {} vs {} WDF summary: C2 ratio={:.2}x, C4 ratio={:.2}x, C4 Lehtonen sympathetic={:+.2} dB (target +6.0..+14.0 dB)",
            report.c2.label,
            report.c4.label,
            report.c2.wdf_two_stage.ratio,
            report.c4.wdf_two_stage.ratio,
            report.wdf_sympathetic_gain_db
        );
        if let (Some(c2_f), Some(c4_f), Some(symp_f)) = (
            report.c2.fixed_two_stage,
            report.c4.fixed_two_stage,
            report.fixed_sympathetic_gain_db,
        ) {
            println!(
                "  Fixed-End summary: C2 ratio={:.2}x, C4 ratio={:.2}x, C4 sympathetic={symp_f:+.2} dB",
                c2_f.ratio, c4_f.ratio
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_two_stage_decay_detects_break_level_and_slope_ratio() {
        let sr = 48_000.0;
        let len = (3.8 * sr) as usize;
        let mut sig = vec![0.0_f32; len];
        // Two-stage envelope: fast mode (~24 dB/s, tau=0.362 s) + slow mode (~2 dB/s, tau=4.34 s)
        // scaled so 0.05..0.10 s -> 0.45..0.65 s drops ~12 dB and prompt/late ratio > 3x.
        for (n, s) in sig.iter_mut().enumerate() {
            let t = (n as f64) / sr;
            let env = 0.88 * (-t / 0.17).exp() + 0.12 * (-t / 4.5).exp();
            *s = (env * (2.0 * std::f64::consts::PI * 261.63 * t).sin()) as f32;
        }
        let m = analyze_two_stage_decay(&sig, sr);
        assert!(
            m.ratio >= 2.5,
            "expected clear two-stage ratio >= 2.5x, got {:.2}",
            m.ratio
        );
        assert!(
            (9.0..=16.0).contains(&m.break_level_db),
            "expected ~12 dB break level, got {:.2}",
            m.break_level_db
        );
    }

    #[test]
    fn compare_fixed_vs_wdf_measures_lehtonen_off_partial_sympathetic_gain() {
        let sr = 48_000.0;
        let len = (3.8 * sr) as usize;
        let mut dry = vec![0.0_f32; len];
        let mut sus = vec![0.0_f32; len];
        let f0 = 261.6256;
        for n in 0..len {
            let t = (n as f64) / sr;
            let main_tone =
                (0.4 * (-t / 1.8).exp() * (2.0 * std::f64::consts::PI * f0 * t).sin()) as f32;
            dry[n] = main_tone;
            // Off-partial sympathetic shimmer at 698.46 Hz (F5) and 1174.66 Hz (D6).
            let symp = 0.035
                * (-t / 2.4).exp()
                * ((2.0 * std::f64::consts::PI * 698.46 * t).sin()
                    + (2.0 * std::f64::consts::PI * 1_174.66 * t).sin());
            sus[n] = main_tone + (symp as f32);
        }
        let report = compare_fixed_vs_wdf(&dry, Some(&dry), &dry, &sus, Some(&dry), Some(&dry), sr);
        assert!(
            report.wdf_sympathetic_gain_db > 5.0,
            "expected positive WDF sympathetic gain, got {:.2}",
            report.wdf_sympathetic_gain_db
        );
        assert!(
            report.fixed_sympathetic_gain_db.unwrap().abs() < 0.2,
            "identical fixed dry/sus should give ~0 dB gain, got {:.2}",
            report.fixed_sympathetic_gain_db.unwrap()
        );
    }
}
