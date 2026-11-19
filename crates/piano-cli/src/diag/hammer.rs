//! `diag-hammer`: C4 `mf` hammer force waveform $F_h(t)$ CSV dump & agraffe peak detection.
//!
//! Note: `PianoEngine::last_hammer_forces(midi_key)` lands during Wave 2 integration.
//! All contact-duration, peak-force, impulse, agraffe-reflection local-maxima detection,
//! and CSV I/O logic is complete here, with explicit Wave 2 hook comments and `--in-csv` support.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use piano_engine::{PerfProfile, PianoEngine};
use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, u7_to_u16, EventKind, SampleEvent,
    VoicingSpec, NUM_KEYS,
};

use super::parse_flag_value;
use crate::cli_profile::CliProfileArgs;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;
const SUB_STEPS: usize = 8;

#[derive(Clone, Debug)]
pub struct HammerForceTrace {
    pub dt_sub_ms: f64,
    pub host_samples: Vec<(f64, f32)>,
    pub sub_samples: Vec<(f64, f32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HammerPeak {
    pub time_ms: f64,
    pub force_n: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HammerForceMetrics {
    pub peak_force_n: f32,
    pub contact_duration_ms: f64,
    pub host_contact_duration_ms: f64,
    pub impulse_ns: f64,
    pub peaks: Vec<HammerPeak>,
    pub mean_inter_peak_ms: Option<f64>,
    pub tau_agg_ms: f64,
}

struct HammerDiagOptions {
    out_csv: Option<PathBuf>,
    in_csv: Option<PathBuf>,
    midi: u8,
    vel_u7: u8,
    duration_ms: f64,
    profile: PerfProfile,
    profile_label: String,
}

fn parse_options(args: &[String]) -> Result<HammerDiagOptions, String> {
    let usage = super::diag_usage();
    let mut out_csv: Option<PathBuf> = None;
    let mut in_csv: Option<PathBuf> = None;
    let mut midi = 60_u8;
    let mut vel_u7 = 64_u8;
    let mut duration_ms = 10.0_f64;
    let mut cli_profile = CliProfileArgs::default();

    let mut i = 0;
    while i < args.len() {
        if cli_profile.try_consume_arg(args, &mut i)? {
            continue;
        }
        if let Some(v) = parse_flag_value(args, &mut i, "--in-csv", usage)? {
            in_csv = Some(PathBuf::from(v));
        } else if let Some(v) = parse_flag_value(args, &mut i, "--midi", usage)? {
            midi = v
                .parse::<u8>()
                .map_err(|_| format!("invalid --midi '{v}'"))?;
            if midi_to_key(midi).is_none() {
                return Err(format!("--midi {midi} out of 88-key range [21..=108]"));
            }
        } else if let Some(v) = parse_flag_value(args, &mut i, "--vel", usage)? {
            vel_u7 = v
                .parse::<u8>()
                .map_err(|_| format!("invalid --vel '{v}'"))?
                .clamp(1, 127);
        } else if let Some(v) = parse_flag_value(args, &mut i, "--duration-ms", usage)? {
            duration_ms = v
                .parse::<f64>()
                .map_err(|_| format!("invalid --duration-ms '{v}'"))?;
            if !duration_ms.is_finite() || !(1.0..=100.0).contains(&duration_ms) {
                return Err(format!("--duration-ms {v} out of range [1.0, 100.0]"));
            }
        } else if args[i].starts_with('-') {
            return Err(String::from(usage));
        } else if out_csv.is_none() {
            out_csv = Some(PathBuf::from(&args[i]));
            i += 1;
        } else {
            return Err(String::from(usage));
        }
    }
    Ok(HammerDiagOptions {
        out_csv,
        in_csv,
        midi,
        vel_u7,
        duration_ms,
        profile: cli_profile.resolve(),
        profile_label: cli_profile.label(),
    })
}

/// Computes expected agraffe round-trip delay `tau_agg = beta / f0` in milliseconds.
#[must_use]
pub fn key_tau_agg_ms(midi_key: u8) -> f64 {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let k = midi_to_key(midi_key).unwrap_or(39);
    let x = (k as f64) / ((NUM_KEYS - 1) as f64);
    let beta = 0.138 - 0.043 * x;
    let f0 = key_tuned_f0_hz(k, &specs, &voicing);
    1_000.0 * beta / f0.max(1.0)
}

/// Measures contact duration, peak force, impulse, and agraffe-reflection local peaks.
#[must_use]
pub fn analyze_hammer_force(trace: &HammerForceTrace, tau_agg_ms: f64) -> HammerForceMetrics {
    let peak_force_n = trace
        .sub_samples
        .iter()
        .map(|&(_, f)| f)
        .fold(0.0_f32, f32::max);
    let dt_sub_s = trace.dt_sub_ms * 1.0e-3;
    let impulse_ns: f64 = trace
        .sub_samples
        .iter()
        .map(|&(_, f)| f64::from(f.max(0.0)) * dt_sub_s)
        .sum();

    let contact_duration_ms = span_above_thresh(&trace.sub_samples, peak_force_n * 0.01);
    let host_peak = trace
        .host_samples
        .iter()
        .map(|&(_, f)| f)
        .fold(0.0_f32, f32::max);
    let host_contact_duration_ms = span_above_thresh(&trace.host_samples, host_peak * 0.01);

    let peaks = detect_prominent_peaks(&trace.sub_samples, peak_force_n, 0.15, 0.03);
    let mean_inter_peak_ms = if peaks.len() >= 2 {
        let span = peaks.last().unwrap().time_ms - peaks.first().unwrap().time_ms;
        Some(span / ((peaks.len() - 1) as f64))
    } else {
        None
    };

    HammerForceMetrics {
        peak_force_n,
        contact_duration_ms,
        host_contact_duration_ms,
        impulse_ns,
        peaks,
        mean_inter_peak_ms,
        tau_agg_ms,
    }
}

fn span_above_thresh(samples: &[(f64, f32)], thresh: f32) -> f64 {
    if thresh <= 0.0 {
        return 0.0;
    }
    let first = samples.iter().position(|&(_, f)| f > thresh);
    let last = samples.iter().rposition(|&(_, f)| f > thresh);
    match (first, last) {
        (Some(i0), Some(i1)) if i1 >= i0 => {
            let dt = if samples.len() >= 2 {
                (samples[1].0 - samples[0].0).abs()
            } else {
                0.0
            };
            (samples[i1].0 - samples[i0].0) + dt
        }
        _ => 0.0,
    }
}

/// Detects local maxima with minimum separation `min_sep_ms` and relative prominence `min_prom_rel`.
#[must_use]
pub fn detect_prominent_peaks(
    samples: &[(f64, f32)],
    global_peak: f32,
    min_sep_ms: f64,
    min_prom_rel: f32,
) -> Vec<HammerPeak> {
    if samples.len() < 3 || global_peak <= 0.0 {
        return Vec::new();
    }
    let min_prom = global_peak * min_prom_rel;
    let min_height = global_peak * 0.10;
    let win_samples =
        ((min_sep_ms / (samples[1].0 - samples[0].0).abs().max(1e-6)).round() as usize).max(2);

    let mut candidates = Vec::new();
    for i in 1..(samples.len() - 1) {
        let f = samples[i].1;
        if f >= min_height && f >= samples[i - 1].1 && f > samples[i + 1].1 {
            let lo = i.saturating_sub(win_samples);
            let hi = (i + win_samples).min(samples.len() - 1);
            let left_min = samples[lo..i].iter().map(|x| x.1).fold(f, f32::min);
            let right_min = samples[i + 1..=hi].iter().map(|x| x.1).fold(f, f32::min);
            let prominence = (f - left_min).min(f - right_min);
            if prominence >= min_prom {
                candidates.push(HammerPeak {
                    time_ms: samples[i].0,
                    force_n: f,
                });
            }
        }
    }

    // Enforce minimum temporal separation by keeping the taller peak in any collision.
    let mut filtered: Vec<HammerPeak> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        if let Some(prev) = filtered.last_mut() {
            if cand.time_ms - prev.time_ms < min_sep_ms {
                if cand.force_n > prev.force_n {
                    *prev = cand;
                }
                continue;
            }
        }
        filtered.push(cand);
    }
    filtered
}

/// Writes host + 8x substep hammer force rows to CSV.
pub fn write_hammer_csv(
    path: &Path,
    rows: &[(usize, f64, f32, [f32; SUB_STEPS])],
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
    }
    let file =
        File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "sample,t_ms,f_host_n,sub0,sub1,sub2,sub3,sub4,sub5,sub6,sub7"
    )
    .map_err(|e| format!("failed writing CSV header: {e}"))?;
    for &(sample, t_ms, f_host, sub) in rows {
        writeln!(
            w,
            "{sample},{t_ms:.6},{f_host:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7]
        )
        .map_err(|e| format!("failed writing CSV row: {e}"))?;
    }
    w.flush().map_err(|e| format!("failed flushing CSV: {e}"))
}

/// Parses a previously dumped `diag-hammer` CSV into a `HammerForceTrace`.
pub fn read_hammer_csv(path: &Path) -> Result<HammerForceTrace, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read CSV '{}': {e}", path.display()))?;
    let mut rows = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        if line_no == 0 && line.starts_with("sample,") {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 11 {
            return Err(format!(
                "invalid hammer CSV row {}: expected 11 columns, got {}",
                line_no + 1,
                parts.len()
            ));
        }
        let t_ms: f64 = parts[1]
            .parse()
            .map_err(|_| format!("invalid t_ms at line {}", line_no + 1))?;
        let f_host: f32 = parts[2]
            .parse()
            .map_err(|_| format!("invalid f_host_n at line {}", line_no + 1))?;
        let mut sub = [0.0_f32; SUB_STEPS];
        for (s_idx, cell) in parts[3..11].iter().enumerate() {
            sub[s_idx] = cell
                .parse()
                .map_err(|_| format!("invalid sub{s_idx} at line {}", line_no + 1))?;
        }
        rows.push((t_ms, f_host, sub));
    }
    Ok(trace_from_rows(&rows, SR64, SUB_STEPS))
}

fn trace_from_rows(
    rows: &[(f64, f32, [f32; SUB_STEPS])],
    sr: f64,
    oversample: usize,
) -> HammerForceTrace {
    let os = oversample.clamp(1, SUB_STEPS);
    let dt_host_ms = 1_000.0 / sr;
    let dt_sub_ms = dt_host_ms / (os as f64);
    let mut host_samples = Vec::with_capacity(rows.len());
    let mut sub_samples = Vec::with_capacity(rows.len() * os);
    for &(t_ms, f_host, sub) in rows {
        host_samples.push((t_ms, f_host));
        for (k, &f_sub) in sub[..os].iter().enumerate() {
            sub_samples.push((t_ms + (k as f64) * dt_sub_ms, f_sub));
        }
    }
    HammerForceTrace {
        dt_sub_ms,
        host_samples,
        sub_samples,
    }
}

/// Executes `diag-hammer` CSV dump / analysis.
pub fn cmd_diag_hammer(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let tau_agg_ms = key_tau_agg_ms(opts.midi);

    if let Some(ref csv_path) = opts.in_csv {
        let trace = read_hammer_csv(csv_path)?;
        let m = analyze_hammer_force(&trace, tau_agg_ms);
        print_hammer_report(opts.midi, opts.vel_u7, "csv", &m);
        return Ok(());
    }

    let total_samples = ((opts.duration_ms * 1.0e-3 * SR64).round() as usize).max(1);
    let mut engine = PianoEngine::new_with_profile(SR, VoicingSpec::default(), opts.profile);
    let ev = [SampleEvent {
        sample_offset: 0,
        kind: EventKind::NoteOn {
            midi_key: opts.midi,
            velocity_u16: u7_to_u16(opts.vel_u7),
        },
    }];

    let mut rows = Vec::with_capacity(total_samples);
    let mut l = [0.0_f32; 1];
    let mut r = [0.0_f32; 1];
    let mut any_nonzero = false;

    for n in 0..total_samples {
        let cur = if n == 0 { &ev[..] } else { &[] };
        engine.render_block(cur, &mut l, &mut r);
        let (f_host_n, sub_forces) = engine.last_hammer_forces(opts.midi);
        if f_host_n > 0.0 || sub_forces.iter().any(|&x| x > 0.0) {
            any_nonzero = true;
        }
        let t_ms = (n as f64) * 1_000.0 / SR64;
        rows.push((n, t_ms, f_host_n, sub_forces));
    }

    if !any_nonzero {
        println!(
            "diag-hammer [midi={}, vel={}, perf={}, tau_agg={tau_agg_ms:.3} ms]: [PENDING Wave 2 hook `engine.last_hammer_forces(midi)` in `diag/hammer.rs`]",
            opts.midi, opts.vel_u7, opts.profile_label
        );
        return Ok(());
    }

    if let Some(ref csv_path) = opts.out_csv {
        write_hammer_csv(csv_path, &rows)?;
    }
    let trace_rows: Vec<_> = rows.iter().map(|&(_, t, fh, s)| (t, fh, s)).collect();
    let os = usize::from(opts.profile.action_oversample);
    let trace = trace_from_rows(&trace_rows, SR64, os);
    let m = analyze_hammer_force(&trace, tau_agg_ms);
    print_hammer_report(opts.midi, opts.vel_u7, &opts.profile_label, &m);
    Ok(())
}

fn print_hammer_report(midi: u8, vel_u7: u8, profile_label: &str, m: &HammerForceMetrics) {
    println!(
        "diag-hammer [midi={midi}, vel={vel_u7}, perf={profile_label}]: peak={:.2} N, contact_sub={:.3} ms (host={:.3} ms, target 1.5..2.5 ms), impulse={:.5} N*s, agraffe_peaks={} (target 2..3), tau_agg={:.3} ms, mean_dt_peak={}",
        m.peak_force_n,
        m.contact_duration_ms,
        m.host_contact_duration_ms,
        m.impulse_ns,
        m.peaks.len(),
        m.tau_agg_ms,
        m.mean_inter_peak_ms
            .map_or_else(|| String::from("n/a"), |v| format!("{v:.3} ms"))
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn analyze_hammer_force_detects_agraffe_peaks_duration_and_impulse() {
        let sr = 48_000.0;
        let tau_agg_ms = key_tau_agg_ms(60);
        let n_host = 480_usize; // 10 ms
        let mut rows = Vec::with_capacity(n_host);
        for n in 0..n_host {
            let t_host = (n as f64) * 1_000.0 / sr;
            let mut sub = [0.0_f32; SUB_STEPS];
            for (k, slot) in sub.iter_mut().enumerate() {
                let t = t_host + (k as f64) * (1_000.0 / (sr * 8.0));
                if (0.2..=2.1).contains(&t) {
                    let phase = (t - 0.2) / 1.9;
                    let base = 28.0 * (std::f64::consts::PI * phase).sin();
                    // Add damped agraffe-reflection ripples spaced by `tau_agg_ms` (2-3 rebounds).
                    let ripple_env = (-((t - 0.5) / 0.8).max(0.0)).exp();
                    let ripple = 5.5
                        * ripple_env
                        * (2.0 * std::f64::consts::PI * (t - 0.35) / tau_agg_ms).sin();
                    *slot = (base + ripple).max(0.0) as f32;
                }
            }
            let f_host = sub.iter().sum::<f32>() / (SUB_STEPS as f32);
            rows.push((n, t_host, f_host, sub));
        }

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let csv_path = std::env::temp_dir().join(format!(
            "pianosim_hammer_diag_{}_{nanos}.csv",
            std::process::id()
        ));
        write_hammer_csv(&csv_path, &rows).expect("write csv");
        let loaded = read_hammer_csv(&csv_path).expect("read csv");
        let _ = fs::remove_file(&csv_path);

        let metrics = analyze_hammer_force(&loaded, tau_agg_ms);
        assert!(
            (1.5..=2.5).contains(&metrics.contact_duration_ms),
            "expected ~2.0 ms contact, got {:.3}",
            metrics.contact_duration_ms
        );
        assert!(metrics.peak_force_n > 25.0);
        assert!(metrics.impulse_ns > 0.02);
        assert!(
            (2..=4).contains(&metrics.peaks.len()),
            "expected 2..4 agraffe peaks, got {} ({:?})",
            metrics.peaks.len(),
            metrics.peaks
        );
        let dt = metrics.mean_inter_peak_ms.expect("inter-peak");
        assert!(
            (dt - tau_agg_ms).abs() < 0.15,
            "inter-peak {dt:.3} ms should track tau_agg {tau_agg_ms:.3} ms"
        );
    }

    #[test]
    fn trace_from_rows_slices_active_substeps_without_trailing_zero_troughs() {
        let rows = [
            (0.0, 10.0, [8.0, 9.0, 10.0, 13.0, 0.0, 0.0, 0.0, 0.0]),
            (
                1.0 / 48.0,
                12.0,
                [11.0, 12.0, 13.0, 12.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ];
        let trace = trace_from_rows(&rows, 48_000.0, 4);
        assert_eq!(trace.sub_samples.len(), 8);
        assert!((trace.dt_sub_ms - (1_000.0 / (48_000.0 * 4.0))).abs() < 1e-12);
        assert!(trace.sub_samples.iter().all(|&(_, f)| f > 0.0));
    }
}
