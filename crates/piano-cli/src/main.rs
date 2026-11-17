//! Native verification, WAV export, Wasm SIMD inspection, and static check CLI.

mod bench_composite;
mod bench_rt;
mod check_lines;
mod scenarios;
mod wasm_check;
mod wav_writer;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use piano_engine::PianoEngine;
use piano_spec::voicing::{A4_FREQ_DEFAULT, A4_FREQ_MAX, A4_FREQ_MIN};
use piano_spec::{u7_to_u16, EventKind, PedalKind, SampleEvent, VoicingSpec};

use check_lines::{check_repo, find_repo_root};
use wasm_check::inspect_wasm_simd;
use wav_writer::{analyze_stereo, write_stereo_pcm16_wav};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench_rt::run_bench(),
        Some("check-lines") => cmd_check_lines(&args[1..]),
        Some("check-wasm-simd") => cmd_check_wasm_simd(&args[1..]),
        Some("render-wav") => cmd_render_wav(&args[1..]),
        Some("scenarios") => scenarios::run_scenarios(&args[1..]),
        _ => Err(usage()),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn usage() -> String {
    String::from(
        "Usage:\n  piano-cli bench\n  piano-cli check-lines [root]\n  piano-cli check-wasm-simd <path.wasm>\n  piano-cli render-wav <out.wav> [--a4 <415..466>]\n  piano-cli scenarios [--json] [--markdown]",
    )
}

fn cmd_check_lines(args: &[String]) -> Result<(), String> {
    let root = match args.first() {
        Some(path) => PathBuf::from(path),
        None => {
            let cwd = env::current_dir().map_err(|e| format!("failed to get cwd: {e}"))?;
            find_repo_root(&cwd)
        }
    };
    let summary = check_repo(&root)?;
    println!(
        "check-lines OK: {} files (max {} lines in {}), AGENTS.md {} bytes (< {}), clean-room {} files",
        summary.checked_files,
        summary.max_lines,
        summary.max_lines_file,
        summary.agents_bytes,
        check_lines::MAX_AGENTS_BYTES,
        summary.clean_room_files,
    );
    Ok(())
}

fn cmd_check_wasm_simd(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or_else(usage)?;
    let bytes = fs::read(path).map_err(|e| format!("failed to read {path}: {e}"))?;
    let report = inspect_wasm_simd(&bytes)?;
    if report.import_count > 0 {
        return Err(format!(
            "check-wasm-simd FAILED for {path}: {} imports (must be 0)",
            report.import_count
        ));
    }
    if report.simd_opcodes == 0 {
        return Err(format!(
            "check-wasm-simd FAILED for {path}: 0 SIMD128 (0xfd) opcodes in {} code bodies",
            report.code_bodies
        ));
    }
    if report.relaxed_simd_opcodes > 0 {
        return Err(format!(
            "check-wasm-simd FAILED for {path}: {} relaxed-simd opcodes forbidden",
            report.relaxed_simd_opcodes
        ));
    }
    println!(
        "check-wasm-simd OK: {path} -> {} SIMD128 (0xfd) opcodes, 0 relaxed-simd, {} v128 locals, {} code bodies, {} imports",
        report.simd_opcodes, report.v128_locals, report.code_bodies, report.import_count
    );
    Ok(())
}

fn parse_a4_hz(raw: &str) -> Result<f32, String> {
    let val: f32 = raw.parse().map_err(|_| {
        format!("invalid --a4 frequency '{raw}' (expected number in 415.0..=466.0)")
    })?;
    if !val.is_finite() || !(A4_FREQ_MIN..=A4_FREQ_MAX).contains(&val) {
        return Err(format!(
            "--a4 frequency {raw} out of supported range [{A4_FREQ_MIN:.1}, {A4_FREQ_MAX:.1}] Hz"
        ));
    }
    Ok(val)
}

fn parse_render_wav_args(args: &[String]) -> Result<(PathBuf, f32), String> {
    let mut out_path: Option<PathBuf> = None;
    let mut a4_hz = A4_FREQ_DEFAULT;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--a4" {
            let next = args.get(i + 1).ok_or_else(usage)?;
            a4_hz = parse_a4_hz(next)?;
            i += 2;
        } else if let Some(rest) = arg.strip_prefix("--a4=") {
            a4_hz = parse_a4_hz(rest)?;
            i += 1;
        } else if arg.starts_with('-') {
            return Err(usage());
        } else if out_path.is_none() {
            out_path = Some(PathBuf::from(arg));
            i += 1;
        } else {
            return Err(usage());
        }
    }
    let path = out_path.ok_or_else(usage)?;
    Ok((path, a4_hz))
}

fn cmd_render_wav(args: &[String]) -> Result<(), String> {
    let (out_path, a4_hz) = parse_render_wav_args(args)?;
    let out_path = Path::new(&out_path);
    let sample_rate = 48_000_u32;
    let total_frames = 96_000_usize; // 2.0s @ 48 kHz
    let block_size = 128_usize;

    let voicing = VoicingSpec {
        a4_freq_hz: a4_hz,
        ..VoicingSpec::default()
    };

    let script = build_m1_demo_phrase();
    let mut engine = PianoEngine::new_with_voicing(sample_rate as f32, &voicing);
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut block_events = Vec::with_capacity(16);

    let mut offset = 0_usize;
    while offset < total_frames {
        let end = (offset + block_size).min(total_frames);
        block_events.clear();
        for &(abs_sample, kind) in &script {
            if abs_sample >= offset && abs_sample < end {
                block_events.push(SampleEvent {
                    sample_offset: (abs_sample - offset) as u32,
                    kind,
                });
            }
        }
        engine.render_block(
            &block_events,
            &mut out_l[offset..end],
            &mut out_r[offset..end],
        );
        offset = end;
    }

    let metrics = analyze_stereo(&out_l, &out_r);
    if metrics.non_finite_count > 0 {
        return Err(format!(
            "render-wav FAILED: detected {} NaN/Inf samples",
            metrics.non_finite_count
        ));
    }
    let peak = metrics.peak_l.max(metrics.peak_r);
    if peak > 1.0 {
        return Err(format!("render-wav FAILED: peak {peak} exceeds 1.0"));
    }
    if metrics.rms_mono <= 0.005 {
        return Err(format!(
            "render-wav FAILED: RMS {} too low (expected audible phrase)",
            metrics.rms_mono
        ));
    }

    write_stereo_pcm16_wav(out_path, sample_rate, &out_l, &out_r)?;
    println!(
        "render-wav OK: {} ({} frames @ {} Hz, A4={:.1} Hz, peak_l={:.4}, peak_r={:.4}, rms={:.4}, non_finite=0)",
        out_path.display(),
        metrics.frames,
        sample_rate,
        a4_hz,
        metrics.peak_l,
        metrics.peak_r,
        metrics.rms_mono
    );
    Ok(())
}

fn build_m1_demo_phrase() -> Vec<(usize, EventKind)> {
    vec![
        (
            0,
            EventKind::NoteOn {
                midi_key: 69,
                velocity_u16: u7_to_u16(108),
            },
        ),
        (
            24_000,
            EventKind::NoteOff {
                midi_key: 69,
                release_vel_u16: 0,
            },
        ),
        (
            31_200,
            EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        ),
        (
            33_600,
            EventKind::NoteOn {
                midi_key: 60,
                velocity_u16: u7_to_u16(96),
            },
        ),
        (
            35_040,
            EventKind::NoteOn {
                midi_key: 64,
                velocity_u16: u7_to_u16(98),
            },
        ),
        (
            36_480,
            EventKind::NoteOn {
                midi_key: 67,
                velocity_u16: u7_to_u16(100),
            },
        ),
        (
            37_920,
            EventKind::NoteOn {
                midi_key: 72,
                velocity_u16: u7_to_u16(104),
            },
        ),
        (
            62_400,
            EventKind::NoteOff {
                midi_key: 60,
                release_vel_u16: 0,
            },
        ),
        (
            62_400,
            EventKind::NoteOff {
                midi_key: 64,
                release_vel_u16: 0,
            },
        ),
        (
            62_400,
            EventKind::NoteOff {
                midi_key: 67,
                release_vel_u16: 0,
            },
        ),
        (
            62_400,
            EventKind::NoteOff {
                midi_key: 72,
                release_vel_u16: 0,
            },
        ),
        (
            79_200,
            EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: 0,
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_render_wav_args_supports_default_and_custom_a4() {
        let args = vec![String::from("out.wav")];
        let (path, a4) = parse_render_wav_args(&args).expect("default a4");
        assert_eq!(path, PathBuf::from("out.wav"));
        assert_eq!(a4, 440.0);

        let args_space = vec![
            String::from("out.wav"),
            String::from("--a4"),
            String::from("442.0"),
        ];
        assert_eq!(parse_render_wav_args(&args_space).unwrap().1, 442.0);

        let args_eq = vec![String::from("--a4=415.0"), String::from("baroque.wav")];
        let (path_eq, a4_eq) = parse_render_wav_args(&args_eq).expect("baroque a4");
        assert_eq!(path_eq, PathBuf::from("baroque.wav"));
        assert_eq!(a4_eq, 415.0);

        assert!(
            parse_render_wav_args(&[String::from("out.wav"), String::from("--a4=399.0")]).is_err()
        );
        assert!(
            parse_render_wav_args(&[String::from("out.wav"), String::from("--a4=480.0")]).is_err()
        );
    }
}
