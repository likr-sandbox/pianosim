//! Native verification, WAV export, Wasm SIMD inspection, and static check CLI.

mod check_lines;
mod wasm_check;
mod wav_writer;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use piano_engine::PianoEngine;
use piano_spec::{u7_to_u16, EventKind, PedalKind, SampleEvent};

use check_lines::{check_repo, find_repo_root};
use wasm_check::inspect_wasm_simd;
use wav_writer::{analyze_stereo, write_stereo_pcm16_wav};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("check-lines") => cmd_check_lines(&args[1..]),
        Some("check-wasm-simd") => cmd_check_wasm_simd(&args[1..]),
        Some("render-wav") => cmd_render_wav(&args[1..]),
        _ => Err(usage()),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn usage() -> String {
    String::from(
        "Usage:\n  piano-cli check-lines [root]\n  piano-cli check-wasm-simd <path.wasm>\n  piano-cli render-wav <out.wav>",
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
    if report.simd_opcodes == 0 {
        return Err(format!(
            "check-wasm-simd FAILED for {path}: 0 SIMD128 (0xfd) opcodes in {} code bodies",
            report.code_bodies
        ));
    }
    println!(
        "check-wasm-simd OK: {path} -> {} SIMD128 (0xfd) opcodes, {} v128 locals, {} code bodies, {} imports",
        report.simd_opcodes, report.v128_locals, report.code_bodies, report.import_count
    );
    Ok(())
}

fn cmd_render_wav(args: &[String]) -> Result<(), String> {
    let out_path = Path::new(args.first().ok_or_else(usage)?);
    let sample_rate = 48_000_u32;
    let total_frames = 96_000_usize; // 2.0s @ 48 kHz
    let block_size = 128_usize;

    let script = build_m1_demo_phrase();
    let mut engine = PianoEngine::new(sample_rate as f32);
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
        "render-wav OK: {} ({} frames @ {} Hz, peak_l={:.4}, peak_r={:.4}, rms={:.4}, non_finite=0)",
        out_path.display(),
        metrics.frames,
        sample_rate,
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
