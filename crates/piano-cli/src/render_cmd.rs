//! Native WAV rendering commands (`render-wav` and `render-demos`).

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use piano_engine::{PerfProfile, PianoEngine};
use piano_spec::voicing::{A4_FREQ_MAX, A4_FREQ_MIN};
use piano_spec::SampleEvent;

use crate::check_lines::find_repo_root;
use crate::cli_profile::CliProfileArgs;
pub(crate) use crate::render_presets::SHOWCASE_DEMOS;
use crate::render_presets::{PresetId, WavFormat};
use crate::smf_parser::parse_smf_file;
use crate::wav_writer::{
    analyze_stereo, write_stereo_f32_wav, write_stereo_pcm16_wav, write_stereo_pcm24_tpdf_wav,
    AudioMetrics,
};

#[derive(Clone, Debug, PartialEq)]
struct RenderOptions {
    target: Option<PathBuf>,
    midi: Option<PathBuf>,
    demos_dir: Option<PathBuf>,
    preset: PresetId,
    sample_rate: u32,
    a4_override: Option<f32>,
    format: WavFormat,
    profile: PerfProfile,
    profile_label: String,
}

pub(crate) fn render_usage() -> String {
    format!(
        "  piano-cli render-wav <out.wav> --midi <path.mid> [--preset <id>] [--sample-rate <48000|96000>] [--a4 <392..466.2>] [--format <pcm16|pcm24|f32>] {}\n  piano-cli render-demos [out_dir] [--demos-dir <dir>] [--preset <id>] [--sample-rate <48000|96000>] [--a4 <392..466.2>] [--format <pcm16|pcm24|f32>] {}",
        CliProfileArgs::usage(),
        CliProfileArgs::usage()
    )
}

pub(crate) fn cmd_render_wav(args: &[String]) -> Result<(), String> {
    let opts = parse_render_options(args, true, WavFormat::Pcm16)?;
    let out_path = opts.target.ok_or_else(render_usage)?;
    let midi_path = opts.midi.ok_or_else(render_usage)?;
    let (mut engine, a4_hz) =
        opts.preset
            .configure_engine(opts.sample_rate, opts.a4_override, opts.profile);
    let metrics = render_midi_to_wav(
        &mut engine,
        &midi_path,
        opts.sample_rate,
        opts.format,
        &out_path,
    )?;
    println!(
        "render-wav OK: {} [midi={}, preset={}, perf={}, {}] ({} frames @ {} Hz, A4={:.1} Hz, peak_l={:.4}, peak_r={:.4}, rms={:.4}, non_finite=0)",
        out_path.display(),
        midi_path.display(),
        opts.preset.slug(),
        opts.profile_label,
        opts.format.label(),
        metrics.frames,
        opts.sample_rate,
        a4_hz,
        metrics.peak_l,
        metrics.peak_r,
        metrics.rms_mono
    );
    Ok(())
}

pub(crate) fn cmd_render_demos(args: &[String]) -> Result<(), String> {
    let opts = parse_render_options(args, false, WavFormat::Pcm24)?;
    let out_dir = opts.target.unwrap_or_else(|| PathBuf::from("demos"));
    let demos_dir = match opts.demos_dir {
        Some(dir) => dir,
        None => {
            let cwd = env::current_dir().map_err(|e| format!("failed to get cwd: {e}"))?;
            find_repo_root(&cwd).join("apps/web/public/demos")
        }
    };
    let start_all = Instant::now();
    let (mut engine, a4_hz) =
        opts.preset
            .configure_engine(opts.sample_rate, opts.a4_override, opts.profile);

    println!(
        "Rendering {} native showcase demos from {} -> {} [preset={}, perf={}, {}, {} Hz, A4={:.1} Hz]",
        SHOWCASE_DEMOS.len(),
        demos_dir.display(),
        out_dir.display(),
        opts.preset.slug(),
        opts.profile_label,
        opts.format.label(),
        opts.sample_rate,
        a4_hz
    );

    for (idx, demo) in SHOWCASE_DEMOS.iter().enumerate() {
        engine.reset();
        let midi_path = demos_dir.join(demo.mid_filename);
        let out_path = out_dir.join(demo.wav_filename);
        let t0 = Instant::now();
        let metrics = render_midi_to_wav(
            &mut engine,
            &midi_path,
            opts.sample_rate,
            opts.format,
            &out_path,
        )?;
        let elapsed_s = t0.elapsed().as_secs_f64();
        let audio_s = (metrics.frames as f64) / f64::from(opts.sample_rate);
        println!(
            "  [{}/{}] {} -> {} ({:.2}s, peak_l={:.4}, peak_r={:.4}, rms={:.4}, rendered in {:.2}s)",
            idx + 1,
            SHOWCASE_DEMOS.len(),
            demo.title,
            out_path.display(),
            audio_s,
            metrics.peak_l,
            metrics.peak_r,
            metrics.rms_mono,
            elapsed_s
        );
    }

    println!(
        "render-demos OK: {} files written in {:.2}s",
        SHOWCASE_DEMOS.len(),
        start_all.elapsed().as_secs_f64()
    );
    Ok(())
}

fn render_midi_to_wav(
    engine: &mut PianoEngine,
    midi_path: &Path,
    sample_rate: u32,
    format: WavFormat,
    out_path: &Path,
) -> Result<AudioMetrics, String> {
    let parsed = parse_smf_file(midi_path, sample_rate)?;
    let total_frames =
        (((parsed.last_event_s + 1.8) * f64::from(sample_rate)).round() as usize).max(128);
    let block_size = 128_usize;

    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut block_events = Vec::with_capacity(32);
    let mut ev_idx = 0_usize;
    let mut offset = 0_usize;

    while offset < total_frames {
        let end = (offset + block_size).min(total_frames);
        block_events.clear();
        while ev_idx < parsed.events.len() && parsed.events[ev_idx].0 < end {
            let (abs_sample, kind) = parsed.events[ev_idx];
            if abs_sample >= offset {
                block_events.push(SampleEvent {
                    sample_offset: (abs_sample - offset) as u32,
                    kind,
                });
            }
            ev_idx += 1;
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
            "render FAILED for {}: detected {} NaN/Inf samples",
            midi_path.display(),
            metrics.non_finite_count
        ));
    }
    let peak = metrics.peak_l.max(metrics.peak_r);
    if peak > 1.0 {
        return Err(format!(
            "render FAILED for {}: peak {peak} exceeds 1.0",
            midi_path.display()
        ));
    }
    if peak < 0.01 || metrics.rms_mono < 5.0e-4 {
        return Err(format!(
            "render FAILED for {}: level too low (peak={peak:.6}, rms={:.6})",
            midi_path.display(),
            metrics.rms_mono
        ));
    }

    match format {
        WavFormat::Pcm16 => write_stereo_pcm16_wav(out_path, sample_rate, &out_l, &out_r)?,
        WavFormat::Pcm24 => write_stereo_pcm24_tpdf_wav(out_path, sample_rate, &out_l, &out_r)?,
        WavFormat::F32 => write_stereo_f32_wav(out_path, sample_rate, &out_l, &out_r)?,
    }
    Ok(metrics)
}

fn parse_a4_hz(raw: &str) -> Result<f32, String> {
    let val: f32 = raw.parse().map_err(|_| {
        format!("invalid --a4 frequency '{raw}' (expected number in {A4_FREQ_MIN:.1}..={A4_FREQ_MAX:.1})")
    })?;
    if !val.is_finite() || !(A4_FREQ_MIN..=A4_FREQ_MAX).contains(&val) {
        return Err(format!(
            "--a4 frequency {raw} out of supported range [{A4_FREQ_MIN:.1}, {A4_FREQ_MAX:.1}] Hz"
        ));
    }
    Ok(val)
}

fn parse_wav_format(raw: &str) -> Result<WavFormat, String> {
    match raw {
        "pcm16" => Ok(WavFormat::Pcm16),
        "pcm24" => Ok(WavFormat::Pcm24),
        "f32" => Ok(WavFormat::F32),
        _ => Err(format!(
            "invalid --format '{raw}' (expected pcm16, pcm24, or f32)"
        )),
    }
}

fn parse_sample_rate(raw: &str) -> Result<u32, String> {
    match raw {
        "48000" => Ok(48_000),
        "96000" => Ok(96_000),
        _ => Err(format!(
            "invalid --sample-rate '{raw}' (expected 48000 or 96000)"
        )),
    }
}

fn parse_flag_value<'a>(
    args: &'a [String],
    idx: &mut usize,
    name: &str,
) -> Result<Option<&'a str>, String> {
    let arg = args[*idx].as_str();
    if arg == name {
        let next = args.get(*idx + 1).ok_or_else(render_usage)?;
        *idx += 2;
        return Ok(Some(next.as_str()));
    }
    if let Some(rest) = arg.strip_prefix(name).and_then(|s| s.strip_prefix('=')) {
        *idx += 1;
        return Ok(Some(rest));
    }
    Ok(None)
}

fn parse_render_options(
    args: &[String],
    is_render_wav: bool,
    default_format: WavFormat,
) -> Result<RenderOptions, String> {
    let mut target: Option<PathBuf> = None;
    let mut midi: Option<PathBuf> = None;
    let mut demos_dir: Option<PathBuf> = None;
    let mut preset = PresetId::ConcertGrand;
    let mut sample_rate = 48_000_u32;
    let mut a4_override = None;
    let mut format = default_format;
    let mut cli_profile = CliProfileArgs::default();

    let mut i = 0;
    while i < args.len() {
        if cli_profile.try_consume_arg(args, &mut i)? {
            continue;
        }
        if is_render_wav {
            if let Some(v) = parse_flag_value(args, &mut i, "--midi")? {
                midi = Some(PathBuf::from(v));
                continue;
            }
        } else if let Some(v) = parse_flag_value(args, &mut i, "--demos-dir")? {
            demos_dir = Some(PathBuf::from(v));
            continue;
        }
        if let Some(v) = parse_flag_value(args, &mut i, "--preset")? {
            preset = PresetId::parse(v)?;
        } else if let Some(v) = parse_flag_value(args, &mut i, "--sample-rate")? {
            sample_rate = parse_sample_rate(v)?;
        } else if let Some(v) = parse_flag_value(args, &mut i, "--a4")? {
            a4_override = Some(parse_a4_hz(v)?);
        } else if let Some(v) = parse_flag_value(args, &mut i, "--format")? {
            format = parse_wav_format(v)?;
        } else if args[i].starts_with('-') {
            return Err(render_usage());
        } else if target.is_none() {
            target = Some(PathBuf::from(&args[i]));
            i += 1;
        } else {
            return Err(render_usage());
        }
    }

    if is_render_wav && (target.is_none() || midi.is_none()) {
        return Err(render_usage());
    }

    Ok(RenderOptions {
        target,
        midi,
        demos_dir,
        preset,
        sample_rate,
        a4_override,
        format,
        profile: cli_profile.resolve(),
        profile_label: cli_profile.label(),
    })
}

#[cfg(test)]
#[path = "render_cmd_tests.rs"]
mod tests;
