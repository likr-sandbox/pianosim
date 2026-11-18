//! Native WAV rendering commands (`render-wav` and `render-demos`).

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use piano_engine::{BodyArchetype, PianoEngine, DEFAULT_REVERB_DECAY_S, DEFAULT_REVERB_MIX};
use piano_spec::voicing::{A4_FREQ_DEFAULT, A4_FREQ_MAX, A4_FREQ_MIN};
use piano_spec::{LidState, ListenerPerspective, SampleEvent, Temperament, VoicingSpec};

use crate::check_lines::find_repo_root;
use crate::smf_parser::parse_smf_file;
use crate::wav_writer::{
    analyze_stereo, write_stereo_f32_wav, write_stereo_pcm16_wav, write_stereo_pcm24_tpdf_wav,
    AudioMetrics,
};

pub(crate) struct ShowcaseDemo {
    pub(crate) title: &'static str,
    pub(crate) mid_filename: &'static str,
    pub(crate) wav_filename: &'static str,
}

pub(crate) const SHOWCASE_DEMOS: [ShowcaseDemo; 6] = [
    ShowcaseDemo {
        title: "Bach: Prelude in C (BWV 846)",
        mid_filename: "01-bach-prelude-bwv846.mid",
        wav_filename: "01-bach-prelude-bwv846.wav",
    },
    ShowcaseDemo {
        title: "Chopin: Nocturne in Eb (Op. 9 No. 2)",
        mid_filename: "02-chopin-nocturne-op9-no2.mid",
        wav_filename: "02-chopin-nocturne-op9-no2.wav",
    },
    ShowcaseDemo {
        title: "Beethoven: Moonlight Sonata (Op. 27 No. 2)",
        mid_filename: "03-beethoven-moonlight-op27-no2.mid",
        wav_filename: "03-beethoven-moonlight-op27-no2.wav",
    },
    ShowcaseDemo {
        title: "Debussy: Clair de Lune",
        mid_filename: "04-debussy-clair-de-lune.mid",
        wav_filename: "04-debussy-clair-de-lune.wav",
    },
    ShowcaseDemo {
        title: "Satie: Gymnopédie No. 1",
        mid_filename: "05-satie-gymnopedie-no1.mid",
        wav_filename: "05-satie-gymnopedie-no1.wav",
    },
    ShowcaseDemo {
        title: "Rachmaninoff: Prelude in C# Minor (Op. 3 No. 2)",
        mid_filename: "06-rachmaninoff-prelude-op3-no2.mid",
        wav_filename: "06-rachmaninoff-prelude-op3-no2.wav",
    },
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WavFormat {
    Pcm16,
    Pcm24,
    F32,
}

impl WavFormat {
    const fn label(self) -> &'static str {
        match self {
            Self::Pcm16 => "pcm16",
            Self::Pcm24 => "pcm24",
            Self::F32 => "f32",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PresetId {
    ConcertGrand,
    IntimateUpright,
    BrightPop,
    VintageSaloon,
}

impl PresetId {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "concert-grand" => Ok(Self::ConcertGrand),
            "intimate-upright" => Ok(Self::IntimateUpright),
            "bright-pop" => Ok(Self::BrightPop),
            "vintage-saloon" => Ok(Self::VintageSaloon),
            _ => Err(format!(
                "invalid --preset '{raw}' (expected concert-grand, intimate-upright, bright-pop, or vintage-saloon)"
            )),
        }
    }

    const fn slug(self) -> &'static str {
        match self {
            Self::ConcertGrand => "concert-grand",
            Self::IntimateUpright => "intimate-upright",
            Self::BrightPop => "bright-pop",
            Self::VintageSaloon => "vintage-saloon",
        }
    }

    fn configure_engine(self, sample_rate: u32, a4_override: Option<f32>) -> (PianoEngine, f32) {
        let (hh, ud, st, sl, a4, cond, body, persp, lid, decay_s, mix) = match self {
            Self::ConcertGrand => (
                0.50,
                1.00,
                1.00,
                1.00,
                A4_FREQ_DEFAULT,
                0.0,
                BodyArchetype::ConcertGrand,
                ListenerPerspective::Player,
                LidState::Open,
                DEFAULT_REVERB_DECAY_S,
                DEFAULT_REVERB_MIX,
            ),
            Self::IntimateUpright => (
                0.25,
                0.60,
                0.70,
                0.75,
                A4_FREQ_DEFAULT,
                0.0,
                BodyArchetype::UprightStudio,
                ListenerPerspective::CloseStereo,
                LidState::Closed,
                0.85,
                0.12,
            ),
            Self::BrightPop => (
                0.80,
                1.40,
                1.30,
                1.15,
                442.0,
                0.0,
                BodyArchetype::ConcertGrand,
                ListenerPerspective::CloseStereo,
                LidState::Open,
                1.25,
                0.22,
            ),
            Self::VintageSaloon => (
                0.70,
                2.50,
                0.20,
                0.65,
                435.0,
                0.25,
                BodyArchetype::VintageSaloon,
                ListenerPerspective::Player,
                LidState::Half,
                0.55,
                0.08,
            ),
        };
        let mut voicing =
            VoicingSpec::from_globals_full(hh, ud, st, sl, a4, Temperament::Equal, cond);
        if let Some(a4_hz) = a4_override {
            voicing = voicing.with_concert_pitch_hz(a4_hz);
        }
        let eff_a4 = voicing.effective_a4_freq_hz();
        let mut engine = PianoEngine::new_with_voicing(sample_rate as f32, &voicing);
        engine.set_body_space(body, persp, lid);
        engine.set_reverb(decay_s, mix);
        (engine, eff_a4)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct RenderOptions {
    target: Option<PathBuf>,
    midi: Option<PathBuf>,
    demos_dir: Option<PathBuf>,
    preset: PresetId,
    sample_rate: u32,
    a4_override: Option<f32>,
    format: WavFormat,
}

pub(crate) fn render_usage() -> String {
    String::from(
        "  piano-cli render-wav <out.wav> --midi <path.mid> [--preset <id>] [--sample-rate <48000|96000>] [--a4 <392..466.2>] [--format <pcm16|pcm24|f32>]\n  piano-cli render-demos [out_dir] [--demos-dir <dir>] [--preset <id>] [--sample-rate <48000|96000>] [--a4 <392..466.2>] [--format <pcm16|pcm24|f32>]",
    )
}

pub(crate) fn cmd_render_wav(args: &[String]) -> Result<(), String> {
    let opts = parse_render_options(args, true, WavFormat::Pcm16)?;
    let out_path = opts.target.ok_or_else(render_usage)?;
    let midi_path = opts.midi.ok_or_else(render_usage)?;
    let (mut engine, a4_hz) = opts
        .preset
        .configure_engine(opts.sample_rate, opts.a4_override);
    let metrics = render_midi_to_wav(
        &mut engine,
        &midi_path,
        opts.sample_rate,
        opts.format,
        &out_path,
    )?;
    println!(
        "render-wav OK: {} [midi={}, preset={}, {}] ({} frames @ {} Hz, A4={:.1} Hz, peak_l={:.4}, peak_r={:.4}, rms={:.4}, non_finite=0)",
        out_path.display(),
        midi_path.display(),
        opts.preset.slug(),
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
    let (mut engine, a4_hz) = opts
        .preset
        .configure_engine(opts.sample_rate, opts.a4_override);

    println!(
        "Rendering {} native showcase demos from {} -> {} [preset={}, {}, {} Hz, A4={:.1} Hz]",
        SHOWCASE_DEMOS.len(),
        demos_dir.display(),
        out_dir.display(),
        opts.preset.slug(),
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

    let mut i = 0;
    while i < args.len() {
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
    })
}

#[cfg(test)]
#[path = "render_cmd_tests.rs"]
mod tests;
