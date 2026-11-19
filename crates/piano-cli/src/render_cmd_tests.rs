//! Unit tests for `render_cmd` CLI option parsing and presets.

use super::*;

#[test]
fn parse_render_wav_args_requires_midi_and_supports_flags() {
    assert!(
        parse_render_options(&[String::from("out.wav")], true, WavFormat::Pcm16).is_err(),
        "render-wav must require --midi"
    );
    assert!(
        parse_render_options(
            &["out.wav", "--score", "bach-prelude"].map(String::from),
            true,
            WavFormat::Pcm16
        )
        .is_err(),
        "render-wav must reject deprecated --score"
    );
    assert!(
        parse_render_options(
            &["--midi", "piece.mid"].map(String::from),
            true,
            WavFormat::Pcm16
        )
        .is_err(),
        "render-wav must require output wav path"
    );

    let parsed = parse_render_options(
        &["out.wav", "--midi", "piece.mid"].map(String::from),
        true,
        WavFormat::Pcm16,
    )
    .expect("minimal render-wav");
    assert_eq!(parsed.target, Some(PathBuf::from("out.wav")));
    assert_eq!(parsed.midi, Some(PathBuf::from("piece.mid")));
    assert_eq!(parsed.preset, PresetId::ConcertGrand);
    assert_eq!(parsed.sample_rate, 48_000);
    assert_eq!(parsed.a4_override, None);
    assert_eq!(parsed.format, WavFormat::Pcm16);

    let args_space = [
        "out.wav",
        "--midi",
        "chopin.mid",
        "--a4",
        "442.0",
        "--format",
        "pcm24",
        "--preset",
        "intimate-upright",
        "--sample-rate",
        "96000",
    ]
    .map(String::from);
    let parsed_space = parse_render_options(&args_space, true, WavFormat::Pcm16).expect("space");
    assert_eq!(parsed_space.midi, Some(PathBuf::from("chopin.mid")));
    assert_eq!(parsed_space.a4_override, Some(442.0));
    assert_eq!(parsed_space.format, WavFormat::Pcm24);
    assert_eq!(parsed_space.preset, PresetId::IntimateUpright);
    assert_eq!(parsed_space.sample_rate, 96_000);

    let args_eq = [
        "--a4=392.0",
        "--format=f32",
        "--midi=bach.mid",
        "baroque.wav",
    ]
    .map(String::from);
    let parsed_eq = parse_render_options(&args_eq, true, WavFormat::Pcm16).expect("eq");
    assert_eq!(parsed_eq.target, Some(PathBuf::from("baroque.wav")));
    assert_eq!(parsed_eq.midi, Some(PathBuf::from("bach.mid")));
    assert_eq!(parsed_eq.a4_override, Some(392.0));
    assert_eq!(parsed_eq.format, WavFormat::F32);

    assert!(parse_render_options(
        &["out.wav", "--midi=a.mid", "--a4=380.0"].map(String::from),
        true,
        WavFormat::Pcm16
    )
    .is_err());
    assert!(parse_render_options(
        &["out.wav", "--midi=a.mid", "--a4=480.0"].map(String::from),
        true,
        WavFormat::Pcm16
    )
    .is_err());
    assert!(parse_render_options(
        &["out.wav", "--midi=a.mid", "--format=pcm32"].map(String::from),
        true,
        WavFormat::Pcm16
    )
    .is_err());
    assert!(parse_render_options(
        &["out.wav", "--midi=a.mid", "--sample-rate=44100"].map(String::from),
        true,
        WavFormat::Pcm16
    )
    .is_err());
}

#[test]
fn parse_render_demos_args_defaults_overrides_and_rejections() {
    let def = parse_render_options(&[], false, WavFormat::Pcm24).expect("defaults");
    assert_eq!(def.target, None);
    assert_eq!(def.demos_dir, None);
    assert_eq!(def.preset, PresetId::ConcertGrand);
    assert_eq!(def.sample_rate, 48_000);
    assert_eq!(def.a4_override, None);
    assert_eq!(def.format, WavFormat::Pcm24);

    let custom = parse_render_options(
        &[
            "out/wav",
            "--demos-dir",
            "custom/demos",
            "--preset=vintage-saloon",
            "--format=f32",
            "--sample-rate=96000",
        ]
        .map(String::from),
        false,
        WavFormat::Pcm24,
    )
    .expect("custom");
    assert_eq!(custom.target, Some(PathBuf::from("out/wav")));
    assert_eq!(custom.demos_dir, Some(PathBuf::from("custom/demos")));
    assert_eq!(custom.preset, PresetId::VintageSaloon);
    assert_eq!(custom.format, WavFormat::F32);
    assert_eq!(custom.sample_rate, 96_000);

    assert!(parse_render_options(
        &["--score=bach-prelude"].map(String::from),
        false,
        WavFormat::Pcm24
    )
    .is_err());
    assert!(parse_render_options(
        &["--midi=bach.mid"].map(String::from),
        false,
        WavFormat::Pcm24
    )
    .is_err());

    let with_prof = parse_render_options(
        &[
            "out.wav",
            "--midi",
            "piece.mid",
            "--perf-profile",
            "realtime",
            "--fir-head-taps",
            "16",
        ]
        .map(String::from),
        true,
        WavFormat::Pcm16,
    )
    .expect("render-wav with perf profile");
    assert_eq!(
        with_prof.profile.ir_duration_secs,
        PerfProfile::realtime().ir_duration_secs
    );
    assert_eq!(with_prof.profile.fir_head_taps, 16);
    assert_eq!(with_prof.profile_label, "realtime+overrides");
}
