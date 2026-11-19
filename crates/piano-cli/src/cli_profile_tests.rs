//! Unit tests for `CliProfileArgs` parsing across `--perf-profile` and all 18 override flags.

use piano_engine::{PerfPreset, PerfProfile};

use super::CliProfileArgs;

fn parse_all(args: &[&str]) -> Result<PerfProfile, String> {
    let owned: Vec<String> = args.iter().map(|s| String::from(*s)).collect();
    let mut cli = CliProfileArgs::default();
    let mut i = 0;
    while i < owned.len() {
        if !cli.try_consume_arg(&owned, &mut i)? {
            return Err(format!("unrecognized arg {}", owned[i]));
        }
    }
    Ok(cli.resolve())
}

#[test]
fn cli_profile_parses_all_presets_and_all_18_override_flags() {
    assert_eq!(parse_all(&[]).unwrap(), PerfProfile::full());
    assert_eq!(
        parse_all(&["--perf-profile", "balanced"]).unwrap(),
        PerfProfile::balanced()
    );
    assert_eq!(
        parse_all(&["--perf-profile=realtime"]).unwrap(),
        PerfProfile::realtime()
    );

    let full_overrides = parse_all(&[
        "--perf-profile",
        "balanced",
        "--horiz-max-hz",
        "4200",
        "--dead-mode-prune-db=-72.5",
        "--energy-horizon-db",
        "-85",
        "--sav-dormancy-db",
        "-36",
        "--sav-max-key",
        "28",
        "--bass-phantom-sparse",
        "false",
        "--passive-res-max-hz",
        "3200",
        "--ir-duration-secs",
        "0.20",
        "--spatial-tail-split-tap",
        "1536",
        "--fir-head-taps",
        "16",
        "--fir-block-fft=true",
        "--hall-fdn-lines",
        "16",
        "--action-oversample",
        "2",
        "--polynomial-adaa",
        "off",
        "--shank-flex",
        "false",
        "--damper-decim=4",
        "--hertz-nonlinear",
        "1",
        "--active-bridge-poles",
        "8",
    ])
    .expect("all 18 overrides");

    assert_eq!(full_overrides.horiz_max_hz, 4200.0);
    assert_eq!(full_overrides.dead_mode_prune_db, -72.5);
    assert_eq!(full_overrides.energy_horizon_db, -85.0);
    assert_eq!(full_overrides.sav_dormancy_db, -36.0);
    assert_eq!(full_overrides.sav_max_key_idx, 28);
    assert!(!full_overrides.bass_phantom_sparse);
    assert_eq!(full_overrides.passive_res_max_hz, 3200.0);
    assert_eq!(full_overrides.ir_duration_secs, 0.20);
    assert_eq!(full_overrides.spatial_tail_split_tap, 1536);
    assert_eq!(full_overrides.fir_head_taps, 16);
    assert!(full_overrides.fir_block_fft);
    assert_eq!(full_overrides.hall_fdn_lines, 16);
    assert_eq!(full_overrides.action_oversample, 2);
    assert!(!full_overrides.polynomial_adaa);
    assert!(!full_overrides.shank_flex_enabled);
    assert_eq!(full_overrides.damper_control_decim, 4);
    assert!(full_overrides.hertz_nonlinear);
    assert_eq!(full_overrides.active_bridge_poles, 8);
}

#[test]
fn cli_profile_rejects_invalid_values_and_reports_label() {
    assert!(parse_all(&["--perf-profile", "turbo"]).is_err());
    assert!(parse_all(&["--fir-head-taps", "48"]).is_err());
    assert!(parse_all(&["--hall-fdn-lines", "12"]).is_err());
    assert!(parse_all(&["--action-oversample", "3"]).is_err());
    assert!(parse_all(&["--damper-decim", "3"]).is_err());
    assert_eq!(
        parse_all(&["--damper-decim", "16"])
            .unwrap()
            .damper_control_decim,
        16
    );
    assert!(parse_all(&["--active-bridge-poles", "12"]).is_err());
    assert!(parse_all(&["--sav-max-key", "80"]).is_err());
    assert!(parse_all(&["--polynomial-adaa", "maybe"]).is_err());
    assert!(parse_all(&["--ir-duration-secs"]).is_err());

    let mut cli = CliProfileArgs {
        preset: Some(PerfPreset::Balanced),
        ..CliProfileArgs::default()
    };
    assert_eq!(cli.label(), "balanced");
    cli.fir_head_taps = Some(16);
    assert_eq!(cli.label(), "balanced+overrides");
}
