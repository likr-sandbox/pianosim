//! Shared CLI parser for `--perf-profile` and all 19 `PerfProfile` override flags.

use piano_engine::{PerfPreset, PerfProfile};

/// Accumulated CLI performance profile preset and per-parameter overrides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CliProfileArgs {
    pub preset: Option<PerfPreset>,
    pub horiz_max_hz: Option<f32>,
    pub dead_mode_prune_db: Option<f32>,
    pub low_order_energy_horizon_db: Option<f32>,
    pub energy_horizon_db: Option<f32>,
    pub sav_dormancy_db: Option<f32>,
    pub sav_max_key_idx: Option<u8>,
    pub bass_phantom_sparse: Option<bool>,
    pub passive_res_max_hz: Option<f32>,
    pub ir_duration_secs: Option<f32>,
    pub spatial_tail_split_tap: Option<u16>,
    pub fir_head_taps: Option<u8>,
    pub fir_block_fft: Option<bool>,
    pub hall_fdn_lines: Option<u8>,
    pub action_oversample: Option<u8>,
    pub polynomial_adaa: Option<bool>,
    pub shank_flex_enabled: Option<bool>,
    pub damper_control_decim: Option<u8>,
    pub hertz_nonlinear: Option<bool>,
    pub active_bridge_poles: Option<u8>,
}

impl CliProfileArgs {
    /// Returns usage summary for performance profile CLI options.
    #[must_use]
    pub const fn usage() -> &'static str {
        "[--perf-profile <full|balanced|realtime>] [--horiz-max-hz <f32>] [--dead-mode-prune-db <f32>] [--low-order-energy-horizon-db <f32>] [--energy-horizon-db <f32>] [--sav-dormancy-db <f32>] [--sav-max-key <0..=52>] [--bass-phantom-sparse <bool>] [--passive-res-max-hz <f32>] [--ir-duration-secs <f32>] [--spatial-tail-split-tap <u16>] [--fir-head-taps <16|32|64>] [--fir-block-fft <bool>] [--hall-fdn-lines <4|8|16>] [--action-oversample <2|4|8>] [--polynomial-adaa <bool>] [--shank-flex <bool>] [--damper-decim <1|2|4|8|16>] [--hertz-nonlinear <bool>] [--active-bridge-poles <4..=10>]"
    }

    /// Resolves base preset + individual overrides into a sanitized [`PerfProfile`].
    #[must_use]
    pub fn resolve(&self) -> PerfProfile {
        let mut p = PerfProfile::from_preset(self.preset.unwrap_or(PerfPreset::Full));
        if let Some(v) = self.horiz_max_hz {
            p.horiz_max_hz = v;
        }
        if let Some(v) = self.dead_mode_prune_db {
            p.dead_mode_prune_db = v;
        }
        if let Some(v) = self.low_order_energy_horizon_db {
            p.low_order_energy_horizon_db = v;
        }
        if let Some(v) = self.energy_horizon_db {
            p.energy_horizon_db = v;
        }
        if let Some(v) = self.sav_dormancy_db {
            p.sav_dormancy_db = v;
        }
        if let Some(v) = self.sav_max_key_idx {
            p.sav_max_key_idx = v;
        }
        if let Some(v) = self.bass_phantom_sparse {
            p.bass_phantom_sparse = v;
        }
        if let Some(v) = self.passive_res_max_hz {
            p.passive_res_max_hz = v;
        }
        if let Some(v) = self.ir_duration_secs {
            p.ir_duration_secs = v;
        }
        if let Some(v) = self.spatial_tail_split_tap {
            p.spatial_tail_split_tap = v;
        }
        if let Some(v) = self.fir_head_taps {
            p.fir_head_taps = v;
        }
        if let Some(v) = self.fir_block_fft {
            p.fir_block_fft = v;
        }
        if let Some(v) = self.hall_fdn_lines {
            p.hall_fdn_lines = v;
        }
        if let Some(v) = self.action_oversample {
            p.action_oversample = v;
        }
        if let Some(v) = self.polynomial_adaa {
            p.polynomial_adaa = v;
        }
        if let Some(v) = self.shank_flex_enabled {
            p.shank_flex_enabled = v;
        }
        if let Some(v) = self.damper_control_decim {
            p.damper_control_decim = v;
        }
        if let Some(v) = self.hertz_nonlinear {
            p.hertz_nonlinear = v;
        }
        if let Some(v) = self.active_bridge_poles {
            p.active_bridge_poles = v;
        }
        p.sanitized()
    }

    /// Human-readable label for banners.
    #[must_use]
    pub fn label(&self) -> String {
        let base = match self.preset.unwrap_or(PerfPreset::Full) {
            PerfPreset::Full => "full",
            PerfPreset::Balanced => "balanced",
            PerfPreset::Realtime => "realtime",
        };
        let has_override = self.horiz_max_hz.is_some()
            || self.dead_mode_prune_db.is_some()
            || self.low_order_energy_horizon_db.is_some()
            || self.energy_horizon_db.is_some()
            || self.sav_dormancy_db.is_some()
            || self.sav_max_key_idx.is_some()
            || self.bass_phantom_sparse.is_some()
            || self.passive_res_max_hz.is_some()
            || self.ir_duration_secs.is_some()
            || self.spatial_tail_split_tap.is_some()
            || self.fir_head_taps.is_some()
            || self.fir_block_fft.is_some()
            || self.hall_fdn_lines.is_some()
            || self.action_oversample.is_some()
            || self.polynomial_adaa.is_some()
            || self.shank_flex_enabled.is_some()
            || self.damper_control_decim.is_some()
            || self.hertz_nonlinear.is_some()
            || self.active_bridge_poles.is_some();
        if has_override {
            format!("{base}+overrides")
        } else {
            String::from(base)
        }
    }

    /// Attempts to consume a profile flag at `args[*idx]`; returns `Ok(false)` if unmatched.
    pub fn try_consume_arg(&mut self, args: &[String], idx: &mut usize) -> Result<bool, String> {
        let Some((flag, val)) = split_flag_if_known(args, idx)? else {
            return Ok(false);
        };
        match flag {
            "--perf-profile" => self.preset = Some(parse_preset(val)?),
            "--horiz-max-hz" => {
                self.horiz_max_hz = Some(parse_f32_range(val, flag, 500.0, 20_000.0)?)
            }
            "--dead-mode-prune-db" => {
                self.dead_mode_prune_db = Some(parse_f32_range(val, flag, -120.0, -20.0)?)
            }
            "--low-order-energy-horizon-db" => {
                self.low_order_energy_horizon_db = Some(parse_f32_range(val, flag, -120.0, -40.0)?)
            }
            "--energy-horizon-db" => {
                self.energy_horizon_db = Some(parse_f32_range(val, flag, -120.0, -40.0)?)
            }
            "--sav-dormancy-db" => {
                self.sav_dormancy_db = Some(parse_f32_range(val, flag, -80.0, -20.0)?)
            }
            "--sav-max-key" => self.sav_max_key_idx = Some(parse_u8_range(val, flag, 0, 52)?),
            "--bass-phantom-sparse" => self.bass_phantom_sparse = Some(parse_bool(val, flag)?),
            "--passive-res-max-hz" => {
                self.passive_res_max_hz = Some(parse_f32_range(val, flag, 1_000.0, 20_000.0)?)
            }
            "--ir-duration-secs" => {
                self.ir_duration_secs = Some(parse_f32_range(val, flag, 0.05, 0.50)?)
            }
            "--spatial-tail-split-tap" => {
                self.spatial_tail_split_tap = Some(parse_u16_range(val, flag, 512, u16::MAX)?)
            }
            "--fir-head-taps" => self.fir_head_taps = Some(parse_u8_set(val, flag, &[16, 32, 64])?),
            "--fir-block-fft" => self.fir_block_fft = Some(parse_bool(val, flag)?),
            "--hall-fdn-lines" => self.hall_fdn_lines = Some(parse_u8_set(val, flag, &[4, 8, 16])?),
            "--action-oversample" => {
                self.action_oversample = Some(parse_u8_set(val, flag, &[2, 4, 8])?)
            }
            "--polynomial-adaa" => self.polynomial_adaa = Some(parse_bool(val, flag)?),
            "--shank-flex" => self.shank_flex_enabled = Some(parse_bool(val, flag)?),
            "--damper-decim" => {
                self.damper_control_decim = Some(parse_u8_set(val, flag, &[1, 2, 4, 8, 16])?)
            }
            "--hertz-nonlinear" => self.hertz_nonlinear = Some(parse_bool(val, flag)?),
            "--active-bridge-poles" => {
                self.active_bridge_poles = Some(parse_u8_range(val, flag, 4, 10)?)
            }
            _ => unreachable!(),
        }
        Ok(true)
    }
}

const KNOWN_PROFILE_FLAGS: &[&str] = &[
    "--perf-profile",
    "--horiz-max-hz",
    "--dead-mode-prune-db",
    "--low-order-energy-horizon-db",
    "--energy-horizon-db",
    "--sav-dormancy-db",
    "--sav-max-key",
    "--bass-phantom-sparse",
    "--passive-res-max-hz",
    "--ir-duration-secs",
    "--spatial-tail-split-tap",
    "--fir-head-taps",
    "--fir-block-fft",
    "--hall-fdn-lines",
    "--action-oversample",
    "--polynomial-adaa",
    "--shank-flex",
    "--damper-decim",
    "--hertz-nonlinear",
    "--active-bridge-poles",
];

fn split_flag_if_known<'a>(
    args: &'a [String],
    idx: &mut usize,
) -> Result<Option<(&'static str, &'a str)>, String> {
    let arg = args[*idx].as_str();
    for &flag in KNOWN_PROFILE_FLAGS {
        if arg == flag {
            let next = args
                .get(*idx + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            *idx += 2;
            return Ok(Some((flag, next.as_str())));
        }
        if let Some(rest) = arg.strip_prefix(flag).and_then(|s| s.strip_prefix('=')) {
            *idx += 1;
            return Ok(Some((flag, rest)));
        }
    }
    Ok(None)
}

fn parse_preset(raw: &str) -> Result<PerfPreset, String> {
    match raw {
        "full" => Ok(PerfPreset::Full),
        "balanced" => Ok(PerfPreset::Balanced),
        "realtime" => Ok(PerfPreset::Realtime),
        _ => Err(format!(
            "invalid --perf-profile '{raw}' (expected full, balanced, or realtime)"
        )),
    }
}

fn parse_bool(raw: &str, flag: &str) -> Result<bool, String> {
    match raw {
        "true" | "1" | "on" => Ok(true),
        "false" | "0" | "off" => Ok(false),
        _ => Err(format!(
            "invalid {flag} '{raw}' (expected true/false, 1/0, or on/off)"
        )),
    }
}

fn parse_f32_range(raw: &str, flag: &str, min: f32, max: f32) -> Result<f32, String> {
    let val: f32 = raw
        .parse()
        .map_err(|_| format!("invalid {flag} '{raw}' (expected number)"))?;
    if !val.is_finite() || !(min..=max).contains(&val) {
        return Err(format!("{flag} {raw} out of range [{min}, {max}]"));
    }
    Ok(val)
}

fn parse_u8_range(raw: &str, flag: &str, min: u8, max: u8) -> Result<u8, String> {
    let val: u8 = raw
        .parse()
        .map_err(|_| format!("invalid {flag} '{raw}' (expected integer)"))?;
    if !(min..=max).contains(&val) {
        return Err(format!("{flag} {raw} out of range [{min}, {max}]"));
    }
    Ok(val)
}

fn parse_u16_range(raw: &str, flag: &str, min: u16, max: u16) -> Result<u16, String> {
    let val: u16 = raw
        .parse()
        .map_err(|_| format!("invalid {flag} '{raw}' (expected integer)"))?;
    if !(min..=max).contains(&val) {
        return Err(format!("{flag} {raw} out of range [{min}, {max}]"));
    }
    Ok(val)
}

fn parse_u8_set(raw: &str, flag: &str, allowed: &[u8]) -> Result<u8, String> {
    let val: u8 = raw
        .parse()
        .map_err(|_| format!("invalid {flag} '{raw}' (expected integer)"))?;
    if !allowed.contains(&val) {
        return Err(format!(
            "invalid {flag} '{raw}' (expected one of {allowed:?})"
        ));
    }
    Ok(val)
}

#[cfg(test)]
#[path = "cli_profile_tests.rs"]
mod tests;
