//! Offline acoustic diagnostic subcommands (`diag-radiator`, `diag-bridge`,
//! `diag-hammer`, `diag-buzz`).

pub mod bridge;
pub mod buzz;
pub mod hammer;
pub mod radiator;

pub use bridge::cmd_diag_bridge;
pub use buzz::cmd_diag_buzz;
pub use hammer::cmd_diag_hammer;
pub use radiator::cmd_diag_radiator;

/// Returns CLI usage lines for all diagnostic subcommands.
#[must_use]
pub(crate) fn diag_usage() -> &'static str {
    "  piano-cli diag-radiator [out_dir] [--body <concert-grand|upright-studio|vintage-saloon>] [--perspective <player|close|hall|binaural>] [--lid <closed|half|open>] [--reverb-mix <f32>] [--perf-profile <full|balanced|realtime> ...]\n  piano-cli diag-bridge [out_dir] [--keys C2,C4]\n  piano-cli diag-hammer [out_csv] [--in-csv <path>] [--midi <u8>] [--vel <u8>] [--duration-ms <f64>]\n  piano-cli diag-buzz [wav_dir] [--demos-dir <dir>]"
}

pub(crate) fn parse_flag_value<'a>(
    args: &'a [String],
    idx: &mut usize,
    name: &str,
    usage_str: &str,
) -> Result<Option<&'a str>, String> {
    let arg = args[*idx].as_str();
    if arg == name {
        let next = args.get(*idx + 1).ok_or_else(|| String::from(usage_str))?;
        *idx += 2;
        return Ok(Some(next.as_str()));
    }
    if let Some(rest) = arg.strip_prefix(name).and_then(|s| s.strip_prefix('=')) {
        *idx += 1;
        return Ok(Some(rest));
    }
    Ok(None)
}
