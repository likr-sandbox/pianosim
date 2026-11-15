//! Offline Layer 1A CLI preset baker: generates a calibrated concert-grand [`BodyPreset`]
//! binary file encoded via [`piano_spec::BodyPreset::encode`].

use piano_body_gen::{bake_concert_grand_with_diagnostics, verify_pole_cancellation};
use piano_spec::BodyPreset;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const DEFAULT_OUTPUT_PATH: &str = "target/concert_grand_body.pbdy";

fn main() {
    let out_path = env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from(DEFAULT_OUTPUT_PATH), PathBuf::from);

    let (preset, diag) = bake_concert_grand_with_diagnostics();
    let encoded = preset.encode();

    if let Err(err) = BodyPreset::decode(&encoded) {
        eprintln!("bake_preset error: self-check decode failed: {err}");
        process::exit(1);
    }

    let report = verify_pole_cancellation(&preset, 48_000.0);
    if report.max_cancelled_excess_db > 1.0 {
        eprintln!(
            "bake_preset error: pole-cancellation excess {:.3} dB exceeds 1.0 dB",
            report.max_cancelled_excess_db
        );
        process::exit(1);
    }

    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(err) = fs::create_dir_all(parent) {
                eprintln!(
                    "bake_preset error: failed to create {}: {err}",
                    parent.display()
                );
                process::exit(1);
            }
        }
    }

    if let Err(err) = fs::write(&out_path, &encoded) {
        eprintln!(
            "bake_preset error: failed to write {}: {err}",
            out_path.display()
        );
        process::exit(1);
    }

    println!(
        "Baked {} modes ({:.1}..{:.1} Hz, T60 {:.3}..{:.3} s), {} bytes, cancel_err={:.4} dB -> {}",
        diag.total_modes,
        diag.min_freq_hz,
        diag.max_freq_hz,
        diag.min_t60_s,
        diag.max_t60_s,
        encoded.len(),
        report.max_cancelled_excess_db,
        out_path.display()
    );
}
