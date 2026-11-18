//! Offline Layer 1A CLI preset baker: generates calibrated [`BodyPreset`] `.pbdy`
//! binary files for one or all body archetypes (`concert_grand`, `upright_studio`,
//! `vintage_saloon`).
//!
//! Usage:
//! - `bake_preset --all [assets_dir]` (defaults to `crates/piano-radiator/assets`)
//! - `bake_preset <archetype> [out_path]`
//! - `bake_preset [out_path]` (bakes `concert_grand`)

use piano_body_gen::{bake_body_with_diagnostics, verify_pole_cancellation, BodyArchetype};
use piano_spec::BodyPreset;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const DEFAULT_ASSETS_DIR: &str = "crates/piano-radiator/assets";

fn bake_and_write(archetype: BodyArchetype, out_path: &Path) {
    let (preset, diag) = bake_body_with_diagnostics(archetype);
    let encoded = preset.encode();

    if let Err(err) = BodyPreset::decode(&encoded) {
        eprintln!(
            "bake_preset error ({}): self-check decode failed: {err}",
            archetype.slug()
        );
        process::exit(1);
    }

    let report = verify_pole_cancellation(&preset, 48_000.0);
    if report.max_inverse_pole_real >= 0.0 {
        eprintln!(
            "bake_preset error ({}): non-passive inverse pole real {:.4}",
            archetype.slug(),
            report.max_inverse_pole_real
        );
        process::exit(1);
    }
    if report.max_cancelled_excess_db > 1.0 {
        eprintln!(
            "bake_preset error ({}): pole-cancellation excess {:.3} dB exceeds 1.0 dB",
            archetype.slug(),
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

    if let Err(err) = fs::write(out_path, &encoded) {
        eprintln!(
            "bake_preset error: failed to write {}: {err}",
            out_path.display()
        );
        process::exit(1);
    }

    println!(
        "Baked [{}] {} modes ({:.1}..{:.1} Hz, T60 {:.3}..{:.3} s), {} bytes, cancel_err={:.4} dB -> {}",
        archetype.slug(),
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

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        let out =
            PathBuf::from(DEFAULT_ASSETS_DIR).join(BodyArchetype::ConcertGrand.asset_filename());
        bake_and_write(BodyArchetype::ConcertGrand, &out);
        return;
    }

    if args[0] == "--all" {
        let dir = args
            .get(1)
            .map_or_else(|| PathBuf::from(DEFAULT_ASSETS_DIR), PathBuf::from);
        for arch in BodyArchetype::ALL {
            bake_and_write(arch, &dir.join(arch.asset_filename()));
        }
        return;
    }

    if let Some(arch) = BodyArchetype::from_slug(&args[0]) {
        let out = args.get(1).map_or_else(
            || PathBuf::from(DEFAULT_ASSETS_DIR).join(arch.asset_filename()),
            PathBuf::from,
        );
        bake_and_write(arch, &out);
        return;
    }

    bake_and_write(BodyArchetype::ConcertGrand, Path::new(&args[0]));
}
