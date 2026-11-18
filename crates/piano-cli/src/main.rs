//! Native verification, WAV export, Wasm SIMD inspection, and static check CLI.

mod bench_composite;
mod bench_rt;
mod check_lines;
mod diag;
mod render_cmd;
mod scenarios;
mod smf_parser;
mod wasm_check;
mod wav_writer;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use check_lines::{check_repo, find_repo_root};
use wasm_check::inspect_wasm_simd;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("bench") => bench_rt::run_bench(),
        Some("check-lines") => cmd_check_lines(&args[1..]),
        Some("check-wasm-simd") => cmd_check_wasm_simd(&args[1..]),
        Some("render-wav") => render_cmd::cmd_render_wav(&args[1..]),
        Some("render-demos") => render_cmd::cmd_render_demos(&args[1..]),
        Some("diag-radiator") => diag::cmd_diag_radiator(&args[1..]),
        Some("diag-bridge") => diag::cmd_diag_bridge(&args[1..]),
        Some("diag-hammer") => diag::cmd_diag_hammer(&args[1..]),
        Some("diag-buzz") => diag::cmd_diag_buzz(&args[1..]),
        Some("scenarios") => scenarios::run_scenarios(&args[1..]),
        _ => Err(usage()),
    };

    if let Err(err) = result {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn usage() -> String {
    format!(
        "Usage:\n  piano-cli bench\n  piano-cli check-lines [root]\n  piano-cli check-wasm-simd <path.wasm>\n  piano-cli scenarios [--json] [--markdown]\n{}\n{}",
        render_cmd::render_usage(),
        diag::diag_usage()
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
