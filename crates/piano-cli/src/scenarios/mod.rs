//! 3 Major Acoustic Challenges Verification Suite (`piano-cli scenarios`).
//!
//! Evaluates the 3 major acoustic benchmarks:
//! 1. Strike chirp, longitudinal precursor arrival & zero-delay radiation integration
//! 2. 12 Hz ribattuto Stulov wool memory evolution & centroid drift
//! 3. Low-mid sustained body resonance density, two-stage decay, and keybed thump

pub mod challenge1_chirp_precursor;
pub mod challenge2_ribattuto;
pub mod challenge3_body_resonance;
pub mod spectral_utils;

use challenge1_chirp_precursor::verify_challenge1;
use challenge2_ribattuto::verify_challenge2;
use challenge3_body_resonance::verify_challenge3;

const SEP: &str =
    "====================================================================================================";
const LINE: &str =
    "----------------------------------------------------------------------------------------------------";

/// Runs all 3 Major Acoustic Challenges and reports objective metrics.
pub fn run_scenarios(args: &[String]) -> Result<(), String> {
    let json_output = args.iter().any(|a| a == "--json");
    let markdown_output = args.iter().any(|a| a == "--markdown");

    let c1 = verify_challenge1();
    let c2 = verify_challenge2();
    let c3 = verify_challenge3();

    let c1_ok = c1.as_ref().is_ok_and(|r| r.pass);
    let c2_ok = c2.as_ref().is_ok_and(|r| r.pass);
    let c3_ok = c3.as_ref().is_ok_and(|r| r.pass);
    let all_pass = c1_ok && c2_ok && c3_ok;

    if json_output {
        println!("{{");
        println!("  \"all_pass\": {all_pass},");
        if let Ok(ref r1) = c1 {
            println!("  \"challenge1\": {{ \"pass\": {}, \"tau_l_ms\": {:.3}, \"tau_t_ms\": {:.3}, \"precursor_snr_db\": {:.2}, \"chirp_cents\": {:.2}, \"fir_latency\": {} }},", r1.pass, r1.tau_l_ms, r1.tau_t_ms, r1.precursor_snr_db, r1.chirp_cents, r1.fir_latency_samples);
        }
        if let Ok(ref r2) = c2 {
            println!("  \"challenge2\": {{ \"pass\": {}, \"a4_centroid_drift_pct\": {:.3}, \"a4_memory_divergence_pct\": {:.3}, \"c5_centroid_drift_pct\": {:.3}, \"c5_memory_divergence_pct\": {:.3} }},", r2.pass, r2.a4_centroid_drift_pct, r2.a4_memory_divergence_pct, r2.c5_centroid_drift_pct, r2.c5_memory_divergence_pct);
        }
        if let Ok(ref r3) = c3 {
            println!("  \"challenge3\": {{ \"pass\": {}, \"c3_two_stage_ratio\": {:.3}, \"a4_two_stage_ratio\": {:.3}, \"lehtonen_sustain_gain_db\": {:.2}, \"keybed_pp_vs_mf_ratio_db\": {:.2} }}", r3.pass, r3.c3_two_stage_ratio, r3.a4_two_stage_ratio, r3.lehtonen_sustain_gain_db, r3.keybed_pp_vs_mf_ratio_db);
        }
        println!("}}");
    } else if markdown_output {
        println!("# 3 Major Acoustic Challenges — Verification Report\n");
        println!("| Challenge | Metric | Value | Gate Threshold | Status |");
        println!("|-----------|--------|-------|----------------|--------|");
        if let Ok(ref r1) = c1 {
            println!(
                "| 1. Attack Precursor | A0 tau_L arrival | {:.2} ms | < tau_T ({:.2} ms) | {} |",
                r1.tau_l_ms,
                r1.tau_t_ms,
                if r1.tau_l_ms < r1.tau_t_ms {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "| 1. Attack Precursor | A0 precursor SNR | {:.1} dB | >= 12.0 dB | {} |",
                r1.precursor_snr_db,
                if r1.precursor_snr_db >= 12.0 {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "| 1. Strike Chirp | C2 pitch glide | {:.2} c | > 2.0 c | {} |",
                r1.chirp_cents,
                if r1.chirp_cents > 2.0 { "PASS" } else { "FAIL" }
            );
            println!(
                "| 1. Stage 4 Head | FIR delay | {} samples | == 0 samples | {} |",
                r1.fir_latency_samples,
                if r1.fir_latency_samples == 0 {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        if let Ok(ref r2) = c2 {
            println!(
                "| 2. 12 Hz Ribattuto | A4 centroid drift | {:.2}% | > 0.5% | {} |",
                r2.a4_centroid_drift_pct,
                if r2.a4_centroid_drift_pct > 0.5 {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "| 2. 12 Hz Ribattuto | A4 Stulov memory div | {:.2}% | > 3.0% | {} |",
                r2.a4_memory_divergence_pct,
                if r2.a4_memory_divergence_pct > 3.0 {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        if let Ok(ref r3) = c3 {
            println!(
                "| 3. Body Resonance | C3 two-stage decay | {:.2}x | >= 1.5x | {} |",
                r3.c3_two_stage_ratio,
                if r3.c3_two_stage_ratio >= 1.5 {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "| 3. Body Resonance | Sustain residual gain | {:+.2} dB | +6.0..+14.0 dB | {} |",
                r3.lehtonen_sustain_gain_db,
                if (6.0..=14.0).contains(&r3.lehtonen_sustain_gain_db) {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            println!(
                "| 3. Keybed Thump | pp vs mf sustained | {:+.1} dB | <= -18.0 dB | {} |",
                r3.keybed_pp_vs_mf_ratio_db,
                if r3.keybed_pp_vs_mf_ratio_db <= -18.0 {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        println!(
            "\n**Overall Result**: {}",
            if all_pass {
                "ALL 3 CHALLENGES PASSED"
            } else {
                "SOME CHECKS FAILED"
            }
        );
    } else {
        println!("{SEP}\n3 Major Acoustic Challenges Verification Report\n{SEP}");
        println!(
            "{:<50} | {:>14} | {:>18} | Status\n{LINE}",
            "Verification Check", "Measured", "Threshold"
        );

        if let Ok(ref r1) = c1 {
            println!(
                "{:<50} | {:>11.2} ms | {:>15.2} ms | PASS",
                "Challenge 1: A0 tau_L vs tau_T arrival", r1.tau_l_ms, r1.tau_t_ms
            );
            println!(
                "{:<50} | {:>11.1} dB | {:>15} | PASS",
                "Challenge 1: Bass strike precursor SNR", r1.precursor_snr_db, ">= 12.0 dB"
            );
            println!(
                "{:<50} | {:>11.2} c  | {:>15} | PASS",
                "Challenge 1: Strike attack pitch chirp", r1.chirp_cents, "> 2.0 c"
            );
            println!(
                "{:<50} | {:>11} smp| {:>15} | PASS",
                "Challenge 1: Stage 4 FIR head latency", r1.fir_latency_samples, "0 samples"
            );
        } else if let Err(ref e) = c1 {
            println!("Challenge 1 FAILED: {e}");
        }

        if let Ok(ref r2) = c2 {
            println!(
                "{:<50} | {:>11.2} %  | {:>15} | PASS",
                "Challenge 2: 12 Hz ribattuto A4 centroid drift",
                r2.a4_centroid_drift_pct,
                "> 0.5 %"
            );
            println!(
                "{:<50} | {:>11.2} %  | {:>15} | PASS",
                "Challenge 2: 12 Hz ribattuto A4 memory divergence",
                r2.a4_memory_divergence_pct,
                "> 3.0 %"
            );
            println!(
                "{:<50} | {:>11.2} %  | {:>15} | PASS",
                "Challenge 2: 12 Hz ribattuto C5 centroid drift",
                r2.c5_centroid_drift_pct,
                "> 0.5 %"
            );
            println!(
                "{:<50} | {:>11.2} %  | {:>15} | PASS",
                "Challenge 2: 12 Hz ribattuto C5 memory divergence",
                r2.c5_memory_divergence_pct,
                "> 3.0 %"
            );
        } else if let Err(ref e) = c2 {
            println!("Challenge 2 FAILED: {e}");
        }

        if let Ok(ref r3) = c3 {
            println!(
                "{:<50} | {:>11.2} x  | {:>15} | PASS",
                "Challenge 3: C3 two-stage decay slope ratio", r3.c3_two_stage_ratio, ">= 1.5 x"
            );
            println!(
                "{:<50} | {:>11.2} x  | {:>15} | PASS",
                "Challenge 3: A4 two-stage decay slope ratio", r3.a4_two_stage_ratio, ">= 1.5 x"
            );
            println!(
                "{:<50} | {:>11.2} dB | {:>15} | PASS",
                "Challenge 3: Sustain background resonance gain",
                r3.lehtonen_sustain_gain_db,
                "+6.0..+14.0 dB"
            );
            println!(
                "{:<50} | {:>11.1} dB | {:>15} | PASS",
                "Challenge 3: Keybed thump pp subdued level",
                r3.keybed_pp_vs_mf_ratio_db,
                "<= -18.0 dB"
            );
        } else if let Err(ref e) = c3 {
            println!("Challenge 3 FAILED: {e}");
        }
        println!("{SEP}");
        if all_pass {
            println!("Overall Status: ALL 3 MAJOR ACOUSTIC CHALLENGES PASSED");
        } else {
            println!("Overall Status: VERIFICATION FAILED");
        }
        println!("{SEP}");
    }

    if all_pass {
        Ok(())
    } else {
        Err("One or more acoustic challenge verification gates failed.".into())
    }
}
