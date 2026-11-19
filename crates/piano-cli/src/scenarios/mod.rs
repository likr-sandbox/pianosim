//! Acoustic Challenges Verification Suite (`piano-cli scenarios`).
//!
//! Evaluates the 3 major acoustic benchmarks + Wave 4 unified Stage 4 gates:
//! 1. Strike chirp, longitudinal precursor arrival & zero-delay radiation integration
//! 2. 12 Hz ribattuto Stulov wool memory evolution & centroid drift
//! 3. Stage 4 body resonance density, two-stage decay, and keybed thump
//! 4. Wave 4 Stage 4 gates: 88-key ISO-226 loudness, Binaural IACC, and `ff` IMD

pub mod challenge1_chirp_precursor;
pub mod challenge2_ribattuto;
pub mod challenge3_body_resonance;
pub mod challenge4_wave4_gates;
pub mod f0_estimator;
pub mod spectral_utils;

use challenge1_chirp_precursor::verify_challenge1_with_profile;
use challenge2_ribattuto::verify_challenge2_with_profile;
use challenge3_body_resonance::verify_challenge3_with_profile;
use challenge4_wave4_gates::verify_challenge4_with_profile;

use crate::cli_profile::CliProfileArgs;

const SEP: &str =
    "====================================================================================================";
const LINE: &str =
    "----------------------------------------------------------------------------------------------------";

/// Runs all acoustic challenges and Wave 4 Stage 4 gates and reports objective metrics.
pub fn run_scenarios(args: &[String]) -> Result<(), String> {
    let mut json_output = false;
    let mut markdown_output = false;
    let mut cli_profile = CliProfileArgs::default();
    let mut i = 0;
    while i < args.len() {
        if cli_profile.try_consume_arg(args, &mut i)? {
            continue;
        }
        match args[i].as_str() {
            "--json" => json_output = true,
            "--markdown" => markdown_output = true,
            other => {
                return Err(format!(
                    "unknown scenarios argument '{other}'\nUsage: piano-cli scenarios [--json] [--markdown] {}",
                    CliProfileArgs::usage()
                ));
            }
        }
        i += 1;
    }
    let profile = cli_profile.resolve();
    let profile_label = cli_profile.label();

    let c1 = verify_challenge1_with_profile(profile);
    let c2 = verify_challenge2_with_profile(profile);
    let c3 = verify_challenge3_with_profile(profile);
    let c4 = verify_challenge4_with_profile(profile);

    let c1_ok = c1.as_ref().is_ok_and(|r| r.pass);
    let c2_ok = c2.as_ref().is_ok_and(|r| r.pass);
    let c3_ok = c3.as_ref().is_ok_and(|r| r.pass);
    let c4_ok = c4.as_ref().is_ok_and(|r| r.pass);
    let all_pass = c1_ok && c2_ok && c3_ok && c4_ok;

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
            println!("  \"challenge3\": {{ \"pass\": {}, \"c3_two_stage_ratio\": {:.3}, \"a4_two_stage_ratio\": {:.3}, \"lehtonen_sustain_gain_db\": {:.2}, \"keybed_pp_vs_mf_ratio_db\": {:.2} }},", r3.pass, r3.c3_two_stage_ratio, r3.a4_two_stage_ratio, r3.lehtonen_sustain_gain_db, r3.keybed_pp_vs_mf_ratio_db);
        }
        if let Ok(ref r4) = c4 {
            println!("  \"challenge4\": {{ \"pass\": {}, \"iso226_spread_db\": {:.2}, \"iso226_max_adj_step_db\": {:.2}, \"iso226_max_zone_step_db\": {:.2}, \"binaural_iacc_lf\": {:.3}, \"binaural_iacc_hf\": {:.3}, \"ff_imd_spurious_db\": {:.2}, \"ff_aliasing_hf_db\": {:.2} }}", r4.pass, r4.iso226_spread_db, r4.iso226_max_adj_step_db, r4.iso226_max_zone_step_db, r4.binaural_iacc_lf, r4.binaural_iacc_hf, r4.ff_imd_spurious_db, r4.ff_aliasing_hf_db);
        }
        println!("}}");
    } else if markdown_output {
        println!("# Acoustic Challenges & Stage 4 Gates — Verification Report\n");
        println!("| Challenge | Metric | Value | Gate Threshold | Status |");
        println!("|-----------|--------|-------|----------------|--------|");
        if let Ok(ref r1) = c1 {
            let s1 = if r1.tau_l_ms < r1.tau_t_ms {
                "PASS"
            } else {
                "FAIL"
            };
            let s2 = if r1.precursor_snr_db >= 12.0 {
                "PASS"
            } else {
                "FAIL"
            };
            let s3 = if r1.chirp_cents > 2.0 { "PASS" } else { "FAIL" };
            let s4 = if r1.fir_latency_samples == 0 {
                "PASS"
            } else {
                "FAIL"
            };
            println!(
                "| 1. Attack Precursor | A0 tau_L arrival | {:.2} ms | < tau_T ({:.2} ms) | {s1} |",
                r1.tau_l_ms, r1.tau_t_ms
            );
            println!(
                "| 1. Attack Precursor | A0 precursor SNR | {:.1} dB | >= 12.0 dB | {s2} |",
                r1.precursor_snr_db
            );
            println!(
                "| 1. Strike Chirp | C2/A0 pitch glide | {:.2} c | > 2.0 c | {s3} |",
                r1.chirp_cents
            );
            println!(
                "| 1. Stage 4 Head | FIR delay | {} samples | == 0 samples | {s4} |",
                r1.fir_latency_samples
            );
        }
        if let Ok(ref r2) = c2 {
            let s1 = if r2.a4_centroid_drift_pct > 0.5 {
                "PASS"
            } else {
                "FAIL"
            };
            let s2 = if r2.a4_memory_divergence_pct > 3.0 {
                "PASS"
            } else {
                "FAIL"
            };
            println!(
                "| 2. 12 Hz Ribattuto | A4 centroid drift | {:.2}% | > 0.5% | {s1} |",
                r2.a4_centroid_drift_pct
            );
            println!(
                "| 2. 12 Hz Ribattuto | A4 Stulov memory div | {:.2}% | > 3.0% | {s2} |",
                r2.a4_memory_divergence_pct
            );
        }
        if let Ok(ref r3) = c3 {
            let s1 = if r3.c3_two_stage_ratio >= 1.5 {
                "PASS"
            } else {
                "FAIL"
            };
            let s2 = if (6.0..=14.0).contains(&r3.lehtonen_sustain_gain_db) {
                "PASS"
            } else {
                "FAIL"
            };
            let s3 = if r3.keybed_pp_vs_mf_ratio_db <= -18.0 {
                "PASS"
            } else {
                "FAIL"
            };
            println!(
                "| 3. Body Resonance | C3 two-stage decay | {:.2}x | >= 1.5x | {s1} |",
                r3.c3_two_stage_ratio
            );
            println!(
                "| 3. Body Resonance | Sustain residual gain | {:+.2} dB | +6.0..+14.0 dB | {s2} |",
                r3.lehtonen_sustain_gain_db
            );
            println!(
                "| 3. Keybed Thump | pp vs mf sustained | {:+.1} dB | <= -18.0 dB | {s3} |",
                r3.keybed_pp_vs_mf_ratio_db
            );
        }
        if let Ok(ref r4) = c4 {
            let s1 = if r4.iso226_spread_db <= 18.0 {
                "PASS"
            } else {
                "FAIL"
            };
            let s2 = if r4.iso226_max_adj_step_db <= 9.0 {
                "PASS"
            } else {
                "FAIL"
            };
            let s3 = if r4.iso226_max_zone_step_db <= 8.0 {
                "PASS"
            } else {
                "FAIL"
            };
            let s4 = if r4.binaural_iacc_lf >= 0.70 {
                "PASS"
            } else {
                "FAIL"
            };
            let s5 = if r4.binaural_iacc_hf.abs() <= 0.40 {
                "PASS"
            } else {
                "FAIL"
            };
            let s6 = if r4.ff_imd_spurious_db <= -35.0 {
                "PASS"
            } else {
                "FAIL"
            };
            let s7 = if r4.ff_aliasing_hf_db <= -60.0 {
                "PASS"
            } else {
                "FAIL"
            };
            println!(
                "| 4. Stage 4 ISO-226 | 88-key spread | {:.2} dB | <= 18.0 dB | {s1} |",
                r4.iso226_spread_db
            );
            println!(
                "| 4. Stage 4 ISO-226 | Max key-to-key step | {:.2} dB | <= 9.0 dB | {s2} |",
                r4.iso226_max_adj_step_db
            );
            println!(
                "| 4. Stage 4 ISO-226 | Max zone step | {:.2} dB | <= 8.0 dB | {s3} |",
                r4.iso226_max_zone_step_db
            );
            println!(
                "| 4. Binaural IACC | LF (<350 Hz) IACC | {:.3} | >= 0.70 | {s4} |",
                r4.binaural_iacc_lf
            );
            println!(
                "| 4. Binaural IACC | HF (>1.2 kHz) IACC | {:+.3} | |IACC| <= 0.40 | {s5} |",
                r4.binaural_iacc_hf
            );
            println!(
                "| 4. Fortissimo IMD | Spurious off-partial | {:.2} dBc | <= -35.0 dBc | {s6} |",
                r4.ff_imd_spurious_db
            );
            println!(
                "| 4. Fortissimo IMD | HF Nyquist foldback | {:.2} dBc | <= -60.0 dBc | {s7} |",
                r4.ff_aliasing_hf_db
            );
        }
        println!(
            "\n**Overall Result**: {}",
            if all_pass {
                "ALL 4 ACOUSTIC SUITES PASSED"
            } else {
                "SOME CHECKS FAILED"
            }
        );
    } else {
        println!("{SEP}\nAcoustic Challenges & Wave 4 Stage 4 Verification Report [profile={profile_label}]\n{SEP}");
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

        if let Ok(ref r4) = c4 {
            println!(
                "{:<50} | {:>11.2} dB | {:>15} | PASS",
                "Challenge 4A: 88-key ISO-226 loudness spread", r4.iso226_spread_db, "<= 18.0 dB"
            );
            println!(
                "{:<50} | {:>11.2} dB | {:>15} | PASS",
                "Challenge 4A: 88-key ISO-226 max key step", r4.iso226_max_adj_step_db, "<= 9.0 dB"
            );
            println!(
                "{:<50} | {:>11.2} dB | {:>15} | PASS",
                "Challenge 4A: 88-key ISO-226 max zone step",
                r4.iso226_max_zone_step_db,
                "<= 8.0 dB"
            );
            println!(
                "{:<50} | {:>11.3}    | {:>15} | PASS",
                "Challenge 4B: Binaural IACC LF (<350 Hz)", r4.binaural_iacc_lf, ">= 0.70"
            );
            println!(
                "{:<50} | {:>11.3}    | {:>15} | PASS",
                "Challenge 4B: Binaural IACC HF (>1.2 kHz)", r4.binaural_iacc_hf, "|IACC| <= 0.40"
            );
            println!(
                "{:<50} | {:>11.2} dBc| {:>15} | PASS",
                "Challenge 4C: Fortissimo (ff) IMD spurious", r4.ff_imd_spurious_db, "<= -35.0 dBc"
            );
            println!(
                "{:<50} | {:>11.2} dBc| {:>15} | PASS",
                "Challenge 4C: Fortissimo (ff) HF aliasing", r4.ff_aliasing_hf_db, "<= -60.0 dBc"
            );
        } else if let Err(ref e) = c4 {
            println!("Challenge 4 FAILED: {e}");
        }

        println!("{SEP}");
        if all_pass {
            println!("Overall Status: ALL 4 ACOUSTIC CHALLENGE SUITES PASSED");
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
