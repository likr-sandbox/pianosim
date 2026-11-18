//! Multi-body `.pbdy` preset verification: distinct physical geometry/spectra,
//! pure-Rust deterministic `.pbdy` asset parity, passivity, and pole cancellation.

use crate::bake::{bake_body_with_diagnostics, BakeDiagnostics};
use crate::pole_cancel::verify_pole_cancellation;
use crate::presets::{body_spec, BodyArchetype};
use crate::ribs_rim::rim_signed_distance_for;
use crate::tests::cached_concert_grand;
use crate::types::{zeta_to_t60, MAX_MODAL_T60_S};
use piano_bridge::NUM_BRIDGE_ZONES;
use piano_spec::BodyPreset;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

fn cached_body(arch: BodyArchetype) -> &'static (BodyPreset, BakeDiagnostics) {
    match arch {
        BodyArchetype::ConcertGrand => cached_concert_grand(),
        BodyArchetype::UprightStudio => {
            static UP: OnceLock<(BodyPreset, BakeDiagnostics)> = OnceLock::new();
            UP.get_or_init(|| bake_body_with_diagnostics(BodyArchetype::UprightStudio))
        }
        BodyArchetype::VintageSaloon => {
            static VS: OnceLock<(BodyPreset, BakeDiagnostics)> = OnceLock::new();
            VS.get_or_init(|| bake_body_with_diagnostics(BodyArchetype::VintageSaloon))
        }
    }
}

fn asset_path(arch: BodyArchetype) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../piano-radiator/assets")
        .join(arch.asset_filename())
}

#[test]
fn all_three_pbdy_assets_decode_cleanly_and_match_deterministic_bake() {
    for arch in BodyArchetype::ALL {
        let path = asset_path(arch);
        let disk_bytes = fs::read(&path).unwrap_or_else(|e| {
            panic!("failed to read {}: {e}", path.display());
        });
        assert!(
            disk_bytes.len() < 500 * 1024,
            "{} exceeds 500 KB ({} bytes)",
            arch.slug(),
            disk_bytes.len()
        );
        let decoded = BodyPreset::decode(&disk_bytes)
            .unwrap_or_else(|e| panic!("{} failed decode: {e}", arch.slug()));
        let (baked, diag) = cached_body(arch);
        assert_eq!(
            disk_bytes,
            baked.encode(),
            "on-disk {} out of sync with deterministic baker",
            arch.asset_filename()
        );
        assert_eq!(decoded.modes.len(), diag.total_modes);
        assert!(
            (1_000..=2_000).contains(&diag.total_modes),
            "{} mode count {} out of range",
            arch.slug(),
            diag.total_modes
        );
        assert!(
            diag.plate.ritz_mode_count >= 120,
            "{} 15x20 Ritz mode count {} unexpectedly low",
            arch.slug(),
            diag.plate.ritz_mode_count
        );
    }
}

#[test]
fn multi_body_pole_cancellation_and_passivity_across_sample_rates() {
    for arch in BodyArchetype::ALL {
        let (preset, diag) = cached_body(arch);
        assert!(
            diag.max_t60_s <= MAX_MODAL_T60_S,
            "{} max T60 {:.3} s > {MAX_MODAL_T60_S}",
            arch.slug(),
            diag.max_t60_s
        );
        for m in &preset.modes {
            let t60 = zeta_to_t60(m.freq_hz as f64, m.zeta as f64);
            assert!(t60 <= MAX_MODAL_T60_S + 1.0e-4);
            assert!(m.freq_hz.is_finite() && m.zeta.is_finite() && m.zeta > 0.0);
        }

        // Stage 3 passivity preconditions.
        assert!(preset
            .y_inf
            .iter()
            .all(|&y| y.is_finite() && y > 0.0 && y < 0.01));
        assert!(preset.reduced_poles.iter().all(|p| {
            p.freq_hz.is_finite()
                && p.freq_hz > 20.0
                && p.zeta.is_finite()
                && p.zeta > 0.0
                && p.modal_mass_kg.is_finite()
                && p.modal_mass_kg > 0.0
        }));

        let decoded = BodyPreset::decode(&preset.encode()).expect("roundtrip decode");
        for sr in [44_100.0_f32, 48_000.0_f32, 96_000.0_f32] {
            let rep = verify_pole_cancellation(&decoded, sr);
            assert!(
                rep.max_inverse_pole_real < 0.0,
                "{} @ {sr}: inverse pole Re = {} >= 0 (non-passive)",
                arch.slug(),
                rep.max_inverse_pole_real
            );
            assert!(
                rep.max_inverse_pole_real < -1.0,
                "{} @ {sr}: inverse pole Re = {} not well inside LHP",
                arch.slug(),
                rep.max_inverse_pole_real
            );
            assert!(
                rep.max_matrix_residual < 1.0e-6,
                "{} @ {sr}: matrix residual {}",
                arch.slug(),
                rep.max_matrix_residual
            );
            assert!(
                rep.max_uncancelled_peaking_db > 3.0,
                "{} @ {sr}: uncancelled peaking {} dB <= 3 dB",
                arch.slug(),
                rep.max_uncancelled_peaking_db
            );
            assert!(
                rep.max_cancelled_excess_db <= 1.0,
                "{} @ {sr}: cancelled excess {} dB > 1.0 dB",
                arch.slug(),
                rep.max_cancelled_excess_db
            );
            assert!(
                rep.max_discrete_cascade_err_db <= 1.0,
                "{} @ {sr}: discrete cascade err {} dB > 1.0 dB",
                arch.slug(),
                rep.max_discrete_cascade_err_db
            );
        }
    }
}

#[test]
fn multi_body_presets_differ_measurably_in_geometry_impedance_and_spectra() {
    let (grand, grand_d) = cached_body(BodyArchetype::ConcertGrand);
    let (upright, upright_d) = cached_body(BodyArchetype::UprightStudio);
    let (saloon, saloon_d) = cached_body(BodyArchetype::VintageSaloon);

    // Bridge zones must lie safely inside each archetype's soundboard rim.
    for arch in BodyArchetype::ALL {
        let spec = body_spec(arch);
        for (z, &[x, y, _]) in spec.zone_pos_m.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let d = rim_signed_distance_for(&spec, x as f64, y as f64);
            assert!(
                d > 0.08,
                "{} zone {z} ({x}, {y}) too close/outside rim: {d}",
                arch.slug()
            );
        }
    }

    // Distinct Stage 3 y_inf: Upright stiffer (lower y_inf), Saloon more compliant (higher y_inf).
    let mean_y =
        |p: &BodyPreset| -> f32 { p.y_inf.iter().sum::<f32>() / (NUM_BRIDGE_ZONES as f32) };
    assert!(
        mean_y(upright) < mean_y(grand) * 0.90,
        "upright y_inf {} should be noticeably lower than grand {}",
        mean_y(upright),
        mean_y(grand)
    );
    assert!(
        mean_y(saloon) > mean_y(grand) * 1.15,
        "saloon y_inf {} should be noticeably higher than grand {}",
        mean_y(saloon),
        mean_y(grand)
    );

    // Distinct reduced pole frequencies.
    assert!(upright.reduced_poles[0].freq_hz > grand.reduced_poles[0].freq_hz + 10.0);
    assert!(saloon.reduced_poles[0].freq_hz < grand.reduced_poles[0].freq_hz - 3.0);
    assert_ne!(grand.reduced_poles, upright.reduced_poles);
    assert_ne!(grand.reduced_poles, saloon.reduced_poles);
    assert_ne!(upright.reduced_poles, saloon.reduced_poles);

    // Distinct modal spectra: compact stiff upright has highest lowest plate mode; grand extends lowest.
    assert!(
        upright_d.plate.min_ritz_hz > grand_d.plate.min_ritz_hz + 12.0,
        "upright min Ritz {:.1} Hz vs grand {:.1} Hz",
        upright_d.plate.min_ritz_hz,
        grand_d.plate.min_ritz_hz
    );
    assert!(
        (34.0..=55.0).contains(&saloon_d.plate.min_ritz_hz),
        "saloon min Ritz {:.1} Hz out of parlor range",
        saloon_d.plate.min_ritz_hz
    );
    assert!(
        (grand_d.plate.min_ritz_hz - saloon_d.plate.min_ritz_hz).abs() > 1.0,
        "grand {:.1} Hz vs saloon {:.1} Hz should differ",
        grand_d.plate.min_ritz_hz,
        saloon_d.plate.min_ritz_hz
    );

    // Sub-50 Hz plate foundation, mid/treble spectral ratio, and HF T60 differ measurably across all 3 bodies.
    let band_drive_rms = |p: &BodyPreset, f_lo: f32, f_hi: f32| -> f64 {
        let mut sum_sq = 0.0_f64;
        for m in &p.modes {
            if m.freq_hz >= f_lo && m.freq_hz < f_hi {
                let rad = m.rad_coeffs[0] as f64;
                for &zd in &m.zone_drive[..NUM_BRIDGE_ZONES] {
                    let v = (zd as f64) * rad;
                    sum_sq += v * v;
                }
            }
        }
        sum_sq.sqrt()
    };
    let sub50_g = band_drive_rms(grand, 28.0, 50.0);
    let sub50_u = band_drive_rms(upright, 28.0, 50.0);
    let sub50_s = band_drive_rms(saloon, 28.0, 50.0);
    assert!(
        sub50_g > 1.0e-4 && sub50_s > 1.0e-4 && sub50_u < 1.0e-9,
        "sub-50 Hz foundation should be active for grand ({sub50_g:.5}) & saloon ({sub50_s:.5}), absent for stiff upright ({sub50_u:.5})"
    );

    let mid_hf_ratio = |p: &BodyPreset| -> f64 {
        band_drive_rms(p, 150.0, 700.0) / band_drive_rms(p, 2_000.0, 6_000.0).max(1.0e-12)
    };
    let hf_mean_t60 = |p: &BodyPreset| -> f64 {
        let (mut sum, mut cnt) = (0.0_f64, 0_usize);
        for m in &p.modes {
            if m.freq_hz >= 2_000.0 {
                sum += zeta_to_t60(m.freq_hz as f64, m.zeta as f64);
                cnt += 1;
            }
        }
        sum / (cnt.max(1) as f64)
    };
    let (r_g, r_u, r_s) = (
        mid_hf_ratio(grand),
        mid_hf_ratio(upright),
        mid_hf_ratio(saloon),
    );
    let (t_g, t_u, t_s) = (
        hf_mean_t60(grand),
        hf_mean_t60(upright),
        hf_mean_t60(saloon),
    );
    eprintln!(
        "Body ratios (mid/HF): grand={r_g:.4}, upright={r_u:.4}, saloon={r_s:.4}; HF T60: grand={t_g:.4}s, upright={t_u:.4}s, saloon={t_s:.4}s"
    );
    assert!(
        r_s > r_g * 1.20 && r_g > r_u * 1.15,
        "mid/HF spectral ordering (saloon > grand > upright) should differ measurably: grand={r_g:.4}, upright={r_u:.4}, saloon={r_s:.4}"
    );
    assert!(
        t_g > t_u + 0.004 && t_u > t_s + 0.004,
        "HF T60 ordering should reflect spruce vs compact vs aged saloon: grand={t_g:.4}, upright={t_u:.4}, saloon={t_s:.4}"
    );
    assert_ne!(grand.zone_pos_m, upright.zone_pos_m);
    assert_ne!(grand.zone_pos_m, saloon.zone_pos_m);
    assert_ne!(grand.direct_zone_gain, upright.direct_zone_gain);
    assert_ne!(grand.direct_zone_gain, saloon.direct_zone_gain);
}
