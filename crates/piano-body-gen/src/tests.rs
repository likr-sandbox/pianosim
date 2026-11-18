//! Comprehensive Layer 1A solver, physics, Zone 8 keybed, pole-cancellation, and codec tests.

use crate::bake::{bake_concert_grand_with_diagnostics, BakeDiagnostics};
use crate::linalg::{
    householder_tridiagonal_ql, invert_8x8_complex, jacobi_symmetric_eigen, real_eigenvalues_qr,
    solve_generalized_symmetric_eigen, Complex64,
};
use crate::orthotropic_plate::{QUAD_NX, QUAD_NY, RITZ_N, RITZ_NX, RITZ_NY};
use crate::pole_cancel::verify_pole_cancellation;
use crate::ribs_rim::{evaluate_plate_site, rim_signed_distance_m, CONCERT_ZONE_POS_M};
use crate::types::{zeta_to_t60, MAX_MODAL_T60_S};
use piano_bridge::{AdmittanceSsm, DEFAULT_Y_INF, KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES};
use piano_spec::{BodyModeRecord, BodyPreset, PRESET_RAD_COEFFS};
use std::f64::consts::PI;
use std::sync::OnceLock;

pub(crate) fn cached_concert_grand() -> &'static (BodyPreset, BakeDiagnostics) {
    static CACHE: OnceLock<(BodyPreset, BakeDiagnostics)> = OnceLock::new();
    CACHE.get_or_init(bake_concert_grand_with_diagnostics)
}

/// Runs a pure second-order modal bank over `modes` for a scalar drive sequence `x[n]`,
/// outputting monopole-summed time series (zero separate envelope decay).
fn simulate_modal_bank_from_zone(
    modes: &[BodyModeRecord],
    zone_idx: usize,
    direct_gain: f32,
    input: &[f64],
    sample_rate: f64,
    freq_filter: impl Fn(f32) -> bool,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; input.len()];
    for (n, &x) in input.iter().enumerate() {
        out[n] += (direct_gain as f64) * x;
    }

    for mode in modes {
        if !freq_filter(mode.freq_hz) {
            continue;
        }
        let drive = (mode.zone_drive[zone_idx] as f64) * (mode.rad_coeffs[0] as f64);
        if drive.abs() < 1.0e-18 {
            continue;
        }
        let w = 2.0 * PI * (mode.freq_hz as f64);
        let z = mode.zeta as f64;
        let ts = 1.0 / sample_rate;
        let wd = w * (1.0 - z * z).max(1.0e-6).sqrt();
        let decay = (-z * w * ts).exp();
        let a1 = -2.0 * decay * (wd * ts).cos();
        let a2 = decay * decay;
        let b1 = 2.0 * z * w * ts * decay;

        let mut y1 = 0.0_f64;
        let mut y2 = 0.0_f64;
        for (n, &x) in input.iter().enumerate() {
            let x_prev = if n > 0 { input[n - 1] } else { 0.0 };
            let y = drive * (x - x_prev * 0.5) * b1.max(1.0e-6) - a1 * y1 - a2 * y2;
            y2 = y1;
            y1 = y;
            out[n] += y;
        }
    }
    out
}

fn band_power_dft(signal: &[f64], sample_rate: f64, f_lo: f64, f_hi: f64, bins: usize) -> f64 {
    let mut total = 0.0;
    for b in 0..bins {
        let f = f_lo + (f_hi - f_lo) * (b as f64) / ((bins - 1).max(1) as f64);
        let w = 2.0 * PI * f / sample_rate;
        let mut re = 0.0;
        let mut im = 0.0;
        for (n, &x) in signal.iter().enumerate() {
            let ang = w * (n as f64);
            re += x * ang.cos();
            im -= x * ang.sin();
        }
        total += re * re + im * im;
    }
    total / (bins as f64)
}

#[test]
fn mode_count_span_and_determinism() {
    let (preset_a, diag) = cached_concert_grand();
    let (preset_b, _) = bake_concert_grand_with_diagnostics();

    assert_eq!((RITZ_NX, RITZ_NY, RITZ_N), (15, 20, 300));
    assert_eq!((QUAD_NX, QUAD_NY), (48, 64));

    assert!(
        (1_000..=2_000).contains(&diag.total_modes),
        "expected 1000..=2000 modes, got {}",
        diag.total_modes
    );
    assert!(
        (28.0..=45.0).contains(&diag.min_freq_hz),
        "lowest mode {} Hz outside concert-grand ~32-42 Hz range",
        diag.min_freq_hz
    );
    assert!(
        (16_000.0..=16_550.0).contains(&diag.max_freq_hz),
        "highest mode {} Hz should reach ~16.5 kHz",
        diag.max_freq_hz
    );
    assert!(diag.plate.ritz_mode_count >= 90);
    assert!(diag.plate.statistical_mode_count >= 1_000);
    assert_eq!(diag.rim_keybed_modes, 32);
    assert_eq!(diag.frame_modes, 40);
    assert_eq!(diag.cavity_modes, 44);

    for m in &preset_a.modes {
        assert!(m.freq_hz.is_finite() && m.freq_hz >= 28.0 && m.freq_hz <= 16_550.0);
        assert!(m.zeta.is_finite() && m.zeta > 0.0 && m.zeta < 0.25);
        assert!(m.cavity_flag <= 1);
        assert!(m.zone_drive.iter().all(|v| v.is_finite()));
        assert!(m.rad_coeffs.iter().all(|v| v.is_finite()));
        if m.cavity_flag == 1 {
            assert!(
                (65.0..=550.0).contains(&m.freq_hz),
                "cavity mode {} Hz out of 65..550 Hz",
                m.freq_hz
            );
            assert!(m.rad_coeffs[0].abs() > 0.05 && m.rad_coeffs[4].abs() > 0.01);
        }
    }

    assert_eq!(preset_a, &preset_b);
    assert_eq!(preset_a.encode(), preset_b.encode());
}

#[test]
fn physical_t60_bounds_and_half_second_natural_decay() {
    let (preset, diag) = cached_concert_grand();
    assert!(
        diag.max_t60_s <= MAX_MODAL_T60_S,
        "max T60 {} s exceeds 0.42 s",
        diag.max_t60_s
    );

    let mut hf_t60_sum = 0.0;
    let mut hf_count = 0usize;
    let mut lf_t60_max = 0.0_f64;
    for m in &preset.modes {
        let t60 = zeta_to_t60(m.freq_hz as f64, m.zeta as f64);
        assert!(
            t60 <= MAX_MODAL_T60_S + 1.0e-4,
            "mode {} Hz T60={t60}",
            m.freq_hz
        );
        if m.freq_hz <= 120.0 {
            lf_t60_max = lf_t60_max.max(t60);
        }
        if m.freq_hz >= 14_000.0 {
            hf_t60_sum += t60;
            hf_count += 1;
        }
    }
    let hf_t60_avg = hf_t60_sum / (hf_count.max(1) as f64);
    assert!((0.25..=MAX_MODAL_T60_S).contains(&lf_t60_max));
    assert!(
        (0.025..=0.055).contains(&hf_t60_avg),
        "HF T60 avg {hf_t60_avg} s should taper near ~0.04 s"
    );

    // Synthesize 1.0 s response of all modes to an impulse at belly Zone 4 (48 kHz)
    // and check natural 0.5 s decay > 60 dB with zero artificial envelope gating.
    let sr = 48_000.0;
    let mut imp = vec![0.0_f64; 48_000];
    imp[0] = 1.0;
    let ir = simulate_modal_bank_from_zone(
        &preset.modes,
        4,
        preset.direct_zone_gain[4],
        &imp,
        sr,
        |_| true,
    );
    let rms = |slice: &[f64]| -> f64 {
        (slice.iter().map(|x| x * x).sum::<f64>() / (slice.len() as f64)).sqrt()
    };
    let attack_rms = rms(&ir[0..2_400]); // first 50 ms
    let tail_rms = rms(&ir[24_000..48_000]); // 0.5 s .. 1.0 s
    let decay_db = 20.0 * (attack_rms / tail_rms.max(1.0e-18)).log10();
    assert!(
        decay_db > 60.0,
        "expected > 60 dB natural decay after 0.5 s, got {decay_db:.2} dB"
    );
}

#[test]
fn zone_geometry_planform_and_bridge_rib_heterogeneity() {
    for (z, &[x, y, _]) in CONCERT_ZONE_POS_M.iter().enumerate().take(NUM_BRIDGE_ZONES) {
        let rim_d = rim_signed_distance_m(x as f64, y as f64);
        assert!(
            rim_d > 0.08,
            "bridge zone {z} ({x}, {y}) too close/outside rim: {rim_d}"
        );
    }
    assert!(CONCERT_ZONE_POS_M[0][0] < 0.0 && CONCERT_ZONE_POS_M[1][0] < 0.0);
    assert!(CONCERT_ZONE_POS_M[6][0] > 0.0 && CONCERT_ZONE_POS_M[7][0] > 0.0);
    assert!(CONCERT_ZONE_POS_M[KEYBED_ZONE_IDX][1] < 0.0);

    let belly = evaluate_plate_site(0.08, 0.64);
    let edge = evaluate_plate_site(-0.62, 1.24);
    assert!(belly.thickness_m > edge.thickness_m);
    assert!(belly.bridge_bending_nm > edge.bridge_bending_nm);
    assert!(belly.areal_mass_kg_m2 > edge.areal_mass_kg_m2);
}

#[test]
fn zone8_keybed_low_freq_resonance_vs_silence_and_no_drive() {
    let (preset, _) = cached_concert_grand();
    let sr = 48_000.0;
    let len = 24_000; // 0.5 s

    // Realistic Stage 1 KeybedBank 3.4 ms sin^4 pulse into Zone 8.
    let mut pulse = vec![0.0_f64; len];
    let pulse_samples = (0.0034 * sr) as usize;
    for (n, slot) in pulse.iter_mut().take(pulse_samples).enumerate() {
        let s = (PI * (n as f64) / (pulse_samples as f64)).sin();
        *slot = 4.5 * s.powi(4);
    }

    let silent_in = vec![0.0_f64; len];
    let silence_out = simulate_modal_bank_from_zone(
        &preset.modes,
        KEYBED_ZONE_IDX,
        preset.direct_zone_gain[KEYBED_ZONE_IDX],
        &silent_in,
        sr,
        |_| true,
    );
    let silence_energy: f64 = silence_out.iter().map(|x| x * x).sum();
    assert_eq!(
        silence_energy, 0.0,
        "no keybed drive must yield zero response"
    );

    // Modal resonance-only response (direct_zone_gain = 0) vs zeroed Zone 8 modal drives.
    let mut zero_z8_modes = preset.modes.clone();
    for m in &mut zero_z8_modes {
        m.zone_drive[KEYBED_ZONE_IDX] = 0.0;
    }
    let no_modal_drive_out =
        simulate_modal_bank_from_zone(&zero_z8_modes, KEYBED_ZONE_IDX, 0.0, &pulse, sr, |_| true);
    assert_eq!(no_modal_drive_out.iter().map(|x| x * x).sum::<f64>(), 0.0);

    let keybed_modal_out =
        simulate_modal_bank_from_zone(&preset.modes, KEYBED_ZONE_IDX, 0.0, &pulse, sr, |_| true);
    let low_band = band_power_dft(&keybed_modal_out[..8_192], sr, 40.0, 380.0, 40);
    let high_band = band_power_dft(&keybed_modal_out[..8_192], sr, 1_000.0, 8_000.0, 40);
    assert!(
        low_band > 1.0e-5,
        "expected strong 40..380 Hz keybed thud resonance, got {low_band}"
    );
    assert!(
        low_band > high_band * 20.0,
        "40..380 Hz keybed power ({low_band}) must dominate 1..8 kHz ({high_band})"
    );
}

#[test]
fn minimum_phase_pole_cancellation_and_zero_double_resonance() {
    let (preset, _) = cached_concert_grand();
    let ssm = AdmittanceSsm::new(48_000.0);
    assert_eq!(preset.y_inf, DEFAULT_Y_INF);
    for (p, s) in preset.reduced_poles.iter().zip(ssm.poles().iter()) {
        assert_eq!(p.freq_hz, s.freq_hz);
        assert_eq!(p.zeta, s.zeta);
        assert_eq!(p.modal_mass_kg, s.modal_mass_kg);
        assert_eq!(p.zone_gains, s.zone_gains);
    }
    assert!(preset
        .direct_zone_gain
        .iter()
        .all(|&g| g.is_finite() && (0.02..=0.25).contains(&g)));

    let decoded = BodyPreset::decode(&preset.encode()).expect("valid encoded preset");
    for sr in [44_100.0_f32, 48_000.0_f32, 96_000.0_f32] {
        let rep = verify_pole_cancellation(&decoded, sr);
        assert!(
            rep.max_inverse_pole_real < -1.0,
            "Z_red_norm poles must stay strictly in LHP, max Re = {}",
            rep.max_inverse_pole_real
        );
        assert!(
            rep.max_matrix_residual < 1.0e-6,
            "Z_red_norm * Y_red residual {} too large at {sr} Hz",
            rep.max_matrix_residual
        );
        assert!(
            rep.max_uncancelled_peaking_db > 3.0,
            "expected > 3 dB peaking without cancellation, got {}",
            rep.max_uncancelled_peaking_db
        );
        assert!(
            rep.max_cancelled_excess_db <= 1.0,
            "cancelled double-resonance excess {} dB > 1.0 dB at {sr} Hz",
            rep.max_cancelled_excess_db
        );
        assert!(
            rep.max_discrete_cascade_err_db <= 1.0,
            "discrete trapezoidal cascade error {} dB > 1.0 dB at {sr} Hz",
            rep.max_discrete_cascade_err_db
        );
    }
}

#[test]
fn binary_encoding_budget_and_lossless_roundtrip() {
    let (preset, _) = cached_concert_grand();
    let bytes = preset.encode();
    assert!(
        bytes.len() < 500 * 1024,
        "encoded preset {} bytes exceeds 500 KB",
        bytes.len()
    );
    assert!(
        (45_000..=150_000).contains(&bytes.len()),
        "encoded preset {} bytes outside ~55-120 KB target envelope",
        bytes.len()
    );

    let dec = BodyPreset::decode(&bytes).expect("preset must decode losslessly");
    assert_eq!(dec.y_inf, preset.y_inf);
    assert_eq!(dec.direct_zone_gain, preset.direct_zone_gain);
    assert_eq!(dec.zone_pos_m, preset.zone_pos_m);
    assert_eq!(dec.reduced_poles, preset.reduced_poles);
    assert_eq!(dec.modes.len(), preset.modes.len());

    let mut non_zero_rad_sum = [0.0_f32; PRESET_RAD_COEFFS];
    for (src, dst) in preset.modes.iter().zip(dec.modes.iter()) {
        assert_eq!(src.freq_hz, dst.freq_hz);
        assert_eq!(src.zeta, dst.zeta);
        assert_eq!(src.cavity_flag, dst.cavity_flag);

        let max_d = src.zone_drive.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        let tol_d = (5.0e-5 * max_d).max(1.0e-7);
        for (&a, &b) in src.zone_drive.iter().zip(dst.zone_drive.iter()) {
            assert!((a - b).abs() <= tol_d);
        }

        let max_r = src.rad_coeffs.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
        let tol_r = (5.0e-5 * max_r).max(1.0e-7);
        for (k, (&a, &b)) in src.rad_coeffs.iter().zip(dst.rad_coeffs.iter()).enumerate() {
            assert!((a - b).abs() <= tol_r);
            non_zero_rad_sum[k] += b.abs();
        }
    }
    assert!(non_zero_rad_sum.iter().all(|&s| s > 10.0));
}

#[test]
fn linalg_generalized_eigen_and_complex_invert_and_qr_match_exact() {
    let k = [4.0, 1.0, 1.0, 3.0];
    let m = [2.0, 0.5, 0.5, 1.5];
    let (evals, vecs) = solve_generalized_symmetric_eigen(2, &k, &m).expect("SPD");
    assert!(evals[0] > 0.0 && evals[0] < evals[1]);
    for col in 0..2 {
        let v0 = vecs[col];
        let v1 = vecs[2 + col];
        let kv0 = k[0] * v0 + k[1] * v1;
        let kv1 = k[2] * v0 + k[3] * v1;
        let mv0 = m[0] * v0 + m[1] * v1;
        let mv1 = m[2] * v0 + m[3] * v1;
        assert!((kv0 - evals[col] * mv0).abs() < 1.0e-11);
        assert!((kv1 - evals[col] * mv1).abs() < 1.0e-11);
    }

    let mut cmat = [[Complex64::ZERO; 8]; 8];
    for (i, row) in cmat.iter_mut().enumerate() {
        row[i] = Complex64::new(2.0 + i as f64, 0.5);
        if i + 1 < 8 {
            row[i + 1] = Complex64::new(-0.3, 0.2);
        }
    }
    let inv = invert_8x8_complex(&cmat).expect("invertible");
    for (i, c_row) in cmat.iter().enumerate() {
        for (j, _) in inv.iter().enumerate() {
            let mut acc = Complex64::ZERO;
            for (&c_il, inv_row) in c_row.iter().zip(inv.iter()) {
                acc += c_il * inv_row[j];
            }
            let exp = if i == j { 1.0 } else { 0.0 };
            assert!((acc.re - exp).abs() < 1.0e-12 && acc.im.abs() < 1.0e-12);
        }
    }

    let (w, z) = (300.0_f64, 0.08_f64);
    let mat = [0.0, w, -w, -2.0 * z * w];
    let poles = real_eigenvalues_qr(2, &mat);
    assert_eq!(poles.len(), 2);
    for p in poles {
        assert!((p.re - (-z * w)).abs() < 1.0e-10);
        assert!((p.im.abs() - w * (1.0 - z * z).sqrt()).abs() < 1.0e-10);
    }

    // Verify Householder+QL matches cyclic Jacobi on a 16x16 dense symmetric matrix.
    let n = 16;
    let mut sym = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = if i == j {
                5.0 + (i as f64) * 0.7
            } else {
                0.25 * (((i * 7 + j * 13) as f64).sin())
            };
            sym[i * n + j] = v;
            sym[j * n + i] = v;
        }
    }
    let (ej, vj) = jacobi_symmetric_eigen(n, &sym);
    let (eh, vh) = householder_tridiagonal_ql(n, &sym);
    for col in 0..n {
        assert!((ej[col] - eh[col]).abs() < 1.0e-11);
        let mut res_sq = 0.0;
        for row in 0..n {
            let mut av = 0.0;
            for k in 0..n {
                av += sym[row * n + k] * vh[k * n + col];
            }
            let diff = av - eh[col] * vh[row * n + col];
            res_sq += diff * diff;
        }
        assert!(res_sq.sqrt() < 1.0e-11, "col {col} residual {res_sq}");
        let dot_j: f64 = (0..n).map(|r| vj[r * n + col] * vh[r * n + col]).sum();
        assert!(dot_j.abs() > 0.999_999_999);
    }
}
