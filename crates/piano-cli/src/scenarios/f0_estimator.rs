//! High-resolution fundamental frequency estimator (`measure_f0_hz`) with
//! harmonic-subspace orthogonalization for short low-frequency windows.

use std::f64::consts::TAU;

use super::spectral_utils::{blackman_harris_window, goertzel_power_f64};

/// Minimum window cycles above which Blackman-Harris Goertzel has >90 dB harmonic isolation.
const SHORT_WINDOW_CYCLE_THRESHOLD: f64 = 5.0;

/// Solves a small symmetric positive-definite linear system `a * x = b` via Gauss-Jordan.
fn solve_linear<const M: usize>(mut a: [[f64; M]; M], mut b: [f64; M]) -> Option<[f64; M]> {
    for col in 0..M {
        let mut piv_row = col;
        let mut piv_val = a[col][col].abs();
        for (r, row) in a.iter().enumerate().skip(col + 1) {
            if row[col].abs() > piv_val {
                piv_val = row[col].abs();
                piv_row = r;
            }
        }
        if piv_val < 1.0e-14 {
            return None;
        }
        if piv_row != col {
            a.swap(col, piv_row);
            b.swap(col, piv_row);
        }
        let inv = 1.0 / a[col][col];
        for v in a[col].iter_mut().skip(col) {
            *v *= inv;
        }
        b[col] *= inv;
        for r in 0..M {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            let (row_r, row_col) = if r < col {
                let (top, bot) = a.split_at_mut(col);
                (&mut top[r], &bot[0])
            } else {
                let (top, bot) = a.split_at_mut(r);
                (&mut bot[0], &top[col])
            };
            for (dst, &src) in row_r[col..M].iter_mut().zip(row_col[col..M].iter()) {
                *dst -= factor * src;
            }
            b[r] -= factor * b[col];
        }
    }
    Some(b)
}

/// Computes `v^T G^{-1} v` for an `M x M` SPD matrix `G`.
fn quad_form_inv<const M: usize>(g: &[[f64; M]; M], v: &[f64; M]) -> f64 {
    let Some(sol) = solve_linear(*g, *v) else {
        return 0.0;
    };
    v.iter().zip(sol.iter()).map(|(&a, &b)| a * b).sum()
}

/// Evaluates weighted harmonic least-squares explained energy for `{1, 1..=6 partials}`
/// with dispersion `f_h = h * f1 * sqrt((1 + B*h^2)/(1 + B))`.
fn harmonic_ls_energy(mono: &[f32], weights: &[f64], f1_hz: f64, b_inharm: f64, sr: f64) -> f64 {
    const H: usize = 6;
    const M: usize = 1 + 2 * H;
    let w0 = TAU * f1_hz / sr;
    let inv_root1 = 1.0 / (1.0 + b_inharm).sqrt();
    let mut dc = [0.0_f64; H];
    let mut ds = [0.0_f64; H];
    for h in 0..H {
        let hf = (h + 1) as f64;
        let ang = hf * ((1.0 + b_inharm * hf * hf).sqrt() * inv_root1) * w0;
        dc[h] = ang.cos();
        ds[h] = ang.sin();
    }
    let mut c = [1.0_f64; H];
    let mut s = [0.0_f64; H];

    let mut g = [[0.0_f64; M]; M];
    let mut rhs = [0.0_f64; M];
    for (&x, &w) in mono.iter().zip(weights.iter()) {
        let mut basis = [0.0_f64; M];
        basis[0] = 1.0;
        for h in 0..H {
            basis[1 + 2 * h] = c[h];
            basis[2 + 2 * h] = s[h];
            let nc = c[h] * dc[h] - s[h] * ds[h];
            s[h] = s[h] * dc[h] + c[h] * ds[h];
            c[h] = nc;
        }
        let wx = w * f64::from(x);
        for i in 0..M {
            rhs[i] += basis[i] * wx;
            let wb = w * basis[i];
            for j in i..M {
                g[i][j] += wb * basis[j];
            }
        }
    }
    symmetrize_and_ridge(&mut g);
    quad_form_inv(&g, &rhs)
}

fn symmetrize_and_ridge<const N: usize>(m: &mut [[f64; N]; N]) {
    for i in 0..N {
        m[i][i] += 1.0e-12;
        let (upper, lower) = m.split_at_mut(i);
        for (j, up_row) in upper.iter().enumerate() {
            lower[0][j] = up_row[i];
        }
    }
}

/// Short-window fundamental estimator using joint harmonic-subspace weighted least squares.
///
/// Evaluates `{1, 1..=6}` dispersion-aware weighted least-squares explained energy across
/// candidate `(f_1, B)` pairs and refines `f_1` directly on the joint objective (avoiding
/// collinearity bias from holding higher partials fixed across short `~1.1`-cycle windows).
fn measure_f0_short_window_hz(mono: &[f32], f_nominal_hz: f64, sr: f64) -> f64 {
    let n = mono.len();
    let denom = (n - 1).max(1) as f64;
    let weights: Vec<f64> = (0..n)
        .map(|i| 0.5 * (1.0 - (TAU * (i as f64) / denom).cos()))
        .collect();

    let mut best_hat = f_nominal_hz;
    let mut best_e = f64::NEG_INFINITY;
    for step in (-200..=200).step_by(8) {
        let f_try = f_nominal_hz * 2.0_f64.powf((step as f64 * 0.5) / 1200.0);
        let e = harmonic_ls_energy(mono, &weights, f_try, 0.00025, sr);
        if e > best_e {
            best_e = e;
            best_hat = f_try;
        }
    }
    let mut f_hat = best_hat;
    let mut b_hat = 0.00025_f64;
    for &b_try in &[0.0, 0.00015, 0.00033, 0.0006, 0.0010, 0.0016] {
        for sub in -12..=12 {
            let f_try = best_hat * 2.0_f64.powf((sub as f64 * 0.5) / 1200.0);
            let e = harmonic_ls_energy(mono, &weights, f_try, b_try, sr);
            if e > best_e {
                best_e = e;
                f_hat = f_try;
                b_hat = b_try;
            }
        }
    }

    let dc = 0.25_f64;
    let mut best_f = f_hat;
    for sub in -8..=8 {
        let f_try = f_hat * 2.0_f64.powf((sub as f64 * dc) / 1200.0);
        let e = harmonic_ls_energy(mono, &weights, f_try, b_hat, sr);
        if e > best_e {
            best_e = e;
            best_f = f_try;
        }
    }
    let r = 2.0_f64.powf(dc / 1200.0);
    let p_lo = harmonic_ls_energy(mono, &weights, best_f / r, b_hat, sr);
    let p_mid = best_e;
    let p_hi = harmonic_ls_energy(mono, &weights, best_f * r, b_hat, sr);
    let denom_q = p_lo - 2.0 * p_mid + p_hi;
    let shift = if denom_q < 0.0 {
        (0.5 * (p_lo - p_hi) / denom_q).clamp(-1.0, 1.0) * dc
    } else {
        0.0
    };
    best_f * 2.0_f64.powf(shift / 1200.0)
}

/// High-resolution fundamental frequency estimator via Goertzel and parabolic peak refinement,
/// with harmonic-subspace orthogonalization for short low-frequency windows (`cycles < 5.0`).
#[must_use]
pub fn measure_f0_hz(mono: &[f32], f_nominal_hz: f64, sr: f64) -> f64 {
    let cycles = f_nominal_hz * (mono.len() as f64) / sr.max(1.0);
    if cycles < SHORT_WINDOW_CYCLE_THRESHOLD && mono.len() >= 16 {
        return measure_f0_short_window_hz(mono, f_nominal_hz, sr);
    }
    let win = blackman_harris_window(mono);
    let mut best_f = f_nominal_hz;
    let mut best_p = 0.0_f64;
    for step in -200..=200 {
        let f_try = f_nominal_hz * 2.0_f64.powf((step as f64 * 0.5) / 1200.0);
        let p = goertzel_power_f64(&win, f_try, sr);
        if p > best_p {
            best_p = p;
            best_f = f_try;
        }
    }
    let dc = 0.5_f64;
    let r = 2.0_f64.powf(dc / 1200.0);
    let p_lo = goertzel_power_f64(&win, best_f / r, sr);
    let p_mid = goertzel_power_f64(&win, best_f, sr);
    let p_hi = goertzel_power_f64(&win, best_f * r, sr);
    let denom_q = p_lo - 2.0 * p_mid + p_hi;
    let shift = if denom_q < 0.0 {
        (0.5 * (p_lo - p_hi) / denom_q).clamp(-1.0, 1.0) * dc
    } else {
        0.0
    };
    best_f * 2.0_f64.powf(shift / 1200.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenarios::spectral_utils::mono_mix;
    use piano_engine::PianoEngine;
    use piano_spec::{
        build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, EventKind,
        SampleEvent, VoicingSpec,
    };

    fn synth_inharmonic_piano_tone_with_phase(
        f1_hz: f64,
        b_inharm: f64,
        sr: f64,
        len: usize,
        dc: f64,
        phase0: f64,
        amps: &[f64],
    ) -> Vec<f32> {
        let f0 = f1_hz / (1.0 + b_inharm).sqrt();
        (0..len)
            .map(|i| {
                let t = (i as f64) / sr;
                let mut v = dc;
                for (idx, &a) in amps.iter().enumerate() {
                    let h = (idx + 1) as f64;
                    let fh = h * f0 * (1.0 + b_inharm * h * h).sqrt();
                    let phase = phase0 + (idx as f64) * 0.63;
                    v += a * (TAU * fh * t + phase).sin();
                }
                v as f32
            })
            .collect()
    }

    fn synth_inharmonic_piano_tone(
        f1_hz: f64,
        b_inharm: f64,
        sr: f64,
        len: usize,
        dc: f64,
        amps: &[f64],
    ) -> Vec<f32> {
        synth_inharmonic_piano_tone_with_phase(f1_hz, b_inharm, sr, len, dc, 0.0, amps)
    }

    fn render_note(
        midi_key: u8,
        nonlinear: bool,
        total_frames: usize,
        velocity_u16: u16,
    ) -> Vec<f32> {
        let mut engine = PianoEngine::new(48_000.0);
        engine.set_nonlinear_coupling(nonlinear, nonlinear);
        let mut out_l = vec![0.0_f32; total_frames];
        let mut out_r = vec![0.0_f32; total_frames];
        let ev = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key,
                velocity_u16,
            },
        }];
        let mut pos = 0;
        while pos < total_frames {
            let n = (total_frames - pos).min(128);
            let cur = if pos == 0 { &ev[..] } else { &[] };
            engine.render_block_bridge(cur, &mut out_l[pos..pos + n], &mut out_r[pos..pos + n]);
            pos += n;
        }
        mono_mix(&out_l, &out_r)
    }

    #[test]
    fn measure_f0_short_window_a0_rejects_2nd_harmonic_leak_and_dc() {
        let sr = 48_000.0_f64;
        let n_40ms = (0.040 * sr).round() as usize; // 1920 samples = 1.1 cycles of A0
        let f1_true = 27.62; // +7.5 cents sharp of 27.5 Hz
        let sig = synth_inharmonic_piano_tone(
            f1_true,
            0.00033,
            sr,
            n_40ms,
            0.05,
            &[0.10, 0.85, 0.45, 0.25],
        );
        let f_est = measure_f0_hz(&sig, 27.5, sr);
        let err_cents = (1200.0 * (f_est / f1_true).log2()).abs();
        assert!(
            err_cents < 1.0,
            "A0 40ms short-window estimate drifted: est={f_est:.4} Hz vs true={f1_true:.4} Hz ({err_cents:.3} cents)"
        );
    }

    #[test]
    fn measure_f0_short_window_a0_phase_and_freq_sweep_invariant() {
        let sr = 48_000.0_f64;
        let n_40ms = (0.040 * sr).round() as usize;
        for &f1_true in &[27.30, 27.62, 27.85, 28.10] {
            for &phase0 in &[0.0, 1.1, 2.3, 4.1] {
                let sig = synth_inharmonic_piano_tone_with_phase(
                    f1_true,
                    0.00033,
                    sr,
                    n_40ms,
                    0.05,
                    phase0,
                    &[0.10, 0.85, 0.45, 0.25],
                );
                let f_est = measure_f0_hz(&sig, 27.5, sr);
                let err_cents = (1200.0 * (f_est / f1_true).log2()).abs();
                assert!(
                    err_cents < 1.0,
                    "A0 8.5x 2nd-harm phase sweep failed at f1={f1_true}, ph={phase0}: est={f_est:.4} Hz ({err_cents:.3} cents)"
                );
            }
        }
    }

    #[test]
    fn measure_f0_short_window_a0_20x_2nd_harmonic_phase_sweep() {
        let sr = 48_000.0_f64;
        let n_40ms = (0.040 * sr).round() as usize;
        for &f1_true in &[27.30, 27.62, 27.85, 28.10] {
            for &phase0 in &[0.0, 1.1, 2.3, 4.1] {
                let sig = synth_inharmonic_piano_tone_with_phase(
                    f1_true,
                    0.00033,
                    sr,
                    n_40ms,
                    0.20,
                    phase0,
                    &[0.05, 1.00, 0.70, 0.40],
                );
                let f_est = measure_f0_hz(&sig, 27.5, sr);
                let err_cents = (1200.0 * (f_est / f1_true).log2()).abs();
                assert!(
                    err_cents < 1.5,
                    "A0 20x 2nd-harm phase sweep failed at f1={f1_true}, ph={phase0}: est={f_est:.4} Hz ({err_cents:.3} cents)"
                );
            }
        }
    }

    #[test]
    fn measure_f0_c2_short_and_long_windows_accurate_with_strong_harmonics() {
        let sr = 48_000.0_f64;
        let f1_true = 65.72; // +8.3 cents above C2 (65.406 Hz)
        let n_40ms = (0.040 * sr).round() as usize;
        let n_long = 16_384_usize;

        for &phase0 in &[0.0, 1.1, 2.3, 4.1] {
            let sig_short = synth_inharmonic_piano_tone_with_phase(
                f1_true,
                0.00018,
                sr,
                n_40ms,
                0.02,
                phase0,
                &[0.25, 0.80, 0.40],
            );
            let sig_long = synth_inharmonic_piano_tone_with_phase(
                f1_true,
                0.00018,
                sr,
                n_long,
                0.02,
                phase0,
                &[0.25, 0.80, 0.40],
            );

            let est_short = measure_f0_hz(&sig_short, 65.406, sr);
            let est_long = measure_f0_hz(&sig_long, 65.406, sr);
            let err_short = (1200.0 * (est_short / f1_true).log2()).abs();
            let err_long = (1200.0 * (est_long / f1_true).log2()).abs();

            assert!(
                err_short < 0.5,
                "C2 40ms ph={phase0} error {err_short:.3} cents (est={est_short:.4})"
            );
            assert!(
                err_long < 0.5,
                "C2 341ms ph={phase0} error {err_long:.3} cents (est={est_long:.4})"
            );
        }
    }

    #[test]
    fn measure_f0_real_a0_and_c2_linear_no_phantom_chirp_and_nonlinear_authentic_chirp() {
        let sr = 48_000.0_f64;
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);

        // A0 (27.127 Hz): >= 2.2 cycles (~81 ms = 3893 samples) spans > 2 bridge reflections.
        let key_a0 = midi_to_key(21).expect("A0");
        let f_nom_a0 = partial_frequency_hz(
            key_tuned_f0_hz(key_a0, &specs, &voicing),
            specs[key_a0].inharmonicity_b,
            1,
        );
        let n_a0_early = ((2.2 / f_nom_a0) * sr).round() as usize;
        let a0_lin = render_note(21, false, 49_152, u16::MAX);
        let a0_nl = render_note(21, true, 49_152, u16::MAX);
        let a0_lin_early = measure_f0_hz(&a0_lin[240..240 + n_a0_early], f_nom_a0, sr);
        let a0_lin_late = measure_f0_hz(&a0_lin[32_768..49_152], f_nom_a0, sr);
        let a0_lin_chirp = 1200.0 * (a0_lin_early / a0_lin_late).log2();
        let a0_nl_early = measure_f0_hz(&a0_nl[240..240 + n_a0_early], f_nom_a0, sr);
        let a0_nl_late = measure_f0_hz(&a0_nl[32_768..49_152], f_nom_a0, sr);
        let a0_nl_chirp = 1200.0 * (a0_nl_early / a0_nl_late).log2();
        assert!(
            a0_lin_chirp.abs() < 2.0,
            "Linear A0 ({n_a0_early} smp) showed phantom chirp: {a0_lin_chirp:+.2}c (early={a0_lin_early:.3}, late={a0_lin_late:.3})"
        );
        assert!(
            a0_nl_chirp > 2.0,
            "Nonlinear ff A0 ({n_a0_early} smp) missing authentic strike chirp: {a0_nl_chirp:+.2}c"
        );

        // C2 (64.96 Hz): 40 ms (1920 samples = 2.60 cycles).
        let key_c2 = midi_to_key(36).expect("C2");
        let f_nom_c2 = partial_frequency_hz(
            key_tuned_f0_hz(key_c2, &specs, &voicing),
            specs[key_c2].inharmonicity_b,
            1,
        );
        let c2_lin = render_note(36, false, 49_152, u16::MAX);
        let c2_nl = render_note(36, true, 49_152, u16::MAX);
        let c2_lin_w1 = measure_f0_hz(&c2_lin[240..2160], f_nom_c2, sr);
        let c2_lin_w2 = measure_f0_hz(&c2_lin[2400..4320], f_nom_c2, sr);
        let c2_lin_late = measure_f0_hz(&c2_lin[32_768..49_152], f_nom_c2, sr);
        let c2_lin_c1 = 1200.0 * (c2_lin_w1 / c2_lin_late).log2();
        let c2_lin_c2 = 1200.0 * (c2_lin_w2 / c2_lin_late).log2();
        let c2_nl_w1 = measure_f0_hz(&c2_nl[240..2160], f_nom_c2, sr);
        let c2_nl_late = measure_f0_hz(&c2_nl[32_768..49_152], f_nom_c2, sr);
        let c2_nl_chirp = 1200.0 * (c2_nl_w1 / c2_nl_late).log2();
        assert!(
            c2_lin_c1.abs() < 2.0 && c2_lin_c2.abs() < 2.0,
            "Linear C2 40ms showed phantom chirp: w1={c2_lin_c1:+.2}c, w2={c2_lin_c2:+.2}c"
        );
        assert!(
            c2_nl_chirp > 2.0,
            "Nonlinear ff C2 40ms missing authentic strike chirp: {c2_nl_chirp:+.2}c"
        );
    }
}
