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

/// Short-window fundamental estimator using harmonic-subspace orthogonalization.
///
/// Step 1 estimates `(f_hat, b_hat)` via `{1, 1..=6}` dispersion-aware weighted least squares.
/// Step 2 projects out `Z = {1, partials 2..=6}` via Schur complement so `f_1` is isolated
/// without 2nd-harmonic leakage, DC bias, or inharmonicity pull.
fn measure_f0_short_window_hz(mono: &[f32], f_nominal_hz: f64, sr: f64) -> f64 {
    const NZ: usize = 11; // DC + {2..=6} partials
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
        for sub in -8..=8 {
            let f_try = best_hat * 2.0_f64.powf((sub as f64 * 0.5) / 1200.0);
            let e = harmonic_ls_energy(mono, &weights, f_try, b_try, sr);
            if e > best_e {
                best_e = e;
                f_hat = f_try;
                b_hat = b_try;
            }
        }
    }

    let inv_root1 = 1.0 / (1.0 + b_hat).sqrt();
    let mut z_cols = vec![[0.0_f64; NZ]; n];
    let mut g_zz = [[0.0_f64; NZ]; NZ];
    let mut rhs_z = [0.0_f64; NZ];
    for (i, (&x, &w)) in mono.iter().zip(weights.iter()).enumerate() {
        let t = (i as f64) / sr;
        let mut row = [0.0_f64; NZ];
        row[0] = 1.0;
        for h in 2..=6 {
            let hf = h as f64;
            let fh = hf * f_hat * (1.0 + b_hat * hf * hf).sqrt() * inv_root1;
            let ang = TAU * fh * t;
            row[1 + 2 * (h - 2)] = ang.cos();
            row[2 + 2 * (h - 2)] = ang.sin();
        }
        z_cols[i] = row;
        let wx = w * f64::from(x);
        for p in 0..NZ {
            rhs_z[p] += row[p] * wx;
            let wr = w * row[p];
            for q in p..NZ {
                g_zz[p][q] += wr * row[q];
            }
        }
    }
    symmetrize_and_ridge(&mut g_zz);
    let coef_z = solve_linear(g_zz, rhs_z).unwrap_or([0.0; NZ]);
    let res: Vec<f64> = mono
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let z_fit: f64 = z_cols[i]
                .iter()
                .zip(coef_z.iter())
                .map(|(&z, &c)| z * c)
                .sum();
            f64::from(x) - z_fit
        })
        .collect();

    let score_f1 = |f_try: f64| -> f64 {
        let w_rad = TAU * f_try / sr;
        let (dc, ds) = (w_rad.cos(), w_rad.sin());
        let (mut c, mut s) = (1.0_f64, 0.0_f64);
        let (mut pc, mut ps) = (0.0_f64, 0.0_f64);
        let (mut cc, mut ss, mut cs) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut qc = [0.0_f64; NZ];
        let mut qs = [0.0_f64; NZ];
        for i in 0..n {
            let wi = weights[i];
            let wc = wi * c;
            let ws = wi * s;
            pc += wc * res[i];
            ps += ws * res[i];
            cc += wc * c;
            ss += ws * s;
            cs += wc * s;
            let zr = &z_cols[i];
            for p in 0..NZ {
                qc[p] += wc * zr[p];
                qs[p] += ws * zr[p];
            }
            let nc = c * dc - s * ds;
            s = s * dc + c * ds;
            c = nc;
        }
        let Some(iqc) = solve_linear(g_zz, qc) else {
            return 0.0;
        };
        let Some(iqs) = solve_linear(g_zz, qs) else {
            return 0.0;
        };
        let gcc = (cc - qc.iter().zip(iqc.iter()).map(|(a, b)| a * b).sum::<f64>()).max(1.0e-12);
        let gss = (ss - qs.iter().zip(iqs.iter()).map(|(a, b)| a * b).sum::<f64>()).max(1.0e-12);
        let gcs = cs - qc.iter().zip(iqs.iter()).map(|(a, b)| a * b).sum::<f64>();
        let det = (gcc * gss - gcs * gcs).max(1.0e-18);
        ((gss * pc * pc - 2.0 * gcs * pc * ps + gcc * ps * ps) / det).max(1.0e-30)
    };

    let mut best_f = f_nominal_hz;
    let mut best_p = 0.0_f64;
    for step in -200..=200 {
        let f_try = f_nominal_hz * 2.0_f64.powf((step as f64 * 0.5) / 1200.0);
        let p = score_f1(f_try);
        if p > best_p {
            best_p = p;
            best_f = f_try;
        }
    }
    let dc = 0.5_f64;
    let r = 2.0_f64.powf(dc / 1200.0);
    let p_lo = score_f1(best_f / r);
    let p_mid = score_f1(best_f);
    let p_hi = score_f1(best_f * r);
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

    fn synth_inharmonic_piano_tone(
        f1_hz: f64,
        b_inharm: f64,
        sr: f64,
        len: usize,
        dc: f64,
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
                    let phase = (idx as f64) * 0.63;
                    v += a * (TAU * fh * t + phase).sin();
                }
                v as f32
            })
            .collect()
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
    fn measure_f0_c2_short_and_long_windows_accurate_with_strong_harmonics() {
        let sr = 48_000.0_f64;
        let f1_true = 65.72; // +8.3 cents above C2 (65.406 Hz)
        let n_40ms = (0.040 * sr).round() as usize;
        let n_long = 16_384_usize;

        let sig_short =
            synth_inharmonic_piano_tone(f1_true, 0.00018, sr, n_40ms, 0.02, &[0.25, 0.80, 0.40]);
        let sig_long =
            synth_inharmonic_piano_tone(f1_true, 0.00018, sr, n_long, 0.02, &[0.25, 0.80, 0.40]);

        let est_short = measure_f0_hz(&sig_short, 65.406, sr);
        let est_long = measure_f0_hz(&sig_long, 65.406, sr);
        let err_short = (1200.0 * (est_short / f1_true).log2()).abs();
        let err_long = (1200.0 * (est_long / f1_true).log2()).abs();

        assert!(
            err_short < 0.15,
            "C2 40ms error {err_short:.3} cents (est={est_short:.4})"
        );
        assert!(
            err_long < 0.10,
            "C2 341ms error {err_long:.3} cents (est={est_long:.4})"
        );
    }
}
