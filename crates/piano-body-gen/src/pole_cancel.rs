//! Stage 3 reduced bridge pole reading, minimum-phase inverse cancellation
//! `Z_red_norm(s) = diag(Y_inf) * Y_red(s)^{-1}`, calibrated `direct_zone_gain`,
//! and full-body zero double-resonance verification across `20..16,000 Hz`.

use crate::linalg::{invert_8x8_complex, real_eigenvalues_qr, Complex64};
use piano_bridge::{AdmittanceSsm, DEFAULT_Y_INF, NUM_BRIDGE_ZONES, NUM_REDUCED_MODES};
use piano_spec::{
    BodyModeRecord, BodyPreset, PresetReducedPole, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES,
    PRESET_REDUCED_POLES,
};
use std::f64::consts::PI;

/// Calibrated physical direct-wave feedthrough weights across 8 bridge zones + Zone 8 keybed.
///
/// Tapering from thinner rim-adjacent bass/treble zones toward the central soundboard belly
/// matches the acoustic direct arrival strength prior to modal reverberation buildup.
pub const CALIBRATED_DIRECT_ZONE_GAIN: [f32; PRESET_RADIATOR_ZONES] = [
    0.100, // Zone 0: Bass bridge tail end
    0.104, // Zone 1: Upper bass bridge
    0.215, // Zone 2: Low tenor belly entry
    0.165, // Zone 3: Tenor center belly
    0.249, // Zone 4: Middle belly apex
    0.245, // Zone 5: Upper middle belly
    0.120, // Zone 6: Lower treble bridge
    0.105, // Zone 7: Top treble capo corner
    0.044, // Zone 8: Wooden keybed direct transient (KEYBED_ZONE_IDX)
];

/// Pole-cancellation diagnostics reported by [`verify_pole_cancellation`].
#[derive(Clone, Debug)]
pub struct PoleCancelReport {
    /// Maximum real part among all 20 continuous poles of `Z_red_norm(s)` (must be `< 0.0`).
    pub max_inverse_pole_real: f64,
    /// Worst-case matrix norm error `||Z_red_norm * Y_red - diag(Y_inf)||_F` across `20..16,000 Hz`.
    pub max_matrix_residual: f64,
    /// Excess peaking [dB] around reduced poles WITHOUT `Z_red_norm` cancellation (`> 3 dB` expected).
    pub max_uncancelled_peaking_db: f64,
    /// Worst-case spectral deviation [dB] of cascaded `H_body * Z_red_norm * Y_red` vs `H_body * diag(Y_inf)`.
    pub max_cancelled_excess_db: f64,
    /// Worst-case discrete `48 kHz` cascade vs direct `Y_inf` difference across reduced pole bands [dB].
    pub max_discrete_cascade_err_db: f64,
}

/// Extracts `PresetReducedPole` array and `y_inf` from `AdmittanceSsm::new(48_000.0)`.
pub fn extract_stage3_reduced_model() -> (
    [f32; PRESET_BRIDGE_ZONES],
    [PresetReducedPole; PRESET_REDUCED_POLES],
) {
    let ssm = AdmittanceSsm::new(48_000.0);
    let poles = ssm.poles();
    let reduced = std::array::from_fn(|i| PresetReducedPole {
        freq_hz: poles[i].freq_hz,
        zeta: poles[i].zeta,
        modal_mass_kg: poles[i].modal_mass_kg,
        zone_gains: poles[i].zone_gains,
    });
    (DEFAULT_Y_INF, reduced)
}

/// Evaluates the 8x8 continuous reduced bridge admittance matrix `Y_red(j*omega)` [m/(N*s)].
pub fn eval_y_red_freq(
    y_inf: &[f32; PRESET_BRIDGE_ZONES],
    poles: &[PresetReducedPole; PRESET_REDUCED_POLES],
    freq_hz: f64,
) -> [[Complex64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
    let omega = 2.0 * PI * freq_hz;
    let jw = Complex64::new(0.0, omega);
    let mut y = [[Complex64::ZERO; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    for z in 0..NUM_BRIDGE_ZONES {
        y[z][z] = Complex64::new(y_inf[z] as f64, 0.0);
    }

    for pole in poles {
        let w_m = 2.0 * PI * (pole.freq_hz as f64);
        let mass = pole.modal_mass_kg as f64;
        let denom = Complex64::new(
            mass * (w_m * w_m - omega * omega),
            mass * 2.0 * (pole.zeta as f64) * w_m * omega,
        );
        let y_m = jw / denom;
        for (y_row, &gr_f) in y.iter_mut().zip(pole.zone_gains.iter()) {
            let gr = gr_f as f64;
            for (y_cell, &gc_f) in y_row.iter_mut().zip(pole.zone_gains.iter()) {
                *y_cell += y_m * (gr * (gc_f as f64));
            }
        }
    }
    y
}

/// Evaluates normalized minimum-phase inverse `Z_red_norm(j*w) = diag(Y_inf) * Y_red(j*w)^{-1}`.
pub fn eval_z_red_norm_freq(
    y_inf: &[f32; PRESET_BRIDGE_ZONES],
    poles: &[PresetReducedPole; PRESET_REDUCED_POLES],
    freq_hz: f64,
) -> Option<[[Complex64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES]> {
    let y_red = eval_y_red_freq(y_inf, poles, freq_hz);
    let inv = invert_8x8_complex(&y_red)?;
    let mut z_norm = [[Complex64::ZERO; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    for r in 0..NUM_BRIDGE_ZONES {
        let scale = y_inf[r] as f64;
        for c in 0..NUM_BRIDGE_ZONES {
            z_norm[r][c] = inv[r][c] * scale;
        }
    }
    Some(z_norm)
}

/// Computes the maximum real part among all 20 continuous poles of `Z_red_norm(s)`.
///
/// Uses mass-normalized modal coordinates `M x'' + (C + Phi Y_inf^{-1} Phi^T) x' + K x = 0`
/// and balanced state scaling `(w_ref * x, x')` for clean `20 x 20` QR eigenvalue convergence.
pub fn max_inverse_pole_real_part(
    y_inf: &[f32; PRESET_BRIDGE_ZONES],
    poles: &[PresetReducedPole; PRESET_REDUCED_POLES],
) -> f64 {
    const N: usize = NUM_REDUCED_MODES;
    const N2: usize = N * 2;
    let w_ref = 2.0 * PI * 300.0;

    let mut d_rad = [0.0_f64; N * N];
    for i in 0..N {
        let inv_sqrt_mi = 1.0 / (poles[i].modal_mass_kg as f64).sqrt();
        for j in 0..N {
            let inv_sqrt_mj = 1.0 / (poles[j].modal_mass_kg as f64).sqrt();
            let mut sum = 0.0;
            for ((&gi_f, &gj_f), &yi_f) in poles[i]
                .zone_gains
                .iter()
                .zip(poles[j].zone_gains.iter())
                .zip(y_inf.iter())
            {
                sum += (gi_f as f64) * (gj_f as f64) / (yi_f as f64);
            }
            d_rad[i * N + j] = sum * inv_sqrt_mi * inv_sqrt_mj;
        }
    }

    let mut a = vec![0.0_f64; N2 * N2];
    for i in 0..N {
        // Upper right: d(w_ref * x_i)/dt = w_ref * v_i
        a[i * N2 + (N + i)] = w_ref;
        let w_i = 2.0 * PI * (poles[i].freq_hz as f64);
        // Lower left: dv_i/dt = -(w_i^2 / w_ref) * (w_ref * x_i)
        a[(N + i) * N2 + i] = -(w_i * w_i) / w_ref;
        // Lower right: -(2*zeta_i*w_i * I + D_rad)
        for j in 0..N {
            let c_ij = if i == j {
                2.0 * (poles[i].zeta as f64) * w_i
            } else {
                0.0
            };
            a[(N + i) * N2 + (N + j)] = -(c_ij + d_rad[i * N + j]);
        }
    }

    let eigvals = real_eigenvalues_qr(N2, &a);
    eigvals
        .iter()
        .map(|lam| lam.re)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Exact trapezoidal discrete inverse `Z_red_norm` paired with `AdmittanceSsm`.
pub struct InverseReducedBridge {
    y_inf: [f32; PRESET_BRIDGE_ZONES],
    poles: [PresetReducedPole; PRESET_REDUCED_POLES],
    sample_rate: f64,
    q: [f64; NUM_REDUCED_MODES],
    q_dot: [f64; NUM_REDUCED_MODES],
    a_br_inv: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
}

impl InverseReducedBridge {
    /// Initializes exact discrete trapezoidal inverse matching `AdmittanceSsm::new(sample_rate)`.
    pub fn from_preset(preset: &BodyPreset, sample_rate: f32) -> Self {
        let sr = sample_rate as f64;
        let h = 0.5 / sr;
        let mut a_br = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        for (z, &yi) in preset.y_inf.iter().enumerate() {
            a_br[z][z] = yi as f64;
        }
        for p in &preset.reduced_poles {
            let m = p.modal_mass_kg as f64;
            let w = 2.0 * PI * (p.freq_hz as f64);
            let c = 2.0 * (p.zeta as f64) * w * m;
            let k = m * w * w;
            let y_m = 1.0 / (m / h + c + h * k);
            for (row_a, &gr_f) in a_br.iter_mut().zip(p.zone_gains.iter()) {
                let gr = gr_f as f64;
                for (cell_a, &gc_f) in row_a.iter_mut().zip(p.zone_gains.iter()) {
                    *cell_a += y_m * gr * (gc_f as f64);
                }
            }
        }
        let a_br_c: [[Complex64; 8]; 8] =
            std::array::from_fn(|r| std::array::from_fn(|c| Complex64::new(a_br[r][c], 0.0)));
        let inv_c = invert_8x8_complex(&a_br_c).expect("SPD A_br matrix");
        let a_br_inv = std::array::from_fn(|r| std::array::from_fn(|c| inv_c[r][c].re));

        Self {
            y_inf: preset.y_inf,
            poles: preset.reduced_poles,
            sample_rate: sr,
            q: [0.0; NUM_REDUCED_MODES],
            q_dot: [0.0; NUM_REDUCED_MODES],
            a_br_inv,
        }
    }

    /// Steps one sample: given Stage 3 midpoint velocity `v`, returns `y_norm = diag(y_inf) * f_recovered`.
    pub fn step(&mut self, v: &[f64; NUM_BRIDGE_ZONES]) -> [f64; NUM_BRIDGE_ZONES] {
        let ts = 1.0 / self.sample_rate;
        let inv_h = 2.0 * self.sample_rate;
        let h = 0.5 * ts;
        let mut y_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut qd_free_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut v_free_z = [0.0_f64; NUM_BRIDGE_ZONES];

        for (i, p) in self.poles.iter().enumerate() {
            let m = p.modal_mass_kg as f64;
            let w = 2.0 * PI * (p.freq_hz as f64);
            let c = 2.0 * (p.zeta as f64) * w * m;
            let k = m * w * w;
            let ym = 1.0 / (m * inv_h + c + h * k);
            y_m[i] = ym;
            let rhs = (m * inv_h) * self.q_dot[i] - k * self.q[i];
            let qdf = ym * rhs;
            qd_free_m[i] = qdf;
            for (z, &g) in p.zone_gains.iter().enumerate() {
                v_free_z[z] += (g as f64) * qdf;
            }
        }

        let mut dv = [0.0_f64; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            dv[z] = v[z] - v_free_z[z];
        }
        let mut f_rec = [0.0_f64; NUM_BRIDGE_ZONES];
        for (f_slot, row_inv) in f_rec.iter_mut().zip(self.a_br_inv.iter()) {
            let mut sum = 0.0;
            for (&a_rc, &dv_c) in row_inv.iter().zip(dv.iter()) {
                sum += a_rc * dv_c;
            }
            *f_slot = sum;
        }

        for (i, p) in self.poles.iter().enumerate() {
            let mut f_m = 0.0;
            for (z, &g) in p.zone_gains.iter().enumerate() {
                f_m += (g as f64) * f_rec[z];
            }
            let qd_mid = qd_free_m[i] + y_m[i] * f_m;
            let qd_next = 2.0 * qd_mid - self.q_dot[i];
            self.q[i] += ts * qd_mid;
            self.q_dot[i] = qd_next;
        }

        std::array::from_fn(|z| (self.y_inf[z] as f64) * f_rec[z])
    }
}

/// Evaluates the full body transfer vector `H_body(j*w)` (8 bridge inputs -> monopole output).
fn eval_body_transfer_vec(
    direct_zone_gain: &[f32; PRESET_RADIATOR_ZONES],
    modes: &[BodyModeRecord],
    freq_hz: f64,
) -> [Complex64; NUM_BRIDGE_ZONES] {
    let w = 2.0 * PI * freq_hz;
    let jw = Complex64::new(0.0, w);
    let mut h = [Complex64::ZERO; NUM_BRIDGE_ZONES];
    for z in 0..NUM_BRIDGE_ZONES {
        h[z] = Complex64::new(direct_zone_gain[z] as f64, 0.0);
    }

    for mode in modes {
        let wm = 2.0 * PI * (mode.freq_hz as f64);
        let denom = Complex64::new(wm * wm - w * w, 2.0 * (mode.zeta as f64) * wm * w);
        // Normalized bandpass velocity transfer peaking near 1.0 * rad * drive.
        let modal_tf = (jw * (2.0 * (mode.zeta as f64) * wm)) / denom;
        let rad_mono = mode.rad_coeffs[0] as f64;
        for (z, slot) in h.iter_mut().enumerate() {
            *slot += modal_tf * (rad_mono * (mode.zone_drive[z] as f64));
        }
    }
    h
}

/// Verifies minimum-phase inverse stability, matrix identity, and zero double-resonance excess peaking.
pub fn verify_pole_cancellation(preset: &BodyPreset, sample_rate: f32) -> PoleCancelReport {
    let ssm_poles_ref = AdmittanceSsm::new(sample_rate);
    let mut ssm_poles = [PresetReducedPole::default(); PRESET_REDUCED_POLES];
    for (i, p) in ssm_poles_ref.poles().iter().enumerate() {
        ssm_poles[i] = PresetReducedPole {
            freq_hz: p.freq_hz,
            zeta: p.zeta,
            modal_mass_kg: p.modal_mass_kg,
            zone_gains: p.zone_gains,
        };
    }

    let max_inverse_pole_real = max_inverse_pole_real_part(&preset.y_inf, &preset.reduced_poles);

    // Dense frequency grid: 180 log points (20..16,000 Hz) + tight clustering around all 10 reduced poles.
    let mut freqs = Vec::with_capacity(420);
    for k in 0..180 {
        let u = (k as f64) / 179.0;
        freqs.push(20.0 * (16_000.0_f64 / 20.0).powf(u));
    }
    for p in &preset.reduced_poles {
        let fp = p.freq_hz as f64;
        for off in -10..=10 {
            let f = fp * (1.0 + 0.015 * (off as f64));
            if (20.0..=16_000.0).contains(&f) {
                freqs.push(f);
            }
        }
    }

    let mut max_matrix_residual = 0.0_f64;
    let mut max_uncancelled_peaking_db = 0.0_f64;
    let mut max_cancelled_excess_db = 0.0_f64;

    for &f in &freqs {
        let y_red = eval_y_red_freq(ssm_poles_ref.y_inf(), &ssm_poles, f);
        let z_norm = eval_z_red_norm_freq(&preset.y_inf, &preset.reduced_poles, f)
            .expect("invertible Y_red");

        // Check Z_red_norm * Y_red == diag(Y_inf).
        let mut mat_err_sq = 0.0;
        for (r, z_row) in z_norm.iter().enumerate() {
            for (c, _) in y_red.iter().enumerate() {
                let mut prod = Complex64::ZERO;
                for (z_rk, y_row) in z_row.iter().zip(y_red.iter()) {
                    prod += *z_rk * y_row[c];
                }
                let target = if r == c {
                    Complex64::new(preset.y_inf[r] as f64, 0.0)
                } else {
                    Complex64::ZERO
                };
                mat_err_sq += (prod - target).norm_sq();
            }
        }
        max_matrix_residual = max_matrix_residual.max(mat_err_sq.sqrt());

        // Compare full-body transfer: target H_body*diag(y_inf), cancelled H_body*Z_norm*Y_red, uncancelled H_body*Y_red.
        let h_body = eval_body_transfer_vec(&preset.direct_zone_gain, &preset.modes, f);
        let mut tgt_sq = 0.0;
        let mut can_sq = 0.0;
        let mut unc_sq = 0.0;

        for c in 0..NUM_BRIDGE_ZONES {
            let t = h_body[c] * (preset.y_inf[c] as f64);
            tgt_sq += t.norm_sq();

            let mut u_sum = Complex64::ZERO;
            let mut c_sum = Complex64::ZERO;
            for r in 0..NUM_BRIDGE_ZONES {
                u_sum += h_body[r] * y_red[r][c];
                let mut z_y_rc = Complex64::ZERO;
                for k in 0..NUM_BRIDGE_ZONES {
                    z_y_rc += z_norm[r][k] * y_red[k][c];
                }
                c_sum += h_body[r] * z_y_rc;
            }
            unc_sq += u_sum.norm_sq();
            can_sq += c_sum.norm_sq();
        }

        if tgt_sq > 1.0e-18 {
            let cancelled_db = 10.0 * (can_sq / tgt_sq).log10().abs();
            let uncancelled_db = 10.0 * (unc_sq / tgt_sq).max(1.0).log10();
            max_cancelled_excess_db = max_cancelled_excess_db.max(cancelled_db);
            max_uncancelled_peaking_db = max_uncancelled_peaking_db.max(uncancelled_db);
        }
    }

    // Also run discrete sample-domain cascade through AdmittanceSsm -> InverseReducedBridge at sample_rate.
    let mut ssm = AdmittanceSsm::new(sample_rate);
    let mut inv = InverseReducedBridge::from_preset(preset, sample_rate);
    let mut max_discrete_err = 0.0_f64;
    for n in 0..512 {
        let mut f_in = [0.0_f64; NUM_BRIDGE_ZONES];
        if n == 0 {
            f_in = [1.0; NUM_BRIDGE_ZONES];
        }
        let (qd_free, v_free) = ssm.free_velocities();
        let mut v_out = v_free;
        for (r, vr) in v_out.iter_mut().enumerate() {
            for (c, &fc) in f_in.iter().enumerate() {
                *vr += ssm.a_br()[r][c] * fc;
            }
        }
        ssm.commit_step(&qd_free, &f_in);
        let y_norm = inv.step(&v_out);
        for z in 0..NUM_BRIDGE_ZONES {
            let target = (preset.y_inf[z] as f64) * f_in[z];
            max_discrete_err = max_discrete_err.max((y_norm[z] - target).abs());
        }
    }
    // Convert discrete residual into worst-case dB relative to minimum y_inf.
    let min_y_inf = preset.y_inf.iter().copied().fold(f32::MAX, f32::min) as f64;
    let max_discrete_cascade_err_db =
        20.0 * (1.0 + max_discrete_err / min_y_inf.max(1.0e-6)).log10();

    PoleCancelReport {
        max_inverse_pole_real,
        max_matrix_residual,
        max_uncancelled_peaking_db,
        max_cancelled_excess_db,
        max_discrete_cascade_err_db,
    }
}
