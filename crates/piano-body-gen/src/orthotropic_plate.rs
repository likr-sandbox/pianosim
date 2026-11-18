//! 2D orthotropic Sitka spruce soundboard Rayleigh-Ritz eigenbasis solver (`15x20` basis,
//! `48x64` quadrature) and high-frequency statistical Rayleigh overlap extension up to ~16.5 kHz.

use crate::linalg::solve_generalized_symmetric_eigen;
use crate::plate_assembly::{assemble_ritz_matrices, build_1d_sine_table};
pub use crate::plate_assembly::{QUAD_NX, QUAD_NY, RITZ_N, RITZ_NX, RITZ_NY};
use crate::presets::{body_plate_t60_s, body_spec, BodyArchetype, BodySpec};
use crate::ribs_rim::rim_signed_distance_for;
use crate::types::{det_unit_f64, t60_to_zeta, ModeSource, SolvedBodyMode, MAX_BODY_MODE_HZ};
use piano_bridge::NUM_BRIDGE_ZONES;
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};
use std::f64::consts::PI;

/// Longitudinal Sitka spruce Young's modulus along grain for Concert Grand [Pa].
pub const SPRUCE_E_L_PA: f64 = 11.5e9;
/// Radial Sitka spruce Young's modulus across grain for Concert Grand [Pa].
pub const SPRUCE_E_R_PA: f64 = 0.95e9;
/// In-plane Sitka spruce shear modulus for Concert Grand [Pa].
pub const SPRUCE_G_LR_PA: f64 = 0.75e9;
/// Major Sitka spruce Poisson's ratio `nu_LR` for Concert Grand.
pub const SPRUCE_NU_LR: f64 = 0.37;

/// Orthotropic plate diagnostic summary returned alongside solved modes.
#[derive(Clone, Debug)]
pub struct PlateSolveStats {
    /// Number of retained Ritz structural plate modes.
    pub ritz_mode_count: usize,
    /// Lowest retained Ritz soundboard mode [Hz].
    pub min_ritz_hz: f64,
    /// Highest retained Ritz soundboard mode [Hz].
    pub max_ritz_hz: f64,
    /// Number of high-frequency statistical Rayleigh plate modes.
    pub statistical_mode_count: usize,
}

/// Solves 2D orthotropic soundboard + statistical extension modes up to `~16.5 kHz` for `spec`.
pub fn solve_soundboard_modes_for(spec: &BodySpec) -> (Vec<SolvedBodyMode>, PlateSolveStats) {
    let bx = build_1d_sine_table(RITZ_NX, QUAD_NX, spec.domain_lx_m);
    let by = build_1d_sine_table(RITZ_NY, QUAD_NY, spec.domain_ly_m);
    let (k_mat, m_mat, m_int_mat) = assemble_ritz_matrices(spec, &bx, &by);

    let (eigvals, eigvecs) =
        solve_generalized_symmetric_eigen(RITZ_N, &k_mat, &m_mat).expect("SPD mass matrix");

    let mut basis_ksq = [0.0_f64; RITZ_N];
    for ix in 0..RITZ_NX {
        for iy in 0..RITZ_NY {
            basis_ksq[ix * RITZ_NY + iy] = bx.k[ix] * bx.k[ix] + by.k[iy] * by.k[iy];
        }
    }

    let norm_xy = (2.0 / spec.domain_lx_m).sqrt() * (2.0 / spec.domain_ly_m).sqrt();
    let mut zone_basis = [[0.0_f64; RITZ_N]; NUM_BRIDGE_ZONES];
    for (z, zb) in zone_basis.iter_mut().enumerate().take(NUM_BRIDGE_ZONES) {
        let zx = (spec.zone_pos_m[z][0] as f64) - spec.x_min_m;
        let zy = spec.zone_pos_m[z][1] as f64;
        for ix in 0..RITZ_NX {
            let sx = ((ix + 1) as f64 * PI * zx / spec.domain_lx_m).sin();
            for iy in 0..RITZ_NY {
                let sy = ((iy + 1) as f64 * PI * zy / spec.domain_ly_m).sin();
                zb[ix * RITZ_NY + iy] = norm_xy * sx * sy;
            }
        }
    }

    let h_avg = spec.thickness_min_m + 0.533 * spec.thickness_span_m;
    let d_avg_approx = h_avg.powi(3) / 12.0 * (spec.spruce_e_l_pa * spec.spruce_e_r_pa).sqrt();
    let mindlin_beta = d_avg_approx / ((5.0 / 6.0) * spec.spruce_g_lr_pa * h_avg);

    let mut modes = Vec::with_capacity(RITZ_N + spec.stat_count);
    let mut min_ritz_hz = f64::MAX;
    let mut max_ritz_hz = 0.0_f64;
    let mut col_vec = [0.0_f64; RITZ_N];

    for (r, &lam) in eigvals.iter().enumerate().take(RITZ_N) {
        if !lam.is_finite() || lam <= 0.0 {
            continue;
        }
        let mut ke_int = 0.0;
        let mut ke_tot = 0.0;
        let mut k_eff_sq = 0.0;
        let mut c_norm_sq = 0.0;
        for i in 0..RITZ_N {
            let ci = eigvecs[i * RITZ_N + r];
            col_vec[i] = ci;
            c_norm_sq += ci * ci;
            k_eff_sq += ci * ci * basis_ksq[i];
        }
        for i in 0..RITZ_N {
            let ci = col_vec[i];
            let m_int_row = &m_int_mat[i * RITZ_N..(i + 1) * RITZ_N];
            let m_row = &m_mat[i * RITZ_N..(i + 1) * RITZ_N];
            let mut s_int = 0.0;
            let mut s_tot = 0.0;
            for j in 0..RITZ_N {
                s_int += m_int_row[j] * col_vec[j];
                s_tot += m_row[j] * col_vec[j];
            }
            ke_int += ci * s_int;
            ke_tot += ci * s_tot;
        }
        if ke_tot <= 1.0e-12 || (ke_int / ke_tot) < 0.35 {
            continue;
        }
        let k_sq = if c_norm_sq > 0.0 {
            k_eff_sq / c_norm_sq
        } else {
            0.0
        };
        let freq_hz = (lam.sqrt() / (2.0 * PI)) / (1.0 + mindlin_beta * k_sq).sqrt();
        if !(28.0..=spec.ritz_max_hz).contains(&freq_hz) {
            continue;
        }

        let mass_scale = 1.0 / ke_tot.sqrt();
        let t60_s = body_plate_t60_s(spec, freq_hz);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            let mut psi_z = 0.0;
            for p in 0..RITZ_N {
                psi_z += col_vec[p] * zone_basis[z][p];
            }
            zone_drive[z] = spec.ritz_gain * psi_z * mass_scale;
        }
        if freq_hz < 360.0 {
            zone_drive[piano_bridge::KEYBED_ZONE_IDX] =
                0.0018 * (det_unit_f64(0x5100 + r as u64) - 0.5) * (360.0 - freq_hz) / 360.0;
        }

        min_ritz_hz = min_ritz_hz.min(freq_hz);
        max_ritz_hz = max_ritz_hz.max(freq_hz);

        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg: 1.0,
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 0,
            source: ModeSource::PlateRitz,
        });
    }

    let ritz_mode_count = modes.len();
    let stat_start_hz = (max_ritz_hz + 10.0).max(950.0);
    extend_statistical_rayleigh_modes(spec, &mut modes, stat_start_hz);

    let stats = PlateSolveStats {
        ritz_mode_count,
        min_ritz_hz,
        max_ritz_hz,
        statistical_mode_count: modes.len() - ritz_mode_count,
    };
    (modes, stats)
}

/// Solves 2D orthotropic concert-grand soundboard + extension modes up to `~16.5 kHz`.
pub fn solve_soundboard_modes() -> (Vec<SolvedBodyMode>, PlateSolveStats) {
    solve_soundboard_modes_for(&body_spec(BodyArchetype::ConcertGrand))
}

fn extend_statistical_rayleigh_modes(
    spec: &BodySpec,
    modes: &mut Vec<SolvedBodyMode>,
    f_start: f64,
) {
    let count = spec.stat_count;
    let span = MAX_BODY_MODE_HZ - f_start;
    let h_avg = spec.thickness_min_m + 0.533 * spec.thickness_span_m;
    let d_eff = (spec.spruce_e_l_pa * spec.spruce_e_r_pa).sqrt() * h_avg.powi(3) / 12.0;
    let m_avg = spec.spruce_rho * 0.01;

    for idx in 0..count {
        let u = (idx as f64 + 0.5) / (count as f64);
        let warp = 0.62 * u + 0.38 * u.powf(1.55);
        let jitter = (det_unit_f64(0x6100 + idx as u64) - 0.5) * 9.5;
        let freq_hz = (f_start + span * warp + jitter).clamp(f_start, MAX_BODY_MODE_HZ);

        let t60_s = body_plate_t60_s(spec, freq_hz);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let omega = 2.0 * PI * freq_hz;
        let k_plate = (m_avg * omega * omega / d_eff).powf(0.25);
        let theta_k = det_unit_f64(0x6200 + idx as u64) * PI;
        let kx = k_plate * theta_k.cos();
        let ky = k_plate * theta_k.sin();
        let phi_x = det_unit_f64(0x6300 + idx as u64) * 2.0 * PI;
        let phi_y = det_unit_f64(0x6400 + idx as u64) * 2.0 * PI;

        let hf_taper = (2_200.0 / (freq_hz + 900.0)).powf(0.18);
        let modal_gain = spec.stat_gain * hf_taper;

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for (z, pos) in spec.zone_pos_m.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let x = pos[0] as f64;
            let y = pos[1] as f64;
            let rim_d = rim_signed_distance_for(spec, x, y).clamp(0.0, 0.25);
            let boundary_fade =
                (rim_d / 0.14).clamp(if z <= 1 { spec.stat_bass_clamp } else { 0.35 }, 1.0);
            let treble_focus = if freq_hz > 1_850.0 && z >= 3 {
                spec.stat_treble_focus
            } else {
                1.0
            };
            let psi = (kx * x + phi_x).sin() * (ky * y + phi_y).sin();
            zone_drive[z] = modal_gain * boundary_fade * treble_focus * psi;
        }

        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg: 1.0,
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 0,
            source: ModeSource::PlateStatistical,
        });
    }
}
