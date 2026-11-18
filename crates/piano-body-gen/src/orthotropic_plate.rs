//! 2D orthotropic Sitka spruce soundboard Rayleigh-Ritz eigenbasis solver (`15x20` basis,
//! `48x64` quadrature) and high-frequency statistical Rayleigh overlap extension up to ~16.5 kHz.

use crate::linalg::solve_generalized_symmetric_eigen;
use crate::plate_assembly::{assemble_ritz_matrices, build_1d_sine_table};
pub use crate::plate_assembly::{QUAD_NX, QUAD_NY, RITZ_N, RITZ_NX, RITZ_NY};
use crate::presets::{body_plate_t60_s, body_spec, BodyArchetype, BodySpec};
use crate::ribs_rim::{bridge_zone_tangents_normals, rim_signed_distance_for};
use crate::types::{det_unit_f64, t60_to_zeta, ModeSource, SolvedBodyMode, MAX_BODY_MODE_HZ};
use piano_bridge::NUM_BRIDGE_ZONES;
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};
use std::f64::consts::PI;

const ROCKING_ALPHA: f64 = 0.35;
const GL_SQRT_3_5: f64 = 0.774_596_669_241_483_4;
const GL_WEIGHTS: [f64; 3] = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0];

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
    let (k_mat, m_mat, m_int_mat, integrals) = assemble_ritz_matrices(spec, &bx, &by);

    let (eigvals, eigvecs) =
        solve_generalized_symmetric_eigen(RITZ_N, &k_mat, &m_mat).expect("SPD mass matrix");

    let mut basis_ksq = [0.0_f64; RITZ_N];
    for ix in 0..RITZ_NX {
        for iy in 0..RITZ_NY {
            basis_ksq[ix * RITZ_NY + iy] = bx.k[ix] * bx.k[ix] + by.k[iy] * by.k[iy];
        }
    }

    let norm_xy = (2.0 / spec.domain_lx_m).sqrt() * (2.0 / spec.domain_ly_m).sqrt();
    let tangents = bridge_zone_tangents_normals(&spec.zone_pos_m);
    let mut zone_basis = [[0.0_f64; RITZ_N]; NUM_BRIDGE_ZONES];
    for (z, zb) in zone_basis.iter_mut().enumerate().take(NUM_BRIDGE_ZONES) {
        let (t, n, half_seg) = tangents[z];
        let h_br = (spec.zone_pos_m[z][2] as f64).max(0.02);
        let xc = spec.zone_pos_m[z][0] as f64;
        let yc = spec.zone_pos_m[z][1] as f64;
        let zx_c = (xc - spec.x_min_m).clamp(0.01, spec.domain_lx_m - 0.01);
        let zy_c = yc.clamp(0.01, spec.domain_ly_m - 0.01);

        let offs = [-GL_SQRT_3_5 * half_seg, 0.0, GL_SQRT_3_5 * half_seg];
        for (q, &s_off) in offs.iter().enumerate() {
            let zx = ((xc + s_off * t[0]) - spec.x_min_m).clamp(0.01, spec.domain_lx_m - 0.01);
            let zy = (yc + s_off * t[1]).clamp(0.01, spec.domain_ly_m - 0.01);
            let w_q = GL_WEIGHTS[q] * norm_xy;
            for ix in 0..RITZ_NX {
                let sx = (bx.k[ix] * zx).sin();
                for iy in 0..RITZ_NY {
                    let sy = (by.k[iy] * zy).sin();
                    zb[ix * RITZ_NY + iy] += w_q * sx * sy;
                }
            }
        }
        for ix in 0..RITZ_NX {
            let kx = bx.k[ix];
            let (sx_c, cx_c) = (kx * zx_c).sin_cos();
            for iy in 0..RITZ_NY {
                let ky = by.k[iy];
                let (sy_c, cy_c) = (ky * zy_c).sin_cos();
                let dpsi_dn = norm_xy * (n[0] * kx * cx_c * sy_c + n[1] * ky * sx_c * cy_c);
                zb[ix * RITZ_NY + iy] += ROCKING_ALPHA * h_br * dpsi_dn;
            }
        }
    }

    let area_in = integrals.inside_area_m2.max(1.0e-6);
    let mean_rho_a = (integrals.inside_mass_kg / area_in).max(1.0);

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

        let int_w2_in = (ke_int / ke_tot) / mean_rho_a;
        let denom_vol = (area_in * int_w2_in).sqrt().max(1.0e-12);
        let (mut vr, mut mxr, mut myr) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (p, &cv) in col_vec.iter().enumerate() {
            let cp = mass_scale * cv;
            vr += cp * integrals.int_w[p];
            mxr += cp * integrals.int_xw[p];
            myr += cp * integrals.int_yw[p];
        }
        let plate_vol_norm = (vr / denom_vol).clamp(-1.0, 1.0);
        let plate_dx_norm = (mxr / (0.55 * denom_vol)).clamp(-1.0, 1.0);
        let plate_dy_norm = (myr / (0.65 * denom_vol)).clamp(-1.0, 1.0);

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
            plate_vol_norm,
            plate_dx_norm,
            plate_dy_norm,
            has_plate_moments: true,
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
    let tangents = bridge_zone_tangents_normals(&spec.zone_pos_m);

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

        let hf_taper = (2_200.0 / (freq_hz + 900.0)).powf(0.38);
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
            let (t, n, half_seg) = tangents[z];
            let kt_l = (kx * t[0] + ky * t[1]) * half_seg;
            let sinc_line = if kt_l.abs() < 1.0e-5 {
                1.0 - kt_l * kt_l / 6.0
            } else {
                kt_l.sin() / kt_l
            };
            let (sx, cx) = (kx * x + phi_x).sin_cos();
            let (sy, cy) = (ky * y + phi_y).sin_cos();
            let psi = sx * sy;
            let dpsi_dn = n[0] * kx * cx * sy + n[1] * ky * sx * cy;
            let h_br = (pos[2] as f64).max(0.02);
            let rock = (ROCKING_ALPHA * h_br * dpsi_dn).clamp(-0.5, 0.5);
            zone_drive[z] = modal_gain * boundary_fade * treble_focus * (psi * sinc_line + rock);
        }

        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg: 1.0,
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 0,
            source: ModeSource::PlateStatistical,
            plate_vol_norm: 0.0,
            plate_dx_norm: 0.0,
            plate_dy_norm: 0.0,
            has_plate_moments: false,
        });
    }
}
