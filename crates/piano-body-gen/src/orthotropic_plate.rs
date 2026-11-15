//! 2D orthotropic Sitka spruce soundboard Rayleigh-Ritz eigenbasis solver and
//! high-frequency statistical/Rayleigh modal overlap extension up to ~16.5 kHz.

use crate::linalg::solve_generalized_symmetric_eigen;
use crate::ribs_rim::{
    evaluate_plate_site, rim_signed_distance_m, PlateSiteProps, CONCERT_ZONE_POS_M,
    SOUNDBOARD_LX_M, SOUNDBOARD_LY_M, SOUNDBOARD_X_MIN_M, SPRUCE_GRAIN_ANGLE_RAD,
};
use crate::types::{
    det_unit_f64, sitka_spruce_t60_s, t60_to_zeta, ModeSource, SolvedBodyMode, MAX_BODY_MODE_HZ,
};
use piano_bridge::NUM_BRIDGE_ZONES;
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};
use std::f64::consts::PI;

/// Longitudinal Sitka spruce Young's modulus along grain [Pa].
pub const SPRUCE_E_L_PA: f64 = 11.5e9;
/// Radial Sitka spruce Young's modulus across grain [Pa].
pub const SPRUCE_E_R_PA: f64 = 0.95e9;
/// In-plane Sitka spruce shear modulus [Pa].
pub const SPRUCE_G_LR_PA: f64 = 0.75e9;
/// Major Sitka spruce Poisson's ratio `nu_LR`.
pub const SPRUCE_NU_LR: f64 = 0.37;
/// Soundboard crown radius of curvature [m] producing membrane-shell downbearing stiffness.
const CROWN_RADIUS_M: f64 = 24.0;
/// Ritz sine basis count along width (`x`) and length (`y`): `9 x 12 = 108` modes.
const RITZ_NX: usize = 9;
const RITZ_NY: usize = 12;
const RITZ_N: usize = RITZ_NX * RITZ_NY;
/// Gauss/midpoint quadrature grid across soundboard domain.
const QUAD_NX: usize = 28;
const QUAD_NY: usize = 36;
/// Target count for high-frequency statistical Rayleigh overlap extension.
const STAT_RAYLEIGH_COUNT: usize = 1_060;

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

/// Computes the symmetric 3x3 flexural rigidity matrix `D` in `(x, y)` coordinates [N*m].
fn rotated_flexural_rigidity(site: &PlateSiteProps) -> [[f64; 3]; 3] {
    let h3_12 = site.thickness_m.powi(3) / 12.0;
    let nu_rl = SPRUCE_NU_LR * (SPRUCE_E_R_PA / SPRUCE_E_L_PA);
    let denom = 1.0 - SPRUCE_NU_LR * nu_rl;
    let q_11 = SPRUCE_E_L_PA / denom;
    let q_22 = SPRUCE_E_R_PA / denom;
    let q_12 = SPRUCE_NU_LR * SPRUCE_E_R_PA / denom;
    let q_66 = SPRUCE_G_LR_PA;

    let c = SPRUCE_GRAIN_ANGLE_RAD.cos();
    let s = SPRUCE_GRAIN_ANGLE_RAD.sin();
    // Strain transformation T_eps from (xx, yy, 2*xy) to (LL, RR, LR).
    let t = [
        [c * c, s * s, c * s],
        [s * s, c * c, -c * s],
        [-2.0 * c * s, 2.0 * c * s, c * c - s * s],
    ];
    let q_0 = [[q_11, q_12, 0.0], [q_12, q_22, 0.0], [0.0, 0.0, q_66]];

    // D_plate = h^3/12 * T_eps^T * Q_0 * T_eps.
    let mut d = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sum += t[k][i] * q_0[k][l] * t[l][j];
                }
            }
            d[i][j] = h3_12 * sum;
        }
    }

    // Add cross-grain transverse spruce rib bending along alpha = theta_grain + pi/2.
    let v_rib = [s * s, c * c, -s * c];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] += site.rib_bending_nm * v_rib[i] * v_rib[j];
        }
    }

    // Add longitudinal rock-maple bass/long bridge bending along bridge tangent.
    let cb = site.bridge_angle_rad.cos();
    let sb = site.bridge_angle_rad.sin();
    let v_br = [cb * cb, sb * sb, cb * sb];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] += site.bridge_bending_nm * v_br[i] * v_br[j];
        }
    }

    d
}

/// Evaluates orthonormalized sine basis `(phi, w_xx, w_yy, 2*w_xy)` and wavenumbers.
struct SineBasis1D {
    val: Vec<f64>,
    d2: Vec<f64>,
    d1: Vec<f64>,
    k: Vec<f64>,
}

fn build_1d_sine_table(n_modes: usize, n_quad: usize, origin: f64, span: f64) -> SineBasis1D {
    let norm = (2.0 / span).sqrt();
    let mut val = vec![0.0; n_modes * n_quad];
    let mut d2 = vec![0.0; n_modes * n_quad];
    let mut d1 = vec![0.0; n_modes * n_quad];
    let mut k_vec = vec![0.0; n_modes];

    for (m, k_slot) in k_vec.iter_mut().enumerate().take(n_modes) {
        let km = ((m + 1) as f64) * PI / span;
        *k_slot = km;
        for q in 0..n_quad {
            let u = ((q as f64) + 0.5) / (n_quad as f64) * span;
            let _x = origin + u;
            let arg = km * u;
            let s = norm * arg.sin();
            let c = norm * arg.cos();
            let idx = m * n_quad + q;
            val[idx] = s;
            d1[idx] = km * c;
            d2[idx] = -km * km * s;
        }
    }
    SineBasis1D {
        val,
        d2,
        d1,
        k: k_vec,
    }
}

/// Solves 2D orthotropic Sitka spruce soundboard + extension modes up to `~16.5 kHz`.
pub fn solve_soundboard_modes() -> (Vec<SolvedBodyMode>, PlateSolveStats) {
    let bx = build_1d_sine_table(RITZ_NX, QUAD_NX, SOUNDBOARD_X_MIN_M, SOUNDBOARD_LX_M);
    let by = build_1d_sine_table(RITZ_NY, QUAD_NY, 0.0, SOUNDBOARD_LY_M);
    let dx = SOUNDBOARD_LX_M / (QUAD_NX as f64);
    let dy = SOUNDBOARD_LY_M / (QUAD_NY as f64);
    let d_area = dx * dy;

    let n_pts = QUAD_NX * QUAD_NY;
    let mut sites = Vec::with_capacity(n_pts);
    let mut d_mats = Vec::with_capacity(n_pts);
    let mut k_crown_pts = Vec::with_capacity(n_pts);
    // Effective crown shell stiffness from harmonic mean orthotropic modulus + string downbearing.
    let e_eff = (SPRUCE_E_L_PA * SPRUCE_E_R_PA).sqrt();
    let inv_r2 = 1.0 / (CROWN_RADIUS_M * CROWN_RADIUS_M);

    for qx in 0..QUAD_NX {
        let x = SOUNDBOARD_X_MIN_M + ((qx as f64) + 0.5) * dx;
        for qy in 0..QUAD_NY {
            let y = ((qy as f64) + 0.5) * dy;
            let site = evaluate_plate_site(x, y);
            let d = rotated_flexural_rigidity(&site);
            // Shallow crown shell term + downbearing membrane pre-stress inside rim contour.
            let inside = if site.rim_signed_dist_m > 0.0 {
                1.0
            } else {
                0.0
            };
            let k_crown = (0.44 * e_eff * site.thickness_m * inv_r2 + 2.8e4) * inside;
            sites.push(site);
            d_mats.push(d);
            k_crown_pts.push(k_crown);
        }
    }

    // Precompute basis evaluations & curvatures at all quadrature points.
    let mut w_tab = vec![0.0_f64; RITZ_N * n_pts];
    let mut kxx_tab = vec![0.0_f64; RITZ_N * n_pts];
    let mut kyy_tab = vec![0.0_f64; RITZ_N * n_pts];
    let mut kxy2_tab = vec![0.0_f64; RITZ_N * n_pts];
    let mut basis_ksq = [0.0_f64; RITZ_N];

    for ix in 0..RITZ_NX {
        for iy in 0..RITZ_NY {
            let p = ix * RITZ_NY + iy;
            basis_ksq[p] = bx.k[ix] * bx.k[ix] + by.k[iy] * by.k[iy];
            let base = p * n_pts;
            for qx in 0..QUAD_NX {
                let xv = bx.val[ix * QUAD_NX + qx];
                let xd1 = bx.d1[ix * QUAD_NX + qx];
                let xd2 = bx.d2[ix * QUAD_NX + qx];
                let row = base + qx * QUAD_NY;
                for qy in 0..QUAD_NY {
                    let yv = by.val[iy * QUAD_NY + qy];
                    let yd1 = by.d1[iy * QUAD_NY + qy];
                    let yd2 = by.d2[iy * QUAD_NY + qy];
                    let q = row + qy;
                    w_tab[q] = xv * yv;
                    kxx_tab[q] = xd2 * yv;
                    kyy_tab[q] = xv * yd2;
                    kxy2_tab[q] = 2.0 * xd1 * yd1;
                }
            }
        }
    }

    // Assemble symmetric stiffness K and mass M matrices (108 x 108).
    let mut k_mat = vec![0.0_f64; RITZ_N * RITZ_N];
    let mut m_mat = vec![0.0_f64; RITZ_N * RITZ_N];
    let mut m_int_mat = vec![0.0_f64; RITZ_N * RITZ_N];

    for q in 0..n_pts {
        let s = &sites[q];
        let dm = &d_mats[q];
        let k_el = (k_crown_pts[q] + s.rim_stiffness_n_m3) * d_area;
        let m_el = s.areal_mass_kg_m2 * d_area;
        let m_in = if s.rim_signed_dist_m > 0.0 { m_el } else { 0.0 };

        for p1 in 0..RITZ_N {
            let i1 = p1 * n_pts + q;
            let w1 = w_tab[i1];
            let kxx1 = kxx_tab[i1];
            let kyy1 = kyy_tab[i1];
            let kxy1 = kxy2_tab[i1];
            // D * kappa1 weighted by d_area.
            let dk0 = (dm[0][0] * kxx1 + dm[0][1] * kyy1 + dm[0][2] * kxy1) * d_area;
            let dk1 = (dm[1][0] * kxx1 + dm[1][1] * kyy1 + dm[1][2] * kxy1) * d_area;
            let dk2 = (dm[2][0] * kxx1 + dm[2][1] * kyy1 + dm[2][2] * kxy1) * d_area;

            for p2 in p1..RITZ_N {
                let i2 = p2 * n_pts + q;
                let w2 = w_tab[i2];
                let ke =
                    kxx_tab[i2] * dk0 + kyy_tab[i2] * dk1 + kxy2_tab[i2] * dk2 + k_el * w1 * w2;
                let me = m_el * w1 * w2;
                let mie = m_in * w1 * w2;
                let idx = p1 * RITZ_N + p2;
                k_mat[idx] += ke;
                m_mat[idx] += me;
                m_int_mat[idx] += mie;
            }
        }
    }
    for p1 in 0..RITZ_N {
        for p2 in 0..p1 {
            let sym = p2 * RITZ_N + p1;
            let dst = p1 * RITZ_N + p2;
            k_mat[dst] = k_mat[sym];
            m_mat[dst] = m_mat[sym];
            m_int_mat[dst] = m_int_mat[sym];
        }
    }

    let (eigvals, eigvecs) =
        solve_generalized_symmetric_eigen(RITZ_N, &k_mat, &m_mat).expect("SPD mass matrix");

    // Precompute basis amplitudes at each physical bridge zone [0..8].
    let norm_xy = (2.0 / SOUNDBOARD_LX_M).sqrt() * (2.0 / SOUNDBOARD_LY_M).sqrt();
    let mut zone_basis = [[0.0_f64; RITZ_N]; NUM_BRIDGE_ZONES];
    for z in 0..NUM_BRIDGE_ZONES {
        let zx = (CONCERT_ZONE_POS_M[z][0] as f64) - SOUNDBOARD_X_MIN_M;
        let zy = CONCERT_ZONE_POS_M[z][1] as f64;
        for ix in 0..RITZ_NX {
            let sx = ((ix + 1) as f64 * PI * zx / SOUNDBOARD_LX_M).sin();
            for iy in 0..RITZ_NY {
                let sy = ((iy + 1) as f64 * PI * zy / SOUNDBOARD_LY_M).sin();
                zone_basis[z][ix * RITZ_NY + iy] = norm_xy * sx * sy;
            }
        }
    }

    // Mindlin transverse shear compliance coefficient: D_avg / (5/6 * G_xz * h_avg).
    let d_avg_approx = 0.0083_f64.powi(3) / 12.0 * (SPRUCE_E_L_PA * SPRUCE_E_R_PA).sqrt();
    let mindlin_beta = d_avg_approx / ((5.0 / 6.0) * SPRUCE_G_LR_PA * 0.0083);

    let mut modes = Vec::with_capacity(RITZ_N + STAT_RAYLEIGH_COUNT);
    let mut min_ritz_hz = f64::MAX;
    let mut max_ritz_hz = 0.0_f64;

    for r in 0..RITZ_N {
        let lam = eigvals[r];
        if !lam.is_finite() || lam <= 0.0 {
            continue;
        }
        // Check interior kinetic energy ratio vs total generalized mass (1.0).
        let mut ke_int = 0.0;
        let mut ke_tot = 0.0;
        let mut k_eff_sq = 0.0;
        let mut c_norm_sq = 0.0;
        for i in 0..RITZ_N {
            let ci = eigvecs[i * RITZ_N + r];
            c_norm_sq += ci * ci;
            k_eff_sq += ci * ci * basis_ksq[i];
            for j in 0..RITZ_N {
                let cj = eigvecs[j * RITZ_N + r];
                ke_int += ci * m_int_mat[i * RITZ_N + j] * cj;
                ke_tot += ci * m_mat[i * RITZ_N + j] * cj;
            }
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
        if !(28.0..=2_800.0).contains(&freq_hz) {
            continue;
        }

        let mass_scale = 1.0 / ke_tot.sqrt();
        let t60_s = sitka_spruce_t60_s(freq_hz);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            let mut psi_z = 0.0;
            for p in 0..RITZ_N {
                psi_z += eigvecs[p * RITZ_N + r] * zone_basis[z][p];
            }
            // Mass-normalized modal projection calibrated into Stage 4 bridge velocity transfer.
            zone_drive[z] = 0.0145 * psi_z * mass_scale;
        }
        // Weak rim-cheek transmission into low-order (< 360 Hz) soundboard modes from Zone 8.
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
    extend_statistical_rayleigh_modes(&mut modes, stat_start_hz, STAT_RAYLEIGH_COUNT);

    let stats = PlateSolveStats {
        ritz_mode_count,
        min_ritz_hz,
        max_ritz_hz,
        statistical_mode_count: modes.len() - ritz_mode_count,
    };
    (modes, stats)
}

/// Smooth high-frequency Rayleigh statistical overlap extension from `f_start` to `16.5 kHz`.
fn extend_statistical_rayleigh_modes(modes: &mut Vec<SolvedBodyMode>, f_start: f64, count: usize) {
    // Power-exponent warped spacing keeps mid/presence region (1.2..5 kHz) richly populated
    // for modal overlap >= 3 while covering continuously to MAX_BODY_MODE_HZ (~16.5 kHz).
    let span = MAX_BODY_MODE_HZ - f_start;
    let d_eff = (SPRUCE_E_L_PA * SPRUCE_E_R_PA).sqrt() * 0.0083_f64.powi(3) / 12.0;
    let m_avg = 4.35; // kg/m^2

    for idx in 0..count {
        let u = (idx as f64 + 0.5) / (count as f64);
        let warp = 0.62 * u + 0.38 * u.powf(1.55);
        let jitter = (det_unit_f64(0x6100 + idx as u64) - 0.5) * 9.5;
        let freq_hz = (f_start + span * warp + jitter).clamp(f_start, MAX_BODY_MODE_HZ);

        let t60_s = sitka_spruce_t60_s(freq_hz);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        // Bending wavenumber k from plate dispersion, used to form correlated zone shapes.
        let omega = 2.0 * PI * freq_hz;
        let k_plate = (m_avg * omega * omega / d_eff).powf(0.25);
        let theta_k = det_unit_f64(0x6200 + idx as u64) * PI;
        let kx = k_plate * theta_k.cos();
        let ky = k_plate * theta_k.sin();
        let phi_x = det_unit_f64(0x6300 + idx as u64) * 2.0 * PI;
        let phi_y = det_unit_f64(0x6400 + idx as u64) * 2.0 * PI;

        // Overlap-normalized amplitude envelope rolling smoothly from Ritz modes up to 16.5 kHz.
        let hf_taper = (1_800.0 / (freq_hz + 900.0)).powf(0.42);
        let modal_gain = 0.0095 * hf_taper;

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for (z, pos) in CONCERT_ZONE_POS_M.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let x = pos[0] as f64;
            let y = pos[1] as f64;
            let rim_d = rim_signed_distance_m(x, y).clamp(0.0, 0.25);
            let boundary_fade = (rim_d / 0.14).clamp(0.35, 1.0);
            // Treble bridge modes localize tighter above 4 kHz due to local bridge impedance.
            let treble_focus = if freq_hz > 3_500.0 && z >= 4 {
                1.18
            } else if freq_hz > 5_000.0 && z <= 1 {
                0.72
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
