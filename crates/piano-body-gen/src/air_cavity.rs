//! 3D interior cabinet & under-board acoustic air cavity modes (65..650 Hz)
//! and physical 3D multipole radiation + concert/upright lid coupling weights.

use crate::presets::{body_spec, BodyArchetype, BodySpec};
use crate::types::{det_unit_f64, t60_to_zeta, ModeSource, SolvedBodyMode};
use piano_bridge::{KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES};
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};
use std::f64::consts::PI;

/// Speed of sound in dry air at ~20 deg C [m/s].
const SPEED_OF_SOUND_M_S: f64 = 343.2;

#[derive(Clone, Copy)]
struct CavityEigenCandidate {
    freq_hz: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    is_upper_bowl: bool,
}

/// Generates 3D interior cabinet & under-board acoustic air cavity modes for `spec` (`cavity_flag = 1`).
#[must_use]
pub fn synthesize_air_cavity_modes_for(spec: &BodySpec) -> Vec<SolvedBodyMode> {
    let mut candidates = Vec::new();
    let f_min = (spec.cavity_helmholtz_hz[0] - 5.0).max(50.0);
    let f_max = 550.0_f64;

    for (i, &fh) in spec.cavity_helmholtz_hz.iter().enumerate() {
        candidates.push(CavityEigenCandidate {
            freq_hz: fh,
            nx: i % 2,
            ny: i / 2,
            nz: 0,
            is_upper_bowl: false,
        });
    }
    let [lx0, ly0, lz0] = spec.cavity_lower_dims_m;
    for nx in 0..=4 {
        for ny in 0..=5 {
            for nz in 0..=1 {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let f = 0.5
                    * SPEED_OF_SOUND_M_S
                    * ((nx as f64 / lx0).powi(2)
                        + (ny as f64 / ly0).powi(2)
                        + (nz as f64 / lz0).powi(2))
                    .sqrt();
                if (f_min..=f_max).contains(&f) {
                    candidates.push(CavityEigenCandidate {
                        freq_hz: f,
                        nx,
                        ny,
                        nz,
                        is_upper_bowl: false,
                    });
                }
            }
        }
    }

    let [lx1, ly1, lz1] = spec.cavity_upper_dims_m;
    let f_upper_min = (f_min + 40.0).min(160.0);
    for nx in 0..=3 {
        for ny in 0..=4 {
            for nz in 0..=1 {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let f = 0.5
                    * SPEED_OF_SOUND_M_S
                    * ((nx as f64 / lx1).powi(2)
                        + (ny as f64 / ly1).powi(2)
                        + (nz as f64 / lz1).powi(2))
                    .sqrt();
                if (f_upper_min..=f_max).contains(&f) {
                    candidates.push(CavityEigenCandidate {
                        freq_hz: f,
                        nx,
                        ny,
                        nz,
                        is_upper_bowl: true,
                    });
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.freq_hz.total_cmp(&b.freq_hz));
    let target = 44;
    let selected: Vec<CavityEigenCandidate> = if candidates.len() <= target {
        candidates
    } else {
        (0..target)
            .map(|i| {
                let idx = (i * (candidates.len() - 1)) / (target - 1);
                candidates[idx]
            })
            .collect()
    };

    let mut modes = Vec::with_capacity(selected.len());
    for (idx, cand) in selected.iter().enumerate() {
        let freq_hz = cand.freq_hz.clamp(65.0, 550.0);
        let u = ((freq_hz - 65.0) / 485.0).clamp(0.0, 1.0);
        let t60_s = (spec.cavity_t60_base_s - 0.21 * u.powf(0.8)
            + 0.02 * (det_unit_f64(0xB100 + idx as u64) - 0.5))
            .clamp(0.10, 0.39);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for (z, pos) in spec.zone_pos_m.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let xn = ((pos[0] as f64) - spec.x_min_m) / spec.domain_lx_m;
            let yn = (pos[1] as f64) / spec.domain_ly_m;
            let z_factor = if cand.nz.is_multiple_of(2) {
                1.0
            } else {
                -0.82
            };
            let p_shape =
                z_factor * (cand.nx as f64 * PI * xn).cos() * (cand.ny as f64 * PI * yn).cos();
            let couple = if cand.is_upper_bowl {
                spec.cavity_upper_couple
            } else {
                spec.cavity_lower_couple
            };
            zone_drive[z] = couple * p_shape * (1.0 - 0.25 * u);
        }
        if !cand.is_upper_bowl && freq_hz < 250.0 {
            let phase = if cand.ny % 2 == 0 { 1.0 } else { -1.0 };
            zone_drive[KEYBED_ZONE_IDX] = 0.0022 * phase * (250.0 - freq_hz) / 185.0;
        }

        let modal_mass_kg = if cand.is_upper_bowl { 0.34 } else { 0.22 };
        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg,
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 1,
            source: ModeSource::AirCavity,
            plate_vol_norm: 0.0,
            plate_dx_norm: 0.0,
            plate_dy_norm: 0.0,
            has_plate_moments: false,
        });
    }
    modes
}

/// Generates concert-grand 3D interior cabinet & under-board air cavity modes.
#[must_use]
pub fn synthesize_air_cavity_modes() -> Vec<SolvedBodyMode> {
    synthesize_air_cavity_modes_for(&body_spec(BodyArchetype::ConcertGrand))
}

/// Computes physical 3D radiation multipole weights `[monopole, dipole_x, dipole_y, dipole_z, lid_coupling]`.
pub fn populate_radiation_multipoles_for(spec: &BodySpec, modes: &mut [SolvedBodyMode]) {
    let x_center = 0.0;
    let y_center = spec.multipole_y_center_m;
    let area_est = spec.domain_lx_m * spec.domain_ly_m * 0.72;
    let h_avg = spec.thickness_min_m + 0.533 * spec.thickness_span_m;
    let crown_shell = (0.045 * area_est / (spec.crown_radius_m * h_avg)).clamp(0.0, 0.85);

    for (idx, mode) in modes.iter_mut().enumerate() {
        let f = mode.freq_hz.max(20.0);
        let mut net = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut norm_abs = 0.0;
        for (z, pos) in spec.zone_pos_m.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let d = mode.zone_drive[z];
            net += d;
            mx += d * ((pos[0] as f64) - x_center);
            my += d * ((pos[1] as f64) - y_center);
            norm_abs += d.abs();
        }
        let inv_n = if norm_abs > 1.0e-12 {
            1.0 / norm_abs
        } else {
            0.0
        };
        let (mono_proj, dx_proj, dy_proj, sign_z) = if mode.has_plate_moments {
            let sgn = if mode.plate_vol_norm.abs() > 1.0e-4 {
                mode.plate_vol_norm.signum()
            } else if net >= 0.0 {
                1.0
            } else {
                -1.0
            };
            (
                mode.plate_vol_norm,
                mode.plate_dx_norm,
                mode.plate_dy_norm,
                sgn,
            )
        } else {
            let sgn = if net >= 0.0 { 1.0 } else { -1.0 };
            (
                (net * inv_n).clamp(-1.0, 1.0),
                (mx * inv_n / 0.55).clamp(-1.0, 1.0),
                (my * inv_n / 0.65).clamp(-1.0, 1.0),
                sgn,
            )
        };

        let coinc_ratio = (f / spec.plate_coincidence_hz).clamp(0.02, 4.0);
        let sigma_rad = (0.20 + 0.80 * coinc_ratio.powf(1.15) / (1.0 + coinc_ratio.powf(1.15)))
            .clamp(0.20, 0.98);
        let k_air = 2.0 * PI * f / SPEED_OF_SOUND_M_S;
        let helmholtz_eff = (k_air * 0.55) / (1.0 + (k_air * 0.55).powi(2)).sqrt();
        let lid_s = spec.lid_coupling_scale;

        let coeffs = match mode.source {
            ModeSource::PlateRitz | ModeSource::PlateStatistical => {
                let lf_weight = 1.0 / (1.0 + (f / spec.plate_coincidence_hz).powi(2));
                let monopole = if mode.has_plate_moments {
                    let base_mono = 0.42 * mono_proj + 0.06 * sign_z;
                    let shell_boost = 1.0 + crown_shell * lf_weight * mono_proj.abs();
                    base_mono * shell_boost * (0.72 + 0.28 * sigma_rad) / (1.0 + 0.18 * coinc_ratio)
                } else {
                    let base_mono = 0.28 * mono_proj + 0.14 * sign_z;
                    base_mono * (0.72 + 0.28 * sigma_rad) / (1.0 + 0.20 * coinc_ratio)
                };
                let dipole_x = 0.24 * dx_proj * (0.40 + 0.60 * helmholtz_eff);
                let dipole_y = 0.20 * dy_proj * (0.40 + 0.60 * helmholtz_eff);
                let dipole_z = sign_z * 0.26 * (1.0 - 0.58 * sigma_rad);
                let mid_band =
                    (f / 260.0) / (1.0 + (f / 260.0)) * (1.0 / (1.0 + (f / 10_500.0).powi(2)));
                let lid_coupling = (0.09 + 0.38 * mid_band) * sign_z * lid_s;
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
            ModeSource::RimKeybed => {
                let sign = if mode.zone_drive[KEYBED_ZONE_IDX] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let monopole = sign * 0.12 * (0.5 + 0.5 * helmholtz_eff);
                let dipole_x = 0.14 * (det_unit_f64(0xC100 + idx as u64) - 0.5) * 2.0;
                let dipole_y = -sign * 0.24;
                let dipole_z = -sign * 0.22;
                let lid_coupling = sign * 0.045 * lid_s;
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
            ModeSource::CastIronFrame => {
                let sign = if net >= 0.0 { 1.0 } else { -1.0 };
                let monopole = sign * 0.085 * sigma_rad;
                let dipole_x = 0.11 * dx_proj;
                let dipole_y = 0.09 * dy_proj;
                let dipole_z = sign * 0.09;
                let lid_coupling = sign * 0.16 * (0.4 + 0.6 * helmholtz_eff) * lid_s;
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
            ModeSource::AirCavity => {
                let is_upper = mode.modal_mass_kg > 0.28;
                let sign = if net >= 0.0 { 1.0 } else { -1.0 };
                let monopole = sign * 0.28 * (0.75 + 0.25 * helmholtz_eff);
                let dipole_x = 0.15 * dx_proj;
                let dipole_y = 0.14 * dy_proj;
                let dipole_z = if is_upper { sign * 0.12 } else { -sign * 0.20 };
                let lid_coupling = if is_upper {
                    sign * 0.30 * lid_s
                } else {
                    sign * 0.07 * lid_s
                };
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
        };

        mode.rad_coeffs = coeffs;
    }
}

/// Computes physical 3D radiation multipole weights for Concert Grand modes.
pub fn populate_radiation_multipoles(modes: &mut [SolvedBodyMode]) {
    populate_radiation_multipoles_for(&body_spec(BodyArchetype::ConcertGrand), modes);
}
