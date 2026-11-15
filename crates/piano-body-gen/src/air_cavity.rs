//! 3D interior cabinet & under-board acoustic air cavity modes (65..550 Hz)
//! and physical 3D multipole radiation + concert lid coupling weights.

use crate::ribs_rim::{CONCERT_ZONE_POS_M, SOUNDBOARD_LX_M, SOUNDBOARD_LY_M, SOUNDBOARD_X_MIN_M};
use crate::types::{det_unit_f64, t60_to_zeta, ModeSource, SolvedBodyMode};
use piano_bridge::{KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES};
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};
use std::f64::consts::PI;

/// Speed of sound in dry air at ~20 deg C [m/s].
const SPEED_OF_SOUND_M_S: f64 = 343.2;
/// Sitka spruce acoustic coincidence frequency (~8.3 mm thickness) [Hz].
const PLATE_COINCIDENCE_HZ: f64 = 1_920.0;

#[derive(Clone, Copy)]
struct CavityEigenCandidate {
    freq_hz: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    is_upper_bowl: bool,
}

/// Generates 3D interior cabinet & under-board acoustic air cavity modes (`65..550 Hz`, `cavity_flag = 1`).
pub fn synthesize_air_cavity_modes() -> Vec<SolvedBodyMode> {
    let mut candidates = Vec::new();

    // Under-board cavity (~1.38 x 1.72 x 0.36 m open underside + Helmholtz port modes).
    // Include lowest Helmholtz / open-underside resonances around 65..110 Hz plus box modes.
    let helmholtz_freqs = [67.5, 84.0, 104.5];
    for (i, &fh) in helmholtz_freqs.iter().enumerate() {
        candidates.push(CavityEigenCandidate {
            freq_hz: fh,
            nx: i % 2,
            ny: i / 2,
            nz: 0,
            is_upper_bowl: false,
        });
    }
    for nx in 0..=4 {
        for ny in 0..=5 {
            for nz in 0..=1 {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let f = 0.5
                    * SPEED_OF_SOUND_M_S
                    * ((nx as f64 / 1.38).powi(2)
                        + (ny as f64 / 1.72).powi(2)
                        + (nz as f64 / 0.42).powi(2))
                    .sqrt();
                if (65.0..=550.0).contains(&f) {
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

    // Upper lid-bowl cavity (~1.34 x 1.68 x 0.56 m angled wedge cavity above soundboard).
    for nx in 0..=3 {
        for ny in 0..=4 {
            for nz in 0..=1 {
                if nx == 0 && ny == 0 && nz == 0 {
                    continue;
                }
                let f = 0.5
                    * SPEED_OF_SOUND_M_S
                    * ((nx as f64 / 1.34).powi(2)
                        + (ny as f64 / 1.68).powi(2)
                        + (nz as f64 / 0.58).powi(2))
                    .sqrt();
                if (110.0..=550.0).contains(&f) {
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
    // Subsample uniformly to 44 distinct non-degenerate cavity modes across 65..550 Hz.
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
        // Open cabinet air losses: low modes ~0.34 s down to ~0.13 s near 550 Hz.
        let u = ((freq_hz - 65.0) / 485.0).clamp(0.0, 1.0);
        let t60_s = (0.35 - 0.21 * u.powf(0.8) + 0.02 * (det_unit_f64(0xB100 + idx as u64) - 0.5))
            .clamp(0.11, 0.38);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for (z, pos) in CONCERT_ZONE_POS_M.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let xn = ((pos[0] as f64) - SOUNDBOARD_X_MIN_M) / SOUNDBOARD_LX_M;
            let yn = (pos[1] as f64) / SOUNDBOARD_LY_M;
            let z_factor = if cand.nz.is_multiple_of(2) {
                1.0
            } else {
                -0.82
            };
            let p_shape =
                z_factor * (cand.nx as f64 * PI * xn).cos() * (cand.ny as f64 * PI * yn).cos();
            let couple = if cand.is_upper_bowl { 0.0046 } else { 0.0058 };
            zone_drive[z] = couple * p_shape * (1.0 - 0.25 * u);
        }
        // Under-board Helmholtz & low longitudinal modes couple weakly to Zone 8 front opening.
        if !cand.is_upper_bowl && freq_hz < 250.0 {
            let phase = if cand.ny % 2 == 0 { 1.0 } else { -1.0 };
            zone_drive[KEYBED_ZONE_IDX] = 0.0022 * phase * (250.0 - freq_hz) / 185.0;
        }

        // Encode upper vs under-board distinction gently in modal_mass_kg sign-free metadata.
        let modal_mass_kg = if cand.is_upper_bowl { 0.34 } else { 0.22 };
        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg,
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 1,
            source: ModeSource::AirCavity,
        });
    }
    modes
}

/// Computes physical 3D radiation multipole weights `[monopole, dipole_x, dipole_y, dipole_z, lid_coupling]`.
pub fn populate_radiation_multipoles(modes: &mut [SolvedBodyMode]) {
    let x_center = 0.0;
    let y_center = 0.72;

    for (idx, mode) in modes.iter_mut().enumerate() {
        let f = mode.freq_hz.max(20.0);
        // Estimate spatial dipole moments directly from zone_drive distribution across the 8 bridge zones.
        let mut net = 0.0;
        let mut mx = 0.0;
        let mut my = 0.0;
        let mut norm_abs = 0.0;
        for (z, pos) in CONCERT_ZONE_POS_M.iter().enumerate().take(NUM_BRIDGE_ZONES) {
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
        let mono_proj = (net * inv_n).clamp(-1.0, 1.0);
        let dx_proj = (mx * inv_n / 0.55).clamp(-1.0, 1.0);
        let dy_proj = (my * inv_n / 0.65).clamp(-1.0, 1.0);

        // Coincidence radiation efficiency curve for Sitka spruce board.
        let coinc_ratio = (f / PLATE_COINCIDENCE_HZ).clamp(0.02, 4.0);
        let sigma_rad = (0.14 + 0.86 * coinc_ratio.powf(1.15) / (1.0 + coinc_ratio.powf(1.15)))
            .clamp(0.14, 0.98);
        let k_air = 2.0 * PI * f / SPEED_OF_SOUND_M_S;
        let helmholtz_eff = (k_air * 0.55) / (1.0 + (k_air * 0.55).powi(2)).sqrt();

        let coeffs = match mode.source {
            ModeSource::PlateRitz | ModeSource::PlateStatistical => {
                // Monopole strengthens near/above coincidence; low modes carry finite volume velocity + dipole_z.
                let base_mono = 0.28 * mono_proj + 0.14 * (if net >= 0.0 { 1.0 } else { -1.0 });
                let monopole = base_mono * (0.45 + 0.55 * sigma_rad);
                let dipole_x = 0.24 * dx_proj * (0.40 + 0.60 * helmholtz_eff);
                let dipole_y = 0.20 * dy_proj * (0.40 + 0.60 * helmholtz_eff);
                // Vertical soundboard dipole dominates below coincidence, decaying smoothly above f_c.
                let sign_z = if net >= 0.0 { 1.0 } else { -1.0 };
                let dipole_z = sign_z * 0.26 * (1.0 - 0.58 * sigma_rad);
                // Open concert lid reflects mid/high upper-board energy toward hall/CloseStereo.
                let mid_band =
                    (f / 260.0) / (1.0 + (f / 260.0)) * (1.0 / (1.0 + (f / 7_500.0).powi(2)));
                let lid_coupling = (0.08 + 0.26 * mid_band) * sign_z;
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
            ModeSource::RimKeybed => {
                let sign = if mode.zone_drive[KEYBED_ZONE_IDX] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                // Keybed sits at y = -0.16, z = -0.07: strong front/downward dipole, weak lid coupling.
                let monopole = sign * 0.12 * (0.5 + 0.5 * helmholtz_eff);
                let dipole_x = 0.14 * (det_unit_f64(0xC100 + idx as u64) - 0.5) * 2.0;
                let dipole_y = -sign * 0.24;
                let dipole_z = -sign * 0.22;
                let lid_coupling = sign * 0.045;
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
            ModeSource::CastIronFrame => {
                let sign = if net >= 0.0 { 1.0 } else { -1.0 };
                // Cast-iron frame faces upward directly beneath open lid; moderate lid & weak air radiation.
                let monopole = sign * 0.085 * sigma_rad;
                let dipole_x = 0.11 * dx_proj;
                let dipole_y = 0.09 * dy_proj;
                let dipole_z = sign * 0.09;
                let lid_coupling = sign * 0.16 * (0.4 + 0.6 * helmholtz_eff);
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
            ModeSource::AirCavity => {
                let is_upper = mode.modal_mass_kg > 0.28;
                let sign = if net >= 0.0 { 1.0 } else { -1.0 };
                // Upper lid bowl radiates strongly via lid_coupling; under-board cavity radiates downward/front.
                let monopole = sign * 0.22 * (0.55 + 0.45 * helmholtz_eff);
                let dipole_x = 0.15 * dx_proj;
                let dipole_y = 0.14 * dy_proj;
                let dipole_z = if is_upper { sign * 0.12 } else { -sign * 0.20 };
                let lid_coupling = if is_upper { sign * 0.30 } else { sign * 0.07 };
                [monopole, dipole_x, dipole_y, dipole_z, lid_coupling]
            }
        };

        mode.rad_coeffs = coeffs;
    }
}
