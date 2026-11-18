//! Body geometry, spruce ribs, bridge heterogeneity, laminated beech
//! rim boundary impedance, Zone 8 wooden keybed coupling, and cast-iron harp frame.

use crate::presets::{body_plate_t60_s, body_spec, BodyArchetype, BodySpec};
use crate::types::{det_unit_f64, t60_to_zeta, ModeSource, SolvedBodyMode};
use piano_bridge::{KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES};
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};

/// Rectangular Ritz integration domain width [m] (`x in [-0.76, 0.76]`) for Concert Grand.
pub const SOUNDBOARD_LX_M: f64 = 1.52;
/// Rectangular Ritz integration domain length [m] (`y in [0.0, 1.98]`) for Concert Grand.
pub const SOUNDBOARD_LY_M: f64 = 1.98;
/// Left boundary of Ritz domain [m] for Concert Grand.
pub const SOUNDBOARD_X_MIN_M: f64 = -0.5 * SOUNDBOARD_LX_M;
/// Number of transverse Sitka spruce ribs under the concert-grand soundboard.
pub const NUM_SPRUCE_RIBS: usize = 14;
/// Spruce grain angle relative to keyboard x-axis [rad] (~32 deg) for Concert Grand.
pub const SPRUCE_GRAIN_ANGLE_RAD: f64 = 32.0 * std::f64::consts::PI / 180.0;

/// 3D physical positions [m] of the 8 bridge zones + Zone 8 keybed (`KEYBED_ZONE_IDX`).
pub const CONCERT_ZONE_POS_M: [[f32; 3]; PRESET_RADIATOR_ZONES] = [
    [-0.44, 1.24, 0.042], // Zone 0: Sub-bass / bass overstrung bridge end
    [-0.28, 1.08, 0.040], // Zone 1: Upper bass overstrung bridge
    [-0.20, 0.96, 0.032], // Zone 2: Low tenor long bridge
    [-0.06, 0.80, 0.032], // Zone 3: Tenor long bridge
    [-0.02, 0.74, 0.032], // Zone 4: Middle register long bridge
    [0.22, 0.48, 0.030],  // Zone 5: Upper middle long bridge
    [0.36, 0.34, 0.028],  // Zone 6: Lower treble long bridge
    [0.50, 0.22, 0.026],  // Zone 7: Top treble bridge / capo termination
    [0.00, -0.16, -0.07], // Zone 8: Wooden keybed front beam (KEYBED_ZONE_IDX)
];

/// Local orthotropic plate material & boundary properties at a quadrature point `(x, y)`.
#[derive(Clone, Copy, Debug)]
pub struct PlateSiteProps {
    /// Positive distance inside soundboard rim (`< 0` outside active planform) [m].
    pub rim_signed_dist_m: f64,
    /// Soundboard thickness `h(x, y)` [m].
    pub thickness_m: f64,
    /// Effective areal mass density (spruce + ribs + bridge maple, attenuated on collar) [kg/m^2].
    pub areal_mass_kg_m2: f64,
    /// Additional cross-grain bending stiffness from spruce ribs [N*m].
    pub rib_bending_nm: f64,
    /// Additional longitudinal bending stiffness along rock-maple bridge tracks [N*m].
    pub bridge_bending_nm: f64,
    /// Bridge orientation angle in `(x, y)` plane [rad].
    pub bridge_angle_rad: f64,
    /// Elastic rim boundary foundation stiffness `k_rim` [N/m^3] (zero well inside plate).
    pub rim_stiffness_n_m3: f64,
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    let u = t.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn planform_x_bounds(spec: &BodySpec, y: f64) -> Option<(f64, f64)> {
    if !(spec.y_min_m..=spec.y_max_m).contains(&y) {
        return None;
    }
    if spec.is_upright_liner {
        let u = ((y - spec.y_join_m) / (spec.y_max_m - spec.y_join_m).max(1.0e-3)).clamp(0.0, 1.0);
        let x_right =
            spec.x_right_front_m * (1.0 - smoothstep(u)) + spec.x_right_join_m * smoothstep(u);
        return Some((spec.x_left_m, x_right));
    }
    if y <= spec.y_join_m {
        let u = (y - spec.y_min_m) / (spec.y_join_m - spec.y_min_m);
        let s = smoothstep(u);
        let waist_pull = spec.waist_pull_m * (std::f64::consts::PI * u).sin().powi(2);
        let x_right = spec.x_right_front_m * (1.0 - s) + spec.x_right_join_m * s - waist_pull;
        Some((spec.x_left_m, x_right))
    } else {
        let dy = y - spec.y_join_m;
        let r_sq = spec.tail_r_m * spec.tail_r_m - dy * dy;
        if r_sq <= 0.0 {
            return None;
        }
        let dx = r_sq.sqrt();
        Some((spec.tail_cx_m - dx, spec.tail_cx_m + dx))
    }
}

/// Signed distance to the laminated rim perimeter for an arbitrary [`BodySpec`] (`> 0` inside) [m].
#[must_use]
pub fn rim_signed_distance_for(spec: &BodySpec, x: f64, y: f64) -> f64 {
    if y <= spec.y_min_m {
        let clamped_x = x.clamp(spec.x_left_m, spec.x_right_front_m);
        return -((x - clamped_x).hypot(spec.y_min_m - y));
    }
    if spec.is_upright_liner {
        if y >= spec.y_max_m {
            let clamped_x = x.clamp(spec.x_left_m, spec.x_right_join_m);
            return -((x - clamped_x).hypot(y - spec.y_max_m));
        }
        let Some((xl, xr)) = planform_x_bounds(spec, y) else {
            return -0.2;
        };
        let d_x = (x - xl).min(xr - x);
        let d_y = (y - spec.y_min_m).min(spec.y_max_m - y);
        return if d_x >= 0.0 {
            d_x.min(d_y)
        } else {
            d_x.max(-0.5)
        };
    }
    if y >= spec.y_max_m {
        return -((x - spec.tail_cx_m).hypot(y - spec.y_join_m) - spec.tail_r_m);
    }
    let Some((xl, xr)) = planform_x_bounds(spec, y) else {
        return -0.2;
    };
    let d_x = (x - xl).min(xr - x);
    let d_front = y - spec.y_min_m;
    let d_tail = if y > spec.y_join_m - 0.35 {
        spec.tail_r_m - (x - spec.tail_cx_m).hypot(y - spec.y_join_m)
    } else {
        1.0
    };
    if d_x >= 0.0 && d_front >= 0.0 && d_tail >= 0.0 {
        d_x.min(d_front).min(d_tail)
    } else {
        let out_x = if x < xl {
            xl - x
        } else if x > xr {
            x - xr
        } else {
            0.0
        };
        let out_tail = if y > spec.y_join_m {
            ((x - spec.tail_cx_m).hypot(y - spec.y_join_m) - spec.tail_r_m).max(0.0)
        } else {
            0.0
        };
        -(out_x.max(out_tail).max(1.0e-6))
    }
}

/// Signed distance to the concert-grand beech rim perimeter (`> 0` inside active soundboard) [m].
#[must_use]
pub fn rim_signed_distance_m(x: f64, y: f64) -> f64 {
    rim_signed_distance_for(&body_spec(BodyArchetype::ConcertGrand), x, y)
}

fn point_to_segment(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64) -> (f64, f64) {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    let t = if len_sq > 1.0e-12 {
        (((x - ax) * dx + (y - ay) * dy) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let px = ax + t * dx;
    let py = ay + t * dy;
    let dist = (x - px).hypot(y - py);
    (dist, dy.atan2(dx))
}

/// Returns `(tangent_unit, normal_unit, half_segment_m)` along the bass bridge (`0..=1`)
/// and long bridge (`2..=7`), keeping the two bridges strictly separated.
pub(crate) fn bridge_zone_tangents_normals(
    zone_pos_m: &[[f32; 3]; PRESET_RADIATOR_ZONES],
) -> [([f64; 2], [f64; 2], f64); NUM_BRIDGE_ZONES] {
    let mut out = [([1.0, 0.0], [0.0, 1.0], 0.04); NUM_BRIDGE_ZONES];
    for (z, slot) in out.iter_mut().enumerate() {
        let (a, b, half_seg) = if z <= 1 {
            (0_usize, 1_usize, 0.045)
        } else {
            let half = if z >= 6 { 0.025 } else { 0.040 };
            if z == 2 {
                (2, 3, half)
            } else if z == NUM_BRIDGE_ZONES - 1 {
                (NUM_BRIDGE_ZONES - 2, NUM_BRIDGE_ZONES - 1, half)
            } else {
                (z - 1, z + 1, half)
            }
        };
        let dx = f64::from(zone_pos_m[b][0]) - f64::from(zone_pos_m[a][0]);
        let dy = f64::from(zone_pos_m[b][1]) - f64::from(zone_pos_m[a][1]);
        let norm = dx.hypot(dy).max(1.0e-6);
        let t = [dx / norm, dy / norm];
        let n = [-t[1], t[0]];
        *slot = (t, n, half_seg);
    }
    out
}

/// Evaluates local soundboard thickness, ribs, bridges, and rim collar at `(x, y)` for `spec`.
#[must_use]
pub fn evaluate_plate_site_for(spec: &BodySpec, x: f64, y: f64) -> PlateSiteProps {
    const MAPLE_RHO: f64 = 705.0;
    const MAPLE_E: f64 = 12.6e9;

    let signed_dist = rim_signed_distance_for(spec, x, y);
    let edge_taper = (signed_dist.max(0.0) / 0.28).clamp(0.0, 1.0);
    let thickness_m = spec.thickness_min_m + spec.thickness_span_m * smoothstep(edge_taper);

    let cos_g = spec.grain_angle_rad.cos();
    let sin_g = spec.grain_angle_rad.sin();
    let u_grain = x * cos_g + y * sin_g;
    let rib_pitch = spec.rib_u_span_m / (spec.num_ribs as f64);
    let mut rib_weight = 0.0;
    for r in 0..spec.num_ribs {
        let u_center = spec.rib_u_min_m + (r as f64 + 0.5) * rib_pitch;
        let du = u_grain - u_center;
        rib_weight += (-0.5 * (du / 0.028).powi(2)).exp();
    }
    let feather = smoothstep((signed_dist / 0.14).clamp(0.0, 1.0));
    let rib_height = spec.rib_height_base_m + spec.rib_height_belly_m * feather;
    let rib_i = spec.rib_width_m * rib_height.powi(3) / 12.0;
    let rib_area_density =
        (spec.rib_width_m * rib_height * spec.spruce_rho / 0.056) * rib_weight * feather;
    let rib_bending_nm = (spec.spruce_e_l_pa * rib_i / 0.056) * rib_weight * feather;

    let (d_bass, ang_bass) = point_to_segment(
        x,
        y,
        spec.zone_pos_m[0][0] as f64,
        spec.zone_pos_m[0][1] as f64,
        spec.zone_pos_m[1][0] as f64,
        spec.zone_pos_m[1][1] as f64,
    );
    let (d_long, ang_long) = point_to_segment(
        x,
        y,
        spec.zone_pos_m[2][0] as f64,
        spec.zone_pos_m[2][1] as f64,
        spec.zone_pos_m[7][0] as f64,
        spec.zone_pos_m[7][1] as f64,
    );
    let w_bass = (-0.5 * (d_bass / 0.036).powi(2)).exp();
    let w_long = (-0.5 * (d_long / 0.032).powi(2)).exp();
    let bridge_mass = MAPLE_RHO * spec.bridge_mass_scale * (0.028 * w_bass + 0.024 * w_long);
    let bridge_i_eff = 0.032 * 0.026_f64.powi(3) / 12.0 / 0.055;
    let bridge_bending_nm =
        MAPLE_E * bridge_i_eff * spec.bridge_stiff_scale * (1.10 * w_bass + w_long);
    let bridge_angle_rad = if w_bass > w_long { ang_bass } else { ang_long };

    let inside_mask = smoothstep((signed_dist + 0.025) / 0.065);
    let rim_collar = 1.0 - smoothstep((signed_dist - 0.005) / 0.055);
    let rim_stiffness_n_m3 = spec.rim_stiffness_n_m3 * rim_collar;
    let interior_mass = spec.spruce_rho * thickness_m + rib_area_density + bridge_mass;
    let areal_mass_kg_m2 = interior_mass * (0.14 + 0.86 * inside_mask);

    PlateSiteProps {
        rim_signed_dist_m: signed_dist,
        thickness_m,
        areal_mass_kg_m2,
        rib_bending_nm,
        bridge_bending_nm,
        bridge_angle_rad,
        rim_stiffness_n_m3,
    }
}

/// Evaluates local concert-grand soundboard properties at `(x, y)`.
#[must_use]
pub fn evaluate_plate_site(x: f64, y: f64) -> PlateSiteProps {
    evaluate_plate_site_for(&body_spec(BodyArchetype::ConcertGrand), x, y)
}

/// Generates laminated rim & Zone 8 wooden keybed thud modes for `spec`.
#[must_use]
pub fn synthesize_rim_keybed_modes_for(spec: &BodySpec) -> Vec<SolvedBodyMode> {
    const COUNT: usize = 32;
    let mut modes = Vec::with_capacity(COUNT);
    let x_span = (spec.x_right_front_m - spec.x_left_m).max(0.5);
    for idx in 0..COUNT {
        let u = (idx as f64 + 0.35) / (COUNT as f64);
        let jitter = (det_unit_f64(0x91A0 + idx as u64) - 0.5) * 4.5;
        let freq_hz = (spec.rim_f_min + spec.rim_f_span * u.powf(1.22) + jitter)
            .clamp(35.0, spec.rim_f_min + spec.rim_f_span + 10.0);
        let t60_s = (spec.rim_t60_base - spec.rim_t60_drop * u
            + 0.015 * det_unit_f64(0x92B0 + idx as u64))
        .clamp(0.042, 0.215);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let kx_n = ((idx % 4) + 1) as f64;
        let ky_n = ((idx / 4) + 1) as f64;
        let phi_center = (kx_n * 0.5 * std::f64::consts::PI).sin().abs().max(0.25);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        let keybed_gain = spec.rim_keybed_gain
            * phi_center
            * (1.0 - 0.42 * u)
            * (0.8 + 0.4 * det_unit_f64(0x93C0 + idx as u64));
        zone_drive[KEYBED_ZONE_IDX] = if idx % 2 == 0 {
            keybed_gain
        } else {
            -keybed_gain
        };

        for (z, pos) in spec.zone_pos_m.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let x_norm = ((pos[0] as f64) - spec.x_left_m) / x_span;
            let rim_prox = 0.5 + 0.5 * (kx_n * std::f64::consts::PI * x_norm).cos();
            let front_prox = (-1.15 * (pos[1] as f64)).exp();
            let sign = if ((idx + z) % 2) == 0 { 1.0 } else { -1.0 };
            zone_drive[z] = sign * 0.0022 * rim_prox * front_prox * (1.0 - 0.3 * u);
        }

        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg: 1.85 + 2.4 * (1.0 - u) + 0.15 * ky_n,
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 0,
            source: ModeSource::RimKeybed,
            plate_vol_norm: 0.0,
            plate_dx_norm: 0.0,
            plate_dy_norm: 0.0,
            has_plate_moments: false,
        });
    }
    modes
}

/// Generates concert-grand laminated beech rim & Zone 8 wooden keybed thud modes.
#[must_use]
pub fn synthesize_rim_keybed_modes() -> Vec<SolvedBodyMode> {
    synthesize_rim_keybed_modes_for(&body_spec(BodyArchetype::ConcertGrand))
}

/// Generates weakly coupled cast-iron harp frame structural modes for `spec`.
#[must_use]
pub fn synthesize_cast_iron_frame_modes_for(spec: &BodySpec) -> Vec<SolvedBodyMode> {
    const COUNT: usize = 40;
    let mut modes = Vec::with_capacity(COUNT);
    for idx in 0..COUNT {
        let u = (idx as f64 + 0.5) / (COUNT as f64);
        let jitter = (det_unit_f64(0xA110 + idx as u64) - 0.5) * 18.0;
        let freq_hz = (spec.frame_f_min * (spec.frame_f_max / spec.frame_f_min).powf(u) + jitter)
            .clamp(spec.frame_f_min - 5.0, spec.frame_f_max + 50.0);
        let t60_s = (body_plate_t60_s(spec, freq_hz) * 0.92).clamp(0.045, 0.36);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for (z, pos) in spec.zone_pos_m.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let phase = (idx as f64 * 0.71 + (pos[0] as f64) * 3.1 + (pos[1] as f64) * 2.4).sin();
            let strut_w = 0.55 + 0.45 * ((pos[0] as f64 * 2.7).cos()).abs();
            zone_drive[z] = spec.frame_gain * strut_w * phase * (1.0 - 0.25 * u);
        }
        if freq_hz < 360.0 {
            zone_drive[KEYBED_ZONE_IDX] = 0.00045 * (det_unit_f64(0xA220 + idx as u64) - 0.5);
        }

        modes.push(SolvedBodyMode {
            freq_hz,
            zeta,
            modal_mass_kg: 18.0 + 16.0 * (1.0 - 0.6 * u),
            zone_drive,
            rad_coeffs: [0.0; PRESET_RAD_COEFFS],
            cavity_flag: 0,
            source: ModeSource::CastIronFrame,
            plate_vol_norm: 0.0,
            plate_dx_norm: 0.0,
            plate_dy_norm: 0.0,
            has_plate_moments: false,
        });
    }
    modes
}

/// Generates concert-grand cast-iron harp frame structural modes.
#[must_use]
pub fn synthesize_cast_iron_frame_modes() -> Vec<SolvedBodyMode> {
    synthesize_cast_iron_frame_modes_for(&body_spec(BodyArchetype::ConcertGrand))
}
