//! Concert-grand geometry, spruce ribs, bridge heterogeneity, laminated beech
//! rim boundary impedance, Zone 8 wooden keybed coupling, and cast-iron harp frame.

use crate::types::{det_unit_f64, sitka_spruce_t60_s, t60_to_zeta, ModeSource, SolvedBodyMode};
use piano_bridge::{KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES};
use piano_spec::{PRESET_RADIATOR_ZONES, PRESET_RAD_COEFFS};

/// Rectangular Ritz integration domain width [m] (`x in [-0.76, 0.76]`).
pub const SOUNDBOARD_LX_M: f64 = 1.52;
/// Rectangular Ritz integration domain length [m] (`y in [0.0, 1.98]`).
pub const SOUNDBOARD_LY_M: f64 = 1.98;
/// Left boundary of Ritz domain [m].
pub const SOUNDBOARD_X_MIN_M: f64 = -0.5 * SOUNDBOARD_LX_M;
/// Number of transverse Sitka spruce ribs under the soundboard.
pub const NUM_SPRUCE_RIBS: usize = 14;
/// Spruce grain angle relative to keyboard x-axis [rad] (~32 deg).
pub const SPRUCE_GRAIN_ANGLE_RAD: f64 = 32.0 * std::f64::consts::PI / 180.0;

/// 3D physical positions [m] of the 8 bridge zones + Zone 8 keybed (`KEYBED_ZONE_IDX`).
///
/// Bass zones 0-1 lie on the left (`x < 0`) and treble zones 6-7 on the right (`x > 0`),
/// aligned with `zone_stereo_pan` and Stage 4 binaural HRTF lateralization.
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
    /// Positive distance inside grand soundboard rim (`< 0` outside active planform) [m].
    pub rim_signed_dist_m: f64,
    /// Soundboard thickness `h(x, y)` tapering 7.5..9.0 mm [m].
    pub thickness_m: f64,
    /// Effective areal mass density (spruce + ribs + bridge maple, attenuated on outer collar) [kg/m^2].
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

/// Right bentside contour `x_right(y)` and left spine contour `x_left(y)` for `y in [0.06, 1.88]`.
fn planform_x_bounds(y: f64) -> Option<(f64, f64)> {
    const Y_MIN: f64 = 0.06;
    const Y_JOIN: f64 = 1.40;
    const Y_MAX: f64 = 1.88;
    if !(Y_MIN..=Y_MAX).contains(&y) {
        return None;
    }
    if y <= Y_JOIN {
        let u = (y - Y_MIN) / (Y_JOIN - Y_MIN);
        let s = smoothstep(u);
        let waist_pull = 0.06 * (std::f64::consts::PI * u).sin().powi(2);
        let x_right = 0.70 * (1.0 - s) + 0.28 * s - waist_pull;
        Some((-0.68, x_right))
    } else {
        // Semicircular tail centered at (-0.20, 1.40) with radius 0.48 joining [-0.68, +0.28].
        let dy = y - Y_JOIN;
        let r_sq = 0.48 * 0.48 - dy * dy;
        if r_sq <= 0.0 {
            return None;
        }
        let dx = r_sq.sqrt();
        Some((-0.20 - dx, -0.20 + dx))
    }
}

/// Signed distance to the laminated beech rim perimeter (`> 0` inside active soundboard) [m].
pub fn rim_signed_distance_m(x: f64, y: f64) -> f64 {
    const Y_MIN: f64 = 0.06;
    const Y_JOIN: f64 = 1.40;
    const Y_MAX: f64 = 1.88;
    if y <= Y_MIN {
        let clamped_x = x.clamp(-0.68, 0.70);
        return -((x - clamped_x).hypot(Y_MIN - y));
    }
    if y >= Y_MAX {
        return -((x + 0.20).hypot(y - 1.40) - 0.48);
    }
    let Some((xl, xr)) = planform_x_bounds(y) else {
        return -0.2;
    };
    let d_x = (x - xl).min(xr - x);
    let d_front = y - Y_MIN;
    let d_tail = if y > Y_JOIN - 0.35 {
        0.48 - (x + 0.20).hypot(y - Y_JOIN)
    } else {
        1.0
    };
    if d_x >= 0.0 && d_front >= 0.0 && d_tail >= 0.0 {
        d_x.min(d_front).min(d_tail)
    } else {
        // Negative distance outside contour.
        let out_x = if x < xl {
            xl - x
        } else if x > xr {
            x - xr
        } else {
            0.0
        };
        let out_tail = if y > Y_JOIN {
            ((x + 0.20).hypot(y - Y_JOIN) - 0.48).max(0.0)
        } else {
            0.0
        };
        -(out_x.max(out_tail).max(1.0e-6))
    }
}

/// Minimum distance from `(x, y)` to a line segment `a -> b` and its tangent unit vector.
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

/// Evaluates local soundboard thickness, 14 transverse ribs, bridges, and beech rim collar at `(x, y)`.
pub fn evaluate_plate_site(x: f64, y: f64) -> PlateSiteProps {
    const SPRUCE_RHO: f64 = 435.0;
    const MAPLE_RHO: f64 = 705.0;
    const SPRUCE_E_L: f64 = 11.5e9;
    const MAPLE_E: f64 = 12.6e9;

    let signed_dist = rim_signed_distance_m(x, y);
    let edge_taper = (signed_dist.max(0.0) / 0.28).clamp(0.0, 1.0);
    // Double-crowned spruce taper: 7.5 mm near rim up to 9.0 mm across belly center.
    let thickness_m = 0.0075 + 0.0015 * smoothstep(edge_taper);

    // 14 transverse spruce ribs oriented perpendicular to grain (`u_g` spacing across [-0.32, 1.48]).
    let cos_g = SPRUCE_GRAIN_ANGLE_RAD.cos();
    let sin_g = SPRUCE_GRAIN_ANGLE_RAD.sin();
    let u_grain = x * cos_g + y * sin_g;
    let rib_pitch = 1.80 / (NUM_SPRUCE_RIBS as f64);
    let mut rib_weight = 0.0;
    for r in 0..NUM_SPRUCE_RIBS {
        let u_center = -0.30 + (r as f64 + 0.5) * rib_pitch;
        let du = u_grain - u_center;
        // Effective Gaussian rib footprint width (~24 mm + bending apron).
        rib_weight += (-0.5 * (du / 0.028).powi(2)).exp();
    }
    let feather = smoothstep((signed_dist / 0.14).clamp(0.0, 1.0));
    let rib_height = 0.017 + 0.007 * feather;
    let rib_width = 0.022;
    let rib_i = rib_width * rib_height.powi(3) / 12.0;
    let rib_area_density = (rib_width * rib_height * SPRUCE_RHO / 0.056) * rib_weight * feather;
    let rib_bending_nm = (SPRUCE_E_L * rib_i / 0.056) * rib_weight * feather;

    // Bass bridge (zones 0..1) and rock-maple long bridge (zones 2..7).
    let (d_bass, ang_bass) = point_to_segment(
        x,
        y,
        CONCERT_ZONE_POS_M[0][0] as f64,
        CONCERT_ZONE_POS_M[0][1] as f64,
        CONCERT_ZONE_POS_M[1][0] as f64,
        CONCERT_ZONE_POS_M[1][1] as f64,
    );
    let (d_long, ang_long) = point_to_segment(
        x,
        y,
        CONCERT_ZONE_POS_M[2][0] as f64,
        CONCERT_ZONE_POS_M[2][1] as f64,
        CONCERT_ZONE_POS_M[7][0] as f64,
        CONCERT_ZONE_POS_M[7][1] as f64,
    );
    let w_bass = (-0.5 * (d_bass / 0.036).powi(2)).exp();
    let w_long = (-0.5 * (d_long / 0.032).powi(2)).exp();
    let bridge_mass = MAPLE_RHO * (0.028 * w_bass + 0.024 * w_long);
    let bridge_i_eff = 0.032 * 0.026_f64.powi(3) / 12.0 / 0.055;
    let bridge_bending_nm = MAPLE_E * bridge_i_eff * (1.10 * w_bass + w_long);
    let bridge_angle_rad = if w_bass > w_long { ang_bass } else { ang_long };

    // Beech rim collar penalty: smooth ramp across +-4 cm around rim boundary.
    // Peak k_rim = 1.15e8 N/m^3 with reduced exterior inertia keeps condition number well-bounded.
    let inside_mask = smoothstep((signed_dist + 0.025) / 0.065);
    let rim_collar = 1.0 - smoothstep((signed_dist - 0.005) / 0.055);
    let rim_stiffness_n_m3 = 1.15e8 * rim_collar;
    let interior_mass = SPRUCE_RHO * thickness_m + rib_area_density + bridge_mass;
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

/// Generates laminated beech rim & Zone 8 (`KEYBED_ZONE_IDX = 8`) wooden keybed thud modes (`40..380 Hz`).
///
/// All modes use pure damped 2nd-order modal physics (`T60 = 0.05..0.18 s`, zero separate
/// envelope decay). `zone_drive[8]` couples directly to physical keybed bottom-out force.
pub fn synthesize_rim_keybed_modes() -> Vec<SolvedBodyMode> {
    const COUNT: usize = 32;
    let mut modes = Vec::with_capacity(COUNT);
    for idx in 0..COUNT {
        let u = (idx as f64 + 0.35) / (COUNT as f64);
        // Cluster lower modes firmly in the wooden keybed thud band (40..380 Hz).
        let jitter = (det_unit_f64(0x91A0 + idx as u64) - 0.5) * 4.5;
        let freq_hz = (42.0 + 334.0 * u.powf(1.22) + jitter).clamp(40.0, 380.0);
        // Keybed / rim laminated beech damping: natural fast wooden thud decay (55..175 ms).
        let t60_s =
            (0.175 - 0.115 * u + 0.015 * det_unit_f64(0x92B0 + idx as u64)).clamp(0.052, 0.185);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let kx_n = ((idx % 4) + 1) as f64;
        let ky_n = ((idx / 4) + 1) as f64;
        let phi_center = (kx_n * 0.5 * std::f64::consts::PI).sin().abs().max(0.25);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        // Direct structural coupling into Zone 8 keybed-to-rim cheek blocks.
        let keybed_gain =
            0.016 * phi_center * (1.0 - 0.42 * u) * (0.8 + 0.4 * det_unit_f64(0x93C0 + idx as u64));
        zone_drive[KEYBED_ZONE_IDX] = if idx % 2 == 0 {
            keybed_gain
        } else {
            -keybed_gain
        };

        // Sympathetic rim cheek-block transmission into bridge zones (stronger near bass/treble rim).
        for (z, pos) in CONCERT_ZONE_POS_M.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            let x_norm = ((pos[0] as f64) + 0.68) / 1.38;
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
        });
    }
    modes
}

/// Generates weakly coupled cast-iron harp frame structural modes (`180..2,800 Hz`).
pub fn synthesize_cast_iron_frame_modes() -> Vec<SolvedBodyMode> {
    const COUNT: usize = 40;
    let mut modes = Vec::with_capacity(COUNT);
    for idx in 0..COUNT {
        let u = (idx as f64 + 0.5) / (COUNT as f64);
        let jitter = (det_unit_f64(0xA110 + idx as u64) - 0.5) * 18.0;
        let freq_hz = (185.0 * (2_750.0_f64 / 185.0).powf(u) + jitter).clamp(180.0, 2_800.0);
        // Cast-iron frame damped by felt/leather shims & rim bolts, well within spruce/wood envelope.
        let t60_s = (sitka_spruce_t60_s(freq_hz) * 0.92).clamp(0.055, 0.36);
        let zeta = t60_to_zeta(freq_hz, t60_s);

        let mut zone_drive = [0.0_f64; PRESET_RADIATOR_ZONES];
        for (z, pos) in CONCERT_ZONE_POS_M.iter().enumerate().take(NUM_BRIDGE_ZONES) {
            // Frame agraffes/struts sit closest to Zones 0, 2, 5, 7 along hitch & capo bars.
            let phase = (idx as f64 * 0.71 + (pos[0] as f64) * 3.1 + (pos[1] as f64) * 2.4).sin();
            let strut_w = 0.55 + 0.45 * ((pos[0] as f64 * 2.7).cos()).abs();
            // Weakly coupled relative to spruce soundboard modes (O(0.0005..0.0022)).
            zone_drive[z] = 0.0014 * strut_w * phase * (1.0 - 0.25 * u);
        }
        // Very weak low-frequency mechanical path from keybed cheek bolts into lower frame struts.
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
        });
    }
    modes
}
