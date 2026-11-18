//! Pure-Rust physical archetypes and parameterization for multi-body `.pbdy` presets:
//! [`BodyArchetype::ConcertGrand`], [`BodyArchetype::UprightStudio`], and
//! [`BodyArchetype::VintageSaloon`].

use crate::pole_cancel::CALIBRATED_DIRECT_ZONE_GAIN;
use crate::ribs_rim::CONCERT_ZONE_POS_M;
use piano_bridge::{AdmittanceSsm, DEFAULT_Y_INF};
use piano_spec::{
    PresetReducedPole, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES, PRESET_REDUCED_POLES,
};
use std::f64::consts::PI;

/// Supported physical piano body archetypes baked into `.pbdy` assets.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BodyArchetype {
    /// 9-foot concert grand piano (Sitka spruce double-crowned soundboard + bentside rim).
    ConcertGrand = 0,
    /// 131 cm upright studio piano (compact rectangular-beveled soundboard + backposts).
    UprightStudio = 1,
    /// Aged late-19th-century parlor / saloon piano (thin compliant board + lively mid bloom).
    VintageSaloon = 2,
}

impl BodyArchetype {
    /// All supported piano body archetypes in canonical discriminant order.
    pub const ALL: [Self; 3] = [Self::ConcertGrand, Self::UprightStudio, Self::VintageSaloon];

    /// Parses a numeric discriminant (`0..=2`).
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ConcertGrand),
            1 => Some(Self::UprightStudio),
            2 => Some(Self::VintageSaloon),
            _ => None,
        }
    }

    /// Parses CLI/slug identifiers (`"concert_grand"`, `"upright_studio"`, `"vintage_saloon"`).
    #[must_use]
    pub fn from_slug(s: &str) -> Option<Self> {
        match s {
            "concert_grand" | "concert-grand" | "grand" => Some(Self::ConcertGrand),
            "upright_studio" | "upright-studio" | "upright" => Some(Self::UprightStudio),
            "vintage_saloon" | "vintage-saloon" | "saloon" => Some(Self::VintageSaloon),
            _ => None,
        }
    }

    /// Canonical lowercase snake_case identifier.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ConcertGrand => "concert_grand",
            Self::UprightStudio => "upright_studio",
            Self::VintageSaloon => "vintage_saloon",
        }
    }

    /// Default `.pbdy` asset filename under `crates/piano-radiator/assets/`.
    #[must_use]
    pub const fn asset_filename(self) -> &'static str {
        match self {
            Self::ConcertGrand => "concert_grand_v1.pbdy",
            Self::UprightStudio => "upright_studio_v1.pbdy",
            Self::VintageSaloon => "vintage_saloon_v1.pbdy",
        }
    }
}

/// Physical soundboard geometry, material, bracing, cavity, and Stage 3 reduced-pole spec.
#[derive(Clone, Debug)]
pub struct BodySpec {
    pub archetype: BodyArchetype,
    pub domain_lx_m: f64,
    pub domain_ly_m: f64,
    pub x_min_m: f64,
    pub y_min_m: f64,
    pub y_join_m: f64,
    pub y_max_m: f64,
    pub x_left_m: f64,
    pub x_right_front_m: f64,
    pub x_right_join_m: f64,
    pub waist_pull_m: f64,
    pub tail_cx_m: f64,
    pub tail_r_m: f64,
    pub is_upright_liner: bool,

    pub spruce_e_l_pa: f64,
    pub spruce_e_r_pa: f64,
    pub spruce_g_lr_pa: f64,
    pub spruce_nu_lr: f64,
    pub spruce_rho: f64,
    pub thickness_min_m: f64,
    pub thickness_span_m: f64,
    pub crown_radius_m: f64,
    pub crown_prestress: f64,
    pub grain_angle_rad: f64,
    pub num_ribs: usize,
    pub rib_u_min_m: f64,
    pub rib_u_span_m: f64,
    pub rib_height_base_m: f64,
    pub rib_height_belly_m: f64,
    pub rib_width_m: f64,
    pub bridge_mass_scale: f64,
    pub bridge_stiff_scale: f64,
    pub rim_stiffness_n_m3: f64,

    pub t60_floor_s: f64,
    pub t60_span_s: f64,
    pub t60_corner_hz: f64,
    pub t60_exp: f64,
    pub ritz_gain: f64,
    pub ritz_max_hz: f64,
    pub stat_count: usize,
    pub stat_gain: f64,
    pub stat_bass_clamp: f64,
    pub stat_treble_focus: f64,

    pub zone_pos_m: [[f32; 3]; PRESET_RADIATOR_ZONES],
    pub direct_zone_gain: [f32; PRESET_RADIATOR_ZONES],

    pub rim_f_min: f64,
    pub rim_f_span: f64,
    pub rim_t60_base: f64,
    pub rim_t60_drop: f64,
    pub rim_keybed_gain: f64,

    pub frame_f_min: f64,
    pub frame_f_max: f64,
    pub frame_gain: f64,

    pub cavity_helmholtz_hz: [f64; 3],
    pub cavity_lower_dims_m: [f64; 3],
    pub cavity_upper_dims_m: [f64; 3],
    pub cavity_lower_couple: f64,
    pub cavity_upper_couple: f64,
    pub cavity_t60_base_s: f64,

    pub plate_coincidence_hz: f64,
    pub lid_coupling_scale: f64,
    pub multipole_y_center_m: f64,

    pub y_inf: [f32; PRESET_BRIDGE_ZONES],
    pub reduced_poles: [PresetReducedPole; PRESET_REDUCED_POLES],
}

/// Evaluates the body-specific Sitka spruce modal decay `T60(f)` [s].
#[must_use]
#[inline]
pub fn body_plate_t60_s(spec: &BodySpec, freq_hz: f64) -> f64 {
    let f = freq_hz.clamp(20.0, 22_000.0);
    let t60 =
        spec.t60_floor_s + spec.t60_span_s / (1.0 + (f / spec.t60_corner_hz).powf(spec.t60_exp));
    t60.clamp(0.022, 0.410)
}

/// Builds the full physical [`BodySpec`] for the requested [`BodyArchetype`].
#[must_use]
pub fn body_spec(archetype: BodyArchetype) -> BodySpec {
    match archetype {
        BodyArchetype::ConcertGrand => concert_grand_spec(),
        BodyArchetype::UprightStudio => upright_studio_spec(),
        BodyArchetype::VintageSaloon => vintage_saloon_spec(),
    }
}

fn concert_grand_spec() -> BodySpec {
    let ssm = AdmittanceSsm::new(48_000.0);
    let poles = ssm.poles();
    let reduced_poles = std::array::from_fn(|i| PresetReducedPole {
        freq_hz: poles[i].freq_hz,
        zeta: poles[i].zeta,
        modal_mass_kg: poles[i].modal_mass_kg,
        zone_gains: poles[i].zone_gains,
    });
    BodySpec {
        archetype: BodyArchetype::ConcertGrand,
        domain_lx_m: 1.52,
        domain_ly_m: 1.98,
        x_min_m: -0.76,
        y_min_m: 0.06,
        y_join_m: 1.40,
        y_max_m: 1.88,
        x_left_m: -0.68,
        x_right_front_m: 0.70,
        x_right_join_m: 0.28,
        waist_pull_m: 0.06,
        tail_cx_m: -0.20,
        tail_r_m: 0.48,
        is_upright_liner: false,

        spruce_e_l_pa: 11.5e9,
        spruce_e_r_pa: 0.95e9,
        spruce_g_lr_pa: 0.75e9,
        spruce_nu_lr: 0.37,
        spruce_rho: 435.0,
        thickness_min_m: 0.0075,
        thickness_span_m: 0.0015,
        crown_radius_m: 24.0,
        crown_prestress: 2.8e4,
        grain_angle_rad: 32.0 * PI / 180.0,
        num_ribs: 14,
        rib_u_min_m: -0.30,
        rib_u_span_m: 1.80,
        rib_height_base_m: 0.017,
        rib_height_belly_m: 0.007,
        rib_width_m: 0.022,
        bridge_mass_scale: 1.0,
        bridge_stiff_scale: 1.0,
        rim_stiffness_n_m3: 1.15e8,

        t60_floor_s: 0.028,
        t60_span_s: 0.365,
        t60_corner_hz: 460.0,
        t60_exp: 1.02,
        ritz_gain: 0.0145,
        ritz_max_hz: 2_800.0,
        stat_count: 1_060,
        stat_gain: 0.0098,
        stat_bass_clamp: 0.50,
        stat_treble_focus: 1.85,

        zone_pos_m: CONCERT_ZONE_POS_M,
        direct_zone_gain: CALIBRATED_DIRECT_ZONE_GAIN,

        rim_f_min: 42.0,
        rim_f_span: 334.0,
        rim_t60_base: 0.175,
        rim_t60_drop: 0.115,
        rim_keybed_gain: 0.016,

        frame_f_min: 185.0,
        frame_f_max: 2_750.0,
        frame_gain: 0.0014,

        cavity_helmholtz_hz: [67.5, 84.0, 104.5],
        cavity_lower_dims_m: [1.38, 1.72, 0.42],
        cavity_upper_dims_m: [1.34, 1.68, 0.58],
        cavity_lower_couple: 0.0058,
        cavity_upper_couple: 0.0046,
        cavity_t60_base_s: 0.35,

        plate_coincidence_hz: 1_550.0,
        lid_coupling_scale: 1.0,
        multipole_y_center_m: 0.72,

        y_inf: DEFAULT_Y_INF,
        reduced_poles,
    }
}

fn upright_studio_spec() -> BodySpec {
    const UPRIGHT_ZONES: [[f32; 3]; PRESET_RADIATOR_ZONES] = [
        [-0.40, 0.72, 0.068],
        [-0.26, 0.64, 0.064],
        [-0.18, 0.56, 0.052],
        [-0.05, 0.48, 0.050],
        [0.04, 0.42, 0.050],
        [0.20, 0.34, 0.046],
        [0.33, 0.26, 0.042],
        [0.46, 0.18, 0.038],
        [0.00, -0.14, -0.05],
    ];
    const UPRIGHT_DIRECT: [f32; PRESET_RADIATOR_ZONES] = [
        0.088, 0.098, 0.228, 0.195, 0.242, 0.232, 0.132, 0.114, 0.052,
    ];
    const UPRIGHT_Y_INF: [f32; PRESET_BRIDGE_ZONES] = [
        0.000_24, 0.000_23, 0.000_11, 0.000_082, 0.000_19, 0.000_14, 0.000_19, 0.000_22,
    ];
    const FREQS: [f32; PRESET_REDUCED_POLES] = [
        74.0, 118.0, 184.0, 268.0, 355.0, 495.0, 650.0, 875.0, 1110.0, 1360.0,
    ];
    const ZETAS: [f32; PRESET_REDUCED_POLES] = [
        0.062, 0.048, 0.034, 0.024, 0.022, 0.017, 0.026, 0.023, 0.022, 0.021,
    ];
    const MASSES: [f32; PRESET_REDUCED_POLES] =
        [3.10, 2.20, 1.15, 0.64, 0.45, 0.36, 0.82, 1.95, 2.55, 3.05];

    let base_ssm = AdmittanceSsm::new(48_000.0);
    let base_poles = base_ssm.poles();
    let reduced_poles = std::array::from_fn(|m| {
        let mut gains = base_poles[m].zone_gains;
        for (z, g) in gains.iter_mut().enumerate() {
            let compact_tilt = 0.94 + 0.14 * ((z as f32) / 7.0);
            *g *= compact_tilt;
        }
        PresetReducedPole {
            freq_hz: FREQS[m],
            zeta: ZETAS[m],
            modal_mass_kg: MASSES[m],
            zone_gains: gains,
        }
    });

    BodySpec {
        archetype: BodyArchetype::UprightStudio,
        domain_lx_m: 1.42,
        domain_ly_m: 1.18,
        x_min_m: -0.71,
        y_min_m: 0.05,
        y_join_m: 0.72,
        y_max_m: 1.13,
        x_left_m: -0.64,
        x_right_front_m: 0.64,
        x_right_join_m: 0.56,
        waist_pull_m: 0.0,
        tail_cx_m: -0.04,
        tail_r_m: 0.60,
        is_upright_liner: true,

        spruce_e_l_pa: 13.0e9,
        spruce_e_r_pa: 1.18e9,
        spruce_g_lr_pa: 0.90e9,
        spruce_nu_lr: 0.36,
        spruce_rho: 445.0,
        thickness_min_m: 0.0084,
        thickness_span_m: 0.0018,
        crown_radius_m: 12.0,
        crown_prestress: 9.5e4,
        grain_angle_rad: 36.0 * PI / 180.0,
        num_ribs: 13,
        rib_u_min_m: -0.34,
        rib_u_span_m: 1.26,
        rib_height_base_m: 0.022,
        rib_height_belly_m: 0.008,
        rib_width_m: 0.024,
        bridge_mass_scale: 1.08,
        bridge_stiff_scale: 1.22,
        rim_stiffness_n_m3: 1.85e8,

        t60_floor_s: 0.024,
        t60_span_s: 0.295,
        t60_corner_hz: 520.0,
        t60_exp: 1.05,
        ritz_gain: 0.0138,
        ritz_max_hz: 3_100.0,
        stat_count: 1_060,
        stat_gain: 0.0094,
        stat_bass_clamp: 0.44,
        stat_treble_focus: 1.98,

        zone_pos_m: UPRIGHT_ZONES,
        direct_zone_gain: UPRIGHT_DIRECT,

        rim_f_min: 55.0,
        rim_f_span: 385.0,
        rim_t60_base: 0.145,
        rim_t60_drop: 0.095,
        rim_keybed_gain: 0.018,

        frame_f_min: 220.0,
        frame_f_max: 3_050.0,
        frame_gain: 0.0016,

        cavity_helmholtz_hz: [88.0, 118.0, 146.0],
        cavity_lower_dims_m: [1.26, 1.06, 0.30],
        cavity_upper_dims_m: [1.22, 1.02, 0.22],
        cavity_lower_couple: 0.0062,
        cavity_upper_couple: 0.0038,
        cavity_t60_base_s: 0.29,

        plate_coincidence_hz: 1_380.0,
        lid_coupling_scale: 0.58,
        multipole_y_center_m: 0.46,

        y_inf: UPRIGHT_Y_INF,
        reduced_poles,
    }
}

fn vintage_saloon_spec() -> BodySpec {
    const SALOON_ZONES: [[f32; 3]; PRESET_RADIATOR_ZONES] = [
        [-0.38, 0.86, 0.048],
        [-0.24, 0.76, 0.045],
        [-0.16, 0.66, 0.038],
        [-0.04, 0.56, 0.038],
        [0.04, 0.48, 0.036],
        [0.19, 0.38, 0.034],
        [0.31, 0.28, 0.032],
        [0.42, 0.19, 0.030],
        [0.00, -0.15, -0.06],
    ];
    const SALOON_DIRECT: [f32; PRESET_RADIATOR_ZONES] = [
        0.096, 0.112, 0.236, 0.205, 0.246, 0.228, 0.108, 0.088, 0.048,
    ];
    const SALOON_Y_INF: [f32; PRESET_BRIDGE_ZONES] = [
        0.000_42, 0.000_39, 0.000_18, 0.000_135, 0.000_32, 0.000_22, 0.000_28, 0.000_31,
    ];
    const FREQS: [f32; PRESET_REDUCED_POLES] = [
        48.0, 78.0, 124.0, 186.0, 254.0, 358.0, 472.0, 648.0, 825.0, 1010.0,
    ];
    const ZETAS: [f32; PRESET_REDUCED_POLES] = [
        0.050, 0.038, 0.027, 0.0175, 0.0165, 0.0125, 0.020, 0.019, 0.021, 0.022,
    ];
    const MASSES: [f32; PRESET_REDUCED_POLES] =
        [2.85, 1.98, 1.02, 0.56, 0.39, 0.31, 0.72, 1.75, 2.30, 2.75];

    let base_ssm = AdmittanceSsm::new(48_000.0);
    let base_poles = base_ssm.poles();
    let reduced_poles = std::array::from_fn(|m| {
        let mut gains = base_poles[m].zone_gains;
        for (z, g) in gains.iter_mut().enumerate() {
            let parlor_bloom = if (2..=5).contains(&z) { 1.12 } else { 0.94 };
            *g *= parlor_bloom;
        }
        PresetReducedPole {
            freq_hz: FREQS[m],
            zeta: ZETAS[m],
            modal_mass_kg: MASSES[m],
            zone_gains: gains,
        }
    });

    BodySpec {
        archetype: BodyArchetype::VintageSaloon,
        domain_lx_m: 1.36,
        domain_ly_m: 1.42,
        x_min_m: -0.68,
        y_min_m: 0.05,
        y_join_m: 0.92,
        y_max_m: 1.36,
        x_left_m: -0.60,
        x_right_front_m: 0.56,
        x_right_join_m: 0.28,
        waist_pull_m: 0.04,
        tail_cx_m: -0.16,
        tail_r_m: 0.44,
        is_upright_liner: false,

        spruce_e_l_pa: 9.4e9,
        spruce_e_r_pa: 0.72e9,
        spruce_g_lr_pa: 0.60e9,
        spruce_nu_lr: 0.38,
        spruce_rho: 418.0,
        thickness_min_m: 0.0064,
        thickness_span_m: 0.0014,
        crown_radius_m: 36.0,
        crown_prestress: 1.4e4,
        grain_angle_rad: 28.0 * PI / 180.0,
        num_ribs: 11,
        rib_u_min_m: -0.28,
        rib_u_span_m: 1.32,
        rib_height_base_m: 0.014,
        rib_height_belly_m: 0.005,
        rib_width_m: 0.020,
        bridge_mass_scale: 0.88,
        bridge_stiff_scale: 0.84,
        rim_stiffness_n_m3: 0.82e8,

        t60_floor_s: 0.019,
        t60_span_s: 0.385,
        t60_corner_hz: 340.0,
        t60_exp: 1.12,
        ritz_gain: 0.0156,
        ritz_max_hz: 2_600.0,
        stat_count: 1_060,
        stat_gain: 0.0092,
        stat_bass_clamp: 0.48,
        stat_treble_focus: 1.55,

        zone_pos_m: SALOON_ZONES,
        direct_zone_gain: SALOON_DIRECT,

        rim_f_min: 38.0,
        rim_f_span: 312.0,
        rim_t60_base: 0.195,
        rim_t60_drop: 0.120,
        rim_keybed_gain: 0.017,

        frame_f_min: 165.0,
        frame_f_max: 2_450.0,
        frame_gain: 0.0017,

        cavity_helmholtz_hz: [76.0, 98.0, 128.0],
        cavity_lower_dims_m: [1.18, 1.16, 0.34],
        cavity_upper_dims_m: [1.12, 1.08, 0.42],
        cavity_lower_couple: 0.0064,
        cavity_upper_couple: 0.0052,
        cavity_t60_base_s: 0.36,

        plate_coincidence_hz: 1_820.0,
        lid_coupling_scale: 0.76,
        multipole_y_center_m: 0.54,

        y_inf: SALOON_Y_INF,
        reduced_poles,
    }
}
