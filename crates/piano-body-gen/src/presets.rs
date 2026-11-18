//! Pure-Rust physical archetypes and parameterization for multi-body `.pbdy` presets:
//! [`BodyArchetype::ConcertGrand`], [`BodyArchetype::UprightStudio`], and
//! [`BodyArchetype::VintageSaloon`].

use crate::preset_bodies::{concert_grand_spec, upright_studio_spec, vintage_saloon_spec};
use piano_spec::{
    PresetReducedPole, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES, PRESET_REDUCED_POLES,
};

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
