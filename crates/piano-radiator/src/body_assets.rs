//! Multi-body acoustic preset registry and embedded `.pbdy` asset lookup.

use crate::CONCERT_GRAND_PRESET_BYTES;

/// Supported physical piano body archetypes (`0..=2`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum BodyArchetype {
    #[default]
    ConcertGrand = 0,
    UprightStudio = 1,
    VintageSaloon = 2,
}

impl BodyArchetype {
    /// Decodes a raw byte discriminant into a [`BodyArchetype`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ConcertGrand),
            1 => Some(Self::UprightStudio),
            2 => Some(Self::VintageSaloon),
            _ => None,
        }
    }
}

/// Returns the canonical `.pbdy` asset filename for `id`.
#[must_use]
pub const fn body_asset_filename(id: BodyArchetype) -> &'static str {
    match id {
        BodyArchetype::ConcertGrand => "concert_grand_v1.pbdy",
        BodyArchetype::UprightStudio => "upright_studio_v1.pbdy",
        BodyArchetype::VintageSaloon => "vintage_saloon_v1.pbdy",
    }
}

/// Returns the embedded serialized `PIANOBOD` bytes for `id`.
///
/// Note: `upright_studio_v1.pbdy` and `vintage_saloon_v1.pbdy` are integrated
/// into `crates/piano-radiator/assets/` by `w3_1_integrator`; until those
/// assets land on `main`, non-grand archetypes fall back cleanly to the
/// embedded concert-grand `.pbdy` bytes.
#[must_use]
pub fn embedded_body_bytes(id: BodyArchetype) -> &'static [u8] {
    match id {
        BodyArchetype::ConcertGrand => CONCERT_GRAND_PRESET_BYTES,
        BodyArchetype::UprightStudio => CONCERT_GRAND_PRESET_BYTES,
        BodyArchetype::VintageSaloon => CONCERT_GRAND_PRESET_BYTES,
    }
}
