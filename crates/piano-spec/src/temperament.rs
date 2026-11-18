//! Classical 12-tone keyboard temperaments (`Equal`, `WerckmeisterIII`,
//! `KirnbergerIII`, `Vallotti`) with exact musicological cent deviations.
//!
//! All tables are anchored at `A = 0.0` cents so concert pitch `a4_freq_hz`
//! remains exact at `A4` (`key_idx = 48`) across all temperaments, while every
//! musical interval preserves its exact historical ratio. C-anchored tables
//! (`C = 0.0` cents) are also exposed via [`Temperament::cent_offset_from_c`].

use crate::scale_design::NUM_KEYS;

/// Pure 3:2 fifth in cents: `1200 * log2(3/2)`.
pub const PURE_FIFTH_CENTS: f64 = 701.955_000_865_387_4;
/// Pythagorean comma in cents: `12 * PURE_FIFTH_CENTS - 8400` (`1200 * log2(531441 / 524288)`).
pub const PYTHAGOREAN_COMMA_CENTS: f64 = 23.460_010_384_649_65;
/// Syntonic comma in cents: `1200 * log2(81 / 80)`.
pub const SYNTONIC_COMMA_CENTS: f64 = 21.506_289_596_714_78;
/// Schisma in cents: `1200 * log2(32805 / 32768)` (`PC - SC`).
pub const SCHISMA_CENTS: f64 = 1.953_720_787_934_16;

const FIFTH_VS_ET: f64 = PYTHAGOREAN_COMMA_CENTS / 12.0; // +1.955000865387471 cents
const PC_QUARTER: f64 = PYTHAGOREAN_COMMA_CENTS * 0.25;
const SC_QUARTER: f64 = SYNTONIC_COMMA_CENTS * 0.25;

// Werckmeister III (1691), C-anchored: C-G, G-D, D-A, B-F# narrowed by 1/4 PC;
// remaining 8 fifths pure 3:2.
// C=0.00, C#=-9.78, D=-7.82, D#=-5.87, E=-9.78, F=-1.96, F#=-11.73,
// G=-3.91, G#=-7.82, A=-11.73, A#=-3.91, B=-7.82.
const WERCKMEISTER_III_FROM_C: [f64; 12] = [
    0.0,
    -5.0 * FIFTH_VS_ET,
    2.0 * (FIFTH_VS_ET - PC_QUARTER),
    -3.0 * FIFTH_VS_ET,
    -5.0 * FIFTH_VS_ET,
    -FIFTH_VS_ET,
    -6.0 * FIFTH_VS_ET,
    FIFTH_VS_ET - PC_QUARTER,
    -4.0 * FIFTH_VS_ET,
    3.0 * (FIFTH_VS_ET - PC_QUARTER),
    -2.0 * FIFTH_VS_ET,
    -4.0 * FIFTH_VS_ET,
];

// A-anchored Werckmeister III (+11.73000519... relative to C-anchored):
// C=+11.73, C#=+1.96, D=+3.91, D#=+5.87, E=+1.96, F=+9.78, F#=0.00,
// G=+7.82, G#=+3.91, A=0.00, A#=+7.82, B=+3.91.
const WERCKMEISTER_III_FROM_A: [f64; 12] = anchor_at_a(WERCKMEISTER_III_FROM_C);

// Kirnberger III (1779), C-anchored: C-G, G-D, D-A, A-E narrowed by 1/4 SC
// (yielding pure C-E 5:4 = -13.6863 cents), F#-C# narrowed by 1 schisma,
// remaining 7 fifths pure 3:2.
// C=0.00, C#=-9.78, D=-6.84, D#=-5.87, E=-13.69, F=-1.96, F#=-9.78,
// G=-3.42, G#=-7.82, A=-10.26, A#=-3.91, B=-11.73.
const KIRNBERGER_III_FROM_C: [f64; 12] = build_kirnberger_iii_from_c();
const KIRNBERGER_III_FROM_A: [f64; 12] = anchor_at_a(KIRNBERGER_III_FROM_C);

// Vallotti (1754), C-anchored: 6 diatonic fifths (F-C, C-G, G-D, D-A, A-E, E-B)
// narrowed by 1/6 PC (-1.9550 cents vs ET); 6 fifths pure 3:2 (+1.9550 cents vs ET).
// C=0.00, C#=-5.87, D=-3.91, D#=-1.96, E=-7.82, F=+1.96, F#=-7.82,
// G=-1.96, G#=-3.91, A=-5.87, A#=0.00, B=-9.78.
const VALLOTTI_FROM_C: [f64; 12] = [
    0.0,
    -3.0 * FIFTH_VS_ET,
    -2.0 * FIFTH_VS_ET,
    -FIFTH_VS_ET,
    -4.0 * FIFTH_VS_ET,
    FIFTH_VS_ET,
    -4.0 * FIFTH_VS_ET,
    -FIFTH_VS_ET,
    -2.0 * FIFTH_VS_ET,
    -3.0 * FIFTH_VS_ET,
    0.0,
    -5.0 * FIFTH_VS_ET,
];

// A-anchored Vallotti (+5.86500260... relative to C-anchored):
// C=+5.87, C#=0.00, D=+1.96, D#=+3.91, E=-1.96, F=+7.82, F#=-1.96,
// G=+3.91, G#=+1.96, A=0.00, A#=+5.87, B=-3.91.
const VALLOTTI_FROM_A: [f64; 12] = anchor_at_a(VALLOTTI_FROM_C);

const fn build_kirnberger_iii_from_c() -> [f64; 12] {
    let q = FIFTH_VS_ET - SC_QUARTER;
    let c = 0.0;
    let g = c + q;
    let d = g + q;
    let a = d + q;
    let e = a + q;
    let b = e + FIFTH_VS_ET;
    let fs = b + FIFTH_VS_ET;
    let cs = fs + FIFTH_VS_ET - SCHISMA_CENTS;
    let gs = cs + FIFTH_VS_ET;
    let ds = gs + FIFTH_VS_ET;
    let as_ = ds + FIFTH_VS_ET;
    let f = as_ + FIFTH_VS_ET;
    [c, cs, d, ds, e, f, fs, g, gs, a, as_, b]
}

const fn anchor_at_a(from_c: [f64; 12]) -> [f64; 12] {
    let a_off = from_c[9];
    let mut out = [0.0; 12];
    let mut i = 0;
    while i < 12 {
        out[i] = if i == 9 { 0.0 } else { from_c[i] - a_off };
        i += 1;
    }
    out
}

/// Classical 12-tone keyboard temperament systems.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Temperament {
    /// 12-tone Equal Temperament (`0.0` cent deviation on all pitch classes).
    #[default]
    Equal = 0,
    /// Andreas Werckmeister III (1691): 4 fifths narrowed by `1/4` Pythagorean comma.
    WerckmeisterIII = 1,
    /// Johann Philipp Kirnberger III (1779): 4 fifths narrowed by `1/4` syntonic comma, 1 schisma fifth.
    KirnbergerIII = 2,
    /// Francesco Antonio Vallotti (1754): 6 diatonic fifths narrowed by `1/6` Pythagorean comma.
    Vallotti = 3,
}

impl Temperament {
    /// Returns the deterministic `u8` ABI identifier (`0..=3`).
    #[must_use]
    #[inline]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Returns the deterministic `u32` ABI identifier (`0..=3`).
    #[must_use]
    #[inline]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }

    /// Decodes a `u8` temperament identifier (`0..=3`), returning `None` if invalid.
    #[must_use]
    #[inline]
    pub const fn from_u8(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Equal),
            1 => Some(Self::WerckmeisterIII),
            2 => Some(Self::KirnbergerIII),
            3 => Some(Self::Vallotti),
            _ => None,
        }
    }

    /// Decodes a `u32` temperament identifier (`0..=3`), returning `None` if invalid.
    #[must_use]
    #[inline]
    pub const fn from_u32(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Equal),
            1 => Some(Self::WerckmeisterIII),
            2 => Some(Self::KirnbergerIII),
            3 => Some(Self::Vallotti),
            _ => None,
        }
    }

    /// Decodes a `u8` temperament identifier, falling back to [`Self::Equal`] if invalid.
    #[must_use]
    #[inline]
    pub const fn from_u8_or_default(id: u8) -> Self {
        match Self::from_u8(id) {
            Some(t) => t,
            None => Self::Equal,
        }
    }

    /// Decodes a `u32` temperament identifier, falling back to [`Self::Equal`] if invalid.
    #[must_use]
    #[inline]
    pub const fn from_u32_or_default(id: u32) -> Self {
        match Self::from_u32(id) {
            Some(t) => t,
            None => Self::Equal,
        }
    }

    /// Human-readable name for UI/CLI display.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Equal => "Equal",
            Self::WerckmeisterIII => "Werckmeister III",
            Self::KirnbergerIII => "Kirnberger III",
            Self::Vallotti => "Vallotti",
        }
    }

    /// Returns the cent deviation relative to Equal Temperament for `pitch_class_from_c`
    /// (`0 = C, 1 = C#, ..., 9 = A, ..., 11 = B`), anchored at `A = 0.0` cents.
    #[must_use]
    #[inline]
    pub const fn cent_offset_from_a4(self, pitch_class_from_c: usize) -> f64 {
        let pc = pitch_class_from_c % 12;
        match self {
            Self::Equal => 0.0,
            Self::WerckmeisterIII => WERCKMEISTER_III_FROM_A[pc],
            Self::KirnbergerIII => KIRNBERGER_III_FROM_A[pc],
            Self::Vallotti => VALLOTTI_FROM_A[pc],
        }
    }

    /// Returns the cent deviation relative to Equal Temperament for `pitch_class_from_c`
    /// (`0 = C, ..., 11 = B`), anchored at `C = 0.0` cents.
    #[must_use]
    #[inline]
    pub const fn cent_offset_from_c(self, pitch_class_from_c: usize) -> f64 {
        let pc = pitch_class_from_c % 12;
        match self {
            Self::Equal => 0.0,
            Self::WerckmeisterIII => WERCKMEISTER_III_FROM_C[pc],
            Self::KirnbergerIII => KIRNBERGER_III_FROM_C[pc],
            Self::Vallotti => VALLOTTI_FROM_C[pc],
        }
    }

    /// Returns the A4-anchored cent deviation relative to 12-TET for piano key `key_idx`
    /// (`0 = A0 .. 48 = A4 .. 87 = C8`). Out-of-range indices clamp to `87`.
    #[must_use]
    #[inline]
    pub const fn cent_offset_for_key(self, key_idx: usize) -> f64 {
        if matches!(self, Self::Equal) {
            return 0.0;
        }
        let k = if key_idx >= NUM_KEYS {
            NUM_KEYS - 1
        } else {
            key_idx
        };
        // Key 0 (A0) is MIDI 21; 21 % 12 == 9 (A), so (k + 9) % 12 gives C-indexed pitch class.
        self.cent_offset_from_a4((k + 9) % 12)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fifth_cents(t: Temperament, root_pc: usize) -> f64 {
        let upper_pc = (root_pc + 7) % 12;
        700.0 + (t.cent_offset_from_a4(upper_pc) - t.cent_offset_from_a4(root_pc))
    }

    fn major_third_cents(t: Temperament, root_pc: usize) -> f64 {
        let third_pc = (root_pc + 4) % 12;
        400.0 + (t.cent_offset_from_a4(third_pc) - t.cent_offset_from_a4(root_pc))
    }

    #[test]
    fn temperament_id_roundtrips_and_default_is_equal() {
        assert_eq!(Temperament::default(), Temperament::Equal);
        for (id, expected) in [
            (0_u8, Temperament::Equal),
            (1, Temperament::WerckmeisterIII),
            (2, Temperament::KirnbergerIII),
            (3, Temperament::Vallotti),
        ] {
            assert_eq!(Temperament::from_u8(id), Some(expected));
            assert_eq!(Temperament::from_u32(u32::from(id)), Some(expected));
            assert_eq!(expected.to_u8(), id);
            assert_eq!(expected.to_u32(), u32::from(id));
            assert_eq!(Temperament::from_u8_or_default(id), expected);
            assert_eq!(Temperament::from_u32_or_default(u32::from(id)), expected);
        }
        assert_eq!(Temperament::from_u8(4), None);
        assert_eq!(Temperament::from_u32(99), None);
        assert_eq!(Temperament::from_u8_or_default(4), Temperament::Equal);
        assert_eq!(Temperament::from_u32_or_default(99), Temperament::Equal);
    }

    #[test]
    fn equal_temperament_is_exact_zero_everywhere() {
        for pc in 0..12 {
            assert_eq!(Temperament::Equal.cent_offset_from_a4(pc), 0.0);
            assert_eq!(Temperament::Equal.cent_offset_from_c(pc), 0.0);
        }
        for k in 0..NUM_KEYS {
            assert_eq!(Temperament::Equal.cent_offset_for_key(k), 0.0);
        }
    }

    #[test]
    fn werckmeister_iii_matches_exact_circle_of_fifths_and_tables() {
        let t = Temperament::WerckmeisterIII;
        assert_eq!(t.cent_offset_from_a4(9), 0.0);
        assert_eq!(t.cent_offset_for_key(48), 0.0);
        assert_eq!(t.cent_offset_from_c(0), 0.0);

        let expected_from_c = [
            0.0, -9.78, -7.82, -5.87, -9.78, -1.96, -11.73, -3.91, -7.82, -11.73, -3.91, -7.82,
        ];
        for (pc, &exp) in expected_from_c.iter().enumerate() {
            assert!((t.cent_offset_from_c(pc) - exp).abs() < 0.01, "pc={pc}");
        }

        // Narrowed by 1/4 PC: C(0)-G(7), G(7)-D(2), D(2)-A(9), B(11)-F#(6).
        let q_narrow = PURE_FIFTH_CENTS - PC_QUARTER;
        for root in [0, 7, 2, 11] {
            assert!((fifth_cents(t, root) - q_narrow).abs() < 1e-11);
        }
        // Remaining 8 fifths pure 3:2.
        for root in [9, 4, 6, 1, 8, 3, 10, 5] {
            assert!((fifth_cents(t, root) - PURE_FIFTH_CENTS).abs() < 1e-11);
        }
    }

    #[test]
    fn kirnberger_iii_yields_pure_c_e_major_third_and_exact_fifths() {
        let t = Temperament::KirnbergerIII;
        assert_eq!(t.cent_offset_from_a4(9), 0.0);
        assert_eq!(t.cent_offset_for_key(48), 0.0);
        assert_eq!(t.cent_offset_from_c(0), 0.0);

        let expected_from_c = [
            0.0, -9.78, -6.84, -5.87, -13.69, -1.96, -9.78, -3.42, -7.82, -10.26, -3.91, -11.73,
        ];
        for (pc, &exp) in expected_from_c.iter().enumerate() {
            assert!((t.cent_offset_from_c(pc) - exp).abs() < 0.01, "pc={pc}");
        }

        // C-E major third must be pure 5:4 = 1200*log2(5/4) = 386.3137138648348 cents.
        let pure_m3 = 386.313_713_864_834_8;
        assert!((major_third_cents(t, 0) - pure_m3).abs() < 1e-11);

        // 4 fifths narrowed by 1/4 SC: C(0), G(7), D(2), A(9).
        let q_narrow = PURE_FIFTH_CENTS - SC_QUARTER;
        for root in [0, 7, 2, 9] {
            assert!((fifth_cents(t, root) - q_narrow).abs() < 1e-11);
        }
        // F#(6)-C#(1) narrowed by 1 schisma.
        assert!((fifth_cents(t, 6) - (PURE_FIFTH_CENTS - SCHISMA_CENTS)).abs() < 1e-11);
        // Remaining 7 fifths pure 3:2.
        for root in [4, 11, 1, 8, 3, 10, 5] {
            assert!((fifth_cents(t, root) - PURE_FIFTH_CENTS).abs() < 1e-11);
        }
    }

    #[test]
    fn vallotti_splits_six_diatonic_and_six_pure_fifths() {
        let t = Temperament::Vallotti;
        assert_eq!(t.cent_offset_from_a4(9), 0.0);
        assert_eq!(t.cent_offset_for_key(48), 0.0);
        assert_eq!(t.cent_offset_from_c(0), 0.0);

        let expected_from_c = [
            0.0, -5.87, -3.91, -1.96, -7.82, 1.96, -7.82, -1.96, -3.91, -5.87, 0.0, -9.78,
        ];
        for (pc, &exp) in expected_from_c.iter().enumerate() {
            assert!((t.cent_offset_from_c(pc) - exp).abs() < 0.01, "pc={pc}");
        }

        // 6 diatonic fifths narrowed by 1/6 PC: F(5), C(0), G(7), D(2), A(9), E(4).
        let sixth_narrow = PURE_FIFTH_CENTS - PYTHAGOREAN_COMMA_CENTS / 6.0;
        for root in [5, 0, 7, 2, 9, 4] {
            assert!((fifth_cents(t, root) - sixth_narrow).abs() < 1e-11);
        }
        // Remaining 6 fifths pure 3:2: B(11), F#(6), C#(1), G#(8), D#(3), A#(10).
        for root in [11, 6, 1, 8, 3, 10] {
            assert!((fifth_cents(t, root) - PURE_FIFTH_CENTS).abs() < 1e-11);
        }
    }
}
