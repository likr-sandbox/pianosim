//! 88-key / 230-string / 4-lane `F32x4` vector-range mapping (D1 seam).
//!
//! Maps each of the 88 piano keys (`CHOIR_LAYOUTS`) to its 1..3 physical
//! unison strings (`TOTAL_STRINGS = 230`) and contiguous `F32x4`-aligned ranges
//! (`vert_vecs`, `horiz_vecs`, `long_vecs`) inside the global SoA bank.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::f64::consts::FRAC_PI_2;
use core::ops::Range;

use piano_spec::{
    analyze_bank_mode_counts, build_scale_design, count_longitudinal_partials,
    count_transverse_partials, effective_string_f0_hz, pad_to_simd4, BankModeCounts, Polarization,
    StringPhysicalParams, VoicingSpec, CHOIR_LAYOUTS, MAX_AUDIBLE_FREQ_HZ, NUM_KEYS, TOTAL_STRINGS,
};

use crate::modal_soa::sin_cos_f64;

/// Contiguous vector-range and stereo-pan descriptor for one physical string.
#[derive(Clone, Debug, PartialEq)]
pub struct StringMapEntry {
    /// Owning piano key index (`0..88`, `0 = A0`, `87 = C8`).
    pub key_idx: u8,
    /// Unison index within the key's choir (`0..choir_size`).
    pub string_in_choir: u8,
    /// Number of strings in this key's choir (`1`, `2`, or `3`).
    pub choir_size: u8,
    /// Whether this string sits under an acoustic damper (`keys 0..70`).
    pub has_damper: bool,
    /// Left stereo panning gain (`pan_l^2 + pan_r^2 ~= 1.0`).
    pub pan_l: f32,
    /// Right stereo panning gain (`pan_l^2 + pan_r^2 ~= 1.0`).
    pub pan_r: f32,
    /// `F32x4` vector range for vertical transverse polarization modes.
    pub vert_vecs: Range<usize>,
    /// `F32x4` vector range for horizontal transverse polarization modes.
    pub horiz_vecs: Range<usize>,
    /// `F32x4` vector range for longitudinal modes (empty for `keys 52..88`).
    pub long_vecs: Range<usize>,
    /// Unpadded active vertical transverse mode count.
    pub vert_modes: usize,
    /// Unpadded active horizontal transverse mode count.
    pub horiz_modes: usize,
    /// Unpadded active longitudinal mode count.
    pub long_modes: usize,
}

impl StringMapEntry {
    /// Full contiguous `F32x4` vector range (`vert_vecs.start..long_vecs.end`).
    #[must_use]
    #[inline]
    pub fn all_vecs(&self) -> Range<usize> {
        self.vert_vecs.start..self.long_vecs.end
    }

    /// Number of `F32x4` vectors owned by this string across all polarizations.
    #[must_use]
    #[inline]
    pub fn vec_len(&self) -> usize {
        self.long_vecs.end - self.vert_vecs.start
    }
}

/// Per-key range mapping into `strings` (`0..230`) and global `F32x4` vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyMapEntry {
    /// Contiguous range in `ChoirMap::strings()` belonging to this key.
    pub string_range: Range<usize>,
    /// Contiguous `F32x4` vector range across all strings of this key.
    pub vec_range: Range<usize>,
    /// Number of unison strings for this key (`1`, `2`, or `3`).
    pub choir_size: u8,
    /// Whether this key has a physical damper (`true` for `0..70`).
    pub has_damper: bool,
}

/// Complete bidirectional layout connecting `88` keys, `230` strings, and
/// SIMD4-aligned SoA vector blocks.
#[derive(Clone, Debug)]
pub struct ChoirMap {
    keys: Box<[KeyMapEntry]>,
    strings: Box<[StringMapEntry]>,
    mode_counts: BankModeCounts,
    total_vecs: usize,
}

impl ChoirMap {
    /// Constructs the choir and vector layout for `voicing`.
    #[must_use]
    pub fn build(voicing: &VoicingSpec) -> Self {
        let specs = build_scale_design(voicing);
        Self::build_with_specs(&specs, voicing)
    }

    /// Constructs the choir and vector layout from precomputed `specs` and `voicing`.
    #[must_use]
    pub fn build_with_specs(
        specs: &[StringPhysicalParams; NUM_KEYS],
        voicing: &VoicingSpec,
    ) -> Self {
        let mut keys = Vec::with_capacity(NUM_KEYS);
        let mut strings = Vec::with_capacity(TOTAL_STRINGS);
        let mut vec_cursor = 0usize;

        let mut t_raw = 0usize;
        let mut t_pad = 0usize;
        let mut l_raw = 0usize;
        let mut l_pad = 0usize;

        for (key_idx, spec) in specs.iter().enumerate() {
            let layout = CHOIR_LAYOUTS[key_idx];
            let choir_size = layout.strings_per_key;
            let str_start = strings.len();
            let key_vec_start = vec_cursor;

            let (pan_l, pan_r) = stereo_pan_for_key(key_idx);
            let long_modes =
                count_longitudinal_partials(key_idx, spec.f_long_hz, MAX_AUDIBLE_FREQ_HZ);
            let long_padded = pad_to_simd4(long_modes);
            let long_vec_len = long_padded / 4;

            for string_in_choir in 0..(choir_size as usize) {
                let f0_vert = effective_string_f0_hz(
                    key_idx,
                    string_in_choir,
                    Polarization::Vertical,
                    specs,
                    voicing,
                );
                let vert_modes =
                    count_transverse_partials(f0_vert, spec.inharmonicity_b, MAX_AUDIBLE_FREQ_HZ);
                let vert_padded = pad_to_simd4(vert_modes);
                let vert_vec_len = vert_padded / 4;

                let f0_horiz = effective_string_f0_hz(
                    key_idx,
                    string_in_choir,
                    Polarization::Horizontal,
                    specs,
                    voicing,
                );
                let horiz_modes =
                    count_transverse_partials(f0_horiz, spec.inharmonicity_b, MAX_AUDIBLE_FREQ_HZ);
                let horiz_padded = pad_to_simd4(horiz_modes);
                let horiz_vec_len = horiz_padded / 4;

                let vert_vecs = vec_cursor..(vec_cursor + vert_vec_len);
                let horiz_vecs = vert_vecs.end..(vert_vecs.end + horiz_vec_len);
                let long_vecs = horiz_vecs.end..(horiz_vecs.end + long_vec_len);
                vec_cursor = long_vecs.end;

                t_raw += vert_modes + horiz_modes;
                t_pad += vert_padded + horiz_padded;
                l_raw += long_modes;
                l_pad += long_padded;

                strings.push(StringMapEntry {
                    key_idx: key_idx as u8,
                    string_in_choir: string_in_choir as u8,
                    choir_size,
                    has_damper: layout.has_damper,
                    pan_l,
                    pan_r,
                    vert_vecs,
                    horiz_vecs,
                    long_vecs,
                    vert_modes,
                    horiz_modes,
                    long_modes,
                });
            }

            let str_end = strings.len();
            keys.push(KeyMapEntry {
                string_range: str_start..str_end,
                vec_range: key_vec_start..vec_cursor,
                choir_size,
                has_damper: layout.has_damper,
            });
        }

        let mode_counts = BankModeCounts {
            transverse_unpadded: t_raw,
            transverse_padded: t_pad,
            longitudinal_unpadded: l_raw,
            longitudinal_padded: l_pad,
            total_unpadded: t_raw + l_raw,
            total_padded: t_pad + l_pad,
        };
        debug_assert_eq!(mode_counts, analyze_bank_mode_counts(voicing));
        debug_assert_eq!(strings.len(), TOTAL_STRINGS);
        debug_assert_eq!(vec_cursor * 4, mode_counts.total_padded);

        Self {
            keys: keys.into_boxed_slice(),
            strings: strings.into_boxed_slice(),
            mode_counts,
            total_vecs: vec_cursor,
        }
    }

    /// Returns the per-key entry (`0..88`).
    #[must_use]
    #[inline]
    pub fn key(&self, key_idx: usize) -> Option<&KeyMapEntry> {
        self.keys.get(key_idx)
    }

    /// Returns the per-string entry (`0..230`).
    #[must_use]
    #[inline]
    pub fn string(&self, str_idx: usize) -> Option<&StringMapEntry> {
        self.strings.get(str_idx)
    }

    /// Slice of all `88` key entries.
    #[must_use]
    #[inline]
    pub fn keys(&self) -> &[KeyMapEntry] {
        &self.keys
    }

    /// Slice of all `230` physical string entries.
    #[must_use]
    #[inline]
    pub fn strings(&self) -> &[StringMapEntry] {
        &self.strings
    }

    /// Mode counts across all transverse and longitudinal bands.
    #[must_use]
    #[inline]
    pub const fn mode_counts(&self) -> BankModeCounts {
        self.mode_counts
    }

    /// Total number of `F32x4` SIMD vectors in the bank.
    #[must_use]
    #[inline]
    pub const fn total_vecs(&self) -> usize {
        self.total_vecs
    }

    /// Total SIMD4-padded modal slots (`total_vecs * 4`).
    #[must_use]
    #[inline]
    pub const fn total_padded_modes(&self) -> usize {
        self.mode_counts.total_padded
    }

    /// Total unpadded active modes across all 230 strings.
    #[must_use]
    #[inline]
    pub const fn active_unpadded_modes(&self) -> usize {
        self.mode_counts.total_unpadded
    }
}

#[inline]
fn stereo_pan_for_key(key_idx: usize) -> (f32, f32) {
    let key_norm = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let pan_angle = (0.22 + 0.56 * key_norm) * FRAC_PI_2;
    let (sin_a, cos_a) = sin_cos_f64(pan_angle);
    (cos_a as f32, sin_a as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use piano_spec::{DAMPED_KEYS, LONGITUDINAL_MAX_KEY_IDX};

    #[test]
    fn choir_map_partitions_88_keys_230_strings_and_matches_spec_counts() {
        let voicing = VoicingSpec::default();
        let map = ChoirMap::build(&voicing);
        let expected = analyze_bank_mode_counts(&voicing);

        assert_eq!(map.keys().len(), NUM_KEYS);
        assert_eq!(map.strings().len(), TOTAL_STRINGS);
        assert_eq!(map.mode_counts(), expected);
        assert_eq!(map.total_vecs() * 4, expected.total_padded);
        assert!((25_000..=26_500).contains(&map.total_padded_modes()));

        let mut vec_cursor = 0usize;
        for (str_idx, entry) in map.strings().iter().enumerate() {
            assert_eq!(entry.vert_vecs.start, vec_cursor);
            assert_eq!(entry.horiz_vecs.start, entry.vert_vecs.end);
            assert_eq!(entry.long_vecs.start, entry.horiz_vecs.end);
            assert_eq!(entry.all_vecs(), entry.vert_vecs.start..entry.long_vecs.end);
            assert_eq!(entry.vec_len(), entry.all_vecs().len());
            vec_cursor = entry.long_vecs.end;

            let k = entry.key_idx as usize;
            let key_entry = map.key(k).expect("valid key");
            assert!(key_entry.string_range.contains(&str_idx));
            assert_eq!(entry.has_damper, k < DAMPED_KEYS);
            if k >= LONGITUDINAL_MAX_KEY_IDX {
                assert!(entry.long_vecs.is_empty());
                assert_eq!(entry.long_modes, 0);
            } else {
                assert!(!entry.long_vecs.is_empty());
                assert!(entry.long_modes > 0);
            }

            let power = entry.pan_l * entry.pan_l + entry.pan_r * entry.pan_r;
            assert!((power - 1.0).abs() < 1e-6, "pan power={power}");
        }
        assert_eq!(vec_cursor, map.total_vecs());

        // Natural A0-left to C8-right stereo image.
        let a0 = &map.strings()[0];
        let c8 = &map.strings()[TOTAL_STRINGS - 1];
        assert!(a0.pan_l > a0.pan_r);
        assert!(c8.pan_r > c8.pan_l);
    }
}
