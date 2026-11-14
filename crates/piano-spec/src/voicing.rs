//! Real-time and per-note voicing parameters (D4 extensibility seam).
//!
//! Provides global piano voicing controls alongside an 88-element
//! zero-initialized [`NoteVoicingOffset`] array so future per-note voicing
//! editors can customize individual keys without changing core DSP data layouts.

/// Number of piano keys represented in [`VoicingSpec::note_offsets`].
const NUM_KEYS: usize = 88;

/// Default global normalized hammer hardness (`[0.0, 1.0]`).
pub const HAMMER_HARDNESS_DEFAULT: f32 = 0.5;
/// Minimum clamped effective hammer hardness.
pub const HAMMER_HARDNESS_MIN: f32 = 0.0;
/// Maximum clamped effective hammer hardness.
pub const HAMMER_HARDNESS_MAX: f32 = 1.0;

/// Default global unison detune multiplier (`0.0` disables detune strictly).
pub const UNISON_DETUNE_SCALE_DEFAULT: f32 = 1.0;
/// Minimum clamped unison detune multiplier.
pub const UNISON_DETUNE_SCALE_MIN: f32 = 0.0;
/// Maximum clamped unison detune multiplier.
pub const UNISON_DETUNE_SCALE_MAX: f32 = 4.0;

/// Default global Railsback stretch tuning multiplier (`0.0` = 12-TET).
pub const STRETCH_AMOUNT_DEFAULT: f32 = 1.0;
/// Minimum clamped stretch tuning multiplier.
pub const STRETCH_AMOUNT_MIN: f32 = 0.0;
/// Maximum clamped stretch tuning multiplier.
pub const STRETCH_AMOUNT_MAX: f32 = 2.0;

/// Default global speaking-length multiplier (`1.0` = nominal concert grand).
pub const STRING_LENGTH_SCALE_DEFAULT: f32 = 1.0;
/// Minimum clamped speaking-length multiplier.
pub const STRING_LENGTH_SCALE_MIN: f32 = 0.5;
/// Maximum clamped speaking-length multiplier.
pub const STRING_LENGTH_SCALE_MAX: f32 = 2.0;

/// Minimum clamped per-note pitch offset in cents.
pub const NOTE_PITCH_CENTS_MIN: f32 = -100.0;
/// Maximum clamped per-note pitch offset in cents.
pub const NOTE_PITCH_CENTS_MAX: f32 = 100.0;

/// Minimum clamped per-note effective inharmonicity multiplier (`1.0 + offset`).
pub const NOTE_INHARMONICITY_SCALE_MIN: f32 = 0.25;
/// Maximum clamped per-note effective inharmonicity multiplier (`1.0 + offset`).
pub const NOTE_INHARMONICITY_SCALE_MAX: f32 = 4.0;

/// Minimum clamped per-note hammer hardness offset.
pub const NOTE_HAMMER_OFFSET_MIN: f32 = -0.5;
/// Maximum clamped per-note hammer hardness offset.
pub const NOTE_HAMMER_OFFSET_MAX: f32 = 0.5;

/// Minimum clamped per-note output level offset in dB.
pub const NOTE_LEVEL_DB_MIN: f32 = -24.0;
/// Maximum clamped per-note output level offset in dB.
pub const NOTE_LEVEL_DB_MAX: f32 = 12.0;

/// Per-key additive voicing adjustments (D4 zero-initialized extension seam).
///
/// All fields default to `0.0` so an unconfigured key adds zero deviation
/// from global physical scaling and voicing parameters.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NoteVoicingOffset {
    /// Fine pitch offset in cents (`[-100.0, 100.0]`, default `0.0`).
    pub pitch_cents: f32,
    /// Additive delta on baseline inharmonicity `B(k)`, where effective
    /// multiplier is `(1.0 + inharmonicity_scale).clamp(0.25, 4.0)` (default `0.0`).
    pub inharmonicity_scale: f32,
    /// Additive delta on global `hammer_hardness` (`[-0.5, 0.5]`, default `0.0`).
    pub hammer_hardness_offset: f32,
    /// Output level trim in decibels (`[-24.0, 12.0]`, default `0.0`).
    pub level_db_offset: f32,
}

impl NoteVoicingOffset {
    /// Zero-initialized per-note offset producing nominal baseline behavior.
    pub const ZERO: Self = Self {
        pitch_cents: 0.0,
        inharmonicity_scale: 0.0,
        hammer_hardness_offset: 0.0,
        level_db_offset: 0.0,
    };
}

impl Default for NoteVoicingOffset {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Global and per-key voicing configuration for the 88-key piano bank.
#[derive(Clone, Debug, PartialEq)]
pub struct VoicingSpec {
    /// Global hammer felt hardness in `[0.0, 1.0]` (default `0.5`).
    pub hammer_hardness: f32,
    /// Unison and polarization micro-detune scale in `[0.0, 4.0]` (default `1.0`).
    pub unison_detune_scale: f32,
    /// Railsback stretch tuning scale in `[0.0, 2.0]` (default `1.0`).
    pub stretch_amount: f32,
    /// Speaking string length scale in `[0.5, 2.0]` (default `1.0`).
    pub string_length_scale: f32,
    /// Per-key additive voicing offsets (default all zero).
    pub note_offsets: [NoteVoicingOffset; NUM_KEYS],
}

impl VoicingSpec {
    /// Constructs a [`VoicingSpec`] with nominal concert grand defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hammer_hardness: HAMMER_HARDNESS_DEFAULT,
            unison_detune_scale: UNISON_DETUNE_SCALE_DEFAULT,
            stretch_amount: STRETCH_AMOUNT_DEFAULT,
            string_length_scale: STRING_LENGTH_SCALE_DEFAULT,
            note_offsets: [NoteVoicingOffset::ZERO; NUM_KEYS],
        }
    }

    /// Returns the per-note offset struct for `key_idx`, or [`NoteVoicingOffset::ZERO`]
    /// if `key_idx >= 88`.
    #[must_use]
    pub fn note_offset(&self, key_idx: usize) -> NoteVoicingOffset {
        self.note_offsets
            .get(key_idx)
            .copied()
            .unwrap_or(NoteVoicingOffset::ZERO)
    }

    /// Returns the clamped global hammer hardness in `[0.0, 1.0]`.
    #[must_use]
    pub fn effective_global_hammer_hardness(&self) -> f32 {
        sanitize_clamp(
            self.hammer_hardness,
            HAMMER_HARDNESS_DEFAULT,
            HAMMER_HARDNESS_MIN,
            HAMMER_HARDNESS_MAX,
        )
    }

    /// Returns the clamped effective hammer hardness for `key_idx` in `[0.0, 1.0]`.
    #[must_use]
    pub fn effective_hammer_hardness(&self, key_idx: usize) -> f32 {
        let base = self.effective_global_hammer_hardness();
        let delta = sanitize_clamp(
            self.note_offset(key_idx).hammer_hardness_offset,
            0.0,
            NOTE_HAMMER_OFFSET_MIN,
            NOTE_HAMMER_OFFSET_MAX,
        );
        (base + delta).clamp(HAMMER_HARDNESS_MIN, HAMMER_HARDNESS_MAX)
    }

    /// Returns the clamped unison detune scale in `[0.0, 4.0]`.
    #[must_use]
    pub fn effective_unison_detune_scale(&self) -> f32 {
        sanitize_clamp(
            self.unison_detune_scale,
            UNISON_DETUNE_SCALE_DEFAULT,
            UNISON_DETUNE_SCALE_MIN,
            UNISON_DETUNE_SCALE_MAX,
        )
    }

    /// Returns the clamped Railsback stretch scale in `[0.0, 2.0]`.
    #[must_use]
    pub fn effective_stretch_amount(&self) -> f32 {
        sanitize_clamp(
            self.stretch_amount,
            STRETCH_AMOUNT_DEFAULT,
            STRETCH_AMOUNT_MIN,
            STRETCH_AMOUNT_MAX,
        )
    }

    /// Returns the clamped string length multiplier in `[0.5, 2.0]`.
    #[must_use]
    pub fn effective_string_length_scale(&self) -> f32 {
        sanitize_clamp(
            self.string_length_scale,
            STRING_LENGTH_SCALE_DEFAULT,
            STRING_LENGTH_SCALE_MIN,
            STRING_LENGTH_SCALE_MAX,
        )
    }

    /// Returns the clamped per-note pitch offset in cents (`[-100.0, 100.0]`).
    #[must_use]
    pub fn effective_pitch_cents(&self, key_idx: usize) -> f32 {
        sanitize_clamp(
            self.note_offset(key_idx).pitch_cents,
            0.0,
            NOTE_PITCH_CENTS_MIN,
            NOTE_PITCH_CENTS_MAX,
        )
    }

    /// Returns the clamped per-note inharmonicity multiplier `1.0 + offset`
    /// in `[0.25, 4.0]`.
    #[must_use]
    pub fn effective_inharmonicity_scale(&self, key_idx: usize) -> f32 {
        let delta = self.note_offset(key_idx).inharmonicity_scale;
        if delta.is_nan() {
            return 1.0;
        }
        (1.0 + delta).clamp(NOTE_INHARMONICITY_SCALE_MIN, NOTE_INHARMONICITY_SCALE_MAX)
    }

    /// Returns the clamped per-note level offset in dB (`[-24.0, 12.0]`).
    #[must_use]
    pub fn effective_level_db_offset(&self, key_idx: usize) -> f32 {
        sanitize_clamp(
            self.note_offset(key_idx).level_db_offset,
            0.0,
            NOTE_LEVEL_DB_MIN,
            NOTE_LEVEL_DB_MAX,
        )
    }
}

impl Default for VoicingSpec {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sanitize_clamp(v: f32, default: f32, min: f32, max: f32) -> f32 {
    if v.is_nan() {
        default
    } else {
        v.clamp(min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_nominal_concert_spec() {
        let spec = VoicingSpec::default();
        assert_eq!(spec.hammer_hardness, 0.5);
        assert_eq!(spec.unison_detune_scale, 1.0);
        assert_eq!(spec.stretch_amount, 1.0);
        assert_eq!(spec.string_length_scale, 1.0);
        assert_eq!(spec.note_offsets.len(), 88);

        for k in 0..NUM_KEYS {
            assert_eq!(spec.note_offsets[k], NoteVoicingOffset::ZERO);
            assert_eq!(spec.effective_hammer_hardness(k), 0.5);
            assert_eq!(spec.effective_pitch_cents(k), 0.0);
            assert_eq!(spec.effective_inharmonicity_scale(k), 1.0);
            assert_eq!(spec.effective_level_db_offset(k), 0.0);
        }
    }

    #[test]
    fn clamping_and_nan_safety_and_per_note_overrides() {
        let mut spec = VoicingSpec {
            hammer_hardness: 0.85,
            unison_detune_scale: -1.5,
            stretch_amount: 5.0,
            string_length_scale: f32::NAN,
            note_offsets: [NoteVoicingOffset::ZERO; NUM_KEYS],
        };
        spec.note_offsets[48] = NoteVoicingOffset {
            pitch_cents: 12.5,
            inharmonicity_scale: 0.25,
            hammer_hardness_offset: 0.4,
            level_db_offset: -40.0,
        };
        spec.note_offsets[0] = NoteVoicingOffset {
            pitch_cents: f32::NAN,
            inharmonicity_scale: -2.0,
            hammer_hardness_offset: f32::NEG_INFINITY,
            level_db_offset: 25.0,
        };

        assert_eq!(spec.effective_unison_detune_scale(), 0.0);
        assert_eq!(spec.effective_stretch_amount(), STRETCH_AMOUNT_MAX);
        assert_eq!(
            spec.effective_string_length_scale(),
            STRING_LENGTH_SCALE_DEFAULT
        );

        assert_eq!(spec.effective_pitch_cents(48), 12.5);
        assert_eq!(spec.effective_inharmonicity_scale(48), 1.25);
        assert_eq!(spec.effective_hammer_hardness(48), 1.0);
        assert_eq!(spec.effective_level_db_offset(48), NOTE_LEVEL_DB_MIN);

        assert_eq!(spec.effective_pitch_cents(0), 0.0);
        assert_eq!(
            spec.effective_inharmonicity_scale(0),
            NOTE_INHARMONICITY_SCALE_MIN
        );
        assert!((spec.effective_hammer_hardness(0) - 0.35).abs() < 1e-6);
        assert_eq!(spec.effective_level_db_offset(0), NOTE_LEVEL_DB_MAX);

        // Out-of-range key index safely falls back without panicking.
        assert_eq!(spec.effective_pitch_cents(999), 0.0);
        assert_eq!(spec.effective_inharmonicity_scale(999), 1.0);
    }
}
