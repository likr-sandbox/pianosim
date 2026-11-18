//! Real-time and per-note voicing parameters (D4 extensibility seam).
//!
//! Provides global piano voicing controls (including concert pitch `A4`,
//! classical 12-tone [`Temperament`], and deterministic aging `condition`)
//! alongside an 88-element zero-initialized [`NoteVoicingOffset`] array.

use crate::condition::clamp_condition;
pub use crate::condition::{
    condition_note_offset, CONDITION_DEFAULT, CONDITION_MAX, CONDITION_MIN,
};
use crate::temperament::Temperament;

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

/// Default global Railsback stretch tuning multiplier (`0.0` = no stretch).
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

/// Default concert pitch reference frequency for `A4` in Hz.
pub const A4_FREQ_DEFAULT: f32 = 440.0;
/// Minimum clamped concert pitch reference frequency for `A4` in Hz (`~-2` semitones, French Baroque).
pub const A4_FREQ_MIN: f32 = 392.0;
/// Maximum clamped concert pitch reference frequency for `A4` in Hz (`~+1` semitone, Chorton).
pub const A4_FREQ_MAX: f32 = 466.2;

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

/// Minimum clamped per-note effective damping multiplier (`1.0 + offset`).
pub const NOTE_DAMPING_SCALE_MIN: f32 = 0.5;
/// Maximum clamped per-note effective damping multiplier (`1.0 + offset`).
pub const NOTE_DAMPING_SCALE_MAX: f32 = 2.0;

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
    /// Additive delta on baseline string damping rate, where effective
    /// multiplier is `(1.0 + damping_scale).clamp(0.5, 2.0)` (default `0.0`).
    pub damping_scale: f32,
}

impl NoteVoicingOffset {
    /// Zero-initialized per-note offset producing nominal baseline behavior.
    pub const ZERO: Self = Self {
        pitch_cents: 0.0,
        inharmonicity_scale: 0.0,
        hammer_hardness_offset: 0.0,
        level_db_offset: 0.0,
        damping_scale: 0.0,
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
    /// Concert pitch reference frequency for `A4` in `[392.0, 466.2]` Hz (default `440.0`).
    pub a4_freq_hz: f32,
    /// Classical 12-tone keyboard temperament (default [`Temperament::Equal`]).
    pub temperament: Temperament,
    /// Deterministic aging condition macro in `[0.0, 1.0]` (default `0.0`).
    pub condition: f32,
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
            a4_freq_hz: A4_FREQ_DEFAULT,
            temperament: Temperament::Equal,
            condition: CONDITION_DEFAULT,
            note_offsets: [NoteVoicingOffset::ZERO; NUM_KEYS],
        }
    }

    /// Returns a [`VoicingSpec`] populated from global parameters (`A4 = 440.0 Hz`,
    /// `Equal`, `condition = 0.0`) with clamped values.
    #[must_use]
    pub fn from_globals(
        hammer_hardness: f32,
        unison_detune_scale: f32,
        stretch_amount: f32,
        string_length_scale: f32,
    ) -> Self {
        Self::from_globals_with_a4(
            hammer_hardness,
            unison_detune_scale,
            stretch_amount,
            string_length_scale,
            A4_FREQ_DEFAULT,
        )
    }

    /// Returns a [`VoicingSpec`] populated from global parameters including custom `A4` concert pitch.
    #[must_use]
    pub fn from_globals_with_a4(
        hammer_hardness: f32,
        unison_detune_scale: f32,
        stretch_amount: f32,
        string_length_scale: f32,
        a4_freq_hz: f32,
    ) -> Self {
        Self::from_globals_full(
            hammer_hardness,
            unison_detune_scale,
            stretch_amount,
            string_length_scale,
            a4_freq_hz,
            Temperament::Equal,
            CONDITION_DEFAULT,
        )
    }

    /// Returns a [`VoicingSpec`] populated from all global voicing, concert pitch,
    /// temperament, and aging `condition` parameters with clamped values.
    #[must_use]
    pub fn from_globals_full(
        hammer_hardness: f32,
        unison_detune_scale: f32,
        stretch_amount: f32,
        string_length_scale: f32,
        a4_freq_hz: f32,
        temperament: Temperament,
        condition: f32,
    ) -> Self {
        let mut s = Self::new();
        s.hammer_hardness = hammer_hardness;
        s.unison_detune_scale = unison_detune_scale;
        s.stretch_amount = stretch_amount;
        s.string_length_scale = string_length_scale;
        s.a4_freq_hz = a4_freq_hz;
        s.temperament = temperament;
        s.condition = condition;
        s.sanitized()
    }

    /// Builder method setting clamped concert pitch `a4_freq_hz` (`[392.0, 466.2]` Hz).
    #[must_use]
    pub fn with_concert_pitch_hz(mut self, concert_pitch_hz: f32) -> Self {
        self.a4_freq_hz =
            sanitize_clamp(concert_pitch_hz, A4_FREQ_DEFAULT, A4_FREQ_MIN, A4_FREQ_MAX);
        self
    }

    /// Builder method setting classical [`Temperament`].
    #[must_use]
    pub const fn with_temperament(mut self, temperament: Temperament) -> Self {
        self.temperament = temperament;
        self
    }

    /// Builder method setting clamped aging `condition` (`[0.0, 1.0]`).
    #[must_use]
    pub fn with_condition(mut self, condition: f32) -> Self {
        self.condition = clamp_condition(condition);
        self
    }

    /// Returns a copy with all global parameters clamped and finite.
    #[must_use]
    pub fn sanitized(&self) -> Self {
        Self {
            hammer_hardness: self.effective_global_hammer_hardness(),
            unison_detune_scale: self.effective_unison_detune_scale(),
            stretch_amount: self.effective_stretch_amount(),
            string_length_scale: self.effective_string_length_scale(),
            a4_freq_hz: self.effective_a4_freq_hz(),
            temperament: self.temperament,
            condition: self.effective_condition(),
            note_offsets: self.note_offsets,
        }
    }

    /// Returns the raw per-note offset struct for `key_idx`, or [`NoteVoicingOffset::ZERO`]
    /// if `key_idx >= 88`.
    #[must_use]
    pub fn note_offset(&self, key_idx: usize) -> NoteVoicingOffset {
        self.note_offsets
            .get(key_idx)
            .copied()
            .unwrap_or(NoteVoicingOffset::ZERO)
    }

    /// Returns the combined per-note offset (explicit `note_offsets[key_idx]` plus
    /// deterministic `condition` perturbation) for `key_idx`.
    #[must_use]
    pub fn effective_note_offset(&self, key_idx: usize) -> NoteVoicingOffset {
        let base = self.note_offset(key_idx);
        let cond = self.effective_condition();
        if cond == 0.0 {
            return base;
        }
        let aging = condition_note_offset(cond, key_idx);
        NoteVoicingOffset {
            pitch_cents: sanitize_zero(base.pitch_cents) + aging.pitch_cents,
            inharmonicity_scale: sanitize_zero(base.inharmonicity_scale)
                + aging.inharmonicity_scale,
            hammer_hardness_offset: sanitize_zero(base.hammer_hardness_offset)
                + aging.hammer_hardness_offset,
            level_db_offset: sanitize_zero(base.level_db_offset) + aging.level_db_offset,
            damping_scale: sanitize_zero(base.damping_scale) + aging.damping_scale,
        }
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
        let cond = self.effective_condition();
        let raw_delta = sanitize_clamp(
            self.note_offset(key_idx).hammer_hardness_offset,
            0.0,
            NOTE_HAMMER_OFFSET_MIN,
            NOTE_HAMMER_OFFSET_MAX,
        );
        let delta = if cond == 0.0 {
            raw_delta
        } else {
            (raw_delta + condition_note_offset(cond, key_idx).hammer_hardness_offset)
                .clamp(NOTE_HAMMER_OFFSET_MIN, NOTE_HAMMER_OFFSET_MAX)
        };
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

    /// Returns the clamped concert pitch reference frequency for `A4` in `[392.0, 466.2]` Hz.
    #[must_use]
    pub fn effective_a4_freq_hz(&self) -> f32 {
        sanitize_clamp(self.a4_freq_hz, A4_FREQ_DEFAULT, A4_FREQ_MIN, A4_FREQ_MAX)
    }

    /// Alias for [`Self::effective_a4_freq_hz`].
    #[must_use]
    #[inline]
    pub fn effective_concert_pitch_hz(&self) -> f32 {
        self.effective_a4_freq_hz()
    }

    /// Returns the active [`Temperament`].
    #[must_use]
    #[inline]
    pub const fn effective_temperament(&self) -> Temperament {
        self.temperament
    }

    /// Returns the clamped aging `condition` macro in `[0.0, 1.0]`.
    #[must_use]
    pub fn effective_condition(&self) -> f32 {
        clamp_condition(self.condition)
    }

    /// Returns the clamped per-note pitch offset in cents (`[-100.0, 100.0]`).
    #[must_use]
    pub fn effective_pitch_cents(&self, key_idx: usize) -> f32 {
        let cond = self.effective_condition();
        if cond == 0.0 {
            return sanitize_clamp(
                self.note_offset(key_idx).pitch_cents,
                0.0,
                NOTE_PITCH_CENTS_MIN,
                NOTE_PITCH_CENTS_MAX,
            );
        }
        let combined = sanitize_zero(self.note_offset(key_idx).pitch_cents)
            + condition_note_offset(cond, key_idx).pitch_cents;
        combined.clamp(NOTE_PITCH_CENTS_MIN, NOTE_PITCH_CENTS_MAX)
    }

    /// Returns the clamped per-note inharmonicity multiplier `1.0 + offset`
    /// in `[0.25, 4.0]`.
    #[must_use]
    pub fn effective_inharmonicity_scale(&self, key_idx: usize) -> f32 {
        let cond = self.effective_condition();
        let delta = self.note_offset(key_idx).inharmonicity_scale;
        if cond == 0.0 {
            if delta.is_nan() {
                return 1.0;
            }
            return (1.0 + delta).clamp(NOTE_INHARMONICITY_SCALE_MIN, NOTE_INHARMONICITY_SCALE_MAX);
        }
        let combined =
            sanitize_zero(delta) + condition_note_offset(cond, key_idx).inharmonicity_scale;
        (1.0 + combined).clamp(NOTE_INHARMONICITY_SCALE_MIN, NOTE_INHARMONICITY_SCALE_MAX)
    }

    /// Returns the clamped per-note level offset in dB (`[-24.0, 12.0]`).
    #[must_use]
    pub fn effective_level_db_offset(&self, key_idx: usize) -> f32 {
        let cond = self.effective_condition();
        if cond == 0.0 {
            return sanitize_clamp(
                self.note_offset(key_idx).level_db_offset,
                0.0,
                NOTE_LEVEL_DB_MIN,
                NOTE_LEVEL_DB_MAX,
            );
        }
        let combined = sanitize_zero(self.note_offset(key_idx).level_db_offset)
            + condition_note_offset(cond, key_idx).level_db_offset;
        combined.clamp(NOTE_LEVEL_DB_MIN, NOTE_LEVEL_DB_MAX)
    }

    /// Returns the clamped per-note string damping multiplier `1.0 + offset`
    /// in `[0.5, 2.0]` (`1.0` when nominal and `condition == 0.0`).
    #[must_use]
    pub fn effective_damping_scale(&self, key_idx: usize) -> f32 {
        let cond = self.effective_condition();
        let delta = self.note_offset(key_idx).damping_scale;
        if cond == 0.0 {
            if delta.is_nan() {
                return 1.0;
            }
            return (1.0 + delta).clamp(NOTE_DAMPING_SCALE_MIN, NOTE_DAMPING_SCALE_MAX);
        }
        let combined = sanitize_zero(delta) + condition_note_offset(cond, key_idx).damping_scale;
        (1.0 + combined).clamp(NOTE_DAMPING_SCALE_MIN, NOTE_DAMPING_SCALE_MAX)
    }
}

impl Default for VoicingSpec {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn sanitize_zero(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v
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
#[path = "voicing_tests.rs"]
mod tests;
