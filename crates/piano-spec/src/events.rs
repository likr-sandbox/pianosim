//! Zero-allocation sample-accurate MIDI 1.0/2.0 event types.

/// Continuous piano pedal kind mapped to MIDI CC64 / CC67 / CC66 semantics.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PedalKind {
    /// Right sustain (damper) pedal (CC64).
    Sustain = 0,
    /// Left una corda (soft) pedal (CC67).
    UnaCorda = 1,
    /// Middle sostenuto pedal (CC66).
    Sostenuto = 2,
}

impl PedalKind {
    /// Converts a raw byte discriminant into a [`PedalKind`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Sustain),
            1 => Some(Self::UnaCorda),
            2 => Some(Self::Sostenuto),
            _ => None,
        }
    }
}

/// Payload of a sample-accurate control or note event.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    /// Key strike with 16-bit normalized velocity (`0..=65535`).
    NoteOn { midi_key: u8, velocity_u16: u16 },
    /// Key release with 16-bit normalized release velocity (`0..=65535`).
    NoteOff { midi_key: u8, release_vel_u16: u16 },
    /// Continuous pedal position (`0..=65535`).
    Pedal { kind: PedalKind, value_u16: u16 },
    /// Panic/reset command releasing all active keys.
    AllNotesOff,
}

/// Sample-accurate event relative to the start of the current render block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SampleEvent {
    /// Sample index within the current render block.
    pub sample_offset: u32,
    /// Event payload.
    pub kind: EventKind,
}

/// Maps a 7-bit MIDI value (`0..=127`, clamping values `> 127` to `127`) onto `0..=65535`.
#[must_use]
pub fn u7_to_u16(v: u8) -> u16 {
    let clamped = if v > 127 { 127 } else { v };
    let val = u32::from(clamped);
    ((val * 65_535) / 127) as u16
}

/// Normalizes a 16-bit unsigned MIDI 2.0 quantity to the unit interval `[0.0, 1.0]`.
#[must_use]
pub fn u16_to_unit_f32(v: u16) -> f32 {
    f32::from(v) / 65_535.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u7_to_u16_endpoints_and_clamping() {
        assert_eq!(u7_to_u16(0), 0);
        assert_eq!(u7_to_u16(127), u16::MAX);
        assert_eq!(u7_to_u16(128), u16::MAX);
        assert_eq!(u7_to_u16(255), u16::MAX);

        let mut prev = u7_to_u16(0);
        for v in 1..=127 {
            let cur = u7_to_u16(v);
            assert!(cur > prev, "expected strict monotonicity at {v}");
            prev = cur;
        }
    }

    #[test]
    fn u16_to_unit_f32_range() {
        assert_eq!(u16_to_unit_f32(0), 0.0);
        assert_eq!(u16_to_unit_f32(u16::MAX), 1.0);
        let mid = u16_to_unit_f32(32_768);
        assert!(mid > 0.49 && mid < 0.51);
    }

    #[test]
    fn pedal_kind_from_u8_roundtrip() {
        assert_eq!(PedalKind::from_u8(0), Some(PedalKind::Sustain));
        assert_eq!(PedalKind::from_u8(1), Some(PedalKind::UnaCorda));
        assert_eq!(PedalKind::from_u8(2), Some(PedalKind::Sostenuto));
        assert_eq!(PedalKind::Sustain as u8, 0);
        assert_eq!(PedalKind::UnaCorda as u8, 1);
        assert_eq!(PedalKind::Sostenuto as u8, 2);
        for invalid in 3..=u8::MAX {
            assert_eq!(PedalKind::from_u8(invalid), None);
        }
    }

    #[test]
    fn sample_event_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<PedalKind>();
        assert_copy::<EventKind>();
        assert_copy::<SampleEvent>();

        let ev = SampleEvent {
            sample_offset: 64,
            kind: EventKind::NoteOn {
                midi_key: 69,
                velocity_u16: u7_to_u16(100),
            },
        };
        let copied = ev;
        assert_eq!(ev, copied);

        let pedal = SampleEvent {
            sample_offset: 12,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: 32_768,
            },
        };
        let pedal_copy = pedal;
        assert_eq!(pedal, pedal_copy);
    }
}
