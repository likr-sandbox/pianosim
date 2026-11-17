//! Zero-allocation fixed-capacity event queue with in-place stable offset sorting.

use piano_spec::{EventKind, SampleEvent};

/// Maximum number of sample-accurate events buffered per render block.
pub const MAX_EVENTS: usize = 512;

const EMPTY_EVENT: SampleEvent = SampleEvent {
    sample_offset: 0,
    kind: EventKind::AllNotesOff,
};

/// Fixed-capacity event buffer requiring zero heap allocation after creation.
#[derive(Clone, Debug)]
pub struct EventQueue {
    events: [SampleEvent; MAX_EVENTS],
    len: usize,
    dropped: u32,
}

impl EventQueue {
    /// Creates an empty fixed-capacity [`EventQueue`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            events: [EMPTY_EVENT; MAX_EVENTS],
            len: 0,
            dropped: 0,
        }
    }

    /// Returns the number of queued events.
    #[must_use]
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no events are queued.
    #[must_use]
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the fixed queue capacity.
    #[must_use]
    #[inline]
    pub const fn capacity(&self) -> usize {
        MAX_EVENTS
    }

    /// Returns the total number of dropped events due to queue saturation.
    #[must_use]
    #[inline]
    pub const fn dropped_count(&self) -> u32 {
        self.dropped
    }

    /// Appends an event without allocating. Returns `false` if the queue is full.
    #[inline]
    pub fn push(&mut self, event: SampleEvent) -> bool {
        if self.len < MAX_EVENTS {
            self.events[self.len] = event;
            self.len += 1;
            true
        } else {
            self.dropped = self.dropped.saturating_add(1);
            false
        }
    }

    /// Clears all queued events without freeing or reallocating storage.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Returns a slice of the currently queued events.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[SampleEvent] {
        &self.events[..self.len]
    }

    /// Sorts queued events stably by `sample_offset` in-place with zero allocation.
    pub fn sort_by_offset(&mut self) {
        // Insertion sort allocates nothing, preserves arrival order for equal offsets,
        // and runs in O(n) on already-ordered blocks.
        for i in 1..self.len {
            let key = self.events[i];
            let mut j = i;
            while j > 0 && self.events[j - 1].sample_offset > key.sample_offset {
                self.events[j] = self.events[j - 1];
                j -= 1;
            }
            self.events[j] = key;
        }
    }

    /// Replaces queue contents from `src` (truncating at `MAX_EVENTS`) and sorts stably.
    pub fn copy_sorted_from(&mut self, src: &[SampleEvent]) {
        let count = src.len().min(MAX_EVENTS);
        if src.len() > MAX_EVENTS {
            let extra = (src.len() - MAX_EVENTS).min(u32::MAX as usize) as u32;
            self.dropped = self.dropped.saturating_add(extra);
        }
        self.events[..count].copy_from_slice(&src[..count]);
        self.len = count;
        self.sort_by_offset();
    }

    /// Stably sorts queued events by `sample_offset` in-place and returns the split index
    /// where `events[..split]` all have `sample_offset < frames`.
    #[must_use]
    pub fn partition_for_block(&mut self, frames: u32) -> usize {
        self.sort_by_offset();
        let mut split = 0_usize;
        while split < self.len && self.events[split].sample_offset < frames {
            split += 1;
        }
        split
    }

    /// Shifts `events[consumed..len]` to the front in-place and subtracts `frames`
    /// from each remaining event's `sample_offset` with zero heap allocation.
    pub fn advance_block(&mut self, frames: u32, consumed: usize) {
        let consumed = consumed.min(self.len);
        let remaining = self.len - consumed;
        for i in 0..remaining {
            let mut ev = self.events[consumed + i];
            ev.sample_offset = ev.sample_offset.saturating_sub(frames);
            self.events[i] = ev;
        }
        self.len = remaining;
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` if `events` is already non-decreasing in `sample_offset`.
#[must_use]
pub fn is_sorted_by_offset(events: &[SampleEvent]) -> bool {
    let mut prev = 0_u32;
    for ev in events {
        if ev.sample_offset < prev {
            return false;
        }
        prev = ev.sample_offset;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use piano_spec::PedalKind;

    fn note_on(offset: u32, key: u8) -> SampleEvent {
        SampleEvent {
            sample_offset: offset,
            kind: EventKind::NoteOn {
                midi_key: key,
                velocity_u16: 40_000,
            },
        }
    }

    fn note_off(offset: u32, key: u8) -> SampleEvent {
        SampleEvent {
            sample_offset: offset,
            kind: EventKind::NoteOff {
                midi_key: key,
                release_vel_u16: 32_768,
            },
        }
    }

    #[test]
    fn partition_and_advance_empty_and_zero_frames() {
        let mut q = EventQueue::new();
        assert_eq!(q.partition_for_block(128), 0);
        q.advance_block(128, 0);
        assert!(q.is_empty());

        assert!(q.push(note_on(0, 60)));
        assert!(q.push(note_on(12, 64)));
        assert_eq!(q.partition_for_block(0), 0);
        q.advance_block(0, 0);
        assert_eq!(q.len(), 2);
        assert_eq!(q.as_slice()[0].sample_offset, 0);
        assert_eq!(q.as_slice()[1].sample_offset, 12);
    }

    #[test]
    fn partition_boundary_equal_to_frames_carries_over() {
        let mut q = EventQueue::new();
        assert!(q.push(note_on(128, 60)));
        assert!(q.push(note_on(127, 62)));
        assert!(q.push(note_on(0, 64)));

        let consumed = q.partition_for_block(128);
        assert_eq!(consumed, 2);
        assert_eq!(q.as_slice()[..consumed].len(), 2);
        assert_eq!(q.as_slice()[0].sample_offset, 0);
        assert_eq!(q.as_slice()[1].sample_offset, 127);

        q.advance_block(128, consumed);
        assert_eq!(q.len(), 1);
        assert_eq!(q.as_slice()[0].sample_offset, 0);
        assert!(matches!(
            q.as_slice()[0].kind,
            EventKind::NoteOn { midi_key: 60, .. }
        ));
    }

    #[test]
    fn multi_block_carryover_across_three_blocks_preserves_order() {
        let mut q = EventQueue::new();
        assert!(q.push(note_on(300, 72)));
        assert!(q.push(SampleEvent {
            sample_offset: 128,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        }));
        assert!(q.push(note_on(10, 60)));
        assert!(q.push(note_off(256, 60)));
        assert!(q.push(note_on(5, 48)));
        assert!(q.push(note_on(128, 64)));

        // Block 1: [0, 128) -> offsets 5, 10
        let c1 = q.partition_for_block(128);
        assert_eq!(c1, 2);
        assert_eq!(q.as_slice()[0].sample_offset, 5);
        assert_eq!(q.as_slice()[1].sample_offset, 10);
        q.advance_block(128, c1);
        assert_eq!(q.len(), 4);

        // Block 2: [128, 256) shifted to [0, 128) -> Pedal@0, NoteOn(64)@0 in arrival order
        let c2 = q.partition_for_block(128);
        assert_eq!(c2, 2);
        assert_eq!(q.as_slice()[0].sample_offset, 0);
        assert!(matches!(
            q.as_slice()[0].kind,
            EventKind::Pedal {
                kind: PedalKind::Sustain,
                ..
            }
        ));
        assert_eq!(q.as_slice()[1].sample_offset, 0);
        assert!(matches!(
            q.as_slice()[1].kind,
            EventKind::NoteOn { midi_key: 64, .. }
        ));
        q.advance_block(128, c2);
        assert_eq!(q.len(), 2);

        // Block 3: [256, 384) shifted to [0, 128) -> NoteOff(60)@0, NoteOn(72)@44
        let c3 = q.partition_for_block(128);
        assert_eq!(c3, 2);
        assert_eq!(q.as_slice()[0].sample_offset, 0);
        assert!(matches!(
            q.as_slice()[0].kind,
            EventKind::NoteOff { midi_key: 60, .. }
        ));
        assert_eq!(q.as_slice()[1].sample_offset, 44);
        assert!(matches!(
            q.as_slice()[1].kind,
            EventKind::NoteOn { midi_key: 72, .. }
        ));
        q.advance_block(128, c3);
        assert!(q.is_empty());
    }

    #[test]
    fn advance_block_defensive_clamping_never_panics() {
        let mut q = EventQueue::new();
        assert!(q.push(note_on(10, 60)));
        q.advance_block(128, 99);
        assert!(q.is_empty());

        assert!(q.push(note_on(10, 60)));
        q.advance_block(128, 0);
        assert_eq!(q.len(), 1);
        assert_eq!(q.as_slice()[0].sample_offset, 0);
    }
}
