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
