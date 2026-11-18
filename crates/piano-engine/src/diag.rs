//! Offline diagnostic inspection hooks for Stage 1 action and Stage 3 bridge.

use piano_spec::midi_to_key;

use crate::PianoEngine;

impl PianoEngine {
    /// Toggles Stage 3 rigid fixed-end diagnostic mode (`v_br = 0`, `b_waves = -a_waves`,
    /// zero WDF inter-string coupling) for F8 two-stage decay comparison.
    pub fn set_bridge_fixed_end(&mut self, fixed_end: bool) {
        self.core.bridge.set_fixed_end(fixed_end);
    }

    /// Returns `(last_host_force_n, prev_sub_forces)` from the most recent strike step
    /// of `midi_key` (`(0.0, [0.0; 8])` when `midi_key` is outside `21..=108`).
    #[must_use]
    pub fn last_hammer_forces(&self, midi_key: u8) -> (f32, [f32; 8]) {
        let Some(k) = midi_to_key(midi_key) else {
            return (0.0, [0.0; 8]);
        };
        self.core.action.last_hammer_forces(k)
    }
}
