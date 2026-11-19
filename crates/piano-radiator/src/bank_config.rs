//! Profile and space reconfiguration helpers for [`crate::RadiatorBank`].

use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PerfProfile, PresetDecodeError, PresetReducedPole,
    PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES, PRESET_REDUCED_POLES,
};

use crate::ir_synth::{
    ir_len_for_sample_rate, ir_len_for_sample_rate_secs, synthesize_irs, IrSynthScratch,
};
use crate::math::clamp_f64;
use crate::RadiatorBank;

extern crate alloc;
use alloc::vec::Vec;

pub(crate) fn empty_fallback_preset() -> BodyPreset {
    BodyPreset {
        y_inf: [1.8e-4; PRESET_BRIDGE_ZONES],
        direct_zone_gain: [0.0; PRESET_RADIATOR_ZONES],
        zone_pos_m: [[0.0; 3]; PRESET_RADIATOR_ZONES],
        reduced_poles: [PresetReducedPole {
            freq_hz: 100.0,
            zeta: 0.02,
            modal_mass_kg: 0.0,
            zone_gains: [0.0; PRESET_BRIDGE_ZONES],
        }; PRESET_REDUCED_POLES],
        modes: Vec::new(),
    }
}

impl RadiatorBank {
    /// Update the active [`PerfProfile`] (rebuilding IR/NUPOLS if structural Stage 4 knobs changed).
    pub fn set_perf_profile(&mut self, profile: &PerfProfile) {
        self.staged_space_stream = None;
        let next = profile.sanitized();
        self.hall_fdn.set_fdn_lines(next.hall_fdn_lines);
        let ir_changed = (next.ir_duration_secs - self.profile.ir_duration_secs).abs() > 1.0e-6;
        let struct_changed = ir_changed
            || next.spatial_tail_split_tap != self.profile.spatial_tail_split_tap
            || next.fir_head_taps != self.profile.fir_head_taps
            || next.fir_block_fft != self.profile.fir_block_fft;
        let block_toggled = next.fir_block_fft != self.fir_block_fft;

        self.profile = next;
        self.profile.hall_fdn_lines = self.hall_fdn.fdn_lines();
        if !struct_changed {
            return;
        }
        let active_ir_len =
            ir_len_for_sample_rate_secs(self.sample_rate, f64::from(next.ir_duration_secs));
        self.fir_block_fft = next.fir_block_fft;
        self.head.set_active_taps(next.fir_head_taps);
        self.tail.set_stage4_config(
            active_ir_len,
            next.spatial_tail_split_tap,
            next.fir_head_taps,
            next.fir_block_fft,
        );
        if block_toggled {
            self.head.reset();
            self.block_fft_head.reset();
        }
        self.rebuild_ir_and_load();
    }

    /// Re-synthesize all 18 spatialized IRs and reload FIR/NUPOLS partitions in-place.
    pub fn reconfigure_space(&mut self, perspective: ListenerPerspective, lid_state: LidState) {
        self.staged_space_stream = None;
        if self.perspective == perspective && self.lid_state == lid_state {
            return;
        }
        self.perspective = perspective;
        self.lid_state = lid_state;
        self.hall_fdn.set_perspective(perspective);
        self.rebuild_ir_and_load();
    }

    /// Decode serialized `PIANOBOD` bytes, switch the active body preset, and
    /// re-synthesize FIR/NUPOLS partitions in-place.
    pub fn reconfigure_from_preset_bytes(
        &mut self,
        bytes: &[u8],
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> Result<(), PresetDecodeError> {
        let preset = BodyPreset::decode(bytes)?;
        self.reconfigure_from_preset(preset, perspective, lid_state);
        Ok(())
    }

    /// Switch the active body preset and re-synthesize FIR/NUPOLS partitions in-place.
    pub fn reconfigure_from_preset(
        &mut self,
        preset: BodyPreset,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) {
        self.staged_space_stream = None;
        self.preset = preset;
        self.perspective = perspective;
        self.lid_state = lid_state;
        self.hall_fdn.set_perspective(perspective);
        self.rebuild_ir_and_load();
    }

    /// Synthesize all 18 spatialized/pole-cancelled IRs (`[zone][L/R][n]`) for
    /// offline inspection and regression tests.
    pub fn synthesize_ir_for_test(
        preset: &BodyPreset,
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> [[Vec<f32>; 2]; PRESET_RADIATOR_ZONES] {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let ir_len = ir_len_for_sample_rate(sr);
        let mut scratch = IrSynthScratch::new(ir_len);
        synthesize_irs(preset, perspective, lid_state, sr, &mut scratch);
        core::array::from_fn(|z| {
            [
                scratch.path_slice(z, 0).to_vec(),
                scratch.path_slice(z, 1).to_vec(),
            ]
        })
    }
}
