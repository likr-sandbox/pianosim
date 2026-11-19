//! `PerfProfile` runtime and structural reconfiguration for [`PianoEngine`].

use piano_spec::PerfProfile;
use piano_strings::StringBank;

use crate::PianoEngine;

#[inline]
pub(crate) fn stage2_structural_changed(a: &PerfProfile, b: &PerfProfile) -> bool {
    (a.horiz_max_hz - b.horiz_max_hz).abs() > f32::EPSILON
        || (a.dead_mode_prune_db - b.dead_mode_prune_db).abs() > f32::EPSILON
        || a.bass_phantom_sparse != b.bass_phantom_sparse
}

#[inline]
pub(crate) fn stage4_structural_changed(a: &PerfProfile, b: &PerfProfile) -> bool {
    (a.ir_duration_secs - b.ir_duration_secs).abs() > 1.0e-6
        || a.spatial_tail_split_tap != b.spatial_tail_split_tap
        || a.fir_head_taps != b.fir_head_taps
        || a.fir_block_fft != b.fir_block_fft
}

impl PianoEngine {
    /// Returns the active sanitized [`PerfProfile`].
    #[must_use]
    #[inline]
    pub const fn perf_profile(&self) -> PerfProfile {
        self.core.perf_profile
    }

    /// Applies `profile` across all 4 synthesis stages, rebuilding Stage 1/2/4 tables
    /// if structural parameters changed while preserving voicing, body, perspective, lid, and reverb.
    pub fn set_perf_profile(&mut self, profile: PerfProfile) {
        let next = profile.sanitized();
        let cur = self.core.perf_profile;

        self.core.action.set_perf_profile(&next);

        if stage2_structural_changed(&cur, &next) {
            let sr = self.core.bank.sample_rate();
            self.core.bank =
                StringBank::new_bridge_coupled_with_profile(sr, &self.core.current_voicing, &next);
            self.core
                .bridge
                .set_port_resistances(self.core.bank.bridge_port_resistances());
        } else {
            self.core.bank.set_perf_profile_runtime(&next);
        }

        self.core.bridge.set_perf_profile_runtime(&next);

        self.core.radiators[0].set_perf_profile(&next);
        self.core.radiators[1].set_perf_profile(&next);

        self.core.perf_profile = next;
    }

    /// Zero-allocation audio-thread profile update for the 10 runtime knobs across
    /// `ActionBank`, `StringBank`, `BridgeBank`, and both `RadiatorBank`s.
    ///
    /// Returns `true` if no structural Stage 1/2/4 rebuild is required, or `false` if
    /// structural fields (or unallocated `PGridLuts`) require off-thread rebuild / `SpaceBlob`.
    pub fn set_perf_profile_runtime(&mut self, profile: PerfProfile) -> bool {
        let next = profile.sanitized();
        let cur = self.core.perf_profile;

        self.core.action.set_perf_profile_runtime(&next);
        self.core.bank.set_perf_profile_runtime(&next);
        self.core.bridge.set_perf_profile_runtime(&next);
        self.core.radiators[0].set_fdn_lines(next.hall_fdn_lines);
        self.core.radiators[1].set_fdn_lines(next.hall_fdn_lines);

        let actual_poly_adaa = self.core.action.perf_profile().polynomial_adaa;
        let structural_mismatch = stage2_structural_changed(&cur, &next)
            || stage4_structural_changed(&cur, &next)
            || next.action_oversample != cur.action_oversample
            || actual_poly_adaa != next.polynomial_adaa;

        self.core.perf_profile.energy_horizon_db = next.energy_horizon_db;
        self.core.perf_profile.sav_dormancy_db = next.sav_dormancy_db;
        self.core.perf_profile.sav_max_key_idx = next.sav_max_key_idx;
        self.core.perf_profile.passive_res_max_hz = next.passive_res_max_hz;
        self.core.perf_profile.damper_control_decim = next.damper_control_decim;
        self.core.perf_profile.hall_fdn_lines = next.hall_fdn_lines;
        self.core.perf_profile.hertz_nonlinear = next.hertz_nonlinear;
        self.core.perf_profile.active_bridge_poles = next.active_bridge_poles;
        self.core.perf_profile.shank_flex_enabled = next.shank_flex_enabled;
        self.core.perf_profile.polynomial_adaa = actual_poly_adaa;

        !structural_mismatch
    }
}
