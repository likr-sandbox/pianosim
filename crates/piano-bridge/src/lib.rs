//! Stage 3: reduced-mode bidirectional bridge coupling, Hertz contact compliance,
//! 8-zone spatial projection (with 9th keybed zone reserved), and 64-mode duplex scale bank.
//!
//! Operates strictly in `no_std` real-time DSP mode with zero heap allocations, zero locks,
//! and zero panics in [`BridgeBank::step_bridge_sample`].

#![cfg_attr(not(test), no_std)]

mod admittance_smooth;
pub mod admittance_ssm;
pub mod bridge_presets;
mod bridge_step;
pub mod duplex;
pub mod hertz_contact;
mod math;
pub mod zone_proj;

#[cfg(test)]
#[path = "duplex_tests.rs"]
mod duplex_tests;
#[cfg(test)]
#[path = "bridge_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "bridge_tests_pbdy.rs"]
mod tests_pbdy;
#[cfg(test)]
#[path = "bridge_tests_pbdy_edge.rs"]
mod tests_pbdy_edge;

use piano_spec::{BodyPreset, PresetReducedPole, VoicingSpec, TOTAL_STRINGS};

pub use admittance_ssm::{
    default_string_port_resistances, AdmittanceSsm, ReducedBridgePole, DEFAULT_Y_INF,
    NUM_REDUCED_MODES,
};
pub use duplex::{DuplexBank, DUPLEX_VEC_ZONES, NUM_DUPLEX_MODES, NUM_DUPLEX_VECS};
pub use hertz_contact::{
    hertz_force_f32, hertz_potential_f32, HertzBridgeContact, C_HERTZ_SAV, DEFAULT_HERTZ_K3,
    DEFAULT_HERTZ_K4,
};
pub use zone_proj::{
    build_zone_weights, gather_zones_to_strings, ZoneWeight, KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES,
    NUM_RADIATOR_ZONES, TREBLE_DUPLEX_ZONE_START,
};

use math::sanitize_sample_rate;
use zone_proj::ZoneRunCache;

/// Integrated Stage 3 bidirectional bridge bank coupling all 230 strings, 10 reduced
/// soundboard/bridge modes, nonlinear Hertz bridge-pin contact, and 64 treble duplex modes
/// across 8 physical bridge zones.
#[derive(Clone, Debug)]
pub struct BridgeBank {
    pub(crate) sample_rate: f32,
    pub(crate) ts: f64,
    pub(crate) zone_weights: [ZoneWeight; TOTAL_STRINGS],
    pub(crate) r0: [f32; TOTAL_STRINGS],
    pub(crate) zone_cache: ZoneRunCache,
    pub(crate) ssm: AdmittanceSsm,
    pub(crate) hertz: HertzBridgeContact,
    pub(crate) duplex: DuplexBank,
    pub(crate) fixed_end: bool,
}

impl BridgeBank {
    fn build(sample_rate: f32, mut ssm: AdmittanceSsm, duplex: DuplexBank) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        let r0 = default_string_port_resistances(safe_sr);
        let zone_weights = build_zone_weights();
        let hertz = HertzBridgeContact::new(safe_sr);
        ssm.set_port_resistances(&r0, &zone_weights, duplex.port_resistances());
        let zone_cache = ZoneRunCache::build(&zone_weights, &r0);
        Self {
            sample_rate: safe_sr,
            ts: 1.0 / f64::from(safe_sr),
            zone_weights,
            r0,
            zone_cache,
            ssm,
            hertz,
            duplex,
            fixed_end: false,
        }
    }

    /// Initializes the Stage 3 bridge bank with physical concert-grand defaults at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_voicing(sample_rate, &VoicingSpec::default())
    }

    /// Initializes the Stage 3 bridge bank for `voicing` at `sample_rate`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        Self::build(
            safe_sr,
            AdmittanceSsm::new(safe_sr),
            DuplexBank::new_with_voicing(safe_sr, voicing),
        )
    }

    /// Initializes the Stage 3 bridge bank in exact lossless mode (`zeta = 0`, `y_inf = 0`,
    /// `duplex_gain = 0`) while keeping conservative nonlinear `HertzBridgeContact` active.
    #[must_use]
    pub fn lossless_for_test(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        Self::build(
            safe_sr,
            AdmittanceSsm::new_lossless(safe_sr),
            DuplexBank::new_with_params(safe_sr, 0.0, true),
        )
    }

    #[cfg(test)]
    pub(crate) fn lossless_with_duplex_for_test(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        Self::build(
            safe_sr,
            AdmittanceSsm::new_lossless(safe_sr),
            DuplexBank::new_with_params(safe_sr, 1.0, true),
        )
    }

    /// Updates per-string WDF bridge port resistances `r0` (`[kg/s]`) and recomputes the
    /// precomputed `8x8` WDF junction matrices offline (`init` / reconfiguration only).
    pub fn set_port_resistances(&mut self, r0: &[f32; TOTAL_STRINGS]) {
        for (dst_r, &src) in self.r0.iter_mut().zip(r0.iter()) {
            *dst_r = if src.is_finite() { src.max(0.0) } else { 0.0 };
        }
        self.zone_cache.update_resistances(&self.r0);
        self.ssm
            .set_port_resistances(&self.r0, &self.zone_weights, self.duplex.port_resistances());
    }

    /// Reconfigures treble duplex scale coupling gain (`0.0` = disabled) and refreshes `8x8` WDF matrices.
    pub fn set_duplex_gain(&mut self, duplex_gain: f32) {
        let voicing = self.duplex.voicing().clone();
        self.duplex = DuplexBank::new_with_params(self.sample_rate, duplex_gain, false);
        self.duplex.retune_voicing(&voicing);
        self.ssm
            .set_port_resistances(&self.r0, &self.zone_weights, self.duplex.port_resistances());
    }

    /// Retunes Stage 3 treble duplex wire-segment resonances in place for `a4_freq_hz` (`[392.0, 466.2]` Hz).
    pub fn set_a4_freq_hz(&mut self, a4_freq_hz: f32) {
        let prev_r = *self.duplex.port_resistances();
        self.duplex.retune_a4(a4_freq_hz);
        if *self.duplex.port_resistances() != prev_r {
            self.ssm.set_port_resistances(
                &self.r0,
                &self.zone_weights,
                self.duplex.port_resistances(),
            );
        }
    }

    /// Retunes Stage 3 treble duplex wire-segment resonances in place for `voicing`
    /// and keeps `AdmittanceSsm` port resistances synchronized if `r_dup` changes.
    pub fn set_voicing(&mut self, voicing: &VoicingSpec) {
        let prev_r = *self.duplex.port_resistances();
        self.duplex.retune_voicing(voicing);
        if *self.duplex.port_resistances() != prev_r {
            self.ssm.set_port_resistances(
                &self.r0,
                &self.zone_weights,
                self.duplex.port_resistances(),
            );
        }
    }

    /// Replaces the Stage 3 reduced soundboard admittance `(y_inf, reduced_poles)` in place,
    /// resynchronizes `solve_8x8`, `solve_abr`, and `solve_phi` via `set_port_resistances`,
    /// and resets Stage 3 modal, Hertz contact, and duplex state to rest equilibrium.
    pub fn set_reduced_model(
        &mut self,
        y_inf: &[f32; NUM_BRIDGE_ZONES],
        reduced_poles: &[PresetReducedPole; NUM_REDUCED_MODES],
    ) {
        self.ssm.set_reduced_model(y_inf, reduced_poles);
        self.ssm
            .set_port_resistances(&self.r0, &self.zone_weights, self.duplex.port_resistances());
        self.reset();
    }

    /// Replaces the Stage 3 reduced soundboard admittance smoothly during active playback,
    /// preserving continuous 8-zone restoring force, velocity, Hertz contact, and duplex shimmer.
    pub fn set_reduced_model_smooth(
        &mut self,
        y_inf: &[f32; NUM_BRIDGE_ZONES],
        reduced_poles: &[PresetReducedPole; NUM_REDUCED_MODES],
    ) {
        self.ssm.set_reduced_model_smooth(y_inf, reduced_poles);
        self.ssm
            .set_port_resistances(&self.r0, &self.zone_weights, self.duplex.port_resistances());
        self.flush_subnormals();
    }

    /// Applies the Stage 3 reduced soundboard admittance `(y_inf, reduced_poles)` from `preset`,
    /// resynchronizes `8x8` WDF junction matrices, and resets Stage 3 dynamic state cleanly.
    pub fn set_body_preset(&mut self, preset: &BodyPreset) {
        self.set_reduced_model(&preset.y_inf, &preset.reduced_poles);
    }

    /// Toggles rigid fixed-end diagnostic mode (`v_br = 0`, `b_waves_out = -a_waves`, zero
    /// WDF inter-string coupling and zero duplex output) for F8 two-stage decay comparison.
    pub fn set_fixed_end(&mut self, fixed_end: bool) {
        self.fixed_end = fixed_end;
    }

    /// Returns `true` when rigid fixed-end diagnostic mode is active.
    #[must_use]
    #[inline]
    pub const fn is_fixed_end(&self) -> bool {
        self.fixed_end
    }

    /// Flushes `< 1e-20` denormals/subnormals across Stage 3 SSM, Hertz contact, and duplex states,
    /// allowing idle bypass to re-engage cleanly after decay.
    pub fn flush_subnormals(&mut self) {
        self.ssm.flush_subnormals();
        self.hertz.flush_subnormals();
        self.duplex.flush_subnormals();
    }

    /// Returns `true` when all Stage 3 states (SSM, Hertz contact, and duplex) are quiescent.
    #[must_use]
    #[inline]
    pub fn is_quiescent(&self) -> bool {
        self.ssm.is_quiescent() && self.hertz.is_quiescent() && self.duplex.is_quiescent()
    }

    /// Resets dynamic bridge, Hertz contact, and duplex states to rest equilibrium.
    pub fn reset(&mut self) {
        self.ssm.reset();
        self.hertz.reset();
        self.duplex.reset();
    }

    /// Configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Configured per-string WDF bridge port resistances `[kg/s]`.
    #[must_use]
    #[inline]
    pub const fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        &self.r0
    }

    /// Partition-of-unity mapping `(z0, w0, z1, w1)` for all 230 strings onto Zones `0..8`.
    #[must_use]
    #[inline]
    pub const fn zone_weights(&self) -> &[ZoneWeight; TOTAL_STRINGS] {
        &self.zone_weights
    }

    /// Public 10 reduced soundboard/bridge poles for M6 `piano-body-gen` pole cancellation.
    #[must_use]
    #[inline]
    pub const fn reduced_poles(&self) -> &[ReducedBridgePole; NUM_REDUCED_MODES] {
        self.ssm.poles()
    }

    /// Direct immutable view of the reduced soundboard admittance model.
    #[must_use]
    #[inline]
    pub const fn ssm(&self) -> &AdmittanceSsm {
        &self.ssm
    }

    /// Direct immutable view of the nonlinear Hertz bridge-pin contact model.
    #[must_use]
    #[inline]
    pub const fn hertz(&self) -> &HertzBridgeContact {
        &self.hertz
    }

    /// Direct immutable view of the 64-mode treble duplex scale resonator bank.
    #[must_use]
    #[inline]
    pub const fn duplex(&self) -> &DuplexBank {
        &self.duplex
    }

    /// Instantaneous stored energy in the 10 reduced soundboard/bridge modes [J].
    #[must_use]
    #[inline]
    pub fn ssm_energy_joules(&self) -> f64 {
        self.ssm.energy_joules()
    }

    /// Instantaneous SAV energy stored in the nonlinear Hertz bridge-pin contact [J].
    #[must_use]
    #[inline]
    pub fn hertz_energy_joules(&self) -> f64 {
        self.hertz.energy_joules()
    }

    /// Instantaneous stored energy across all 64 treble duplex scale modes [J].
    #[must_use]
    #[inline]
    pub fn duplex_energy_joules(&self) -> f64 {
        self.duplex.energy_joules()
    }

    /// Total instantaneous stored mechanical energy across bridge, Hertz contact, and duplex [J].
    #[must_use]
    #[inline]
    pub fn total_energy_joules(&self) -> f64 {
        self.ssm.energy_joules() + self.hertz.energy_joules() + self.duplex.energy_joules()
    }

    /// Cumulative dissipated energy in the reduced soundboard admittance [J].
    #[must_use]
    #[inline]
    pub fn dissipated_energy_joules(&self) -> f64 {
        self.ssm.dissipated_energy_joules()
    }
}
