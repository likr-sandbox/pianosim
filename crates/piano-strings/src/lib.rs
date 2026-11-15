//! Stage 2: 230-string ZOH modal bank, SAV nonlinearity, and longitudinal coupling.
//!
//! M2 implements all 88 keys, 230 physical strings, and ~26,184 SIMD4-padded
//! modes using Exact Coupled-Form Incremental ZOH updated unconditionally on
//! every audio sample with zero heap allocations after [`StringBank::new`].

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod bridge_coupled;
mod bridge_diag;
pub(crate) mod bridge_kernels;
pub(crate) mod bridge_port;
pub(crate) mod bridge_sav;
pub(crate) mod bridge_tails;
mod bridge_zones;
pub mod choir_map;
pub mod contact;
pub(crate) mod contact_ports;
pub(crate) mod longitudinal;
pub mod modal_soa;
pub(crate) mod phantom;
pub(crate) mod sav_drift;
pub(crate) mod sav_pair;
pub(crate) mod sav_spec;
pub(crate) mod sav_tile;
mod voicing_update;
pub mod zoh_kernel;

pub use bridge_port::BridgePortDriver;
pub use contact::KeyActionDriver;

use piano_spec::{build_scale_design, VoicingSpec, TOTAL_STRINGS};

use crate::bridge_port::BridgePorts;
use crate::choir_map::ChoirMap;
use crate::contact_ports::ContactPorts;
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::{SavBank, C0_SAV};
use crate::sav_pair::SavPairQueue;
use crate::sav_tile::step_ringing_string_tile;
use crate::zoh_kernel::{
    accum_tile_stereo, flush_subnormals_range, inject_excitation, step_all_strings,
    step_string_subblock, step_string_tile, TILE,
};

/// All-88-key, 230-string, ~26k-mode continuous SIMD coupled-form ZOH bank.
#[derive(Clone, Debug)]
pub struct StringBank {
    sample_rate: f32,
    choir_map: ChoirMap,
    soa: ModalSoa,
    ports: ContactPorts,
    bridge_ports: BridgePorts,
    long_bank: LongitudinalBank,
    sav: SavBank,
    longitudinal_enabled: bool,
    sav_phantom_enabled: bool,
    soa_dirty: bool,
    bp_dirty: bool,
    lossless: bool,
}

impl StringBank {
    fn build_internal(sample_rate: f32, voicing: &VoicingSpec, lossless: bool) -> Self {
        let safe_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let sr64 = f64::from(safe_rate);
        let specs = build_scale_design(voicing);
        let choir_map = ChoirMap::build_with_specs(&specs, voicing);
        let soa = ModalSoa::build_with_lossless(sr64, voicing, &specs, &choir_map, lossless);
        let ports = ContactPorts::build(sr64, voicing, &specs, &choir_map);
        let bridge_ports = BridgePorts::build(sr64, voicing, &specs, &choir_map, &ports, &soa);
        let long_bank = LongitudinalBank::build(sr64, &specs, &choir_map, &ports.gamma_s);
        let sav = SavBank::build(
            sr64,
            voicing,
            &specs,
            &choir_map,
            &ports.gamma_s,
            &ports.beta_s,
        );

        Self {
            sample_rate: safe_rate,
            choir_map,
            soa,
            ports,
            bridge_ports,
            long_bank,
            sav,
            longitudinal_enabled: true,
            sav_phantom_enabled: true,
            soa_dirty: false,
            bp_dirty: false,
            lossless,
        }
    }

    /// Allocates and precomputes permanent undamped SoA mode tables and physical SI
    /// dual-port projections for `sample_rate` and `voicing`.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::build_internal(sample_rate, voicing, false)
    }

    /// Allocates a lossless (`r_hold = 1.0`) bank for SAV symplectic conservation checks.
    #[must_use]
    pub fn new_lossless(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::build_internal(sample_rate, voicing, true)
    }

    /// Enables or disables Stage 2 longitudinal coupling and Bank-Sujbert SAV phantom coupling.
    pub fn set_nonlinear_coupling(
        &mut self,
        longitudinal_enabled: bool,
        sav_phantom_enabled: bool,
    ) {
        self.longitudinal_enabled = longitudinal_enabled;
        self.sav_phantom_enabled = sav_phantom_enabled;
    }

    /// Returns total physical SAV Hamiltonian of string `s_idx` in Joules:
    /// `0.5 * beta_s * (sum(u^2 + v^2) + r_s^2 - C_0)`.
    #[must_use]
    pub fn string_sav_energy_joules(&self, s_idx: usize) -> f64 {
        if s_idx >= TOTAL_STRINGS {
            return 0.0;
        }
        let r = self.sav.r_sav[s_idx];
        self.string_linear_energy_joules(s_idx) + 0.5 * self.ports.beta_s[s_idx] * (r * r - C0_SAV)
    }

    /// Returns summed physical SAV Hamiltonian of all strings belonging to `key_idx` in Joules.
    #[must_use]
    pub fn key_sav_energy_joules(&self, key_idx: usize) -> f64 {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return 0.0;
        };
        key_entry
            .string_range
            .clone()
            .map(|s_idx| self.string_sav_energy_joules(s_idx))
            .sum()
    }

    /// Returns summed physical SAV Hamiltonian across all 230 strings in Joules.
    #[must_use]
    pub fn total_sav_energy_joules(&self) -> f64 {
        (0..TOTAL_STRINGS)
            .map(|s_idx| self.string_sav_energy_joules(s_idx))
            .sum()
    }

    /// Returns instantaneous nonlinear strain/phantom potential energy `beta_s * E_nl(v)` of
    /// `key_idx` in Joules.
    #[must_use]
    pub fn key_nonlinear_energy_joules(&self, key_idx: usize) -> f64 {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return 0.0;
        };
        let strings = self.choir_map.strings();
        key_entry
            .string_range
            .clone()
            .map(|s_idx| {
                if self.soa_dirty {
                    self.sav.string_nonlinear_energy_joules_bp(
                        s_idx,
                        &self.bridge_ports,
                        self.ports.beta_s[s_idx],
                        self.longitudinal_enabled,
                    )
                } else {
                    self.sav.string_nonlinear_energy_joules(
                        s_idx,
                        &strings[s_idx],
                        &self.soa,
                        self.ports.beta_s[s_idx],
                        self.longitudinal_enabled,
                    )
                }
            })
            .sum()
    }

    /// Resets all modal state buffers, cached contact ports, and `r_sav = 1.0` without reallocating.
    pub fn reset(&mut self) {
        self.soa.reset_resting();
        self.ports.reset_resting();
        self.bridge_ports.reset_resting();
        self.sav.reset_all();
        self.soa_dirty = false;
        self.bp_dirty = false;
    }

    /// Returns the configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the total SIMD4-padded modal slot count (`~26,184`).
    #[must_use]
    #[inline]
    pub const fn total_modes(&self) -> usize {
        self.choir_map.total_padded_modes()
    }

    /// Returns the total unpadded active mode count (`~25,184`).
    #[must_use]
    #[inline]
    pub const fn active_unpadded_modes(&self) -> usize {
        self.choir_map.active_unpadded_modes()
    }

    /// Returns the total `F32x4` vector count (`~6,546`).
    #[must_use]
    #[inline]
    pub const fn total_vecs(&self) -> usize {
        self.choir_map.total_vecs()
    }

    /// Returns the 88-key / 230-string / vector-range mapping.
    #[must_use]
    #[inline]
    pub const fn choir_map(&self) -> &ChoirMap {
        &self.choir_map
    }

    /// Marks all strings belonging to `key_idx` as potentially carrying non-zero
    /// modal state and invalidates cached dual-port readouts until resynchronized.
    pub fn mark_key_possibly_active(&mut self, key_idx: usize) {
        if let Some(key_entry) = self.choir_map.key(key_idx) {
            for s_idx in key_entry.string_range.clone() {
                self.ports.may_be_nonzero[s_idx] = true;
                self.ports.tail_nonzero[s_idx] = true;
                self.ports.struck_active[s_idx] = true;
                self.ports.ports_synced[s_idx] = false;
            }
        }
    }

    /// Injects a single-sample force impulse into `key_idx` across its choir strings
    /// weighted by `string_weights`. Intended for sample-accurate test/external impulses.
    pub fn apply_excitation_sample(
        &mut self,
        key_idx: usize,
        string_weights: [f32; 3],
        force: f32,
    ) {
        let Some(string_range) = self.choir_map.key(key_idx).map(|k| k.string_range.clone()) else {
            return;
        };
        if !force.is_finite() || force == 0.0 {
            return;
        }
        self.ensure_soa_mode();
        self.bp_dirty = true;
        let total_vecs = self.choir_map.total_vecs();
        let u_vecs = &mut self.soa.u.as_f32x4_slice_mut()[..total_vecs];
        let b_vecs = &self.soa.b_exc.as_f32x4_slice()[..total_vecs];
        for (str_in_choir, str_idx) in string_range.enumerate() {
            let w = string_weights[str_in_choir];
            if w.is_finite() && w != 0.0 {
                let range = self.choir_map.strings()[str_idx].all_vecs();
                inject_excitation(&mut u_vecs[range.clone()], &b_vecs[range], force * w);
                self.ports.may_be_nonzero[str_idx] = true;
                self.ports.tail_nonzero[str_idx] = true;
                self.ports.struck_active[str_idx] = true;
                self.ports.ports_synced[str_idx] = false;
            }
        }
    }

    /// Steps all 230 strings unconditionally for one audio frame and returns `(L, R)`.
    /// Call [`StringBank::flush_subnormals`] once per subblock after sample loops.
    #[must_use]
    #[inline]
    pub fn render_frame(&mut self) -> (f32, f32) {
        self.ensure_soa_mode();
        self.bp_dirty = true;
        if !self.sav_phantom_enabled && self.longitudinal_enabled {
            let frame = step_all_strings(&mut self.soa, &self.choir_map);
            for (s_idx, flag) in self.ports.ports_synced.iter_mut().enumerate() {
                if self.ports.may_be_nonzero[s_idx] {
                    *flag = false;
                }
            }
            return frame;
        }
        let (mut l1, mut r1) = ([0.0_f32; 1], [0.0_f32; 1]);
        for (s_idx, entry) in self.choir_map.strings().iter().enumerate() {
            if self.ports.may_be_nonzero[s_idx] {
                step_ringing_string_tile(
                    &mut self.soa,
                    &mut self.sav,
                    s_idx,
                    entry,
                    self.longitudinal_enabled,
                    self.sav_phantom_enabled,
                    &mut l1,
                    &mut r1,
                );
                self.ports.ports_synced[s_idx] = false;
            } else {
                let acc = step_string_tile(&mut self.soa, entry.all_vecs(), 1);
                let (mut tile_l, mut tile_r) = ([0.0_f32; TILE], [0.0_f32; TILE]);
                accum_tile_stereo(&acc, entry.pan_l, entry.pan_r, 1, &mut tile_l, &mut tile_r);
                l1[0] += tile_l[0];
                r1[0] += tile_r[0];
            }
        }
        (l1[0], r1[0])
    }

    /// Flushes subnormals (`< 1e-20`) in `u` and `v` and clears dormant string flags.
    #[inline]
    pub fn flush_subnormals(&mut self) {
        self.ensure_soa_mode();
        for (s_idx, entry) in self.choir_map.strings().iter().enumerate() {
            if !flush_subnormals_range(&mut self.soa, entry.all_vecs()) {
                self.ports.clear_string_state(s_idx);
                self.sav.reset_string(s_idx);
            }
        }
    }

    /// Renders an uncoupled stereo subblock across all 230 strings in `SUB = 128` mode
    /// and flushes subnormals once at the end of the subblock.
    pub fn render_subblock(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        self.ensure_soa_mode();
        self.bp_dirty = true;
        let frames = out_l.len().min(out_r.len());
        out_l.fill(0.0);
        out_r.fill(0.0);
        let strings = self.choir_map.strings();
        let mut base = 0usize;
        while base < frames {
            let sub = (frames - base).min(128);
            let (sub_l, sub_r) = (&mut out_l[base..base + sub], &mut out_r[base..base + sub]);
            let mut sav_pairs =
                SavPairQueue::new(self.longitudinal_enabled, self.sav_phantom_enabled);
            for (s_idx, entry) in strings.iter().enumerate() {
                if self.ports.may_be_nonzero[s_idx] {
                    sav_pairs.push_ringing(
                        &mut self.soa,
                        &mut self.sav,
                        strings,
                        s_idx,
                        sub_l,
                        sub_r,
                    );
                    self.ports.ports_synced[s_idx] = false;
                } else {
                    step_string_subblock(
                        &mut self.soa,
                        entry.all_vecs(),
                        entry.pan_l,
                        entry.pan_r,
                        sub_l,
                        sub_r,
                    );
                }
            }
            sav_pairs.flush(&mut self.soa, &mut self.sav, strings, sub_l, sub_r);
            base += sub;
        }
        self.flush_subnormals();
    }

    /// Steps all 88 keys / 230 strings across `out_l`/`out_r` with closed-loop Stage 1
    /// hammer and unilateral damper port coupling (`Newtons`, `meters`, `m/s`).
    pub fn render_subblock_coupled(
        &mut self,
        driver: &mut impl KeyActionDriver,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        self.ensure_soa_mode();
        self.bp_dirty = true;
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }
        contact::render_subblock_coupled_core(
            &mut self.soa,
            &mut self.ports,
            &self.long_bank,
            &mut self.sav,
            &self.choir_map,
            driver,
            self.longitudinal_enabled,
            self.sav_phantom_enabled,
            &mut out_l[..frames],
            &mut out_r[..frames],
        );
    }

    /// Steps all 88 keys / 230 strings unconditionally across `out_l`/`out_r` with
    /// Stage 1 action coupling and Stage 3 per-sample WDF bridge scattering.
    pub fn render_subblock_bridge_coupled(
        &mut self,
        action: &mut impl KeyActionDriver,
        bridge: &mut impl BridgePortDriver,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }
        if self.bp_dirty {
            self.bridge_ports
                .pack_all_from_soa(&self.soa, self.choir_map.strings());
            self.bp_dirty = false;
        }
        self.soa_dirty = true;
        bridge_coupled::render_subblock_bridge_coupled_core(
            &mut self.ports,
            &mut self.bridge_ports,
            &self.long_bank,
            &mut self.sav,
            &self.choir_map,
            action,
            bridge,
            self.longitudinal_enabled,
            self.sav_phantom_enabled,
            true,
            &mut out_l[..frames],
            &mut out_r[..frames],
            None,
        );
    }
}

#[cfg(test)]
#[path = "bank_tests.rs"]
mod tests;

#[cfg(test)]
mod m5_tests;
