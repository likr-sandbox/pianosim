//! Stage 2: 230-string ZOH modal bank, SAV nonlinearity, and longitudinal coupling.
//!
//! M2 implements all 88 keys, 230 physical strings, and ~26,184 SIMD4-padded
//! modes using Exact Coupled-Form Incremental ZOH updated unconditionally on
//! every audio sample with zero heap allocations after [`StringBank::new`].

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod action_driver;
mod bank_energy;
pub mod bridge_coupled;
pub(crate) mod bridge_damp_kernels;
mod bridge_diag;
pub(crate) mod bridge_kernels;
pub(crate) mod bridge_open_kernels;
pub(crate) mod bridge_port;
mod bridge_port_build;
mod bridge_prefix_math;
mod bridge_prefix_step;
mod bridge_prefix_wdf;
pub(crate) mod bridge_sample_prep;
pub(crate) mod bridge_sav;
pub(crate) mod bridge_tails;
mod bridge_zones;
pub mod choir_map;
pub mod contact;
mod contact_port_sync;
pub(crate) mod contact_ports;
pub(crate) mod damping_law;
pub(crate) mod longitudinal;
pub(crate) mod math;
pub mod modal_soa;
pub(crate) mod phantom;
mod phantom_3x6;
pub(crate) mod sav_drift;
mod sav_kernels;
pub(crate) mod sav_pair;
pub(crate) mod sav_spec;
pub(crate) mod sav_tile;
mod sav_tile_3x6;
mod voicing_blob;
mod voicing_update;
pub mod zoh_kernel;

pub use bridge_port::BridgePortDriver;
pub use contact::KeyActionDriver;
pub use voicing_blob::{voicing_blob_len, VoicingPrepContext};

use piano_spec::{build_scale_design, VoicingSpec};

use crate::bridge_port::BridgePorts;
use crate::choir_map::ChoirMap;
use crate::contact_ports::ContactPorts;
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::sav_pair::SavPairQueue;
use crate::sav_tile::step_ringing_string_tile;
use crate::zoh_kernel::{
    accum_tile_stereo, inject_excitation, step_all_strings, step_string_subblock, step_string_tile,
    TILE,
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
    bridge_coupled: bool,
}

impl StringBank {
    fn build_internal(
        sample_rate: f32,
        voicing: &VoicingSpec,
        lossless: bool,
        bridge_coupled: bool,
    ) -> Self {
        let safe_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let sr64 = f64::from(safe_rate);
        let specs = build_scale_design(voicing);
        let choir_map = ChoirMap::build_with_specs(&specs, voicing);
        let soa =
            ModalSoa::build_with_mode(sr64, voicing, &specs, &choir_map, lossless, bridge_coupled);
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
            bridge_coupled,
        }
    }

    /// Allocates and precomputes permanent standalone SoA mode tables and physical SI
    /// dual-port projections for `sample_rate` and `voicing`.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::build_internal(sample_rate, voicing, false, false)
    }

    /// Allocates the string bank configured with intrinsic-only wire/air damping for
    /// coupled Stage 3 `BridgeBank` radiation (eliminating standalone double damping).
    #[must_use]
    pub fn new_bridge_coupled(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::build_internal(sample_rate, voicing, false, true)
    }

    /// Allocates a lossless (`r_hold = 1.0`) bank for SAV symplectic conservation checks.
    #[must_use]
    pub fn new_lossless(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::build_internal(sample_rate, voicing, true, false)
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
                self.ports.tl_v_active[str_idx] = u16::MAX;
                self.ports.tl_h_active[str_idx] = u16::MAX;
                self.ports.sav_dormant[str_idx] = false;
                self.ports.sav_low_nl_chunks[str_idx] = 0;
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
            self.lossless,
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
