//! Stage 2: 230-string ZOH modal bank, SAV nonlinearity, and longitudinal coupling.
//!
//! M2 implements all 88 keys, 230 physical strings, and ~26,184 SIMD4-padded
//! modes using Exact Coupled-Form Incremental ZOH updated unconditionally on
//! every audio sample with zero heap allocations after [`StringBank::new`].

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod choir_map;
pub mod contact;
pub(crate) mod contact_ports;
pub mod modal_soa;
pub mod zoh_kernel;

pub use contact::KeyActionDriver;

use piano_spec::{build_scale_design, VoicingSpec};

use crate::choir_map::ChoirMap;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::ModalSoa;
use crate::zoh_kernel::{flush_subnormals_range, inject_excitation, step_all_strings};

/// All-88-key, 230-string, ~26k-mode continuous SIMD coupled-form ZOH bank.
#[derive(Clone, Debug)]
pub struct StringBank {
    sample_rate: f32,
    choir_map: ChoirMap,
    soa: ModalSoa,
    ports: ContactPorts,
}

impl StringBank {
    /// Allocates and precomputes permanent undamped SoA mode tables and physical SI
    /// dual-port projections for `sample_rate` and `voicing`.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let safe_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let specs = build_scale_design(voicing);
        let choir_map = ChoirMap::build_with_specs(&specs, voicing);
        let soa = ModalSoa::build(f64::from(safe_rate), voicing, &specs, &choir_map);
        let ports = ContactPorts::build(f64::from(safe_rate), voicing, &specs, &choir_map);

        Self {
            sample_rate: safe_rate,
            choir_map,
            soa,
            ports,
        }
    }

    /// Resets all modal state buffers and cached contact ports without reallocating.
    pub fn reset(&mut self) {
        self.soa.reset_resting();
        self.ports.reset_resting();
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
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return;
        };
        if !force.is_finite() || force == 0.0 {
            return;
        }
        let total_vecs = self.choir_map.total_vecs();
        let u_vecs = &mut self.soa.u.as_f32x4_slice_mut()[..total_vecs];
        let b_vecs = &self.soa.b_exc.as_f32x4_slice()[..total_vecs];
        for (str_in_choir, str_idx) in key_entry.string_range.clone().enumerate() {
            let w = string_weights[str_in_choir];
            if w.is_finite() && w != 0.0 {
                let range = self.choir_map.strings()[str_idx].all_vecs();
                inject_excitation(&mut u_vecs[range.clone()], &b_vecs[range], force * w);
                self.ports.may_be_nonzero[str_idx] = true;
                self.ports.ports_synced[str_idx] = false;
            }
        }
    }

    /// Steps all 230 strings unconditionally for one audio frame and returns `(L, R)`.
    /// Call [`StringBank::flush_subnormals`] once per subblock after sample loops.
    #[must_use]
    #[inline]
    pub fn render_frame(&mut self) -> (f32, f32) {
        let frame = step_all_strings(&mut self.soa, &self.choir_map);
        for (s_idx, flag) in self.ports.ports_synced.iter_mut().enumerate() {
            if self.ports.may_be_nonzero[s_idx] {
                *flag = false;
            }
        }
        frame
    }

    /// Flushes subnormals (`< 1e-20`) in `u` and `v` and clears dormant string flags.
    #[inline]
    pub fn flush_subnormals(&mut self) {
        for (s_idx, entry) in self.choir_map.strings().iter().enumerate() {
            if !flush_subnormals_range(&mut self.soa, entry.all_vecs()) {
                self.ports.clear_string_state(s_idx);
            }
        }
    }

    /// Renders an uncoupled stereo subblock across all 230 strings in `TILE = 8` mode
    /// and flushes subnormals once at the end of the subblock.
    pub fn render_subblock(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }
        zoh_kernel::step_all_tiles(
            &mut self.soa,
            &self.choir_map,
            &mut out_l[..frames],
            &mut out_r[..frames],
        );
        for (s_idx, flag) in self.ports.ports_synced.iter_mut().enumerate() {
            if self.ports.may_be_nonzero[s_idx] {
                *flag = false;
            }
        }
    }

    /// Steps all 88 keys / 230 strings across `out_l`/`out_r` with closed-loop Stage 1
    /// hammer and unilateral damper port coupling (`Newtons`, `meters`, `m/s`).
    pub fn render_subblock_coupled(
        &mut self,
        driver: &mut impl KeyActionDriver,
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
        contact::render_subblock_coupled_core(
            &mut self.soa,
            &mut self.ports,
            &self.choir_map,
            driver,
            &mut out_l[..frames],
            &mut out_r[..frames],
        );
    }

    /// Diagnostic helper returning `(vert_rms, horiz_rms)` of modal bridge velocity
    /// `v * c_bridge` across `key_idx`'s strings at the current instant.
    #[must_use]
    pub fn measure_key_polarization_rms(&self, key_idx: usize) -> (f32, f32) {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return (0.0, 0.0);
        };
        let v = self.soa.v.as_f32x4_slice();
        let c = self.soa.c_bridge.as_f32x4_slice();
        let strings = &self.choir_map.strings()[key_entry.string_range.clone()];
        let calc = |sel: fn(&choir_map::StringMapEntry) -> core::ops::Range<usize>| {
            let (mut sum, mut cnt) = (0.0_f64, 0usize);
            for entry in strings {
                for idx in sel(entry) {
                    for val in (v[idx] * c[idx]).to_array() {
                        sum += f64::from(val) * f64::from(val);
                    }
                    cnt += 4;
                }
            }
            if cnt > 0 {
                crate::modal_soa::sqrt_f64(sum / cnt as f64) as f32
            } else {
                0.0
            }
        };
        (
            calc(|e| e.vert_vecs.clone()),
            calc(|e| e.horiz_vecs.clone()),
        )
    }

    #[cfg(test)]
    pub(crate) fn render_key_subblock(
        &mut self,
        key_idx: usize,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let key_entry = self.choir_map.key(key_idx).expect("valid key");
        zoh_kernel::step_key_subblock(
            &mut self.soa,
            key_entry,
            self.choir_map.strings(),
            out_l,
            out_r,
        );
        for s_idx in key_entry.string_range.clone() {
            if self.ports.may_be_nonzero[s_idx] {
                self.ports.ports_synced[s_idx] = false;
            }
        }
    }
}

#[cfg(test)]
#[path = "bank_tests.rs"]
mod tests;
