//! Energy queries, accessors, and key lifecycle helpers for [`StringBank`] (`no_std`).

use piano_spec::TOTAL_STRINGS;

use crate::choir_map::ChoirMap;
use crate::sav_drift::C0_SAV;
use crate::zoh_kernel::flush_subnormals_range;
use crate::StringBank;

impl StringBank {
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
                self.ports.tl_v_active[s_idx] = u16::MAX;
                self.ports.tl_h_active[s_idx] = u16::MAX;
                self.ports.sav_dormant[s_idx] = false;
                self.ports.sav_low_nl_chunks[s_idx] = 0;
            }
        }
    }

    /// Retires struck contact state for all strings belonging to `key_idx` once
    /// vibration decays below silence, transitioning the key from ActiveContact to DampedSymp.
    #[inline]
    pub fn retire_key_struck_contact(&mut self, key_idx: usize) {
        if let Some(entry) = self.choir_map.key(key_idx) {
            for s_idx in entry.string_range.clone() {
                self.ports.struck_active[s_idx] = false;
                self.ports.sav_dormant[s_idx] = false;
                self.ports.sav_low_nl_chunks[s_idx] = 0;
                self.sav.reset_string(s_idx);
            }
        }
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
}
