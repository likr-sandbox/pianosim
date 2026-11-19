//! Off-thread Stage 2/3 tuning table precomputation and zero-allocation
//! linear-memory installer for non-blocking `AudioWorklet` voicing updates.

use alloc::vec::Vec;
use piano_spec::{PerfProfile, VoicingSpec, LONGITUDINAL_MAX_KEY_IDX, TOTAL_STRINGS};

use crate::voicing_blob_sav::{push_copy, push_f64_bits, push_sav_entry, SAV_ENTRY_FLOATS};
use crate::StringBank;

pub(crate) const VOICING_BLOB_MAGIC: u32 = 0x564F_4943; // "VOIC"
pub(crate) const VOICING_HEADER_LEN: usize = 11;

/// Computes the exact serialized `f32` length of a `VoicingBlob` for a bridge-coupled bank.
#[must_use]
pub fn voicing_blob_len(sample_rate: f32) -> usize {
    VoicingPrepContext::new(sample_rate).blob_len()
}

/// Computes the exact serialized `f32` length of a `VoicingBlob` under `perf`.
#[must_use]
pub fn voicing_blob_len_with_profile(sample_rate: f32, perf: &PerfProfile) -> usize {
    VoicingPrepContext::new_with_profile(sample_rate, perf).blob_len()
}

/// Off-thread Stage 2/3 tuning table builder for non-blocking `AudioWorklet` swaps.
pub struct VoicingPrepContext {
    bank: StringBank,
    out_blob: Vec<f32>,
}

impl VoicingPrepContext {
    /// Initializes a headless Stage 2/3 precomputation context at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_profile(sample_rate, &PerfProfile::full())
    }

    /// Initializes a headless Stage 2/3 precomputation context honoring `perf`.
    #[must_use]
    pub fn new_with_profile(sample_rate_hz: f32, perf: &PerfProfile) -> Self {
        let bank = StringBank::new_bridge_coupled_with_profile(
            sample_rate_hz,
            &VoicingSpec::default(),
            perf,
        );
        let len = Self::compute_blob_len(&bank);
        Self {
            bank,
            out_blob: Vec::with_capacity(len),
        }
    }

    /// Updates the Stage 2 `PerfProfile`, rebuilding internal tables if structural mode counts change.
    pub fn set_perf_profile(&mut self, perf: &PerfProfile) {
        let p = perf.sanitized();
        let cur = self.bank.perf_profile();
        if (p.horiz_max_hz - cur.horiz_max_hz).abs() > f32::EPSILON
            || (p.dead_mode_prune_db - cur.dead_mode_prune_db).abs() > f32::EPSILON
            || p.bass_phantom_sparse != cur.bass_phantom_sparse
        {
            self.bank = StringBank::new_bridge_coupled_with_profile(
                self.bank.sample_rate(),
                &VoicingSpec::default(),
                &p,
            );
            let len = Self::compute_blob_len(&self.bank);
            if self.out_blob.capacity() < len {
                self.out_blob.reserve(len - self.out_blob.capacity());
            }
        } else {
            self.bank.set_perf_profile_runtime(&p);
        }
    }

    /// Exact number of `f32` elements produced by [`Self::prepare_voicing`].
    #[must_use]
    pub fn blob_len(&self) -> usize {
        Self::compute_blob_len(&self.bank)
    }

    pub(crate) fn compute_blob_len(bank: &StringBank) -> usize {
        let soa_floats = bank.soa.d_coeff.len();
        let tl_floats = bank.bridge_ports.tl_soa.d_coeff.len();
        let coef_floats = bank.bridge_ports.coef.len() * 12;
        let long_floats = bank.long_bank.b_long_prec.len();
        let sav_count = bank
            .choir_map
            .strings()
            .iter()
            .filter(|e| usize::from(e.key_idx) < LONGITUDINAL_MAX_KEY_IDX)
            .count();
        VOICING_HEADER_LEN
            + 10 * soa_floats
            + 21 * TOTAL_STRINGS
            + long_floats
            + coef_floats
            + 3 * tl_floats
            + sav_count * SAV_ENTRY_FLOATS
    }

    /// Recomputes all tuned modal/port/SAV/WDF tables off-thread and packs them into `f32` blob.
    pub fn prepare_voicing(&mut self, voicing: &VoicingSpec) -> &[f32] {
        let v = voicing.sanitized();
        self.bank.update_voicing_in_place(&v);
        let b = &self.bank;
        self.out_blob.clear();
        let total_vecs = b.choir_map.total_vecs();
        let tl_vecs = b.bridge_ports.tl_soa.d_coeff.len() / 4;
        self.out_blob.extend_from_slice(&[
            f32::from_bits(VOICING_BLOB_MAGIC),
            b.sample_rate,
            v.hammer_hardness,
            v.unison_detune_scale,
            v.stretch_amount,
            v.string_length_scale,
            v.a4_freq_hz,
            f32::from_bits(total_vecs as u32),
            f32::from_bits(tl_vecs as u32),
            f32::from_bits(b.bridge_ports.coef.len() as u32),
            f32::from_bits(b.long_bank.b_long_prec.len() as u32),
        ]);
        for s in [
            b.soa.d_coeff.as_slice(),
            b.soa.s_coeff.as_slice(),
            b.soa.c_bridge.as_slice(),
            b.soa.b_exc.as_slice(),
            b.ports.b_strike.as_slice(),
            b.ports.a_strike.as_slice(),
            b.ports.b_strike_diff.as_slice(),
            b.ports.a_strike_diff.as_slice(),
            b.ports.b_damp.as_slice(),
            b.ports.a_damp.as_slice(),
        ] {
            push_copy(&mut self.out_blob, s);
        }
        for s in [
            &b.ports.alpha_vel[..],
            &b.ports.r_port_strike[..],
            &b.ports.r_strike_00[..],
            &b.ports.r_strike_01[..],
            &b.ports.r_strike_11[..],
            &b.ports.r_port_damp_v[..],
            &b.ports.r_port_damp_h[..],
        ] {
            push_copy(&mut self.out_blob, s);
        }
        for &g in &b.ports.gamma_s {
            push_f64_bits(&mut self.out_blob, g);
        }
        for &bs in &b.ports.beta_s {
            push_f64_bits(&mut self.out_blob, bs);
        }
        push_copy(&mut self.out_blob, b.long_bank.b_long_prec.as_slice());
        for &x in &b.long_bank.prec_force_scale {
            push_f64_bits(&mut self.out_blob, x);
        }
        for &x in &b.long_bank.prec_force_cap_n {
            push_f64_bits(&mut self.out_blob, x);
        }
        for c in &b.bridge_ports.coef {
            self.out_blob.extend_from_slice(&c.d.to_array());
            self.out_blob.extend_from_slice(&c.s.to_array());
            self.out_blob.extend_from_slice(&c.beta.to_array());
        }
        for s in [
            b.bridge_ports.tl_soa.d_coeff.as_slice(),
            b.bridge_ports.tl_soa.s_coeff.as_slice(),
            b.bridge_ports.tl_soa.c_bridge.as_slice(),
        ] {
            push_copy(&mut self.out_blob, s);
        }
        for s in [
            &b.bridge_ports.r0[..],
            &b.bridge_ports.a_wave_gains[..],
            &b.bridge_ports.r_damp_prefix_v[..],
            &b.bridge_ports.r_damp_prefix_h[..],
            &b.bridge_ports.tail_n_trans[..],
            &b.bridge_ports.tail_n_long[..],
        ] {
            push_copy(&mut self.out_blob, s);
        }
        for (s_idx, entry) in b.choir_map.strings().iter().enumerate() {
            if usize::from(entry.key_idx) < LONGITUDINAL_MAX_KEY_IDX {
                push_sav_entry(
                    &mut self.out_blob,
                    &b.sav.specs[s_idx],
                    &b.sav.slopes[s_idx],
                );
            }
        }
        &self.out_blob
    }

    /// Returns the most recently prepared voicing blob slice.
    #[must_use]
    pub fn out_blob(&self) -> &[f32] {
        &self.out_blob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voicing_blob_install_matches_in_place_and_keeps_live_ringing_parity() {
        let sr = 48_000.0;
        let mut ref_bank = StringBank::new_bridge_coupled(sr, &VoicingSpec::default());
        let mut fast_bank = StringBank::new_bridge_coupled(sr, &VoicingSpec::default());
        let mut prep = VoicingPrepContext::new(sr);

        let k_a4 = 48usize;
        ref_bank.apply_excitation_sample(k_a4, [1.0, 1.0, 1.0], 8.0);
        fast_bank.apply_excitation_sample(k_a4, [1.0, 1.0, 1.0], 8.0);

        for a4_hz in [415.0_f32, 442.0] {
            let v = VoicingSpec::from_globals_with_a4(0.75, 1.85, 1.35, 0.92, a4_hz);
            ref_bank.update_voicing_in_place(&v);
            let blob = prep.prepare_voicing(&v);
            assert_eq!(blob.len(), voicing_blob_len(sr));
            assert!(fast_bank.install_precomputed_voicing(&v, blob));

            assert_eq!(
                ref_bank.bridge_port_resistances(),
                fast_bank.bridge_port_resistances()
            );
            assert_eq!(
                ref_bank.soa.d_coeff.as_slice(),
                fast_bank.soa.d_coeff.as_slice()
            );
            assert_eq!(
                ref_bank.soa.s_coeff.as_slice(),
                fast_bank.soa.s_coeff.as_slice()
            );
            assert_eq!(ref_bank.ports.gamma_s, fast_bank.ports.gamma_s);
            assert_eq!(ref_bank.ports.r_strike_00, fast_bank.ports.r_strike_00);
            assert_eq!(ref_bank.ports.r_strike_01, fast_bank.ports.r_strike_01);
            assert_eq!(ref_bank.ports.r_strike_11, fast_bank.ports.r_strike_11);
            assert_eq!(
                ref_bank.ports.b_strike_diff.as_slice(),
                fast_bank.ports.b_strike_diff.as_slice()
            );
            assert_eq!(
                ref_bank.ports.a_strike_diff.as_slice(),
                fast_bank.ports.a_strike_diff.as_slice()
            );
            assert_eq!(
                ref_bank.string_linear_energy_joules(144),
                fast_bank.string_linear_energy_joules(144)
            );
            let mismatched = VoicingSpec::from_globals_with_a4(0.75, 1.85, 1.35, 0.92, 440.0);
            assert!(!fast_bank.install_precomputed_voicing(&mismatched, blob));
            assert!(!fast_bank.install_precomputed_voicing(&v, &[0.0; 12]));
        }
    }

    #[test]
    fn time_sliced_per_key_install_defers_ports_until_finish_and_matches_batch() {
        let sr = 48_000.0;
        let perf = PerfProfile::balanced();
        let mut batch_bank =
            StringBank::new_bridge_coupled_with_profile(sr, &VoicingSpec::default(), &perf);
        let mut sliced_bank =
            StringBank::new_bridge_coupled_with_profile(sr, &VoicingSpec::default(), &perf);
        let mut direct_bank =
            StringBank::new_bridge_coupled_with_profile(sr, &VoicingSpec::default(), &perf);
        let mut prep = VoicingPrepContext::new_with_profile(sr, &perf);

        batch_bank.apply_excitation_sample(0, [1.0, 0.0, 0.0], 6.0);
        sliced_bank.apply_excitation_sample(0, [1.0, 0.0, 0.0], 6.0);
        batch_bank.apply_excitation_sample(48, [1.0, 1.0, 1.0], 6.0);
        sliced_bank.apply_excitation_sample(48, [1.0, 1.0, 1.0], 6.0);

        let v = VoicingSpec::from_globals_with_a4(0.80, 1.75, 1.25, 0.94, 415.0);
        let blob = prep.prepare_voicing(&v);
        let orig_r0 = *sliced_bank.bridge_port_resistances();
        let orig_aw = sliced_bank.bridge_ports.a_wave_gains;

        assert!(batch_bank.install_precomputed_voicing(&v, blob));

        let scale_specs = piano_spec::build_scale_design(&v);
        assert!(sliced_bank.begin_voicing_install(&v, &scale_specs, blob));
        assert!(!sliced_bank.install_voicing_key(88, blob));
        assert!(!sliced_bank.install_voicing_key(0, &[0.0; 12]));
        assert!(!sliced_bank.finish_voicing_ports(&[0.0; 12]));

        for chunk_start in (0..piano_spec::NUM_KEYS).step_by(8) {
            for k in chunk_start..(chunk_start + 8).min(piano_spec::NUM_KEYS) {
                assert!(sliced_bank.install_voicing_key(k, blob));
            }
            // r0 and a_wave_gains must remain atomic (unchanged) until finish_voicing_ports.
            assert_eq!(sliced_bank.bridge_port_resistances(), &orig_r0);
            assert_eq!(sliced_bank.bridge_ports.a_wave_gains, orig_aw);
        }
        sliced_bank.bridge_ports.a_waves_dirty = false;
        assert!(sliced_bank.finish_voicing_ports(blob));
        assert!(sliced_bank.bridge_ports.a_waves_dirty);

        // Verify bit-exact parity between batch install and 8-keys/block sliced install.
        assert_eq!(
            batch_bank.bridge_port_resistances(),
            sliced_bank.bridge_port_resistances()
        );
        assert_eq!(
            batch_bank.bridge_ports.a_wave_gains,
            sliced_bank.bridge_ports.a_wave_gains
        );
        assert_eq!(
            batch_bank.soa.d_coeff.as_slice(),
            sliced_bank.soa.d_coeff.as_slice()
        );
        assert_eq!(
            batch_bank.soa.s_coeff.as_slice(),
            sliced_bank.soa.s_coeff.as_slice()
        );
        assert_eq!(
            batch_bank.bridge_ports.tl_soa.d_coeff.as_slice(),
            sliced_bank.bridge_ports.tl_soa.d_coeff.as_slice()
        );
        assert_eq!(
            batch_bank.long_bank.b_long_prec.as_slice(),
            sliced_bank.long_bank.b_long_prec.as_slice()
        );
        assert_eq!(batch_bank.ports.gamma_s, sliced_bank.ports.gamma_s);
        assert_eq!(batch_bank.passive_v_vecs, sliced_bank.passive_v_vecs);
        assert_eq!(batch_bank.passive_h_vecs, sliced_bank.passive_h_vecs);
        assert_eq!(
            batch_bank.string_linear_energy_joules(0),
            sliced_bank.string_linear_energy_joules(0)
        );
        assert_eq!(
            batch_bank.string_linear_energy_joules(144),
            sliced_bank.string_linear_energy_joules(144)
        );

        // Verify direct install_voicing_key + finish_voicing_ports without begin_voicing_install.
        for k in 0..piano_spec::NUM_KEYS {
            assert!(direct_bank.install_voicing_key(k, blob));
        }
        assert!(direct_bank.finish_voicing_ports(blob));
        assert_eq!(
            batch_bank.bridge_port_resistances(),
            direct_bank.bridge_port_resistances()
        );
        assert_eq!(batch_bank.last_voicing.a4_freq_hz, 415.0);
        assert_eq!(direct_bank.last_voicing.a4_freq_hz, 415.0);
    }
}
