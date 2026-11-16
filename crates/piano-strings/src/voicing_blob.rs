//! Off-thread Stage 2/3 tuning table precomputation and zero-allocation
//! linear-memory installer for non-blocking `AudioWorklet` voicing updates.

use alloc::vec::Vec;
use piano_simd::F32x4;
use piano_spec::{VoicingSpec, LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

use crate::phantom::{
    MAX_PHANTOM_LONG_MODES, MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_MODES,
    MAX_PHANTOM_TRANSVERSE_VECS,
};
use crate::sav_spec::{SavEnergySlopes, SavStringSpec};
use crate::voicing_update::rescale_active_string_uv;
use crate::StringBank;

const VOICING_BLOB_MAGIC: u32 = 0x564F_4943; // "VOIC"
const VOICING_HEADER_LEN: usize = 10;

#[inline]
fn push_f64_bits(out: &mut Vec<f32>, val: f64) {
    let bits = val.to_bits();
    out.push(f32::from_bits(bits as u32));
    out.push(f32::from_bits((bits >> 32) as u32));
}

#[inline]
fn read_f64_bits(src: &[f32], cur: &mut usize) -> f64 {
    let lo = u64::from(src[*cur].to_bits());
    let hi = u64::from(src[*cur + 1].to_bits());
    *cur += 2;
    f64::from_bits(lo | (hi << 32))
}

#[inline]
fn push_vec4s(out: &mut Vec<f32>, slice: &[F32x4]) {
    for v in slice {
        out.extend_from_slice(&v.to_array());
    }
}

#[inline]
fn read_vec4s(src: &[f32], cur: &mut usize, dst: &mut [F32x4]) {
    for d in dst {
        *d = F32x4::from_slice(&src[*cur..*cur + 4]);
        *cur += 4;
    }
}

#[inline]
fn push_copy(out: &mut Vec<f32>, src: &[f32]) {
    out.extend_from_slice(src);
}

#[inline]
fn read_copy(src: &[f32], cur: &mut usize, dst: &mut [f32]) {
    dst.copy_from_slice(&src[*cur..*cur + dst.len()]);
    *cur += dst.len();
}

fn push_sav_entry(out: &mut Vec<f32>, sp: &SavStringSpec, sl: &SavEnergySlopes) {
    for &u in &[
        sp.n_vh, sp.m_ph, sp.nv_vecs, sp.nh_vecs, sp.nl_vecs, sl.nv_c, sl.nh_c,
    ] {
        out.push(f32::from_bits(u as u32));
    }
    for &f in &[sp.length_m, sp.quarter_l, sp.kappa_0, sp.kappa_ph] {
        push_f64_bits(out, f);
    }
    for arr in [
        &sp.sigma_v4,
        &sp.omega_sigma_v4,
        &sp.mask_v4,
        &sp.sigma_h4,
        &sp.omega_sigma_h4,
        &sp.mask_h4,
    ] {
        push_vec4s(out, arr);
    }
    for arr in [
        &sp.alpha_l4,
        &sp.half_ql_mask4,
        &sp.mask_l4,
        &sp.omega_alpha_l_norm4,
    ] {
        push_vec4s(out, arr);
    }
    for &f in &sl.sigma_v {
        push_f64_bits(out, f);
    }
    for &f in &sl.sigma_h {
        push_f64_bits(out, f);
    }
    for &f in &sl.alpha_l {
        push_f64_bits(out, f);
    }
}

fn read_sav_entry(src: &[f32], cur: &mut usize, sp: &mut SavStringSpec, sl: &mut SavEnergySlopes) {
    let ru = |c: &mut usize| -> usize {
        let v = src[*c].to_bits() as usize;
        *c += 1;
        v
    };
    (
        sp.n_vh, sp.m_ph, sp.nv_vecs, sp.nh_vecs, sp.nl_vecs, sl.nv_c, sl.nh_c,
    ) = (
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
        ru(cur),
    );
    sp.length_m = read_f64_bits(src, cur);
    sp.quarter_l = read_f64_bits(src, cur);
    sp.kappa_0 = read_f64_bits(src, cur);
    sp.kappa_ph = read_f64_bits(src, cur);
    for arr in [
        &mut sp.sigma_v4,
        &mut sp.omega_sigma_v4,
        &mut sp.mask_v4,
        &mut sp.sigma_h4,
        &mut sp.omega_sigma_h4,
        &mut sp.mask_h4,
    ] {
        read_vec4s(src, cur, arr);
    }
    for arr in [
        &mut sp.alpha_l4,
        &mut sp.half_ql_mask4,
        &mut sp.mask_l4,
        &mut sp.omega_alpha_l_norm4,
    ] {
        read_vec4s(src, cur, arr);
    }
    for d in &mut sl.sigma_v {
        *d = read_f64_bits(src, cur);
    }
    for d in &mut sl.sigma_h {
        *d = read_f64_bits(src, cur);
    }
    for d in &mut sl.alpha_l {
        *d = read_f64_bits(src, cur);
    }
}

const SAV_ENTRY_FLOATS: usize = 7
    + 4 * 2
    + (6 * MAX_PHANTOM_TRANSVERSE_VECS + 4 * MAX_PHANTOM_LONG_VECS) * 4
    + (2 * MAX_PHANTOM_TRANSVERSE_MODES + MAX_PHANTOM_LONG_MODES) * 2;

/// Computes the exact serialized `f32` length of a `VoicingBlob` for a bridge-coupled bank.
#[must_use]
pub fn voicing_blob_len(sample_rate: f32) -> usize {
    VoicingPrepContext::new(sample_rate).blob_len()
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
        let bank = StringBank::new_bridge_coupled(sample_rate, &VoicingSpec::default());
        let len = Self::compute_blob_len(&bank);
        Self {
            bank,
            out_blob: Vec::with_capacity(len),
        }
    }

    /// Exact number of `f32` elements produced by [`Self::prepare_voicing`].
    #[must_use]
    pub fn blob_len(&self) -> usize {
        Self::compute_blob_len(&self.bank)
    }

    fn compute_blob_len(bank: &StringBank) -> usize {
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
            + 8 * soa_floats
            + 18 * TOTAL_STRINGS
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
            b.ports.b_damp.as_slice(),
            b.ports.a_damp.as_slice(),
        ] {
            push_copy(&mut self.out_blob, s);
        }
        for s in [
            &b.ports.alpha_vel[..],
            &b.ports.r_port_strike[..],
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

impl StringBank {
    /// Installs a precomputed Stage 2/3 tuning table blob with zero allocations and
    /// continuous active string state rescaling (`gamma_new / gamma_old`).
    pub fn install_precomputed_voicing(&mut self, voicing: &VoicingSpec, blob: &[f32]) -> bool {
        let expected_len = VoicingPrepContext::compute_blob_len(self);
        let total_vecs = self.choir_map.total_vecs();
        let tl_vecs = self.bridge_ports.tl_soa.d_coeff.len() / 4;
        let v = voicing.sanitized();
        if blob.len() != expected_len
            || blob[0].to_bits() != VOICING_BLOB_MAGIC
            || (blob[1] - self.sample_rate).abs() > 0.5
            || (blob[3] - v.unison_detune_scale).abs() > 1.0e-4
            || (blob[4] - v.stretch_amount).abs() > 1.0e-4
            || (blob[5] - v.string_length_scale).abs() > 1.0e-4
            || blob[6].to_bits() != total_vecs as u32
            || blob[7].to_bits() != tl_vecs as u32
            || blob[8].to_bits() != self.bridge_ports.coef.len() as u32
            || blob[9].to_bits() != self.long_bank.b_long_prec.len() as u32
        {
            return false;
        }
        let mut cur = VOICING_HEADER_LEN;
        for d in [
            self.soa.d_coeff.as_mut_slice(),
            self.soa.s_coeff.as_mut_slice(),
            self.soa.c_bridge.as_mut_slice(),
            self.soa.b_exc.as_mut_slice(),
            self.ports.b_strike.as_mut_slice(),
            self.ports.a_strike.as_mut_slice(),
            self.ports.b_damp.as_mut_slice(),
            self.ports.a_damp.as_mut_slice(),
        ] {
            read_copy(blob, &mut cur, d);
        }
        for d in [
            &mut self.ports.alpha_vel[..],
            &mut self.ports.r_port_strike[..],
            &mut self.ports.r_port_damp_v[..],
            &mut self.ports.r_port_damp_h[..],
        ] {
            read_copy(blob, &mut cur, d);
        }
        let gamma_old = self.ports.gamma_s;
        for d in &mut self.ports.gamma_s {
            *d = read_f64_bits(blob, &mut cur);
        }
        for d in &mut self.ports.beta_s {
            *d = read_f64_bits(blob, &mut cur);
        }
        for (s_idx, entry) in self.choir_map.strings().iter().enumerate() {
            if self.ports.may_be_nonzero[s_idx] {
                self.ports.ports_synced[s_idx] = false;
                rescale_active_string_uv(
                    &mut self.soa,
                    &mut self.bridge_ports,
                    self.soa_dirty,
                    &mut self.bp_dirty,
                    s_idx,
                    entry,
                    gamma_old[s_idx],
                    self.ports.gamma_s[s_idx],
                );
            }
        }
        read_copy(blob, &mut cur, self.long_bank.b_long_prec.as_mut_slice());
        for d in &mut self.long_bank.prec_force_scale {
            *d = read_f64_bits(blob, &mut cur);
        }
        for d in &mut self.long_bank.prec_force_cap_n {
            *d = read_f64_bits(blob, &mut cur);
        }
        for c in &mut self.bridge_ports.coef {
            c.d = F32x4::from_slice(&blob[cur..cur + 4]);
            c.s = F32x4::from_slice(&blob[cur + 4..cur + 8]);
            c.beta = F32x4::from_slice(&blob[cur + 8..cur + 12]);
            cur += 12;
        }
        for d in [
            self.bridge_ports.tl_soa.d_coeff.as_mut_slice(),
            self.bridge_ports.tl_soa.s_coeff.as_mut_slice(),
            self.bridge_ports.tl_soa.c_bridge.as_mut_slice(),
        ] {
            read_copy(blob, &mut cur, d);
        }
        for d in [
            &mut self.bridge_ports.r0[..],
            &mut self.bridge_ports.a_wave_gains[..],
            &mut self.bridge_ports.r_damp_prefix_v[..],
            &mut self.bridge_ports.r_damp_prefix_h[..],
            &mut self.bridge_ports.tail_n_trans[..],
            &mut self.bridge_ports.tail_n_long[..],
        ] {
            read_copy(blob, &mut cur, d);
        }
        for s_idx in 0..TOTAL_STRINGS {
            let g = self.bridge_ports.a_wave_gains[s_idx];
            self.bridge_ports.meta[s_idx].a_wave_gain = g;
            self.bridge_ports.fast_meta[s_idx].a_wave_gain = g;
        }
        for k in 0..NUM_KEYS {
            let s0 = self.choir_map.keys()[k].string_range.start;
            self.bridge_ports.key_tails[k].tail_n_trans = self.bridge_ports.tail_n_trans[s0];
            self.bridge_ports.key_tails[k].tail_n_long = self.bridge_ports.tail_n_long[s0];
        }
        for (s_idx, entry) in self.choir_map.strings().iter().enumerate() {
            if usize::from(entry.key_idx) < LONGITUDINAL_MAX_KEY_IDX {
                read_sav_entry(
                    blob,
                    &mut cur,
                    &mut self.sav.specs[s_idx],
                    &mut self.sav.slopes[s_idx],
                );
            }
        }
        debug_assert_eq!(cur, expected_len);
        self.bridge_ports.sync_damp_ports(&self.ports);
        true
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

        let v = VoicingSpec::from_globals(0.75, 1.85, 1.35, 0.92);
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
        assert_eq!(
            ref_bank.string_linear_energy_joules(144),
            fast_bank.string_linear_energy_joules(144)
        );
        assert!(!fast_bank.install_precomputed_voicing(&v, &[0.0; 12]));
    }
}
