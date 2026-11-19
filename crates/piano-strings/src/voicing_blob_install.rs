//! Per-key time-sliced and atomic-port installation for `VoicingBlob` (`no_std`).

use piano_simd::F32x4;
use piano_spec::{
    build_scale_design, string_params_for_key, StringPhysicalParams, VoicingSpec,
    LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS,
};

use crate::voicing_blob::{VoicingPrepContext, VOICING_BLOB_MAGIC, VOICING_HEADER_LEN};
use crate::voicing_blob_sav::{read_f64_bits, read_sav_entry, SAV_ENTRY_FLOATS};
use crate::voicing_update::rescale_active_string_uv;
use crate::StringBank;

struct VoicingBlobLayout {
    soa_base: usize,
    soa_floats: usize,
    ports7_base: usize,
    gamma_base: usize,
    beta_base: usize,
    long_b_base: usize,
    long_scale_base: usize,
    long_cap_base: usize,
    coef_base: usize,
    tl_base: usize,
    tl_floats: usize,
    bp6_base: usize,
    sav_base: usize,
}

impl VoicingBlobLayout {
    fn from_bank(bank: &StringBank) -> Self {
        let soa_base = VOICING_HEADER_LEN;
        let soa_floats = bank.soa.d_coeff.len();
        let ports7_base = soa_base + 10 * soa_floats;
        let gamma_base = ports7_base + 7 * TOTAL_STRINGS;
        let beta_base = gamma_base + 2 * TOTAL_STRINGS;
        let long_b_base = beta_base + 2 * TOTAL_STRINGS;
        let long_floats = bank.long_bank.b_long_prec.len();
        let long_scale_base = long_b_base + long_floats;
        let long_cap_base = long_scale_base + 2 * TOTAL_STRINGS;
        let coef_base = long_cap_base + 2 * TOTAL_STRINGS;
        let coef_vecs = bank.bridge_ports.coef.len();
        let tl_base = coef_base + coef_vecs * 12;
        let tl_floats = bank.bridge_ports.tl_soa.d_coeff.len();
        let bp6_base = tl_base + 3 * tl_floats;
        let sav_base = bp6_base + 6 * TOTAL_STRINGS;
        Self {
            soa_base,
            soa_floats,
            ports7_base,
            gamma_base,
            beta_base,
            long_b_base,
            long_scale_base,
            long_cap_base,
            coef_base,
            tl_base,
            tl_floats,
            bp6_base,
            sav_base,
        }
    }
}

#[inline]
fn validate_blob_header(bank: &StringBank, blob: &[f32]) -> bool {
    let expected_len = VoicingPrepContext::compute_blob_len(bank);
    let total_vecs = bank.choir_map.total_vecs();
    let tl_vecs = bank.bridge_ports.tl_soa.d_coeff.len() / 4;
    blob.len() == expected_len
        && blob[0].to_bits() == VOICING_BLOB_MAGIC
        && (blob[1] - bank.sample_rate).abs() <= 0.5
        && blob[7].to_bits() == total_vecs as u32
        && blob[8].to_bits() == tl_vecs as u32
        && blob[9].to_bits() == bank.bridge_ports.coef.len() as u32
        && blob[10].to_bits() == bank.long_bank.b_long_prec.len() as u32
}

#[inline]
fn voicing_from_blob_header(base: &VoicingSpec, blob: &[f32]) -> VoicingSpec {
    let mut v = base.clone();
    v.hammer_hardness = blob[2];
    v.unison_detune_scale = blob[3];
    v.stretch_amount = blob[4];
    v.string_length_scale = blob[5];
    v.a4_freq_hz = blob[6];
    v.sanitized()
}

impl StringBank {
    /// Validates `blob` header, length, structural vector counts, and matching global voicing parameters.
    #[must_use]
    pub fn validate_voicing_blob(&self, voicing: &VoicingSpec, blob: &[f32]) -> bool {
        let v = voicing.sanitized();
        validate_blob_header(self, blob)
            && (blob[3] - v.unison_detune_scale).abs() <= 1.0e-4
            && (blob[4] - v.stretch_amount).abs() <= 1.0e-4
            && (blob[5] - v.string_length_scale).abs() <= 1.0e-4
            && (blob[6] - v.a4_freq_hz).abs() <= 1.0e-3
    }

    /// Validates `blob` and latches `voicing` + precomputed `scale_specs` for subsequent
    /// time-sliced [`Self::install_voicing_key`] and [`Self::finish_voicing_ports`] calls.
    pub fn begin_voicing_install(
        &mut self,
        voicing: &VoicingSpec,
        scale_specs: &[StringPhysicalParams; NUM_KEYS],
        blob: &[f32],
    ) -> bool {
        if !self.validate_voicing_blob(voicing, blob) {
            return false;
        }
        self.pending_voicing = voicing.sanitized();
        *self.pending_scale_specs = *scale_specs;
        self.pending_voicing_active = true;
        true
    }

    /// Installs all Stage 2/3 voicing tables for key `k` directly from `blob` while deferring
    /// `bridge_ports.r0` and `bridge_ports.a_wave_gains` to [`Self::finish_voicing_ports`].
    pub fn install_voicing_key(&mut self, k: usize, blob: &[f32]) -> bool {
        if k >= NUM_KEYS || !validate_blob_header(self, blob) {
            return false;
        }
        let layout = VoicingBlobLayout::from_bank(self);
        let key = self.choir_map.keys()[k].clone();
        let sr = key.string_range;
        let vr_f = (key.vec_range.start * 4)..(key.vec_range.end * 4);

        for (i, dst) in [
            self.soa.d_coeff.as_mut_slice(),
            self.soa.s_coeff.as_mut_slice(),
            self.soa.c_bridge.as_mut_slice(),
            self.soa.b_exc.as_mut_slice(),
            self.ports.b_strike.as_mut_slice(),
            self.ports.a_strike.as_mut_slice(),
            self.ports.b_strike_diff.as_mut_slice(),
            self.ports.a_strike_diff.as_mut_slice(),
            self.ports.b_damp.as_mut_slice(),
            self.ports.a_damp.as_mut_slice(),
        ]
        .into_iter()
        .enumerate()
        {
            let off = layout.soa_base + i * layout.soa_floats;
            dst[vr_f.clone()].copy_from_slice(&blob[(off + vr_f.start)..(off + vr_f.end)]);
        }

        for (i, dst) in [
            &mut self.ports.alpha_vel[..],
            &mut self.ports.r_port_strike[..],
            &mut self.ports.r_strike_00[..],
            &mut self.ports.r_strike_01[..],
            &mut self.ports.r_strike_11[..],
            &mut self.ports.r_port_damp_v[..],
            &mut self.ports.r_port_damp_h[..],
        ]
        .into_iter()
        .enumerate()
        {
            let off = layout.ports7_base + i * TOTAL_STRINGS;
            dst[sr.clone()].copy_from_slice(&blob[(off + sr.start)..(off + sr.end)]);
        }

        for s_idx in sr.clone() {
            let mut g_cur = layout.gamma_base + s_idx * 2;
            let mut b_cur = layout.beta_base + s_idx * 2;
            let gamma_old = self.ports.gamma_s[s_idx];
            let gamma_new = read_f64_bits(blob, &mut g_cur);
            self.ports.gamma_s[s_idx] = gamma_new;
            self.ports.beta_s[s_idx] = read_f64_bits(blob, &mut b_cur);
            if self.ports.may_be_nonzero[s_idx] {
                self.ports.ports_synced[s_idx] = false;
                let entry = &self.choir_map.strings()[s_idx];
                rescale_active_string_uv(
                    &mut self.soa,
                    &mut self.bridge_ports,
                    self.soa_dirty,
                    &mut self.bp_dirty,
                    s_idx,
                    entry,
                    gamma_old,
                    gamma_new,
                );
            }
        }

        for s_idx in sr.clone() {
            let l_vecs = self.choir_map.strings()[s_idx].long_vecs.len();
            if l_vecs > 0 {
                let l_off = self.long_bank.vec_offset[s_idx] * 4;
                let l_len = l_vecs * 4;
                self.long_bank.b_long_prec.as_mut_slice()[l_off..l_off + l_len].copy_from_slice(
                    &blob[(layout.long_b_base + l_off)..(layout.long_b_base + l_off + l_len)],
                );
            }
            let mut sc = layout.long_scale_base + s_idx * 2;
            let mut cc = layout.long_cap_base + s_idx * 2;
            self.long_bank.prec_force_scale[s_idx] = read_f64_bits(blob, &mut sc);
            self.long_bank.prec_force_cap_n[s_idx] = read_f64_bits(blob, &mut cc);
        }

        for s_idx in sr.clone() {
            let m = &self.bridge_ports.meta[s_idx];
            for c_idx in m.beta_v_off..(m.beta_v_off + m.n_vh) {
                let off = layout.coef_base + c_idx * 12;
                let c = &mut self.bridge_ports.coef[c_idx];
                c.d = F32x4::from_slice(&blob[off..off + 4]);
                c.s = F32x4::from_slice(&blob[off + 4..off + 8]);
                c.beta = F32x4::from_slice(&blob[off + 8..off + 12]);
            }
            let (tl_v, tl_h, tl_l) = (m.tl_v.clone(), m.tl_h.clone(), m.tl_l.clone());
            for r in [tl_v, tl_h, tl_l] {
                if r.is_empty() {
                    continue;
                }
                let fr = (r.start * 4)..(r.end * 4);
                for (i, dst) in [
                    self.bridge_ports.tl_soa.d_coeff.as_mut_slice(),
                    self.bridge_ports.tl_soa.s_coeff.as_mut_slice(),
                    self.bridge_ports.tl_soa.c_bridge.as_mut_slice(),
                ]
                .into_iter()
                .enumerate()
                {
                    let off = layout.tl_base + i * layout.tl_floats;
                    dst[fr.clone()].copy_from_slice(&blob[(off + fr.start)..(off + fr.end)]);
                }
            }
        }

        for (slot_idx, dst) in [
            (2usize, &mut self.bridge_ports.r_damp_prefix_v[..]),
            (3, &mut self.bridge_ports.r_damp_prefix_h[..]),
            (4, &mut self.bridge_ports.tail_n_trans[..]),
            (5, &mut self.bridge_ports.tail_n_long[..]),
        ] {
            let off = layout.bp6_base + slot_idx * TOTAL_STRINGS;
            dst[sr.clone()].copy_from_slice(&blob[(off + sr.start)..(off + sr.end)]);
        }
        let s0 = sr.start;
        self.bridge_ports.key_tails[k].tail_n_trans = self.bridge_ports.tail_n_trans[s0];
        self.bridge_ports.key_tails[k].tail_n_long = self.bridge_ports.tail_n_long[s0];

        if k < LONGITUDINAL_MAX_KEY_IDX {
            for s_idx in sr.clone() {
                let mut sav_cur = layout.sav_base + s_idx * SAV_ENTRY_FLOATS;
                read_sav_entry(
                    blob,
                    &mut sav_cur,
                    &mut self.sav.specs[s_idx],
                    &mut self.sav.slopes[s_idx],
                );
            }
        }

        self.bridge_ports.sync_damp_ports_range(&self.ports, sr);
        if self.perf.passive_res_max_hz < 20_000.0 {
            if !self.pending_voicing_active {
                let v = voicing_from_blob_header(&self.last_voicing, blob);
                self.pending_scale_specs[48] = string_params_for_key(48, &v);
                self.pending_scale_specs[k] = string_params_for_key(k, &v);
                self.recompute_passive_res_caps_key_from_pending(k, &v);
            } else {
                self.recompute_passive_res_caps_pending(k);
            }
        } else {
            self.recompute_passive_res_caps_pending(k);
        }
        true
    }

    /// Atomically installs `bridge_ports.r0` and `bridge_ports.a_wave_gains` across all 230
    /// strings, marks `a_waves_dirty = true`, and finalizes `last_voicing`.
    pub fn finish_voicing_ports(&mut self, blob: &[f32]) -> bool {
        if !validate_blob_header(self, blob) {
            return false;
        }
        let layout = VoicingBlobLayout::from_bank(self);
        let r0_off = layout.bp6_base;
        let aw_off = layout.bp6_base + TOTAL_STRINGS;
        self.bridge_ports
            .r0
            .copy_from_slice(&blob[r0_off..r0_off + TOTAL_STRINGS]);
        self.bridge_ports
            .a_wave_gains
            .copy_from_slice(&blob[aw_off..aw_off + TOTAL_STRINGS]);
        for s_idx in 0..TOTAL_STRINGS {
            let g = self.bridge_ports.a_wave_gains[s_idx];
            self.bridge_ports.meta[s_idx].a_wave_gain = g;
            self.bridge_ports.fast_meta[s_idx].a_wave_gain = g;
        }
        self.bridge_ports.a_waves_dirty = true;
        if self.pending_voicing_active {
            self.last_voicing = self.pending_voicing.clone();
            self.pending_voicing_active = false;
        } else {
            self.last_voicing = voicing_from_blob_header(&self.last_voicing, blob);
        }
        true
    }

    /// Installs a precomputed Stage 2/3 tuning table blob with zero allocations and
    /// continuous active string state rescaling (`gamma_new / gamma_old`).
    pub fn install_precomputed_voicing(&mut self, voicing: &VoicingSpec, blob: &[f32]) -> bool {
        let v = voicing.sanitized();
        if !self.validate_voicing_blob(&v, blob) {
            return false;
        }
        let specs = build_scale_design(&v);
        if !self.begin_voicing_install(&v, &specs, blob) {
            return false;
        }
        for k in 0..NUM_KEYS {
            if !self.install_voicing_key(k, blob) {
                return false;
            }
        }
        self.finish_voicing_ports(blob)
    }
}
