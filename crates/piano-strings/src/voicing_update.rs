//! Zero-allocation in-place voicing/tuning update across active Stage 2/3 tables (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::{
    bridge_pin_couplings, build_scale_design, count_transverse_partials, effective_string_f0_hz,
    Polarization, StringPhysicalParams, VoicingSpec, MAX_AUDIBLE_FREQ_HZ, NUM_KEYS,
};

use crate::bridge_port::{
    count_prefix_modes, fill_pol_beta, BridgePorts, FastPrefixMeta, PrefixCoef,
};
use crate::choir_map::StringMapEntry;
use crate::contact_ports::populate_string_ports;
use crate::modal_soa::{populate_string_modes_with_mode, sqrt_f64, ModalSoa, NYQUIST_CUTOFF_RATIO};
use crate::sav_spec::SavStringSpec;
use crate::StringBank;

#[inline]
fn rescale_vec_uv(u_slice: &mut [F32x4], v_slice: &mut [F32x4], r: Range<usize>, s4: F32x4) {
    for u in &mut u_slice[r.clone()] {
        *u = *u * s4;
    }
    for v in &mut v_slice[r] {
        *v = *v * s4;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rescale_active_string_uv(
    soa: &mut ModalSoa,
    bp: &mut BridgePorts,
    soa_dirty: bool,
    bp_dirty: &mut bool,
    s_idx: usize,
    entry: &StringMapEntry,
    gamma_old: f64,
    gamma_new: f64,
) {
    if gamma_old <= 0.0 || gamma_new <= 0.0 {
        return;
    }
    let ratio = gamma_new / gamma_old;
    if !ratio.is_finite() || (ratio - 1.0).abs() <= 1.0e-7 {
        return;
    }
    let scale4 = F32x4::splat(ratio as f32);
    if soa_dirty {
        let m = &bp.meta[s_idx];
        for st in &mut bp.state[m.beta_v_off..(m.beta_v_off + m.n_vh)] {
            st.u = st.u * scale4;
            st.v = st.v * scale4;
        }
        let (tu, tv) = (
            bp.tl_soa.u.as_f32x4_slice_mut(),
            bp.tl_soa.v.as_f32x4_slice_mut(),
        );
        for r in [m.tl_v.clone(), m.tl_h.clone(), m.tl_l.clone()] {
            rescale_vec_uv(tu, tv, r, scale4);
        }
    } else {
        let (su, sv) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
        rescale_vec_uv(su, sv, entry.all_vecs(), scale4);
        *bp_dirty = true;
    }
}

#[allow(clippy::too_many_arguments)]
fn update_prefix_pol(
    coef: &mut [PrefixCoef],
    d_src: &[F32x4],
    s_src: &[F32x4],
    br: &Range<usize>,
    beta_off: usize,
    n_cut: usize,
    gain: f64,
    b_inharm: f64,
) -> f64 {
    let mut beta_tmp = [0.0_f32; 24];
    let n_vecs = br.len().min(6);
    let clamped_cut = n_cut.min(n_vecs * 4);
    let sum_sq = fill_pol_beta(&mut beta_tmp, clamped_cut, gain, b_inharm);
    for vi in 0..n_vecs {
        coef[beta_off + vi] = PrefixCoef {
            d: d_src[br.start + vi],
            s: s_src[br.start + vi],
            beta: F32x4::from_slice(&beta_tmp[(vi * 4)..(vi * 4 + 4)]),
        };
    }
    sum_sq
}

fn copy_tail_coeffs(tl_soa: &mut ModalSoa, soa: &ModalSoa, src: &Range<usize>, dst: &Range<usize>) {
    let count = src.len().min(dst.len());
    let (sd, ss, sc) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    tl_soa.d_coeff.as_f32x4_slice_mut()[dst.start..dst.start + count]
        .copy_from_slice(&sd[src.start..src.start + count]);
    tl_soa.s_coeff.as_f32x4_slice_mut()[dst.start..dst.start + count]
        .copy_from_slice(&ss[src.start..src.start + count]);
    tl_soa.c_bridge.as_f32x4_slice_mut()[dst.start..dst.start + count]
        .copy_from_slice(&sc[src.start..src.start + count]);
}

#[allow(clippy::too_many_arguments)]
fn update_string_bridge_ports(
    bp: &mut BridgePorts,
    soa: &ModalSoa,
    entry: &StringMapEntry,
    s_idx: usize,
    k: usize,
    spec: &StringPhysicalParams,
    voicing_specs: &[StringPhysicalParams; NUM_KEYS],
    voicing: &VoicingSpec,
    gamma: f64,
    alpha_vel: f32,
    b_damp: &[f32],
    ts: f64,
    nyquist: f64,
) {
    let s_choir = usize::from(entry.string_in_choir);
    let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, voicing_specs, voicing);
    let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, voicing_specs, voicing);
    let v_modes = count_transverse_partials(f0_v, spec.inharmonicity_b, MAX_AUDIBLE_FREQ_HZ)
        .min(entry.vert_vecs.len() * 4);
    let h_modes = count_transverse_partials(f0_h, spec.inharmonicity_b, MAX_AUDIBLE_FREQ_HZ)
        .min(entry.horiz_vecs.len() * 4);

    let (v_br, h_br, v_tail, h_tail, tl_v, tl_h, tl_l, beta_v_off, beta_h_off) = {
        let m = &bp.meta[s_idx];
        (
            m.v_br.clone(),
            m.h_br.clone(),
            m.v_tail.clone(),
            m.h_tail.clone(),
            m.tl_v.clone(),
            m.tl_h.clone(),
            m.tl_l.clone(),
            m.beta_v_off,
            m.beta_h_off,
        )
    };
    let nv_cut = count_prefix_modes(k, v_modes, f0_v, spec.inharmonicity_b, nyquist)
        .min(v_br.len() * 4)
        .min(24);
    let nh_cut = count_prefix_modes(k, h_modes, f0_h, spec.inharmonicity_b, nyquist)
        .min(h_br.len() * 4)
        .min(24);

    let l_m = spec.speaking_length_m;
    let (cv, ch) = bridge_pin_couplings(k, s_choir, entry.choir_size);
    let gain_v = gamma * (2.0 * (2.0 * l_m * f0_v) * ts / l_m) * cv;
    let gain_h = gamma * (2.0 * (2.0 * l_m * f0_h) * ts / l_m) * ch;
    let (d_src, s_src) = (soa.d_coeff.as_f32x4_slice(), soa.s_coeff.as_f32x4_slice());
    let sum_sq_v = update_prefix_pol(
        &mut bp.coef,
        d_src,
        s_src,
        &v_br,
        beta_v_off,
        nv_cut,
        gain_v,
        spec.inharmonicity_b,
    );
    let sum_sq_h = update_prefix_pol(
        &mut bp.coef,
        d_src,
        s_src,
        &h_br,
        beta_h_off,
        nh_cut,
        gain_h,
        spec.inharmonicity_b,
    );

    let alpha64 = f64::from(alpha_vel);
    let r0_val = (0.5 * alpha64 * (sum_sq_v + sum_sq_h)) as f32;
    let a_wave_gain = if r0_val > 0.0 && r0_val.is_finite() {
        bp.r0[s_idx] = r0_val;
        (alpha64 / f64::from(r0_val)) as f32
    } else {
        bp.r0[s_idx] = 1.0e-9;
        0.0
    };
    bp.meta[s_idx].a_wave_gain = a_wave_gain;
    bp.a_wave_gains[s_idx] = a_wave_gain;
    let cv32 = cv as f32;
    let pan_l_v = entry.pan_l * cv32;
    let pan_r_v = entry.pan_r * cv32;
    bp.c_vert[s_idx] = cv32;
    bp.c_horiz[s_idx] = ch as f32;
    bp.h_proj[s_idx] = (ch / cv) as f32;
    bp.pan_l[s_idx] = pan_l_v;
    bp.pan_r[s_idx] = pan_r_v;
    bp.fast_meta[s_idx] = FastPrefixMeta {
        a_wave_gain,
        pan_l: pan_l_v,
        pan_r: pan_r_v,
    };

    let base_n = if gamma > 0.0 {
        2.0 * spec.linear_density_kg_m * l_m * f0_v / gamma
    } else {
        0.0
    };
    let cut_j = (nv_cut + 1) as f64;
    bp.tail_n_trans[s_idx] = (base_n * sqrt_f64(1.0 + spec.inharmonicity_b * cut_j * cut_j)) as f32;
    bp.tail_n_long[s_idx] = base_n as f32;

    if entry.has_damper {
        let sum_sq_slice = |vr: &Range<usize>| -> f64 {
            b_damp[(vr.start * 4)..(vr.end * 4)]
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum()
        };
        bp.r_damp_prefix_v[s_idx] = (alpha64 * sum_sq_slice(&v_br)) as f32;
        bp.r_damp_prefix_h[s_idx] = (alpha64 * sum_sq_slice(&h_br)) as f32;
    } else {
        bp.r_damp_prefix_v[s_idx] = 0.0;
        bp.r_damp_prefix_h[s_idx] = 0.0;
    }

    copy_tail_coeffs(&mut bp.tl_soa, soa, &v_tail, &tl_v);
    copy_tail_coeffs(&mut bp.tl_soa, soa, &h_tail, &tl_h);
    copy_tail_coeffs(&mut bp.tl_soa, soa, &entry.long_vecs, &tl_l);
}

impl StringBank {
    /// Updates tuned modal poles, port projections, longitudinal tables, SAV specs, and
    /// WDF bridge coefficients in place with zero heap allocations and continuous live states.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        let sr64 = f64::from(self.sample_rate);
        let ts = 1.0 / sr64;
        let nyquist = NYQUIST_CUTOFF_RATIO * sr64;
        let specs = build_scale_design(voicing);
        let strings = self.choir_map.strings();

        for (k, key) in self.choir_map.keys().iter().enumerate() {
            let spec = specs[k];
            for s_idx in key.string_range.clone() {
                let entry = &strings[s_idx];
                let gamma_old = self.ports.gamma_s[s_idx];

                populate_string_modes_with_mode(
                    &mut self.soa,
                    entry,
                    sr64,
                    voicing,
                    &specs,
                    self.lossless,
                    self.bridge_coupled,
                );
                populate_string_ports(&mut self.ports, s_idx, entry, sr64, voicing, &specs);

                let gamma_new = self.ports.gamma_s[s_idx];
                if self.ports.may_be_nonzero[s_idx] {
                    self.ports.ports_synced[s_idx] = false;
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

                self.long_bank
                    .populate_string(s_idx, entry, sr64, &specs, gamma_new);
                let (sav_spec, sav_slopes) = SavStringSpec::build(
                    entry,
                    sr64,
                    voicing,
                    &specs,
                    gamma_new,
                    self.ports.beta_s[s_idx],
                );
                self.sav.specs[s_idx] = sav_spec;
                self.sav.slopes[s_idx] = sav_slopes;

                update_string_bridge_ports(
                    &mut self.bridge_ports,
                    &self.soa,
                    entry,
                    s_idx,
                    k,
                    &spec,
                    &specs,
                    voicing,
                    gamma_new,
                    self.ports.alpha_vel[s_idx],
                    self.ports.b_damp.as_slice(),
                    ts,
                    nyquist,
                );
            }
            let s0 = key.string_range.start;
            self.bridge_ports.key_tails[k].tail_n_trans = self.bridge_ports.tail_n_trans[s0];
            self.bridge_ports.key_tails[k].tail_n_long = self.bridge_ports.tail_n_long[s0];
        }
        self.bridge_ports.sync_damp_ports(&self.ports);
    }
}

#[cfg(test)]
#[path = "voicing_update_tests.rs"]
mod tests;
