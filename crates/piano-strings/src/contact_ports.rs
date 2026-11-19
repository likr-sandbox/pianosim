//! Stage 1 <-> Stage 2 SI contact port precomputation (`Newtons`, `meters`, `m/s`)
//! with orthogonal vertical and horizontal damper projections.

use core::f64::consts::{PI, TAU};

use piano_simd::AlignedBuf;
use piano_spec::{
    bridge_pin_couplings, effective_string_f0_hz, partial_frequency_hz, PerfProfile, Polarization,
    StringPhysicalParams, VoicingSpec, NUM_KEYS, TOTAL_STRINGS,
};

use crate::choir_map::{compute_string_pol_modes, ChoirMap, StringMapEntry};
#[allow(unused_imports)]
pub(crate) use crate::contact_phi::{
    damper_phi, hammer_contact_time_s, hammer_contact_width_bounds_m, hammer_contact_width_m,
    strike_phi, strike_phi_with_width, DAMPER_CLEAR_THRESHOLD_M, HORIZ_DAMP_SCALE,
    HORIZ_STRIKE_SCALE,
};
#[allow(unused_imports)]
pub(crate) use crate::contact_port_sync::{
    accum_prefix_damp_vel_bp, refresh_prefix_damp_ports_bp, refresh_prefix_ports_with_tail_bp,
    refresh_string_ports, refresh_string_ports_bp,
};
use crate::modal_soa::{sin_cos_f64, sqrt_f64, NYQUIST_CUTOFF_RATIO};

/// Precomputed SI port projection arrays and per-string dual-port state cache.
#[derive(Clone, Debug)]
pub(crate) struct ContactPorts {
    pub b_strike: AlignedBuf,
    pub a_strike: AlignedBuf,
    pub b_strike_diff: AlignedBuf,
    pub a_strike_diff: AlignedBuf,
    pub b_damp: AlignedBuf,
    pub a_damp: AlignedBuf,
    pub gamma_s: [f64; TOTAL_STRINGS],
    pub beta_s: [f64; TOTAL_STRINGS],
    pub alpha_vel: [f32; TOTAL_STRINGS],
    pub r_port_strike: [f32; TOTAL_STRINGS],
    pub r_strike_00: [f32; TOTAL_STRINGS],
    pub r_strike_01: [f32; TOTAL_STRINGS],
    pub r_strike_11: [f32; TOTAL_STRINGS],
    pub r_port_damp_v: [f32; TOTAL_STRINGS],
    pub r_port_damp_h: [f32; TOTAL_STRINGS],
    pub y_h: [f32; TOTAL_STRINGS],
    pub v_h: [f32; TOTAL_STRINGS],
    pub y_dv: [f32; TOTAL_STRINGS],
    pub v_dv: [f32; TOTAL_STRINGS],
    pub y_dh: [f32; TOTAL_STRINGS],
    pub v_dh: [f32; TOTAL_STRINGS],
    pub may_be_nonzero: [bool; TOTAL_STRINGS],
    pub tail_nonzero: [bool; TOTAL_STRINGS],
    pub struck_active: [bool; TOTAL_STRINGS],
    pub ports_synced: [bool; TOTAL_STRINGS],
    pub tl_v_active: [u16; TOTAL_STRINGS],
    pub tl_h_active: [u16; TOTAL_STRINGS],
    pub tl_damp_active: [bool; TOTAL_STRINGS],
    pub sav_dormant: [bool; TOTAL_STRINGS],
    pub sav_low_nl_chunks: [u8; TOTAL_STRINGS],
    pub y_clear_scratch: alloc::vec::Vec<[f32; 128]>,
}

impl ContactPorts {
    /// Precomputes exact SI strike and orthogonal V/H damper projections and mobilities.
    #[cfg(test)]
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
    ) -> Self {
        Self::build_with_profile(sample_rate, voicing, specs, map, &PerfProfile::full())
    }

    /// Precomputes exact SI strike and damper projections honoring `perf` mode limits.
    #[must_use]
    pub fn build_with_profile(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        perf: &PerfProfile,
    ) -> Self {
        let len = map.total_padded_modes();
        let mut ports = Self {
            b_strike: AlignedBuf::zeroed(len),
            a_strike: AlignedBuf::zeroed(len),
            b_strike_diff: AlignedBuf::zeroed(len),
            a_strike_diff: AlignedBuf::zeroed(len),
            b_damp: AlignedBuf::zeroed(len),
            a_damp: AlignedBuf::zeroed(len),
            gamma_s: [1.0; TOTAL_STRINGS],
            beta_s: [0.0; TOTAL_STRINGS],
            alpha_vel: [0.0; TOTAL_STRINGS],
            r_port_strike: [0.0; TOTAL_STRINGS],
            r_strike_00: [0.0; TOTAL_STRINGS],
            r_strike_01: [0.0; TOTAL_STRINGS],
            r_strike_11: [0.0; TOTAL_STRINGS],
            r_port_damp_v: [0.0; TOTAL_STRINGS],
            r_port_damp_h: [0.0; TOTAL_STRINGS],
            y_h: [0.0; TOTAL_STRINGS],
            v_h: [0.0; TOTAL_STRINGS],
            y_dv: [0.0; TOTAL_STRINGS],
            v_dv: [0.0; TOTAL_STRINGS],
            y_dh: [0.0; TOTAL_STRINGS],
            v_dh: [0.0; TOTAL_STRINGS],
            may_be_nonzero: [false; TOTAL_STRINGS],
            tail_nonzero: [false; TOTAL_STRINGS],
            struck_active: [false; TOTAL_STRINGS],
            ports_synced: [true; TOTAL_STRINGS],
            tl_v_active: [u16::MAX; TOTAL_STRINGS],
            tl_h_active: [u16::MAX; TOTAL_STRINGS],
            tl_damp_active: [false; TOTAL_STRINGS],
            sav_dormant: [false; TOTAL_STRINGS],
            sav_low_nl_chunks: [0; TOTAL_STRINGS],
            y_clear_scratch: alloc::vec![[0.0_f32; 128]; NUM_KEYS],
        };
        for (s_idx, entry) in map.strings().iter().enumerate() {
            populate_string_ports_with_profile(
                &mut ports,
                s_idx,
                entry,
                sample_rate,
                voicing,
                specs,
                perf,
            );
        }
        ports
    }

    pub fn reset_resting(&mut self) {
        self.y_h.fill(0.0);
        self.v_h.fill(0.0);
        self.y_dv.fill(0.0);
        self.v_dv.fill(0.0);
        self.y_dh.fill(0.0);
        self.v_dh.fill(0.0);
        self.may_be_nonzero.fill(false);
        self.tail_nonzero.fill(false);
        self.struck_active.fill(false);
        self.ports_synced.fill(true);
        self.tl_v_active.fill(u16::MAX);
        self.tl_h_active.fill(u16::MAX);
        self.tl_damp_active.fill(false);
        self.sav_dormant.fill(false);
        self.sav_low_nl_chunks.fill(0);
    }

    #[inline]
    pub fn clear_string_state(&mut self, s_idx: usize) {
        self.may_be_nonzero[s_idx] = false;
        self.tail_nonzero[s_idx] = false;
        self.struck_active[s_idx] = false;
        self.tl_v_active[s_idx] = u16::MAX;
        self.tl_h_active[s_idx] = u16::MAX;
        self.tl_damp_active[s_idx] = false;
        self.sav_dormant[s_idx] = false;
        self.sav_low_nl_chunks[s_idx] = 0;
        self.y_h[s_idx] = 0.0;
        self.v_h[s_idx] = 0.0;
        self.y_dv[s_idx] = 0.0;
        self.v_dv[s_idx] = 0.0;
        self.y_dh[s_idx] = 0.0;
        self.v_dh[s_idx] = 0.0;
        self.ports_synced[s_idx] = true;
    }

    /// Computes physical linear Hamiltonian `0.5 * beta_s * sum(u^2 + v^2)` in Joules over `vec_range`.
    #[inline]
    #[must_use]
    pub fn linear_energy_joules(
        &self,
        s_idx: usize,
        soa: &crate::modal_soa::ModalSoa,
        vec_range: core::ops::Range<usize>,
    ) -> f64 {
        let u = &soa.u.as_f32x4_slice()[vec_range.clone()];
        let v = &soa.v.as_f32x4_slice()[vec_range];
        let mut sum_sq = 0.0_f64;
        for (&uv, &vv) in u.iter().zip(v.iter()) {
            let ua = uv.to_array();
            let va = vv.to_array();
            for lane in 0..4 {
                let u64v = f64::from(ua[lane]);
                let v64v = f64::from(va[lane]);
                sum_sq += u64v * u64v + v64v * v64v;
            }
        }
        0.5 * self.beta_s[s_idx] * sum_sq
    }

    /// Exact quadratic dual strike port resistance for compression blend `alpha in [0.0, 1.0]`.
    #[inline]
    #[must_use]
    pub(crate) fn effective_strike_r_port(&self, s_idx: usize, alpha: f32) -> f32 {
        let a = alpha.clamp(0.0, 1.0);
        if a == 0.0 {
            return self.r_strike_00[s_idx];
        }
        let om = 1.0 - a;
        om * om * self.r_strike_00[s_idx]
            + 2.0 * a * om * self.r_strike_01[s_idx]
            + a * a * self.r_strike_11[s_idx]
    }
}

pub(crate) fn populate_string_ports_with_profile(
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
    sample_rate: f64,
    voicing: &VoicingSpec,
    specs: &[StringPhysicalParams; NUM_KEYS],
    perf: &PerfProfile,
) {
    let k = entry.key_idx as usize;
    let s_choir = entry.string_in_choir as usize;
    let spec = specs[k];
    let m_str = spec.linear_density_kg_m * spec.speaking_length_m;
    let ts = 1.0 / sample_rate;
    let nyquist = NYQUIST_CUTOFF_RATIO * sample_rate;
    let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
    let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);

    let x = (k as f64) / ((NUM_KEYS - 1) as f64);
    let m_h = 0.0112 - 0.0077 * x;
    let tau_c = hammer_contact_time_s(k);
    let z_choir = (entry.choir_size as f64) * 2.0 * f0_v * m_str;
    let i_mf = (1.55 * m_h * 1.8).min(3.3 * z_choir * tau_c * 1.8);

    let mut s_raw_sq = 0.0_f64;
    let mut s_eff_sq = 0.0_f64;
    let (vm_raw, hm_raw) = compute_string_pol_modes(k, &spec, f0_v, f0_h, perf);
    let (vb, vm) = (
        entry.vert_vecs.start * 4,
        vm_raw.min(entry.vert_vecs.len() * 4),
    );
    let (hb, hm) = (
        entry.horiz_vecs.start * 4,
        hm_raw.min(entry.horiz_vecs.len() * 4),
    );
    let (cv, ch) = bridge_pin_couplings(k, s_choir, entry.choir_size);
    let (w_narrow, w_wide) = hammer_contact_width_bounds_m(k);
    let pols = [
        (vb, vm, f0_v, 1.0, 1.0, cv, false),
        (hb, hm, f0_h, HORIZ_STRIKE_SCALE, HORIZ_DAMP_SCALE, ch, true),
    ];
    for &(_, modes, f0, hs, _, c_b, _) in &pols {
        for j in 1..=modes {
            let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
            if f_j > 0.0 && f_j < nyquist {
                let ph_raw = hs * strike_phi_with_width(k, j, spec.speaking_length_m, w_narrow);
                s_raw_sq += ph_raw * ph_raw;
                let ph_nom = hs * strike_phi(k, j, spec.speaking_length_m);
                let ft = 2.0 * f_j * tau_c;
                let sinc = 1.0 / sqrt_f64(1.0 + ft * ft);
                let w = ph_nom * c_b * sinc;
                s_eff_sq += w * w;
            }
        }
    }

    let r_raw_48k = (2.0 / (48_000.0 * m_str)) * s_raw_sq;
    let c_cap_hi = if r_raw_48k > 0.33 {
        sqrt_f64(0.33 / r_raw_48k)
    } else {
        1.0
    };
    let r_excess = (r_raw_48k - 0.17).max(0.0);
    let c_cap_17 = 1.0 / sqrt_f64(1.0 + r_excess / 0.10);
    let treble_t = ((x - 0.62) / 0.10).clamp(0.0, 1.0);
    let smooth = treble_t * treble_t * (3.0 - 2.0 * treble_t);
    let c_port = c_cap_hi + (c_cap_17 - c_cap_hi) * smooth;
    s_eff_sq *= c_port * c_port;

    let mid_lift = sin_cos_f64(PI * x).0;
    let target_rms = 0.068 * (1.0 + 0.62 * mid_lift * mid_lift) * (1.0 + 0.45 * x * x);
    let gamma_k = if s_eff_sq > 0.0 {
        target_rms * m_str / (i_mf * sqrt_f64(s_eff_sq))
    } else {
        1.0
    };
    let b_gain = gamma_k * (2.0 * ts / m_str);
    ports.gamma_s[s_idx] = gamma_k;
    ports.beta_s[s_idx] = m_str / (2.0 * gamma_k * gamma_k);
    ports.alpha_vel[s_idx] = (m_str / (2.0 * ts * gamma_k * gamma_k)) as f32;

    let full_slots = (entry.vert_vecs.start * 4)..(entry.long_vecs.end * 4);
    ports.b_strike.as_mut_slice()[full_slots.clone()].fill(0.0);
    ports.a_strike.as_mut_slice()[full_slots.clone()].fill(0.0);
    ports.b_strike_diff.as_mut_slice()[full_slots.clone()].fill(0.0);
    ports.a_strike_diff.as_mut_slice()[full_slots.clone()].fill(0.0);
    ports.b_damp.as_mut_slice()[full_slots.clone()].fill(0.0);
    ports.a_damp.as_mut_slice()[full_slots].fill(0.0);

    let (mut sum_b0_sq, mut sum_b0_b1, mut sum_b1_sq) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut sum_bd_v_sq, mut sum_bd_h_sq) = (0.0_f64, 0.0_f64);
    let bh_s = ports.b_strike.as_mut_slice();
    let ah_s = ports.a_strike.as_mut_slice();
    let bh_diff_s = ports.b_strike_diff.as_mut_slice();
    let ah_diff_s = ports.a_strike_diff.as_mut_slice();
    let bd_s = ports.b_damp.as_mut_slice();
    let ad_s = ports.a_damp.as_mut_slice();

    for &(base, modes, f0, hs, ds, _, is_horiz) in &pols {
        for j in 1..=modes {
            let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
            if f_j > 0.0 && f_j < nyquist {
                let omega = TAU * f_j;
                let ph0 =
                    hs * c_port * strike_phi_with_width(k, j, spec.speaking_length_m, w_narrow);
                let ph1 = hs * c_port * strike_phi_with_width(k, j, spec.speaking_length_m, w_wide);
                let (bh0, ah0) = ((b_gain * ph0) as f32, (ph0 / (omega * gamma_k)) as f32);
                let (bh1, ah1) = ((b_gain * ph1) as f32, (ph1 / (omega * gamma_k)) as f32);
                let (d_bh, d_ah) = (bh1 - bh0, ah1 - ah0);
                let slot = base + (j - 1);
                bh_s[slot] = bh0;
                ah_s[slot] = ah0;
                bh_diff_s[slot] = d_bh;
                ah_diff_s[slot] = d_ah;
                let (b0, b1) = (f64::from(bh0), f64::from(bh0 + d_bh));
                sum_b0_sq += b0 * b0;
                sum_b0_b1 += b0 * b1;
                sum_b1_sq += b1 * b1;
                if entry.has_damper {
                    let pd = ds * damper_phi(k, j, spec.speaking_length_m, f_j);
                    let (bd, ad) = ((b_gain * pd) as f32, (pd / (omega * gamma_k)) as f32);
                    bd_s[slot] = bd;
                    ad_s[slot] = ad;
                    let bd_sq = f64::from(bd) * f64::from(bd);
                    if is_horiz {
                        sum_bd_h_sq += bd_sq;
                    } else {
                        sum_bd_v_sq += bd_sq;
                    }
                }
            }
        }
    }
    // Longitudinal modes remain strictly orthogonal to the transverse damper pads (`b_damp = a_damp = 0`).
    let alpha64 = f64::from(ports.alpha_vel[s_idx]);
    let r00 = (alpha64 * sum_b0_sq) as f32;
    ports.r_strike_00[s_idx] = r00;
    ports.r_strike_01[s_idx] = (alpha64 * sum_b0_b1) as f32;
    ports.r_strike_11[s_idx] = (alpha64 * sum_b1_sq) as f32;
    ports.r_port_strike[s_idx] = r00;
    ports.r_port_damp_v[s_idx] = (alpha64 * sum_bd_v_sq) as f32;
    ports.r_port_damp_h[s_idx] = (alpha64 * sum_bd_h_sq) as f32;
}

#[cfg(test)]
#[path = "contact_port_tests.rs"]
mod port_tests;
