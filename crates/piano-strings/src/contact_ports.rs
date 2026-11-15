//! Stage 1 <-> Stage 2 SI contact port precomputation (`Newtons`, `meters`, `m/s`)
//! with orthogonal vertical and horizontal damper projections.

use core::f64::consts::{PI, TAU};
use core::ops::Range;

use piano_simd::{AlignedBuf, F32x4};
use piano_spec::{
    effective_string_f0_hz, partial_frequency_hz, Polarization, StringPhysicalParams, VoicingSpec,
    NUM_KEYS, TOTAL_STRINGS,
};

use crate::choir_map::{ChoirMap, StringMapEntry};
use crate::modal_soa::{
    exp_f64, sin_cos_f64, sqrt_f64, ModalSoa, C_HORIZ, C_VERT, NYQUIST_CUTOFF_RATIO,
};

pub(crate) const HORIZ_STRIKE_SCALE: f64 = 0.65;
pub(crate) const HORIZ_DAMP_SCALE: f64 = 0.85;
pub(crate) const DAMPER_CLEAR_THRESHOLD_M: f32 = 4.5e-3;

/// Precomputed SI port projection arrays and per-string dual-port state cache.
#[derive(Clone, Debug)]
pub(crate) struct ContactPorts {
    pub b_strike: AlignedBuf,
    pub a_strike: AlignedBuf,
    pub b_damp: AlignedBuf,
    pub a_damp: AlignedBuf,
    pub gamma_s: [f64; TOTAL_STRINGS],
    pub beta_s: [f64; TOTAL_STRINGS],
    pub alpha_vel: [f32; TOTAL_STRINGS],
    pub r_port_strike: [f32; TOTAL_STRINGS],
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
    pub y_clear_scratch: alloc::vec::Vec<[f32; 128]>,
}

impl ContactPorts {
    /// Precomputes exact SI strike and orthogonal V/H damper projections and mobilities.
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
    ) -> Self {
        let len = map.total_padded_modes();
        let mut ports = Self {
            b_strike: AlignedBuf::zeroed(len),
            a_strike: AlignedBuf::zeroed(len),
            b_damp: AlignedBuf::zeroed(len),
            a_damp: AlignedBuf::zeroed(len),
            gamma_s: [1.0; TOTAL_STRINGS],
            beta_s: [0.0; TOTAL_STRINGS],
            alpha_vel: [0.0; TOTAL_STRINGS],
            r_port_strike: [0.0; TOTAL_STRINGS],
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
            y_clear_scratch: alloc::vec![[0.0_f32; 128]; NUM_KEYS],
        };
        for (s_idx, entry) in map.strings().iter().enumerate() {
            populate_string_ports(&mut ports, s_idx, entry, sample_rate, voicing, specs);
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
    }

    #[inline]
    pub fn clear_string_state(&mut self, s_idx: usize) {
        self.may_be_nonzero[s_idx] = false;
        self.tail_nonzero[s_idx] = false;
        self.struck_active[s_idx] = false;
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
}

#[inline]
pub(crate) fn strike_phi(key_idx: usize, partial_index: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let beta = 0.137 - 0.042 * x;
    let s = sin_cos_f64((partial_index as f64) * PI * beta).0;
    s / (1.0 + (0.25 + 4.40 * x) * ((partial_index - 1) as f64))
}

#[inline]
pub(crate) fn damper_phi(partial_index: usize) -> f64 {
    let jf = partial_index as f64;
    let (x_d, w) = (0.125_f64, 0.082_f64);
    let pw = jf * PI * w;
    let sinc_w = sin_cos_f64(pw).0 / pw;
    let cos_2xd = sin_cos_f64(2.0 * jf * PI * x_d).1;
    let env_rms = sqrt_f64((0.5 - 0.5 * cos_2xd * sinc_w).max(0.0));
    env_rms * (1.0 + 0.44 * (1.0 - exp_f64(-0.48 * ((partial_index - 1) as f64))))
}

fn populate_string_ports(
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
    sample_rate: f64,
    voicing: &VoicingSpec,
    specs: &[StringPhysicalParams; NUM_KEYS],
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
    let tau_c = (0.0032 * exp_f64(-0.35 * x - 2.75 * x * x * x)).max(0.00014);
    let z_choir = (entry.choir_size as f64) * 2.0 * f0_v * m_str;
    let i_mf = (1.55 * m_h * 1.8).min(1.45 * z_choir * tau_c * 1.8);

    let mut s_eff_sq = 0.0_f64;
    let (vb, vm) = (entry.vert_vecs.start * 4, entry.vert_modes);
    let (hb, hm) = (entry.horiz_vecs.start * 4, entry.horiz_modes);
    let pols = [
        (vb, vm, f0_v, 1.0, 1.0, f64::from(C_VERT), false),
        (
            hb,
            hm,
            f0_h,
            HORIZ_STRIKE_SCALE,
            HORIZ_DAMP_SCALE,
            f64::from(C_HORIZ),
            true,
        ),
    ];
    for &(_, modes, f0, hs, _, c_b, _) in &pols {
        for j in 1..=modes {
            let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
            if f_j > 0.0 && f_j < nyquist {
                let sinc = 1.0 / sqrt_f64(1.0 + (2.0 * f_j * tau_c) * (2.0 * f_j * tau_c));
                let w = hs * strike_phi(k, j) * c_b * sinc;
                s_eff_sq += w * w;
            }
        }
    }

    let target_rms = 0.062 * (1.0 + 0.28 * x * x);
    let gamma_k = if s_eff_sq > 0.0 {
        target_rms * m_str / (i_mf * sqrt_f64(s_eff_sq))
    } else {
        1.0
    };
    let b_gain = gamma_k * (2.0 * ts / m_str);
    ports.gamma_s[s_idx] = gamma_k;
    ports.beta_s[s_idx] = m_str / (2.0 * gamma_k * gamma_k);
    ports.alpha_vel[s_idx] = (m_str / (2.0 * ts * gamma_k * gamma_k)) as f32;

    let (mut sum_bh_sq, mut sum_bd_v_sq, mut sum_bd_h_sq) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (bh_s, ah_s) = (ports.b_strike.as_mut_slice(), ports.a_strike.as_mut_slice());
    let (bd_s, ad_s) = (ports.b_damp.as_mut_slice(), ports.a_damp.as_mut_slice());

    for &(base, modes, f0, hs, ds, _, is_horiz) in &pols {
        for j in 1..=modes {
            let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
            if f_j > 0.0 && f_j < nyquist {
                let omega = TAU * f_j;
                let ph = hs * strike_phi(k, j);
                let (bh, ah) = ((b_gain * ph) as f32, (ph / (omega * gamma_k)) as f32);
                bh_s[base + (j - 1)] = bh;
                ah_s[base + (j - 1)] = ah;
                sum_bh_sq += f64::from(bh) * f64::from(bh);
                if entry.has_damper {
                    let pd = ds * damper_phi(j);
                    let (bd, ad) = ((b_gain * pd) as f32, (pd / (omega * gamma_k)) as f32);
                    bd_s[base + (j - 1)] = bd;
                    ad_s[base + (j - 1)] = ad;
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
    ports.r_port_strike[s_idx] = (alpha64 * sum_bh_sq) as f32;
    ports.r_port_damp_v[s_idx] = (alpha64 * sum_bd_v_sq) as f32;
    ports.r_port_damp_h[s_idx] = (alpha64 * sum_bd_h_sq) as f32;
}

#[inline]
fn accum_pol_ports(
    soa: &ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    has_damper: bool,
) -> (f32, f32, f32, f32) {
    let (u, v) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (mut ayh, mut avh, mut ayd, mut avd) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in vr {
        ayh = v[i].mul_add(ah[i], ayh);
        avh = u[i].mul_add(bh[i], avh);
        if has_damper {
            ayd = v[i].mul_add(ad[i], ayd);
            avd = u[i].mul_add(bd[i], avd);
        }
    }
    (
        ayh.reduce_sum(),
        avh.reduce_sum(),
        ayd.reduce_sum(),
        avd.reduce_sum(),
    )
}

/// Recomputes cached `(y_h, v_h, y_dv, v_dv, y_dh, v_dh)` from `soa.(u, v)` after fast-tile steps.
#[inline]
pub(crate) fn refresh_string_ports(
    soa: &ModalSoa,
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
) {
    let d = entry.has_damper;
    let (yh_v, vh_v, ydv, vdv) = accum_pol_ports(soa, ports, entry.vert_vecs.clone(), d);
    let (yh_h, vh_h, ydh, vdh) = accum_pol_ports(soa, ports, entry.horiz_vecs.clone(), d);
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = yh_v + yh_h;
    ports.v_h[s_idx] = alpha * (vh_v + vh_h);
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = if d { (ydv, alpha * vdv) } else { (0.0, 0.0) };
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = if d { (ydh, alpha * vdh) } else { (0.0, 0.0) };
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn refresh_string_ports_bp(
    bp: &crate::bridge_port::BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
    has_damper: bool,
) {
    let m = &bp.meta[s_idx];
    let (bh, ah) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (tu, tv) = (bp.tl_soa.u.as_f32x4_slice(), bp.tl_soa.v.as_f32x4_slice());

    let accum_pol =
        |p_off: usize, vr_br: Range<usize>, tl_r: Range<usize>, vr_tail: Range<usize>| {
            let (mut ayh, mut avh, mut ayd, mut avd) =
                (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
            for (k, i) in vr_br.enumerate() {
                let st = bp.state[p_off + k];
                ayh = st.v.mul_add(ah[i], ayh);
                avh = st.u.mul_add(bh[i], avh);
                if has_damper {
                    ayd = st.v.mul_add(ad[i], ayd);
                    avd = st.u.mul_add(bd[i], avd);
                }
            }
            for (k, i) in vr_tail.enumerate() {
                let (u_v, v_v) = (tu[tl_r.start + k], tv[tl_r.start + k]);
                ayh = v_v.mul_add(ah[i], ayh);
                avh = u_v.mul_add(bh[i], avh);
                if has_damper {
                    ayd = v_v.mul_add(ad[i], ayd);
                    avd = u_v.mul_add(bd[i], avd);
                }
            }
            (
                ayh.reduce_sum(),
                avh.reduce_sum(),
                ayd.reduce_sum(),
                avd.reduce_sum(),
            )
        };
    let (yh_v, vh_v, ydv, vdv) = accum_pol(
        m.beta_v_off,
        m.v_br.clone(),
        m.tl_v.clone(),
        m.v_tail.clone(),
    );
    let (yh_h, vh_h, ydh, vdh) = accum_pol(
        m.beta_h_off,
        m.h_br.clone(),
        m.tl_h.clone(),
        m.h_tail.clone(),
    );
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = yh_v + yh_h;
    ports.v_h[s_idx] = alpha * (vh_v + vh_h);
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = if has_damper {
        (ydv, alpha * vdv)
    } else {
        (0.0, 0.0)
    };
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = if has_damper {
        (ydh, alpha * vdh)
    } else {
        (0.0, 0.0)
    };
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn refresh_prefix_damp_ports_bp(
    bp: &crate::bridge_port::BridgePorts,
    ports: &mut ContactPorts,
    s_idx: usize,
) {
    let m = &bp.meta[s_idx];
    let (bd, ad) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let accum = |p_off: usize, vr: Range<usize>| {
        let (mut ayd, mut avd) = (F32x4::zero(), F32x4::zero());
        for (k, i) in vr.enumerate() {
            let st = bp.state[p_off + k];
            ayd = st.v.mul_add(ad[i], ayd);
            avd = st.u.mul_add(bd[i], avd);
        }
        (ayd.reduce_sum(), avd.reduce_sum())
    };
    let (ydv, vdv) = accum(m.beta_v_off, m.v_br.clone());
    let (ydh, vdh) = accum(m.beta_h_off, m.h_br.clone());
    let alpha = ports.alpha_vel[s_idx];
    (ports.y_dv[s_idx], ports.v_dv[s_idx]) = (ydv, alpha * vdv);
    (ports.y_dh[s_idx], ports.v_dh[s_idx]) = (ydh, alpha * vdh);
    ports.ports_synced[s_idx] = true;
}

#[inline]
pub(crate) fn accum_prefix_damp_vel_bp(
    bp: &crate::bridge_port::BridgePorts,
    ports: &ContactPorts,
    p_off: usize,
    vr: Range<usize>,
) -> f32 {
    let bd = ports.b_damp.as_f32x4_slice();
    let mut acc = F32x4::zero();
    for (k, i) in vr.enumerate() {
        acc = bp.state[p_off + k].u.mul_add(bd[i], acc);
    }
    acc.reduce_sum()
}
