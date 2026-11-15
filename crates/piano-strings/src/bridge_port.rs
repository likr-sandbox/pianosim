//! Stage 2 <-> Stage 3 Wave Digital Filter (WDF) displacement-bridge port
//! precomputation and `< 5,500 Hz` SIMD prefix stepping (`no_std`).

use alloc::vec::Vec;
use core::ops::Range;

use piano_simd::{AlignedBuf, F32x4};
use piano_spec::{
    effective_string_f0_hz, partial_frequency_hz, Polarization, StringPhysicalParams, VoicingSpec,
    LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS,
};

use crate::choir_map::ChoirMap;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::{sqrt_f64, ModalSoa, C_HORIZ, C_VERT, NYQUIST_CUTOFF_RATIO};
use crate::phantom::MAX_PHANTOM_TRANSVERSE_VECS;
use crate::zoh_kernel::zoh_rotate;

/// Upper frequency cutoff (Hz) for sample-by-sample WDF bridge prefix modes.
pub const BRIDGE_PREFIX_CUTOFF_HZ: f64 = 5_500.0;

/// Stage 3 bidirectional reduced bridge and duplex scale driver (`0` runtime allocs).
pub trait BridgePortDriver {
    /// Per-string WDF mechanical port resistances `R_0,s > 0` (`N*s/m`).
    fn port_resistances(&self) -> &[f32; TOTAL_STRINGS];

    /// Partition-of-unity 8-zone bridge projection `(z0, w0, z1, w1)` per string.
    fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS];

    /// Advances bridge admittance, 8-zone velocity projection, and duplex bank by 1 sample.
    fn step_bridge_sample(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
    ) -> (f32, f32);
}

/// Precomputed per-string prefix/tail slice bounds and WDF bridge port weights.
#[derive(Clone, Debug)]
pub(crate) struct StringBridgeMeta {
    pub v_br: Range<usize>,
    pub h_br: Range<usize>,
    pub v_tail: Range<usize>,
    pub h_tail: Range<usize>,
    pub beta_v_off: usize,
    pub beta_h_off: usize,
    pub a_wave_gain: f32,
}

impl StringBridgeMeta {
    const fn empty() -> Self {
        Self {
            v_br: 0..0,
            h_br: 0..0,
            v_tail: 0..0,
            h_tail: 0..0,
            beta_v_off: 0,
            beta_h_off: 0,
            a_wave_gain: 0.0,
        }
    }
}

/// Precomputed compact WDF bridge port couplings for `< 5,500 Hz` string prefixes.
#[derive(Clone, Debug)]
pub(crate) struct BridgePorts {
    pub beta_br: AlignedBuf,
    pub r0: [f32; TOTAL_STRINGS],
    pub r_damp_prefix_v: [f32; TOTAL_STRINGS],
    pub r_damp_prefix_h: [f32; TOTAL_STRINGS],
    pub tail_n_trans: [f32; TOTAL_STRINGS],
    pub tail_n_long: [f32; TOTAL_STRINGS],
    pub meta: [StringBridgeMeta; TOTAL_STRINGS],
}

fn count_prefix_modes(
    modes: usize,
    f0: f64,
    inharmonicity_b: f64,
    cutoff_hz: f64,
    nyquist_hz: f64,
) -> usize {
    let lim = cutoff_hz.min(nyquist_hz);
    let mut n = 0usize;
    for j in 1..=modes {
        let f_j = partial_frequency_hz(f0, inharmonicity_b, j);
        if f_j > 0.0 && f_j < lim {
            n = j;
        } else {
            break;
        }
    }
    n
}

fn fill_pol_beta(
    out: &mut [f32],
    modes_in_prefix: usize,
    gain_64: f64,
    inharmonicity_b: f64,
) -> f64 {
    let mut sum_sq = 0.0_f64;
    for j in 1..=modes_in_prefix {
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        let jf = j as f64;
        let b_f32 = (sign * gain_64 * sqrt_f64(1.0 + inharmonicity_b * jf * jf)) as f32;
        out[j - 1] = b_f32;
        let bf = f64::from(b_f32);
        sum_sq += bf * bf;
    }
    sum_sq
}

impl BridgePorts {
    /// Precomputes masked per-prefix-vec `beta_br` and exact WDF port resistances `R_0,s`.
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        ports: &ContactPorts,
    ) -> Self {
        let ts = 1.0 / sample_rate;
        let nyquist = NYQUIST_CUTOFF_RATIO * sample_rate;
        let mut meta = [const { StringBridgeMeta::empty() }; TOTAL_STRINGS];
        let mut r0 = [0.0_f32; TOTAL_STRINGS];
        let mut r_damp_prefix_v = [0.0_f32; TOTAL_STRINGS];
        let mut r_damp_prefix_h = [0.0_f32; TOTAL_STRINGS];
        let mut tail_n_trans = [0.0_f32; TOTAL_STRINGS];
        let mut tail_n_long = [0.0_f32; TOTAL_STRINGS];
        let mut counts = [(0usize, 0usize, 0usize, 0usize); TOTAL_STRINGS];
        let mut total_prefix_vecs = 0usize;

        for (s_idx, entry) in map.strings().iter().enumerate() {
            let k = usize::from(entry.key_idx);
            let s_choir = usize::from(entry.string_in_choir);
            let spec = specs[k];
            let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
            let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
            let nv_cut = count_prefix_modes(
                entry.vert_modes,
                f0_v,
                spec.inharmonicity_b,
                BRIDGE_PREFIX_CUTOFF_HZ,
                nyquist,
            );
            let nh_cut = count_prefix_modes(
                entry.horiz_modes,
                f0_h,
                spec.inharmonicity_b,
                BRIDGE_PREFIX_CUTOFF_HZ,
                nyquist,
            );
            let sav_min = if k < LONGITUDINAL_MAX_KEY_IDX {
                MAX_PHANTOM_TRANSVERSE_VECS
            } else {
                0
            };
            let nv_vecs = nv_cut.div_ceil(4).max(sav_min).min(entry.vert_vecs.len());
            let nh_vecs = nh_cut.div_ceil(4).max(sav_min).min(entry.horiz_vecs.len());
            counts[s_idx] = (nv_cut, nv_vecs, nh_cut, nh_vecs);
            total_prefix_vecs += nv_vecs + nh_vecs;
        }

        let mut beta_vals: Vec<f32> = Vec::with_capacity(total_prefix_vecs * 4);
        beta_vals.resize(total_prefix_vecs * 4, 0.0);
        let mut vec_cursor = 0usize;

        for (s_idx, entry) in map.strings().iter().enumerate() {
            let k = usize::from(entry.key_idx);
            let s_choir = usize::from(entry.string_in_choir);
            let spec = specs[k];
            let (nv_cut, nv_vecs, nh_cut, nh_vecs) = counts[s_idx];

            let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
            let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
            let l_m = spec.speaking_length_m;
            let gamma = ports.gamma_s[s_idx];
            let gain_v = gamma * (2.0 * (2.0 * l_m * f0_v) * ts / l_m) * f64::from(C_VERT);
            let gain_h = gamma * (2.0 * (2.0 * l_m * f0_h) * ts / l_m) * f64::from(C_HORIZ);

            let beta_v_off = vec_cursor;
            let v_slot_off = beta_v_off * 4;
            let sum_sq_v = fill_pol_beta(
                &mut beta_vals[v_slot_off..v_slot_off + nv_vecs * 4],
                nv_cut,
                gain_v,
                spec.inharmonicity_b,
            );
            vec_cursor += nv_vecs;

            let beta_h_off = vec_cursor;
            let h_slot_off = beta_h_off * 4;
            let sum_sq_h = fill_pol_beta(
                &mut beta_vals[h_slot_off..h_slot_off + nh_vecs * 4],
                nh_cut,
                gain_h,
                spec.inharmonicity_b,
            );
            vec_cursor += nh_vecs;

            let sum_sq = sum_sq_v + sum_sq_h;
            let alpha64 = f64::from(ports.alpha_vel[s_idx]);
            let r0_val = (0.5 * alpha64 * sum_sq) as f32;
            if r0_val > 0.0 && r0_val.is_finite() {
                r0[s_idx] = r0_val;
                meta[s_idx].a_wave_gain = (alpha64 / f64::from(r0_val)) as f32;
            } else {
                r0[s_idx] = 1.0e-9;
                meta[s_idx].a_wave_gain = 0.0;
            }

            let base_n = if gamma > 0.0 {
                2.0 * spec.linear_density_kg_m * l_m * f0_v / gamma
            } else {
                0.0
            };
            let cut_j = (nv_cut + 1) as f64;
            tail_n_trans[s_idx] =
                (base_n * sqrt_f64(1.0 + spec.inharmonicity_b * cut_j * cut_j)) as f32;
            tail_n_long[s_idx] = base_n as f32;

            meta[s_idx].v_br = entry.vert_vecs.start..(entry.vert_vecs.start + nv_vecs);
            meta[s_idx].v_tail = (entry.vert_vecs.start + nv_vecs)..entry.vert_vecs.end;
            meta[s_idx].h_br = entry.horiz_vecs.start..(entry.horiz_vecs.start + nh_vecs);
            meta[s_idx].h_tail = (entry.horiz_vecs.start + nh_vecs)..entry.horiz_vecs.end;
            meta[s_idx].beta_v_off = beta_v_off;
            meta[s_idx].beta_h_off = beta_h_off;

            // Prefix damper port resistances match `r_port_damp_v/h` (`alpha * sum(b_damp^2)`)
            // so `1 + 0.5 * rp * c` preserves exact discrete passivity on `< 5,500 Hz` prefixes.
            if entry.has_damper {
                let bd = ports.b_damp.as_slice();
                let sum_sq_slice = |vr: &Range<usize>| -> f64 {
                    bd[(vr.start * 4)..(vr.end * 4)]
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum()
                };
                r_damp_prefix_v[s_idx] = (alpha64 * sum_sq_slice(&meta[s_idx].v_br)) as f32;
                r_damp_prefix_h[s_idx] = (alpha64 * sum_sq_slice(&meta[s_idx].h_br)) as f32;
            }
        }

        let mut beta_br = AlignedBuf::zeroed(total_prefix_vecs.max(1) * 4);
        beta_br.as_mut_slice()[..beta_vals.len()].copy_from_slice(&beta_vals);
        Self {
            beta_br,
            r0,
            r_damp_prefix_v,
            r_damp_prefix_h,
            tail_n_trans,
            tail_n_long,
            meta,
        }
    }

    /// Computes incident velocity waves `a_waves[s]` from current modal `soa.v`.
    pub fn compute_a_waves(
        &self,
        soa: &ModalSoa,
        may_be_nonzero: &[bool; TOTAL_STRINGS],
        a_waves: &mut [f32; TOTAL_STRINGS],
    ) {
        let v = soa.v.as_f32x4_slice();
        let beta = self.beta_br.as_f32x4_slice();
        for (s_idx, m) in self.meta.iter().enumerate() {
            if !may_be_nonzero[s_idx] {
                a_waves[s_idx] = 0.0;
                continue;
            }
            let sum_v = dot_beta_v(v, beta, m.v_br.clone(), m.beta_v_off);
            let sum_h = dot_beta_v(v, beta, m.h_br.clone(), m.beta_h_off);
            a_waves[s_idx] = m.a_wave_gain * (sum_v + sum_h);
        }
    }
}

#[inline(always)]
fn dot_beta_v(v: &[F32x4], beta: &[F32x4], vr: Range<usize>, beta_off: usize) -> f32 {
    let n = vr.end - vr.start;
    let vs = &v[vr];
    let bs = &beta[beta_off..beta_off + n];
    let mut acc = F32x4::zero();
    for i in 0..n {
        acc = vs[i].mul_add(bs[i], acc);
    }
    acc.reduce_sum()
}

/// Applies `v -= beta_br * v_br`, steps prefix ZOH rotation, accumulates `c_bridge` audio
/// readout, and computes the next-sample `beta_br . v_next` dot product in a single SIMD pass.
#[inline(always)]
pub(crate) fn step_pol_prefix_wdf(
    soa: &mut ModalSoa,
    beta_buf: &[F32x4],
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
) -> (f32, f32) {
    let n = vr.end - vr.start;
    let (u_all, v_all) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let u = &mut u_all[vr.clone()];
    let v = &mut v_all[vr.clone()];
    let d = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c = &soa.c_bridge.as_f32x4_slice()[vr];
    let beta = &beta_buf[beta_off..beta_off + n];

    let (mut acc_c, mut acc_br) = (F32x4::zero(), F32x4::zero());
    for i in 0..n {
        let vc = beta[i].mul_add(neg_vbr, v[i]);
        let (un, vn) = zoh_rotate(u[i], vc, d[i], s[i]);
        u[i] = un;
        v[i] = vn;
        acc_c = vn.mul_add(c[i], acc_c);
        acc_br = vn.mul_add(beta[i], acc_br);
    }
    (acc_c.reduce_sum(), acc_br.reduce_sum())
}

/// Same as [`step_pol_prefix_wdf`] with simultaneous hammer/damper force injection into `u`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_inj(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    beta_buf: &[F32x4],
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_h: f32,
    f_d: f32,
    has_damper: bool,
) -> (f32, f32) {
    let n = vr.end - vr.start;
    let u = &mut soa.u.as_f32x4_slice_mut()[vr.clone()];
    let v = &mut soa.v.as_f32x4_slice_mut()[vr.clone()];
    let d = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c = &soa.c_bridge.as_f32x4_slice()[vr.clone()];
    let bh = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr];
    let beta = &beta_buf[beta_off..beta_off + n];
    let (fh_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(f_d));

    let (mut acc_c, mut acc_br) = (F32x4::zero(), F32x4::zero());
    for i in 0..n {
        let u_inj = if has_damper {
            fd_v.mul_add(bd[i], fh_v.mul_add(bh[i], u[i]))
        } else {
            fh_v.mul_add(bh[i], u[i])
        };
        let bi = beta[i];
        let vc = bi.mul_add(neg_vbr, v[i]);
        let (un, vn) = zoh_rotate(u_inj, vc, d[i], s[i]);
        u[i] = un;
        v[i] = vn;
        acc_c = vn.mul_add(c[i], acc_c);
        acc_br = vn.mul_add(bi, acc_br);
    }
    (acc_c.reduce_sum(), acc_br.reduce_sum())
}

/// Single-pass prefix WDF step for damped sympathetic strings (`KeyCat::DampedSymp`):
/// injects optional `f_d * b_damp` into `u`, couples `-beta_br * v_br` onto `v`,
/// advances ZOH rotation, and accumulates `(y_d, v_d)` alongside `(acc_c, acc_br)` in registers.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_pol_prefix_wdf_symp_damp(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    beta_buf: &[F32x4],
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_d: f32,
) -> (f32, f32, f32, f32) {
    if f_d != 0.0 {
        step_pol_prefix_wdf_symp_damp_inner::<true>(
            soa, ports, beta_buf, vr, beta_off, neg_vbr, f_d,
        )
    } else {
        step_pol_prefix_wdf_symp_damp_inner::<false>(
            soa, ports, beta_buf, vr, beta_off, neg_vbr, 0.0,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_pol_prefix_wdf_symp_damp_inner<const HAS_FD: bool>(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    beta_buf: &[F32x4],
    vr: Range<usize>,
    beta_off: usize,
    neg_vbr: F32x4,
    f_d: f32,
) -> (f32, f32, f32, f32) {
    let n = vr.end - vr.start;
    let (u_all, v_all) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let u = &mut u_all[vr.clone()];
    let v = &mut v_all[vr.clone()];
    let d = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c = &soa.c_bridge.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr.clone()];
    let ad = &ports.a_damp.as_f32x4_slice()[vr];
    let beta = &beta_buf[beta_off..beta_off + n];
    let fd_v = F32x4::splat(f_d);

    let (mut acc_c, mut acc_br, mut acc_yd, mut acc_vd) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in 0..n {
        let bdi = bd[i];
        let u_inj = if HAS_FD {
            fd_v.mul_add(bdi, u[i])
        } else {
            u[i]
        };
        let bi = beta[i];
        let vc = bi.mul_add(neg_vbr, v[i]);
        let (un, vn) = zoh_rotate(u_inj, vc, d[i], s[i]);
        u[i] = un;
        v[i] = vn;
        acc_c = vn.mul_add(c[i], acc_c);
        acc_br = vn.mul_add(bi, acc_br);
        acc_yd = vn.mul_add(ad[i], acc_yd);
        acc_vd = un.mul_add(bdi, acc_vd);
    }
    (
        acc_c.reduce_sum(),
        acc_br.reduce_sum(),
        acc_yd.reduce_sum(),
        acc_vd.reduce_sum(),
    )
}
