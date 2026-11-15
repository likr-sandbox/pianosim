//! Stage 2 <-> Stage 3 Wave Digital Filter (WDF) displacement-bridge port
//! precomputation and compact `< 4,500 Hz` prefix table packing (`no_std`).

use alloc::boxed::Box;
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

/// Upper frequency cutoff (Hz) for sample-by-sample WDF bridge prefix modes.
pub const BRIDGE_PREFIX_CUTOFF_HZ: f64 = 4_500.0;

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

/// Compact per-string prefix metadata for contiguous L1 uncoupled WDF stepping.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct FastPrefixMeta {
    pub n_vh: u32,
    pub a_wave_gain: f32,
    pub pan_l: f32,
    pub pan_r: f32,
}

impl FastPrefixMeta {
    pub const ZERO: Self = Self {
        n_vh: 0,
        a_wave_gain: 0.0,
        pan_l: 0.0,
        pan_r: 0.0,
    };
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
    pub n_vh: usize,
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
            n_vh: 0,
            a_wave_gain: 0.0,
        }
    }
}

/// Precomputed compact WDF bridge port couplings and L1-packed ZOH state/tables.
#[derive(Clone, Debug)]
pub(crate) struct BridgePorts {
    pub total_prefix_vecs: usize,
    pub beta_br: AlignedBuf,
    pub d_br: AlignedBuf,
    pub s_br: AlignedBuf,
    pub c_br: AlignedBuf,
    pub u_br: AlignedBuf,
    pub v_br: AlignedBuf,
    pub c_tail_l: AlignedBuf,
    pub c_tail_r: AlignedBuf,
    pub tail_idx_scratch: Box<[u16]>,
    pub fast_meta: [FastPrefixMeta; TOTAL_STRINGS],
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
        if f_j > 0.0 && f_j < lim && (f_j < 1_800.0 || j <= 24) {
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
    /// Precomputes masked per-prefix-vec `beta_br` and packed prefix ZOH arrays `(d_br, s_br, c_br)`.
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        ports: &ContactPorts,
        soa: &ModalSoa,
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

        assert!(map.total_vecs() <= usize::from(u16::MAX));
        let mut beta_vals: Vec<f32> = Vec::with_capacity(total_prefix_vecs * 4);
        beta_vals.resize(total_prefix_vecs * 4, 0.0);
        let mut d_br = AlignedBuf::zeroed(total_prefix_vecs.max(1) * 4);
        let mut s_br = AlignedBuf::zeroed(total_prefix_vecs.max(1) * 4);
        let mut c_br = AlignedBuf::zeroed(total_prefix_vecs.max(1) * 4);
        let u_br = AlignedBuf::zeroed(total_prefix_vecs.max(1) * 4);
        let v_br = AlignedBuf::zeroed(total_prefix_vecs.max(1) * 4);
        let mut c_tail_l = AlignedBuf::zeroed(map.total_vecs().max(1) * 4);
        let mut c_tail_r = AlignedBuf::zeroed(map.total_vecs().max(1) * 4);
        let tail_idx_scratch = alloc::vec![0_u16; map.total_vecs()].into_boxed_slice();
        let mut fast_meta = [FastPrefixMeta::ZERO; TOTAL_STRINGS];
        let d_src = soa.d_coeff.as_f32x4_slice();
        let s_src = soa.s_coeff.as_f32x4_slice();
        let c_src = soa.c_bridge.as_f32x4_slice();
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
            meta[s_idx].n_vh = nv_vecs + nh_vecs;

            fast_meta[s_idx] = FastPrefixMeta {
                n_vh: (nv_vecs + nh_vecs) as u32,
                a_wave_gain: meta[s_idx].a_wave_gain,
                pan_l: entry.pan_l,
                pan_r: entry.pan_r,
            };

            let (v_br_r, h_br_r) = (meta[s_idx].v_br.clone(), meta[s_idx].h_br.clone());
            d_br.as_f32x4_slice_mut()[beta_v_off..beta_v_off + nv_vecs]
                .copy_from_slice(&d_src[v_br_r.clone()]);
            s_br.as_f32x4_slice_mut()[beta_v_off..beta_v_off + nv_vecs]
                .copy_from_slice(&s_src[v_br_r.clone()]);
            c_br.as_f32x4_slice_mut()[beta_v_off..beta_v_off + nv_vecs]
                .copy_from_slice(&c_src[v_br_r]);
            d_br.as_f32x4_slice_mut()[beta_h_off..beta_h_off + nh_vecs]
                .copy_from_slice(&d_src[h_br_r.clone()]);
            s_br.as_f32x4_slice_mut()[beta_h_off..beta_h_off + nh_vecs]
                .copy_from_slice(&s_src[h_br_r.clone()]);
            c_br.as_f32x4_slice_mut()[beta_h_off..beta_h_off + nh_vecs]
                .copy_from_slice(&c_src[h_br_r]);

            let (pl, pr) = (F32x4::splat(entry.pan_l), F32x4::splat(entry.pan_r));
            let (cl_sl, cr_sl) = (c_tail_l.as_f32x4_slice_mut(), c_tail_r.as_f32x4_slice_mut());
            for i in entry.all_vecs() {
                cl_sl[i] = c_src[i] * pl;
                cr_sl[i] = c_src[i] * pr;
            }

            // Prefix damper port resistances match `r_port_damp_v/h` (`alpha * sum(b_damp^2)`)
            // so `1 + 0.5 * rp * c` preserves exact discrete passivity on `< 4,500 Hz` prefixes.
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
            total_prefix_vecs,
            beta_br,
            d_br,
            s_br,
            c_br,
            u_br,
            v_br,
            c_tail_l,
            c_tail_r,
            tail_idx_scratch,
            fast_meta,
            r0,
            r_damp_prefix_v,
            r_damp_prefix_h,
            tail_n_trans,
            tail_n_long,
            meta,
        }
    }

    /// Zeroes packed L1 prefix state buffers without reallocating.
    #[inline]
    pub fn reset_resting(&mut self) {
        self.u_br.as_mut_slice().fill(0.0);
        self.v_br.as_mut_slice().fill(0.0);
    }

    /// Copies all 230 strings' prefix `(u, v)` state from `soa` into L1-packed buffers.
    pub fn pack_from_soa(&mut self, soa: &ModalSoa) {
        let (u_src, v_src) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
        let (u_dst, v_dst) = (
            self.u_br.as_f32x4_slice_mut(),
            self.v_br.as_f32x4_slice_mut(),
        );
        for m in &self.meta {
            let (nv, nh) = (m.v_br.len(), m.h_br.len());
            u_dst[m.beta_v_off..m.beta_v_off + nv].copy_from_slice(&u_src[m.v_br.clone()]);
            v_dst[m.beta_v_off..m.beta_v_off + nv].copy_from_slice(&v_src[m.v_br.clone()]);
            u_dst[m.beta_h_off..m.beta_h_off + nh].copy_from_slice(&u_src[m.h_br.clone()]);
            v_dst[m.beta_h_off..m.beta_h_off + nh].copy_from_slice(&v_src[m.h_br.clone()]);
        }
    }

    /// Copies all 230 strings' prefix `(u, v)` state from L1-packed buffers back to `soa`.
    pub fn unpack_to_soa(&self, soa: &mut ModalSoa) {
        let (u_dst, v_dst) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
        let (u_src, v_src) = (self.u_br.as_f32x4_slice(), self.v_br.as_f32x4_slice());
        for m in &self.meta {
            let (nv, nh) = (m.v_br.len(), m.h_br.len());
            u_dst[m.v_br.clone()].copy_from_slice(&u_src[m.beta_v_off..m.beta_v_off + nv]);
            v_dst[m.v_br.clone()].copy_from_slice(&v_src[m.beta_v_off..m.beta_v_off + nv]);
            u_dst[m.h_br.clone()].copy_from_slice(&u_src[m.beta_h_off..m.beta_h_off + nh]);
            v_dst[m.h_br.clone()].copy_from_slice(&v_src[m.beta_h_off..m.beta_h_off + nh]);
        }
    }

    /// Syncs a single string's packed prefix `(u, v)` out to `soa`.
    #[inline(always)]
    pub fn sync_string_to_soa(&self, soa: &mut ModalSoa, s_idx: usize) {
        let m = &self.meta[s_idx];
        let (u_dst, v_dst) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
        let (u_src, v_src) = (self.u_br.as_f32x4_slice(), self.v_br.as_f32x4_slice());
        let (nv, nh) = (m.v_br.len(), m.h_br.len());
        u_dst[m.v_br.clone()].copy_from_slice(&u_src[m.beta_v_off..m.beta_v_off + nv]);
        v_dst[m.v_br.clone()].copy_from_slice(&v_src[m.beta_v_off..m.beta_v_off + nv]);
        u_dst[m.h_br.clone()].copy_from_slice(&u_src[m.beta_h_off..m.beta_h_off + nh]);
        v_dst[m.h_br.clone()].copy_from_slice(&v_src[m.beta_h_off..m.beta_h_off + nh]);
    }

    /// Syncs a single string's prefix `(u, v)` from `soa` into L1-packed buffers.
    #[inline(always)]
    pub fn sync_string_from_soa(&mut self, soa: &ModalSoa, s_idx: usize) {
        let m = &self.meta[s_idx];
        let (u_src, v_src) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
        let (u_dst, v_dst) = (
            self.u_br.as_f32x4_slice_mut(),
            self.v_br.as_f32x4_slice_mut(),
        );
        let (nv, nh) = (m.v_br.len(), m.h_br.len());
        u_dst[m.beta_v_off..m.beta_v_off + nv].copy_from_slice(&u_src[m.v_br.clone()]);
        v_dst[m.beta_v_off..m.beta_v_off + nv].copy_from_slice(&v_src[m.v_br.clone()]);
        u_dst[m.beta_h_off..m.beta_h_off + nh].copy_from_slice(&u_src[m.h_br.clone()]);
        v_dst[m.beta_h_off..m.beta_h_off + nh].copy_from_slice(&v_src[m.h_br.clone()]);
    }

    /// Computes incident velocity waves `a_waves[s]` from L1-packed `v_br`.
    pub fn compute_a_waves(
        &self,
        may_be_nonzero: &[bool; TOTAL_STRINGS],
        a_waves: &mut [f32; TOTAL_STRINGS],
    ) {
        let v = self.v_br.as_f32x4_slice();
        let beta = self.beta_br.as_f32x4_slice();
        for (s_idx, m) in self.meta.iter().enumerate() {
            if !may_be_nonzero[s_idx] {
                a_waves[s_idx] = 0.0;
                continue;
            }
            let mut acc = F32x4::zero();
            for i in m.beta_v_off..(m.beta_v_off + m.n_vh) {
                acc = v[i].mul_add(beta[i], acc);
            }
            a_waves[s_idx] = m.a_wave_gain * acc.reduce_sum();
        }
    }
}
