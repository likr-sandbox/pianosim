//! Stage 2 <-> Stage 3 Wave Digital Filter (WDF) displacement-bridge port
//! precomputation, L1d-packed `< 3,800 Hz` prefix tables, and contiguous tail SoA (`no_std`).

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

/// Alternating spatial mode sign `[+C_VERT, -C_VERT, +C_VERT, -C_VERT]` for vertical prefix vectors.
pub(crate) const C_SIGN_V_ARR: [f32; 4] = [C_VERT, -C_VERT, C_VERT, -C_VERT];
/// Alternating spatial mode sign `[+C_HORIZ, -C_HORIZ, +C_HORIZ, -C_HORIZ]` for horizontal prefix vectors.
pub(crate) const C_SIGN_H_ARR: [f32; 4] = [C_HORIZ, -C_HORIZ, C_HORIZ, -C_HORIZ];

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

    /// Advances bridge admittance, 8-zone velocity projection, and per-zone duplex bank (`Zones 2..7`).
    #[inline]
    fn step_bridge_sample_with_duplex_zones(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
        dup_zone_out: &mut [f32; 8],
    ) -> (f32, f32) {
        dup_zone_out.fill(0.0);
        self.step_bridge_sample(a_waves, zone_str_audio, b_waves_out, zone_out)
    }
}

/// 32-byte L1d-packed coupled-form modal state `(u, v)` for one 4-mode prefix vector.
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct PrefixState {
    pub u: F32x4,
    pub v: F32x4,
}

/// 48-byte L1d-packed read-only ZOH poles + WDF port vector `(d, s, beta)` (`c` is analytic!).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct PrefixCoef {
    pub d: F32x4,
    pub s: F32x4,
    pub beta: F32x4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct FastPrefixMeta {
    pub a_wave_gain: f32,
    pub pan_l: f32,
    pub pan_r: f32,
}

#[rustfmt::skip]
impl FastPrefixMeta {
    pub const ZERO: Self = Self { a_wave_gain: 0.0, pan_l: 0.0, pan_r: 0.0 };
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PrefixRun {
    pub start_s: u8,
    pub count: u8,
    pub n_pol: u8,
    pub p_off: u16,
}

#[rustfmt::skip]
impl PrefixRun {
    pub const ZERO: Self = Self { start_s: 0, count: 0, n_pol: 0, p_off: 0 };
}

pub(crate) const MAX_PREFIX_RUNS: usize = 16;

/// Per-key contiguous ranges in `BridgePorts::tl_soa` across all unison strings of key `k`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct KeyTailRanges {
    pub th_end: u16,
    pub l_end: u16,
    pub pan_l: f32,
    pub pan_r: f32,
    pub tail_n_trans: f32,
    pub tail_n_long: f32,
}

#[rustfmt::skip]
impl KeyTailRanges {
    pub const ZERO: Self = Self {
        th_end: 0, l_end: 0, pan_l: 0.0, pan_r: 0.0, tail_n_trans: 0.0, tail_n_long: 0.0,
    };
}

/// Precomputed per-string prefix/tail slice bounds and WDF bridge port weights.
#[derive(Clone, Debug)]
pub(crate) struct StringBridgeMeta {
    pub v_br: Range<usize>,
    pub h_br: Range<usize>,
    pub v_tail: Range<usize>,
    pub h_tail: Range<usize>,
    pub tl_v: Range<usize>,
    pub tl_h: Range<usize>,
    pub tl_l: Range<usize>,
    pub beta_v_off: usize,
    pub beta_h_off: usize,
    pub n_vh: usize,
    pub a_wave_gain: f32,
}

#[rustfmt::skip]
impl StringBridgeMeta {
    const fn empty() -> Self {
        Self {
            v_br: 0..0, h_br: 0..0, v_tail: 0..0, h_tail: 0..0, tl_v: 0..0, tl_h: 0..0,
            tl_l: 0..0, beta_v_off: 0, beta_h_off: 0, n_vh: 0, a_wave_gain: 0.0,
        }
    }
}

/// Precomputed L1d-resident WDF prefix state/coefficients and key-contiguous tail `ModalSoa`.
#[derive(Clone, Debug)]
pub(crate) struct BridgePorts {
    pub state: Box<[PrefixState]>,
    pub coef: Box<[PrefixCoef]>,
    pub damp_bd: Box<[F32x4]>,
    pub damp_ad: Box<[F32x4]>,
    pub tl_soa: ModalSoa,
    pub fast_meta: [FastPrefixMeta; TOTAL_STRINGS],
    pub a_wave_gains: [f32; TOTAL_STRINGS],
    pub pan_l: [f32; TOTAL_STRINGS],
    pub pan_r: [f32; TOTAL_STRINGS],
    pub prefix_runs: [PrefixRun; MAX_PREFIX_RUNS],
    pub n_prefix_runs: usize,
    pub key_tails: [KeyTailRanges; NUM_KEYS],
    pub r0: [f32; TOTAL_STRINGS],
    pub r_damp_prefix_v: [f32; TOTAL_STRINGS],
    pub r_damp_prefix_h: [f32; TOTAL_STRINGS],
    pub tail_n_trans: [f32; TOTAL_STRINGS],
    pub tail_n_long: [f32; TOTAL_STRINGS],
    pub meta: [StringBridgeMeta; TOTAL_STRINGS],
}

pub(crate) fn count_prefix_modes(
    key_idx: usize,
    modes: usize,
    f0: f64,
    inharmonicity_b: f64,
    nyquist_hz: f64,
) -> usize {
    let (cutoff_hz, max_j) = if key_idx < 16 {
        (1_250.0_f64, 12usize)
    } else if key_idx < 28 {
        (2_500.0_f64, 12usize)
    } else if key_idx < 40 {
        (3_800.0_f64, 8usize)
    } else {
        (3_800.0_f64, 4usize)
    };
    let lim = cutoff_hz.min(nyquist_hz);
    let mut n = 0usize;
    for j in 1..=modes.min(max_j) {
        let f_j = partial_frequency_hz(f0, inharmonicity_b, j);
        if f_j > 0.0 && f_j < lim {
            n = j;
        } else {
            break;
        }
    }
    n
}

pub(crate) fn fill_pol_beta(
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
    /// Precomputes L1d-packed `state`/`coef` and key-grouped contiguous `tl_soa`.
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
        let mut total_tail_vecs = 0usize;

        for (s_idx, entry) in map.strings().iter().enumerate() {
            let k = usize::from(entry.key_idx);
            let s_choir = usize::from(entry.string_in_choir);
            let spec = specs[k];
            let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
            let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
            let nv_cut =
                count_prefix_modes(k, entry.vert_modes, f0_v, spec.inharmonicity_b, nyquist);
            let nh_cut =
                count_prefix_modes(k, entry.horiz_modes, f0_h, spec.inharmonicity_b, nyquist);
            let sav_min = if k < LONGITUDINAL_MAX_KEY_IDX {
                MAX_PHANTOM_TRANSVERSE_VECS
            } else {
                0
            };
            let n_pol = nv_cut
                .max(nh_cut)
                .div_ceil(4)
                .max(sav_min)
                .min(entry.vert_vecs.len())
                .min(entry.horiz_vecs.len());
            let (nv_vecs, nh_vecs) = (n_pol, n_pol);
            counts[s_idx] = (nv_cut, nv_vecs, nh_cut, nh_vecs);
            total_prefix_vecs += nv_vecs + nh_vecs;
            total_tail_vecs += (entry.vert_vecs.len() - nv_vecs)
                + (entry.horiz_vecs.len() - nh_vecs)
                + entry.long_vecs.len();
        }

        assert!(total_tail_vecs <= usize::from(u16::MAX));
        let cap_p = total_prefix_vecs.max(1);
        let z = F32x4::zero();
        let state = alloc::vec![PrefixState { u: z, v: z }; cap_p].into_boxed_slice();
        let mut coef = alloc::vec![
            PrefixCoef {
                d: F32x4::splat(1.0),
                s: z,
                beta: z,
            };
            cap_p
        ]
        .into_boxed_slice();

        let tl_slots = total_tail_vecs.max(1) * 4;
        let mut tl_soa = ModalSoa {
            u: AlignedBuf::zeroed(tl_slots),
            v: AlignedBuf::zeroed(tl_slots),
            d_coeff: AlignedBuf::filled(tl_slots, 1.0),
            s_coeff: AlignedBuf::zeroed(tl_slots),
            c_bridge: AlignedBuf::zeroed(tl_slots),
            b_exc: AlignedBuf::zeroed(4),
        };

        let mut fast_meta = [FastPrefixMeta::ZERO; TOTAL_STRINGS];
        let mut a_wave_gains = [0.0_f32; TOTAL_STRINGS];
        let mut pan_l = [0.0_f32; TOTAL_STRINGS];
        let mut pan_r = [0.0_f32; TOTAL_STRINGS];
        let mut key_tails = [KeyTailRanges::ZERO; NUM_KEYS];
        let d_src = soa.d_coeff.as_f32x4_slice();
        let s_src = soa.s_coeff.as_f32x4_slice();
        let c_src = soa.c_bridge.as_f32x4_slice();

        let mut p_cur = 0usize;
        let mut tl_cur = 0usize;

        for (k, key) in map.keys().iter().enumerate() {
            let spec = specs[k];
            let l_m = spec.speaking_length_m;

            for s_idx in key.string_range.clone() {
                let entry = &map.strings()[s_idx];
                let s_choir = usize::from(entry.string_in_choir);
                let (nv_cut, nv_vecs, nh_cut, nh_vecs) = counts[s_idx];
                let f0_v =
                    effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
                let f0_h =
                    effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
                let gamma = ports.gamma_s[s_idx];
                let gain_v = gamma * (2.0 * (2.0 * l_m * f0_v) * ts / l_m) * f64::from(C_VERT);
                let gain_h = gamma * (2.0 * (2.0 * l_m * f0_h) * ts / l_m) * f64::from(C_HORIZ);

                let beta_v_off = p_cur;
                let mut bv_tmp = Vec::new();
                bv_tmp.resize(nv_vecs * 4, 0.0_f32);
                let sum_sq_v = fill_pol_beta(&mut bv_tmp, nv_cut, gain_v, spec.inharmonicity_b);
                for vi in 0..nv_vecs {
                    let src_i = entry.vert_vecs.start + vi;
                    coef[p_cur + vi] = PrefixCoef {
                        d: d_src[src_i],
                        s: s_src[src_i],
                        beta: F32x4::from_slice(&bv_tmp[vi * 4..vi * 4 + 4]),
                    };
                }
                p_cur += nv_vecs;

                let beta_h_off = p_cur;
                let mut bh_tmp = Vec::new();
                bh_tmp.resize(nh_vecs * 4, 0.0_f32);
                let sum_sq_h = fill_pol_beta(&mut bh_tmp, nh_cut, gain_h, spec.inharmonicity_b);
                for vi in 0..nh_vecs {
                    let src_i = entry.horiz_vecs.start + vi;
                    coef[p_cur + vi] = PrefixCoef {
                        d: d_src[src_i],
                        s: s_src[src_i],
                        beta: F32x4::from_slice(&bh_tmp[vi * 4..vi * 4 + 4]),
                    };
                }
                p_cur += nh_vecs;

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

                let v_br = entry.vert_vecs.start..(entry.vert_vecs.start + nv_vecs);
                let v_tail = (entry.vert_vecs.start + nv_vecs)..entry.vert_vecs.end;
                let h_br = entry.horiz_vecs.start..(entry.horiz_vecs.start + nh_vecs);
                let h_tail = (entry.horiz_vecs.start + nh_vecs)..entry.horiz_vecs.end;

                let tl_v = tl_cur..(tl_cur + v_tail.len());
                tl_cur = tl_v.end;
                let tl_h = tl_cur..(tl_cur + h_tail.len());
                tl_cur = tl_h.end;

                tl_soa.d_coeff.as_f32x4_slice_mut()[tl_v.clone()]
                    .copy_from_slice(&d_src[v_tail.clone()]);
                tl_soa.s_coeff.as_f32x4_slice_mut()[tl_v.clone()]
                    .copy_from_slice(&s_src[v_tail.clone()]);
                tl_soa.c_bridge.as_f32x4_slice_mut()[tl_v.clone()]
                    .copy_from_slice(&c_src[v_tail.clone()]);
                tl_soa.d_coeff.as_f32x4_slice_mut()[tl_h.clone()]
                    .copy_from_slice(&d_src[h_tail.clone()]);
                tl_soa.s_coeff.as_f32x4_slice_mut()[tl_h.clone()]
                    .copy_from_slice(&s_src[h_tail.clone()]);
                tl_soa.c_bridge.as_f32x4_slice_mut()[tl_h.clone()]
                    .copy_from_slice(&c_src[h_tail.clone()]);

                meta[s_idx] = StringBridgeMeta {
                    v_br: v_br.clone(),
                    h_br: h_br.clone(),
                    v_tail,
                    h_tail,
                    tl_v,
                    tl_h,
                    tl_l: 0..0,
                    beta_v_off,
                    beta_h_off,
                    n_vh: nv_vecs + nh_vecs,
                    a_wave_gain: meta[s_idx].a_wave_gain,
                };
                a_wave_gains[s_idx] = meta[s_idx].a_wave_gain;
                pan_l[s_idx] = entry.pan_l;
                pan_r[s_idx] = entry.pan_r;
                fast_meta[s_idx] = FastPrefixMeta {
                    a_wave_gain: meta[s_idx].a_wave_gain,
                    pan_l: entry.pan_l,
                    pan_r: entry.pan_r,
                };

                if entry.has_damper {
                    let bd = ports.b_damp.as_slice();
                    let sum_sq_slice = |vr: &Range<usize>| -> f64 {
                        bd[(vr.start * 4)..(vr.end * 4)]
                            .iter()
                            .map(|&x| f64::from(x) * f64::from(x))
                            .sum()
                    };
                    r_damp_prefix_v[s_idx] = (alpha64 * sum_sq_slice(&v_br)) as f32;
                    r_damp_prefix_h[s_idx] = (alpha64 * sum_sq_slice(&h_br)) as f32;
                }
            }
            let th_end = tl_cur;

            for s_idx in key.string_range.clone() {
                let entry = &map.strings()[s_idx];
                let l_src = entry.long_vecs.clone();
                let tl_l = tl_cur..(tl_cur + l_src.len());
                tl_cur = tl_l.end;
                tl_soa.d_coeff.as_f32x4_slice_mut()[tl_l.clone()]
                    .copy_from_slice(&d_src[l_src.clone()]);
                tl_soa.s_coeff.as_f32x4_slice_mut()[tl_l.clone()]
                    .copy_from_slice(&s_src[l_src.clone()]);
                tl_soa.c_bridge.as_f32x4_slice_mut()[tl_l.clone()].copy_from_slice(&c_src[l_src]);
                meta[s_idx].tl_l = tl_l;
            }
            let s0 = key.string_range.start;
            let e0 = &map.strings()[s0];
            key_tails[k] = KeyTailRanges {
                th_end: th_end as u16,
                l_end: tl_cur as u16,
                pan_l: e0.pan_l,
                pan_r: e0.pan_r,
                tail_n_trans: tail_n_trans[s0],
                tail_n_long: tail_n_long[s0],
            };
        }

        let (mut prefix_runs, mut n_prefix_runs, mut s_run) =
            ([PrefixRun::ZERO; MAX_PREFIX_RUNS], 0usize, 0usize);
        while s_run < TOTAL_STRINGS {
            let (n_pol, p_off, has_damp) = (
                counts[s_run].1,
                meta[s_run].beta_v_off as u16,
                map.strings()[s_run].has_damper,
            );
            let mut end_s = s_run + 1;
            while end_s < TOTAL_STRINGS
                && counts[end_s].1 == n_pol
                && map.strings()[end_s].has_damper == has_damp
            {
                end_s += 1;
            }
            assert!(n_prefix_runs < MAX_PREFIX_RUNS && n_pol <= 3);
            prefix_runs[n_prefix_runs] = PrefixRun {
                start_s: s_run as u8,
                count: (end_s - s_run) as u8,
                n_pol: n_pol as u8,
                p_off,
            };
            n_prefix_runs += 1;
            s_run = end_s;
        }
        let mut built = Self {
            state,
            coef,
            damp_bd: alloc::vec![z; cap_p].into_boxed_slice(),
            damp_ad: alloc::vec![z; cap_p].into_boxed_slice(),
            tl_soa,
            fast_meta,
            a_wave_gains,
            pan_l,
            pan_r,
            prefix_runs,
            n_prefix_runs,
            key_tails,
            r0,
            r_damp_prefix_v,
            r_damp_prefix_h,
            tail_n_trans,
            tail_n_long,
            meta,
        };
        built.sync_damp_ports(ports);
        built
    }
}
