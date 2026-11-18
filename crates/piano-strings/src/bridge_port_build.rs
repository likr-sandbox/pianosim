//! Stage 2 <-> Stage 3 WDF bridge port table construction (`no_std`).

use alloc::vec::Vec;
use core::ops::Range;

use piano_simd::{AlignedBuf, F32x4};
use piano_spec::{
    effective_string_f0_hz, Polarization, StringPhysicalParams, VoicingSpec, NUM_KEYS,
    TOTAL_STRINGS,
};

use crate::bridge_port::{
    count_prefix_modes, fill_pol_beta, BridgePorts, FastPrefixMeta, KeyTailRanges, PrefixCoef,
    PrefixRun, PrefixState, StringBridgeMeta, MAX_PREFIX_RUNS,
};
use crate::choir_map::ChoirMap;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::{sqrt_f64, ModalSoa, C_HORIZ, C_VERT, NYQUIST_CUTOFF_RATIO};
use crate::phantom::max_phantom_transverse_vecs_for_key;

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
            let sav_min = max_phantom_transverse_vecs_for_key(k);
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
            assert!(n_prefix_runs < MAX_PREFIX_RUNS && n_pol <= 6);
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
