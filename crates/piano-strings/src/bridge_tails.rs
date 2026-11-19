//! Stage 2 <-> Stage 3 key-batched tail/damper/hammer helpers (`CHUNK = 128`, `1x` active contact).

use piano_simd::F32x4;
use piano_spec::{NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::BridgePorts;
pub(crate) use crate::bridge_sample_prep::prepare_sample_tails_and_hammers;
#[allow(unused_imports)]
pub(crate) use crate::bridge_tail_routing::{
    key_clear_of_open_damper, key_needs_active_tail, route_damped_symp_key,
    try_demote_half_pedal_key,
};
use crate::choir_map::{KeyMapEntry, StringMapEntry};
use crate::contact_ports::ContactPorts;
use crate::longitudinal::LongitudinalBank;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::accum_soa_range_in;

pub(crate) const SUB: usize = 128;

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum KeyCat {
    Uncoupled = 0,
    DampedSymp = 1,
    ActiveContact = 2,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct HammerStepForce {
    pub fh: f32,
    pub fd_v: f32,
    pub fd_h: f32,
    pub alpha_w: f32,
}

#[rustfmt::skip]
impl HammerStepForce {
    pub const ZERO: Self = Self { fh: 0.0, fd_v: 0.0, fd_h: 0.0, alpha_w: 0.0 };
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct SavLongSlice {
    pub s_idx: u8,
    pub start: u16,
    pub end: u16,
}

#[rustfmt::skip]
impl SavLongSlice {
    pub const ZERO: Self = Self { s_idx: 0, start: 0, end: 0 };
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct TailPortPartial {
    pub yhv: f32,
    pub vhv_raw: f32,
    pub yhh: f32,
    pub vhh_raw: f32,
    pub ydv: f32,
    pub vdv_raw: f32,
    pub ydh: f32,
    pub vdh_raw: f32,
}

#[rustfmt::skip]
impl TailPortPartial {
    pub const ZERO: Self = Self {
        yhv: 0.0, vhv_raw: 0.0, yhh: 0.0, vhh_raw: 0.0,
        ydv: 0.0, vdv_raw: 0.0, ydh: 0.0, vdh_raw: 0.0,
    };
}

#[derive(Clone, Debug)]
pub(crate) struct ChunkPlan {
    pub key_cat: [KeyCat; NUM_KEYS],
    pub key_has_hammer: [bool; NUM_KEYS],
    pub key_has_mid_ev: [bool; NUM_KEYS],
    pub key_alpha_post: [f32; NUM_KEYS],
    pub was_nz: [bool; TOTAL_STRINGS],
    pub was_tail_nz: [bool; TOTAL_STRINGS],
    pub str_coupled: [bool; TOTAL_STRINGS],
    pub coupled_strs: [u8; TOTAL_STRINGS],
    pub active_keys: [u8; NUM_KEYS],
    pub sav_long_slices: [SavLongSlice; TOTAL_STRINGS],
    pub sav_strings: [u8; TOTAL_STRINGS],
    pub sav_coupled_damp_strs: [u8; TOTAL_STRINGS],
    pub hammer_forces: [HammerStepForce; TOTAL_STRINGS],
    pub tail_partial_ports: [TailPortPartial; TOTAL_STRINGS],
    pub symp_gv: [f32; TOTAL_STRINGS],
    pub symp_gh: [f32; TOTAL_STRINGS],
    pub symp_mask: [f32; TOTAL_STRINGS],
    pub symp_y_lim: [f32; TOTAL_STRINGS],
    pub symp_has_gains: [bool; TOTAL_STRINGS],
    pub symp_dirty_strs: [u8; TOTAL_STRINGS],
    pub zone_tail_n: [[f32; 8]; SUB],
    pub n_a: usize,
    pub n_sl: usize,
    pub n_sav: usize,
    pub n_sav_cd: usize,
    pub n_coupled_strs: usize,
    pub n_symp_dirty: usize,
}

impl ChunkPlan {
    #[inline(always)]
    pub fn push_coupled(&mut self, s_idx: usize) {
        self.str_coupled[s_idx] = true;
        self.coupled_strs[self.n_coupled_strs] = s_idx as u8;
        self.n_coupled_strs += 1;
    }
}

pub(crate) struct BridgeRun<'a> {
    pub ports: &'a mut ContactPorts,
    pub bridge_ports: &'a mut BridgePorts,
    pub long_bank: &'a LongitudinalBank,
    pub sav: &'a mut SavBank,
    pub keys: &'a [KeyMapEntry],
    pub strings: &'a [StringMapEntry],
    pub zone_weights: [(u8, f32, u8, f32); TOTAL_STRINGS],
    pub passive_v_vecs: &'a [u16; TOTAL_STRINGS],
    pub passive_h_vecs: &'a [u16; TOTAL_STRINGS],
    pub sav_max_key_idx: usize,
    pub sav_dormancy_ratio: f64,
    pub passive_res_enabled: bool,
    pub long_on: bool,
    pub sav_on: bool,
    pub lossless: bool,
}

#[inline]
pub(crate) fn add_zone_n(zone_buf: &mut [f32; 8], zw: (u8, f32, u8, f32), force_n: f32) {
    zone_buf[usize::from(zw.0) & 7] += force_n * zw.1;
    zone_buf[usize::from(zw.2) & 7] += force_n * zw.3;
}

#[inline]
pub(crate) fn sav_long_vecs(r: &BridgeRun<'_>, s_idx: usize) -> usize {
    if !r.ports.struck_active[s_idx]
        || r.ports.sav_dormant[s_idx]
        || !r.sav_on
        || !r.long_on
        || r.sav.string_n_vh(s_idx) == 0
    {
        return 0;
    }
    let (_, _, nl, m_ph) = r.sav.string_prefix_bounds(s_idx, true);
    if m_ph > 0 {
        nl
    } else {
        0
    }
}

pub(crate) fn step_uncoupled_tails(
    r: &mut BridgeRun<'_>,
    plan: &mut ChunkPlan,
    acc_th: &mut [F32x4; SUB],
    acc_l: &mut [F32x4; SUB],
    sub_l: &mut [f32],
    sub_r: &mut [f32],
) {
    let sub = sub_l.len();

    for k in 0..NUM_KEYS {
        let step_transverse = match plan.key_cat[k] {
            KeyCat::Uncoupled | KeyCat::DampedSymp => true,
            KeyCat::ActiveContact if !plan.key_has_hammer[k] && !plan.key_has_mid_ev[k] => false,
            KeyCat::ActiveContact => continue,
        };
        let kt = r.bridge_ports.key_tails[k];
        let sr = r.keys[k].string_range.clone();
        let any_tail_nz = plan.was_tail_nz[sr.clone()].iter().any(|&b| b);
        if !any_tail_nz {
            continue;
        }
        let (th_e, l_e) = (usize::from(kt.th_end), usize::from(kt.l_end));
        let has_l = th_e < l_e;
        if !step_transverse && !has_l {
            continue;
        }
        if step_transverse {
            acc_th[..sub].fill(F32x4::zero());
        }
        if has_l {
            acc_l[..sub].fill(F32x4::zero());
        }
        for s_idx in sr {
            if !plan.was_tail_nz[s_idx] {
                continue;
            }
            let (tl_v, tl_h, tl_l) = {
                let m = &r.bridge_ports.meta[s_idx];
                (m.tl_v.clone(), m.tl_h.clone(), m.tl_l.clone())
            };
            if step_transverse {
                let mut nv = usize::from(r.ports.tl_v_active[s_idx]).min(tl_v.len());
                let mut nh = usize::from(r.ports.tl_h_active[s_idx]).min(tl_h.len());
                if !r.ports.struck_active[s_idx] {
                    nv = nv.min(usize::from(r.passive_v_vecs[s_idx]));
                    nh = nh.min(usize::from(r.passive_h_vecs[s_idx]));
                }
                accum_soa_range_in(
                    &mut r.bridge_ports.tl_soa,
                    tl_v.start..(tl_v.start + nv),
                    &mut acc_th[..sub],
                );
                accum_soa_range_in(
                    &mut r.bridge_ports.tl_soa,
                    tl_h.start..(tl_h.start + nh),
                    &mut acc_th[..sub],
                );
            }
            if has_l {
                let nl_sav = sav_long_vecs(r, s_idx);
                accum_soa_range_in(
                    &mut r.bridge_ports.tl_soa,
                    (tl_l.start + nl_sav)..tl_l.end,
                    &mut acc_l[..sub],
                );
            }
        }

        let zw = r.zone_weights[r.keys[k].string_range.start];
        let (nt, nl, use_l) = (kt.tail_n_trans, kt.tail_n_long, has_l && r.long_on);
        for t in 0..sub {
            let th_v = if step_transverse {
                acc_th[t].reduce_sum()
            } else {
                0.0
            };
            let fl = if use_l { acc_l[t].reduce_sum() } else { 0.0 };
            let audio = th_v + fl;
            sub_l[t] += audio * kt.pan_l;
            sub_r[t] += audio * kt.pan_r;
            add_zone_n(&mut plan.zone_tail_n[t], zw, th_v * nt + fl * nl);
        }
    }
}
