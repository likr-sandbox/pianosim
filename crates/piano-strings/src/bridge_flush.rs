//! Stage 2 <-> Stage 3 subnormal flush, energy-horizon shrinking, and SAV dormancy (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::{LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::{BridgePorts, PrefixState};
use crate::choir_map::StringMapEntry;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::zoh_kernel::{flush_subnormals_range, flush_subnormals_slices};
use crate::{BridgePortDriver, KeyActionDriver, StringBank};

const TAIL_DAMP_HORIZON_SQ: f32 = 4.0e-12; // (half-pedal tail-grazing horizon; closed damper damps to -120 dBFS)
pub(crate) const TAIL_DAMP_HIGH_BAND_SQ: f32 = 1.0e-9;
const SAV_DORMANCY_CHUNKS: u8 = 12;

#[inline]
pub(crate) fn tail_damp_thresholds(
    passive_res_max_hz: f32,
    high_order_horizon_sq: f32,
) -> (f32, f32) {
    if passive_res_max_hz >= 20_000.0 && high_order_horizon_sq <= 1.0e-18 {
        (TAIL_DAMP_HORIZON_SQ, TAIL_DAMP_HORIZON_SQ)
    } else {
        (
            TAIL_DAMP_HORIZON_SQ,
            TAIL_DAMP_HIGH_BAND_SQ.max(high_order_horizon_sq),
        )
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn shrink_tail_horizon(
    tl_soa: &mut ModalSoa,
    range: Range<usize>,
    active: &mut u16,
    low_tl_vecs: usize,
    low_order_horizon_sq: f32,
    high_order_horizon_sq: f32,
    passive_vecs: usize,
    tail_damp_low_sq: f32,
    tail_damp_high_sq: f32,
    check_damp: bool,
) -> bool {
    let mut n = usize::from(*active).min(range.len());
    let u = &mut tl_soa.u.as_f32x4_slice_mut()[range.clone()];
    let v = &mut tl_soa.v.as_f32x4_slice_mut()[range];
    while n > 0 {
        let idx = n - 1;
        let thresh = if idx < low_tl_vecs {
            low_order_horizon_sq
        } else {
            high_order_horizon_sq
        };
        let e4 = (u[idx] * u[idx] + v[idx] * v[idx]).to_array();
        if e4[0] < thresh && e4[1] < thresh && e4[2] < thresh && e4[3] < thresh {
            u[idx] = F32x4::zero();
            v[idx] = F32x4::zero();
            n -= 1;
        } else {
            break;
        }
    }
    *active = n as u16;
    if !check_damp {
        return false;
    }
    (0..n).any(|idx| {
        let damp_thresh = if idx < passive_vecs {
            tail_damp_low_sq
        } else {
            tail_damp_high_sq
        };
        (u[idx] * u[idx] + v[idx] * v[idx])
            .to_array()
            .iter()
            .any(|&e| e >= damp_thresh)
    })
}

pub(crate) fn update_sav_dormancy(
    ports: &mut ContactPorts,
    bp: &BridgePorts,
    sav: &mut SavBank,
    s_idx: usize,
    sav_dormancy_ratio: f64,
) {
    if !ports.struck_active[s_idx] || ports.sav_dormant[s_idx] {
        return;
    }
    let m = &bp.meta[s_idx];
    let mut e_sq4 = F32x4::zero();
    for st in &bp.state[m.beta_v_off..(m.beta_v_off + m.n_vh)] {
        e_sq4 = st.u.mul_add(st.u, st.v.mul_add(st.v, e_sq4));
    }
    let e_sq = f64::from(e_sq4.reduce_sum());
    let r = sav.r_sav[s_idx];
    let e_nl = 0.5 * (r * r - 1.0).max(0.0);
    if (r - 1.0).abs() < 1.0e-4 && e_nl <= sav_dormancy_ratio * e_sq.max(1.0e-14) {
        let cnt = ports.sav_low_nl_chunks[s_idx].saturating_add(1);
        ports.sav_low_nl_chunks[s_idx] = cnt;
        if cnt >= SAV_DORMANCY_CHUNKS {
            ports.sav_dormant[s_idx] = true;
            sav.reset_string(s_idx);
        }
    } else {
        ports.sav_low_nl_chunks[s_idx] = 0;
    }
}

#[inline]
fn flush_prefix_slice(
    slice: &mut [PrefixState],
    clear_horizon: bool,
    energy_horizon_sq: f32,
) -> bool {
    let mut any_nz = false;
    let mut max_e = 0.0_f32;
    for st in slice.iter_mut() {
        let mut u_a = st.u.to_array();
        let mut v_a = st.v.to_array();
        any_nz |= flush_subnormals_slices(&mut u_a, &mut v_a);
        if clear_horizon {
            for i in 0..4 {
                max_e = max_e.max(u_a[i] * u_a[i] + v_a[i] * v_a[i]);
            }
        }
        st.u = F32x4::from_array(u_a);
        st.v = F32x4::from_array(v_a);
    }
    if clear_horizon && max_e < energy_horizon_sq {
        for st in slice {
            st.u = F32x4::zero();
            st.v = F32x4::zero();
        }
        return false;
    }
    any_nz
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_bridge_subnormals_core(
    ports: &mut ContactPorts,
    bridge_ports: &mut BridgePorts,
    sav: &mut SavBank,
    strings: &[StringMapEntry],
    key_has_hammer: &[bool; NUM_KEYS],
    key_closed_damp: &[bool; NUM_KEYS],
    passive_v_vecs: &[u16; TOTAL_STRINGS],
    passive_h_vecs: &[u16; TOTAL_STRINGS],
    low_order_horizon_sq: f32,
    high_order_horizon_sq: f32,
    tail_damp_low_sq: f32,
    tail_damp_high_sq: f32,
    lossless: bool,
) {
    for (s_idx, entry) in strings.iter().enumerate() {
        if ports.may_be_nonzero[s_idx] {
            let k = usize::from(entry.key_idx);
            let (p_range, tl_v, tl_h, tl_l, v_br_len, h_br_len) = {
                let m = &bridge_ports.meta[s_idx];
                (
                    m.beta_v_off..(m.beta_v_off + m.n_vh),
                    m.tl_v.clone(),
                    m.tl_h.clone(),
                    m.tl_l.clone(),
                    m.v_br.len(),
                    m.h_br.len(),
                )
            };
            let nz_br = flush_prefix_slice(
                &mut bridge_ports.state[p_range],
                !lossless && key_closed_damp[k] && !ports.tail_nonzero[s_idx],
                low_order_horizon_sq,
            );
            let nz = if !ports.tail_nonzero[s_idx] {
                ports.tl_damp_active[s_idx] = false;
                nz_br
            } else {
                if !lossless && !key_has_hammer[k] {
                    let check_d = ports.tl_damp_active[s_idx];
                    let low_tl_v = 2usize.saturating_sub(v_br_len);
                    let low_tl_h = 2usize.saturating_sub(h_br_len);
                    let dv = shrink_tail_horizon(
                        &mut bridge_ports.tl_soa,
                        tl_v.clone(),
                        &mut ports.tl_v_active[s_idx],
                        low_tl_v,
                        low_order_horizon_sq,
                        high_order_horizon_sq,
                        usize::from(passive_v_vecs[s_idx]),
                        tail_damp_low_sq,
                        tail_damp_high_sq,
                        check_d,
                    );
                    let dh = shrink_tail_horizon(
                        &mut bridge_ports.tl_soa,
                        tl_h.clone(),
                        &mut ports.tl_h_active[s_idx],
                        low_tl_h,
                        low_order_horizon_sq,
                        high_order_horizon_sq,
                        usize::from(passive_h_vecs[s_idx]),
                        tail_damp_low_sq,
                        tail_damp_high_sq,
                        check_d,
                    );
                    if check_d && !dv && !dh {
                        ports.tl_damp_active[s_idx] = false;
                    }
                }
                let nv = usize::from(ports.tl_v_active[s_idx]).min(tl_v.len());
                let nh = usize::from(ports.tl_h_active[s_idx]).min(tl_h.len());
                let nz_tail =
                    flush_subnormals_range(&mut bridge_ports.tl_soa, tl_v.start..(tl_v.start + nv))
                        | flush_subnormals_range(
                            &mut bridge_ports.tl_soa,
                            tl_h.start..(tl_h.start + nh),
                        )
                        | flush_subnormals_range(&mut bridge_ports.tl_soa, tl_l);
                ports.tail_nonzero[s_idx] = nz_tail;
                nz_br | nz_tail
            };
            if !nz {
                ports.clear_string_state(s_idx);
                sav.reset_string(s_idx);
                bridge_ports.a_waves[s_idx] = 0.0;
            }
        }
    }
}

impl StringBank {
    /// Steps a subblock with Stage 1 + Stage 3 WDF coupling, deferring subnormal flush to block end.
    pub fn render_subblock_bridge_coupled_no_flush(
        &mut self,
        action: &mut impl KeyActionDriver,
        bridge: &mut impl BridgePortDriver,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }
        if self.bp_dirty {
            self.bridge_ports
                .pack_all_from_soa(&self.soa, self.choir_map.strings());
            self.bp_dirty = false;
        }
        self.soa_dirty = true;
        let sav_max_key_idx = usize::from(self.perf.sav_max_key_idx).min(LONGITUDINAL_MAX_KEY_IDX);
        crate::bridge_coupled::render_subblock_bridge_coupled_core(
            &mut self.ports,
            &mut self.bridge_ports,
            &self.long_bank,
            &mut self.sav,
            &self.choir_map,
            &mut self.scratch,
            action,
            bridge,
            &self.passive_v_vecs,
            &self.passive_h_vecs,
            sav_max_key_idx,
            self.perf.sav_dormancy_ratio(),
            self.perf.low_order_energy_horizon_sq(),
            self.perf.energy_horizon_sq(),
            self.perf.passive_res_max_hz,
            self.longitudinal_enabled,
            self.sav_phantom_enabled,
            self.lossless,
            false,
            &mut out_l[..frames],
            &mut out_r[..frames],
            None,
        );
    }

    /// Flushes `< 1e-20` subnormals across `BridgePorts` once at the end of an audio block.
    pub fn flush_bridge_subnormals(&mut self, key_has_hammer: &[bool; NUM_KEYS]) {
        self.flush_bridge_subnormals_with_dampers(key_has_hammer, &[false; NUM_KEYS]);
    }

    /// Flushes subnormals and retires `< -120 dBFS` closed-damper strings to zero-skip idle.
    pub fn flush_bridge_subnormals_with_dampers(
        &mut self,
        key_has_hammer: &[bool; NUM_KEYS],
        key_closed_damp: &[bool; NUM_KEYS],
    ) {
        if self.soa_dirty {
            let high_sq = self.perf.energy_horizon_sq();
            let (td_low, td_high) = tail_damp_thresholds(self.perf.passive_res_max_hz, high_sq);
            flush_bridge_subnormals_core(
                &mut self.ports,
                &mut self.bridge_ports,
                &mut self.sav,
                self.choir_map.strings(),
                key_has_hammer,
                key_closed_damp,
                &self.passive_v_vecs,
                &self.passive_h_vecs,
                self.perf.low_order_energy_horizon_sq(),
                high_sq,
                td_low,
                td_high,
                self.lossless,
            );
        }
    }
}
