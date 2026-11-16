//! Cold state synchronization (`ModalSoa <-> BridgePorts`) and energy/RMS diagnostics (`no_std`).

use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::{NUM_KEYS, TOTAL_STRINGS};

use crate::bridge_port::{BridgePorts, PrefixState, C_SIGN_H_ARR, C_SIGN_V_ARR};
use crate::choir_map::StringMapEntry;
use crate::contact_ports::ContactPorts;
use crate::modal_soa::{sqrt_f64, ModalSoa};
use crate::phantom::{convolve_transverse_slopes, MAX_PHANTOM_TRANSVERSE_MODES};
use crate::sav_drift::SavBank;
use crate::zoh_kernel::{flush_subnormals_range, flush_subnormals_slices};
use crate::{BridgePortDriver, KeyActionDriver, StringBank};

impl BridgePorts {
    pub fn pack_all_from_soa(&mut self, soa: &ModalSoa, strings: &[StringMapEntry]) {
        let (u_src, v_src) = (soa.u.as_f32x4_slice(), soa.v.as_f32x4_slice());
        let (tu_dst, tv_dst) = (
            self.tl_soa.u.as_f32x4_slice_mut(),
            self.tl_soa.v.as_f32x4_slice_mut(),
        );
        for (s_idx, m) in self.meta.iter().enumerate() {
            for (i, src_i) in m.v_br.clone().enumerate() {
                self.state[m.beta_v_off + i] = PrefixState {
                    u: u_src[src_i],
                    v: v_src[src_i],
                };
            }
            for (i, src_i) in m.h_br.clone().enumerate() {
                self.state[m.beta_h_off + i] = PrefixState {
                    u: u_src[src_i],
                    v: v_src[src_i],
                };
            }
            tu_dst[m.tl_v.clone()].copy_from_slice(&u_src[m.v_tail.clone()]);
            tv_dst[m.tl_v.clone()].copy_from_slice(&v_src[m.v_tail.clone()]);
            tu_dst[m.tl_h.clone()].copy_from_slice(&u_src[m.h_tail.clone()]);
            tv_dst[m.tl_h.clone()].copy_from_slice(&v_src[m.h_tail.clone()]);
            let lr = strings[s_idx].long_vecs.clone();
            tu_dst[m.tl_l.clone()].copy_from_slice(&u_src[lr.clone()]);
            tv_dst[m.tl_l.clone()].copy_from_slice(&v_src[lr]);
        }
    }

    pub fn unpack_all_to_soa(&self, soa: &mut ModalSoa, strings: &[StringMapEntry]) {
        let (u_dst, v_dst) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
        let (tu_src, tv_src) = (
            self.tl_soa.u.as_f32x4_slice(),
            self.tl_soa.v.as_f32x4_slice(),
        );
        for (s_idx, m) in self.meta.iter().enumerate() {
            for (i, dst_i) in m.v_br.clone().enumerate() {
                let st = self.state[m.beta_v_off + i];
                u_dst[dst_i] = st.u;
                v_dst[dst_i] = st.v;
            }
            for (i, dst_i) in m.h_br.clone().enumerate() {
                let st = self.state[m.beta_h_off + i];
                u_dst[dst_i] = st.u;
                v_dst[dst_i] = st.v;
            }
            u_dst[m.v_tail.clone()].copy_from_slice(&tu_src[m.tl_v.clone()]);
            v_dst[m.v_tail.clone()].copy_from_slice(&tv_src[m.tl_v.clone()]);
            u_dst[m.h_tail.clone()].copy_from_slice(&tu_src[m.tl_h.clone()]);
            v_dst[m.h_tail.clone()].copy_from_slice(&tv_src[m.tl_h.clone()]);
            let lr = strings[s_idx].long_vecs.clone();
            u_dst[lr.clone()].copy_from_slice(&tu_src[m.tl_l.clone()]);
            v_dst[lr].copy_from_slice(&tv_src[m.tl_l.clone()]);
        }
    }

    #[must_use]
    pub fn string_linear_energy_joules(&self, s_idx: usize, beta_s: f64) -> f64 {
        let m = &self.meta[s_idx];
        let mut sum_sq = 0.0_f64;
        for st in &self.state[m.beta_v_off..m.beta_v_off + m.n_vh] {
            for (&u_l, &v_l) in st.u.to_array().iter().zip(st.v.to_array().iter()) {
                let (u64v, v64v) = (f64::from(u_l), f64::from(v_l));
                sum_sq += u64v * u64v + v64v * v64v;
            }
        }
        let (tu, tv) = (
            self.tl_soa.u.as_f32x4_slice(),
            self.tl_soa.v.as_f32x4_slice(),
        );
        for r in [m.tl_v.clone(), m.tl_h.clone(), m.tl_l.clone()] {
            for (&uv, &vv) in tu[r.clone()].iter().zip(&tv[r]) {
                for (&u_l, &v_l) in uv.to_array().iter().zip(vv.to_array().iter()) {
                    let (u64v, v64v) = (f64::from(u_l), f64::from(v_l));
                    sum_sq += u64v * u64v + v64v * v64v;
                }
            }
        }
        0.5 * beta_s * sum_sq
    }

    #[must_use]
    pub fn measure_key_polarization_rms(&self, string_range: Range<usize>) -> (f32, f32) {
        let tv = self.tl_soa.v.as_f32x4_slice();
        let tc = self.tl_soa.c_bridge.as_f32x4_slice();
        let cv = F32x4::from_array(C_SIGN_V_ARR);
        let ch = F32x4::from_array(C_SIGN_H_ARR);
        let (mut acc_v, mut cv_n) = (F32x4::zero(), 0usize);
        let (mut acc_h, mut ch_n) = (F32x4::zero(), 0usize);
        for s_idx in string_range {
            let m = &self.meta[s_idx];
            for st in &self.state[m.beta_v_off..m.beta_v_off + m.v_br.len()] {
                let v = st.v * cv;
                acc_v = v.mul_add(v, acc_v);
                cv_n += 4;
            }
            for idx in m.tl_v.clone() {
                let v = tv[idx] * tc[idx];
                acc_v = v.mul_add(v, acc_v);
                cv_n += 4;
            }
            for st in &self.state[m.beta_h_off..m.beta_h_off + m.h_br.len()] {
                let v = st.v * ch;
                acc_h = v.mul_add(v, acc_h);
                ch_n += 4;
            }
            for idx in m.tl_h.clone() {
                let v = tv[idx] * tc[idx];
                acc_h = v.mul_add(v, acc_h);
                ch_n += 4;
            }
        }
        let rms = |acc: F32x4, n: usize| {
            if n > 0 {
                sqrt_f64((acc.reduce_sum() as f64) / n as f64) as f32
            } else {
                0.0
            }
        };
        (rms(acc_v, cv_n), rms(acc_h, ch_n))
    }
}

impl SavBank {
    #[must_use]
    pub fn string_nonlinear_energy_joules_bp(
        &self,
        s_idx: usize,
        bp: &BridgePorts,
        beta_s: f64,
        longitudinal_enabled: bool,
    ) -> f64 {
        let sp = &self.specs[s_idx];
        let sl = &self.slopes[s_idx];
        if sp.n_vh == 0 {
            return 0.0;
        }
        let m = &bp.meta[s_idx];
        let mut d_v = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        let mut d_h = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        for (a, dv) in d_v.iter_mut().enumerate().take(sl.nv_c) {
            let va = bp.state[m.beta_v_off + (a >> 2)].v.to_array()[a & 3];
            *dv = sl.sigma_v[a] * f64::from(va);
        }
        for (a, dh) in d_h.iter_mut().enumerate().take(sl.nh_c) {
            let ha = bp.state[m.beta_h_off + (a >> 2)].v.to_array()[a & 3];
            *dh = sl.sigma_h[a] * f64::from(ha);
        }
        let m_ph = if longitudinal_enabled { sp.m_ph } else { 0 };
        let (s_val, phi_tilde) = convolve_transverse_slopes(sp.length_m, &d_v, &d_h, m_ph);
        let lv_sl = bp.tl_soa.v.as_slice();
        let lb = m.tl_l.start * 4;
        let mut sum_res_sq = 0.0_f64;
        for (idx, &phi) in phi_tilde[1..=m_ph].iter().enumerate() {
            let res = phi + sl.alpha_l[idx] * f64::from(lv_sl[lb + idx]);
            sum_res_sq += res * res;
        }
        beta_s * 0.5 * (sp.kappa_0 * s_val * s_val + sp.kappa_ph * sum_res_sq)
    }
}

impl StringBank {
    #[inline]
    pub(crate) fn ensure_soa_mode(&mut self) {
        if self.soa_dirty {
            self.bridge_ports
                .unpack_all_to_soa(&mut self.soa, self.choir_map.strings());
            self.soa_dirty = false;
        }
    }

    /// Per-string WDF mechanical port resistances `R_0,s > 0` (`N*s/m`).
    #[must_use]
    pub fn bridge_port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        &self.bridge_ports.r0
    }

    /// Wave-link stored energy (`0.0` for instantaneous memoryless WDF port scattering).
    #[must_use]
    pub fn bridge_link_energy_joules(&self, _r0: &[f32; TOTAL_STRINGS]) -> f64 {
        0.0
    }

    /// Returns instantaneous linear modal energy `0.5 * beta_s * sum(u^2 + v^2)` for string `s_idx`.
    #[must_use]
    pub fn string_linear_energy_joules(&self, s_idx: usize) -> f64 {
        let Some(entry) = self.choir_map.strings().get(s_idx) else {
            return 0.0;
        };
        if self.soa_dirty {
            self.bridge_ports
                .string_linear_energy_joules(s_idx, self.ports.beta_s[s_idx])
        } else {
            self.ports
                .linear_energy_joules(s_idx, &self.soa, entry.all_vecs())
        }
    }

    /// Diagnostic helper returning `(vert_rms, horiz_rms)` of modal bridge velocity
    /// `v * c_bridge` across `key_idx`'s strings at the current instant.
    #[must_use]
    pub fn measure_key_polarization_rms(&self, key_idx: usize) -> (f32, f32) {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return (0.0, 0.0);
        };
        if self.soa_dirty {
            return self
                .bridge_ports
                .measure_key_polarization_rms(key_entry.string_range.clone());
        }
        let v = self.soa.v.as_f32x4_slice();
        let c = self.soa.c_bridge.as_f32x4_slice();
        let strings = &self.choir_map.strings()[key_entry.string_range.clone()];
        let calc = |sel: fn(&StringMapEntry) -> Range<usize>| {
            let (mut acc, mut cnt) = (F32x4::zero(), 0usize);
            for entry in strings {
                for idx in sel(entry) {
                    let prod = v[idx] * c[idx];
                    acc = prod.mul_add(prod, acc);
                    cnt += 4;
                }
            }
            if cnt > 0 {
                sqrt_f64((acc.reduce_sum() as f64) / cnt as f64) as f32
            } else {
                0.0
            }
        };
        (
            calc(|e| e.vert_vecs.clone()),
            calc(|e| e.horiz_vecs.clone()),
        )
    }
}

#[inline]
fn flush_prefix_slice(slice: &mut [PrefixState]) -> bool {
    let mut any_nz = false;
    for st in slice {
        let mut u_a = st.u.to_array();
        let mut v_a = st.v.to_array();
        any_nz |= flush_subnormals_slices(&mut u_a, &mut v_a);
        st.u = F32x4::from_array(u_a);
        st.v = F32x4::from_array(v_a);
    }
    any_nz
}

pub(crate) fn flush_bridge_subnormals_core(
    ports: &mut ContactPorts,
    bridge_ports: &mut BridgePorts,
    sav: &mut SavBank,
    strings: &[StringMapEntry],
    key_has_hammer: &[bool; NUM_KEYS],
) {
    for (s_idx, entry) in strings.iter().enumerate() {
        if ports.may_be_nonzero[s_idx] {
            let (p_range, tl_v, tl_h, tl_l) = {
                let m = &bridge_ports.meta[s_idx];
                (
                    m.beta_v_off..(m.beta_v_off + m.n_vh),
                    m.tl_v.clone(),
                    m.tl_h.clone(),
                    m.tl_l.clone(),
                )
            };
            let nz_br = flush_prefix_slice(&mut bridge_ports.state[p_range]);
            let nz = if !ports.tail_nonzero[s_idx] {
                nz_br
            } else {
                let nz_tail = flush_subnormals_range(&mut bridge_ports.tl_soa, tl_v)
                    | flush_subnormals_range(&mut bridge_ports.tl_soa, tl_h)
                    | flush_subnormals_range(&mut bridge_ports.tl_soa, tl_l);
                ports.tail_nonzero[s_idx] = nz_tail;
                nz_br | nz_tail
            };
            let k = usize::from(entry.key_idx);
            if !nz {
                ports.clear_string_state(s_idx);
                sav.reset_string(s_idx);
            } else if !key_has_hammer[k] {
                ports.ports_synced[s_idx] = false;
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
        crate::bridge_coupled::render_subblock_bridge_coupled_core(
            &mut self.ports,
            &mut self.bridge_ports,
            &self.long_bank,
            &mut self.sav,
            &self.choir_map,
            action,
            bridge,
            self.longitudinal_enabled,
            self.sav_phantom_enabled,
            false,
            &mut out_l[..frames],
            &mut out_r[..frames],
            None,
        );
    }

    /// Flushes `< 1e-20` subnormals across `BridgePorts` once at the end of an audio block.
    pub fn flush_bridge_subnormals(&mut self, key_has_hammer: &[bool; NUM_KEYS]) {
        if self.soa_dirty {
            flush_bridge_subnormals_core(
                &mut self.ports,
                &mut self.bridge_ports,
                &mut self.sav,
                self.choir_map.strings(),
                key_has_hammer,
            );
        }
    }
}
