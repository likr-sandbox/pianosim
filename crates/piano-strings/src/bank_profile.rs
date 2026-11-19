//! Stage 2 `StringBank` profile construction and zero-alloc runtime `PerfProfile` updates.

use piano_spec::{
    build_scale_design, effective_string_f0_hz, partial_frequency_hz, PerfProfile, Polarization,
    StringPhysicalParams, VoicingSpec, LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, TOTAL_STRINGS,
};

use crate::bridge_port::BridgePorts;
use crate::bridge_scratch::BridgeScratch;
use crate::choir_map::ChoirMap;
use crate::contact_ports::ContactPorts;
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::StringBank;

#[inline]
fn count_passive_tail_vecs(
    prefix_vecs: usize,
    tail_vecs: usize,
    f0_hz: f64,
    b_inharm: f64,
    lim_hz: f64,
) -> u16 {
    let mut keep = 0usize;
    for vi in 0..tail_vecs {
        let j_low = (prefix_vecs + vi) * 4 + 1;
        let f_low = partial_frequency_hz(f0_hz, b_inharm, j_low);
        if f_low > 0.0 && f_low <= lim_hz {
            keep = vi + 1;
        } else {
            break;
        }
    }
    keep as u16
}

impl StringBank {
    pub(crate) fn build_internal(
        sample_rate: f32,
        voicing: &VoicingSpec,
        lossless: bool,
        bridge_coupled: bool,
        perf: &PerfProfile,
    ) -> Self {
        let safe_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let p = perf.sanitized();
        let v = voicing.sanitized();
        let sr64 = f64::from(safe_rate);
        let specs = build_scale_design(&v);
        let choir_map = ChoirMap::build_with_profile(&specs, &v, &p);
        let soa = ModalSoa::build_with_profile(
            sr64,
            &v,
            &specs,
            &choir_map,
            lossless,
            bridge_coupled,
            &p,
        );
        let ports = ContactPorts::build_with_profile(sr64, &v, &specs, &choir_map, &p);
        let bridge_ports = BridgePorts::build_with_profile(
            sr64,
            &v,
            &specs,
            &choir_map,
            &ports,
            &soa,
            p.bass_phantom_sparse,
        );
        let long_bank = LongitudinalBank::build(sr64, &specs, &choir_map, &ports.gamma_s);
        let sav = SavBank::build_with_profile(
            sr64,
            &v,
            &specs,
            &choir_map,
            &ports.gamma_s,
            &ports.beta_s,
            p.bass_phantom_sparse,
        );

        let mut bank = Self {
            sample_rate: safe_rate,
            perf: p,
            last_voicing: v.clone(),
            pending_voicing: v.clone(),
            pending_scale_specs: alloc::boxed::Box::new(specs),
            pending_voicing_active: false,
            choir_map,
            soa,
            ports,
            bridge_ports,
            long_bank,
            sav,
            passive_v_vecs: [u16::MAX; TOTAL_STRINGS],
            passive_h_vecs: [u16::MAX; TOTAL_STRINGS],
            scratch: BridgeScratch::new_boxed(),
            longitudinal_enabled: true,
            sav_phantom_enabled: true,
            soa_dirty: false,
            bp_dirty: false,
            lossless,
            bridge_coupled,
        };
        bank.recompute_passive_res_caps(&v, &specs);
        bank
    }

    pub(crate) fn recompute_passive_res_caps_key(
        &mut self,
        k: usize,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
    ) {
        let Some(key_entry) = self.choir_map.key(k) else {
            return;
        };
        let sr = key_entry.string_range.clone();
        if self.perf.passive_res_max_hz >= 20_000.0 {
            self.passive_v_vecs[sr.clone()].fill(u16::MAX);
            self.passive_h_vecs[sr].fill(u16::MAX);
            return;
        }
        let lim_hz = f64::from(self.perf.passive_res_max_hz).clamp(1_000.0, 20_000.0);
        let spec = &specs[k];
        for s_idx in sr {
            let entry = &self.choir_map.strings()[s_idx];
            let s_choir = usize::from(entry.string_in_choir);
            let m = &self.bridge_ports.meta[s_idx];
            let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
            let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
            self.passive_v_vecs[s_idx] = count_passive_tail_vecs(
                m.v_br.len(),
                m.tl_v.len(),
                f0_v,
                spec.inharmonicity_b,
                lim_hz,
            );
            self.passive_h_vecs[s_idx] = count_passive_tail_vecs(
                m.h_br.len(),
                m.tl_h.len(),
                f0_h,
                spec.inharmonicity_b,
                lim_hz,
            );
        }
    }

    pub(crate) fn recompute_passive_res_caps(
        &mut self,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
    ) {
        if self.perf.passive_res_max_hz >= 20_000.0 {
            self.passive_v_vecs.fill(u16::MAX);
            self.passive_h_vecs.fill(u16::MAX);
            return;
        }
        for k in 0..NUM_KEYS {
            self.recompute_passive_res_caps_key(k, voicing, specs);
        }
    }

    pub(crate) fn recompute_passive_res_caps_pending(&mut self, k: usize) {
        let Some(key_entry) = self.choir_map.key(k) else {
            return;
        };
        let sr = key_entry.string_range.clone();
        if self.perf.passive_res_max_hz >= 20_000.0 {
            self.passive_v_vecs[sr.clone()].fill(u16::MAX);
            self.passive_h_vecs[sr].fill(u16::MAX);
            return;
        }
        let lim_hz = f64::from(self.perf.passive_res_max_hz).clamp(1_000.0, 20_000.0);
        let spec = &self.pending_scale_specs[k];
        for s_idx in sr {
            let entry = &self.choir_map.strings()[s_idx];
            let s_choir = usize::from(entry.string_in_choir);
            let m = &self.bridge_ports.meta[s_idx];
            let f0_v = effective_string_f0_hz(
                k,
                s_choir,
                Polarization::Vertical,
                &self.pending_scale_specs,
                &self.pending_voicing,
            );
            let f0_h = effective_string_f0_hz(
                k,
                s_choir,
                Polarization::Horizontal,
                &self.pending_scale_specs,
                &self.pending_voicing,
            );
            self.passive_v_vecs[s_idx] = count_passive_tail_vecs(
                m.v_br.len(),
                m.tl_v.len(),
                f0_v,
                spec.inharmonicity_b,
                lim_hz,
            );
            self.passive_h_vecs[s_idx] = count_passive_tail_vecs(
                m.h_br.len(),
                m.tl_h.len(),
                f0_h,
                spec.inharmonicity_b,
                lim_hz,
            );
        }
    }

    pub(crate) fn recompute_passive_res_caps_key_from_pending(
        &mut self,
        k: usize,
        voicing: &VoicingSpec,
    ) {
        let Some(key_entry) = self.choir_map.key(k) else {
            return;
        };
        let sr = key_entry.string_range.clone();
        if self.perf.passive_res_max_hz >= 20_000.0 {
            self.passive_v_vecs[sr.clone()].fill(u16::MAX);
            self.passive_h_vecs[sr].fill(u16::MAX);
            return;
        }
        let lim_hz = f64::from(self.perf.passive_res_max_hz).clamp(1_000.0, 20_000.0);
        let spec = &self.pending_scale_specs[k];
        for s_idx in sr {
            let entry = &self.choir_map.strings()[s_idx];
            let s_choir = usize::from(entry.string_in_choir);
            let m = &self.bridge_ports.meta[s_idx];
            let f0_v = effective_string_f0_hz(
                k,
                s_choir,
                Polarization::Vertical,
                &self.pending_scale_specs,
                voicing,
            );
            let f0_h = effective_string_f0_hz(
                k,
                s_choir,
                Polarization::Horizontal,
                &self.pending_scale_specs,
                voicing,
            );
            self.passive_v_vecs[s_idx] = count_passive_tail_vecs(
                m.v_br.len(),
                m.tl_v.len(),
                f0_v,
                spec.inharmonicity_b,
                lim_hz,
            );
            self.passive_h_vecs[s_idx] = count_passive_tail_vecs(
                m.h_br.len(),
                m.tl_h.len(),
                f0_h,
                spec.inharmonicity_b,
                lim_hz,
            );
        }
    }

    /// Allocates and precomputes permanent standalone SoA mode tables and physical SI
    /// dual-port projections for `sample_rate` and `voicing`.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::new_with_profile(sample_rate, voicing, &PerfProfile::full())
    }

    /// Allocates a standalone `StringBank` configured with `perf`.
    #[must_use]
    pub fn new_with_profile(
        sample_rate_hz: f32,
        voicing: &VoicingSpec,
        perf: &PerfProfile,
    ) -> Self {
        Self::build_internal(sample_rate_hz, voicing, false, false, perf)
    }

    /// Allocates the string bank configured with intrinsic-only wire/air damping for
    /// coupled Stage 3 `BridgeBank` radiation (eliminating standalone double damping).
    #[must_use]
    pub fn new_bridge_coupled(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::new_bridge_coupled_with_profile(sample_rate, voicing, &PerfProfile::full())
    }

    /// Allocates a bridge-coupled `StringBank` configured with `perf`.
    #[must_use]
    pub fn new_bridge_coupled_with_profile(
        sample_rate_hz: f32,
        voicing: &VoicingSpec,
        perf: &PerfProfile,
    ) -> Self {
        Self::build_internal(sample_rate_hz, voicing, false, true, perf)
    }

    /// Allocates a lossless (`r_hold = 1.0`) bank for SAV symplectic conservation checks.
    #[must_use]
    pub fn new_lossless(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self::new_lossless_with_profile(sample_rate, voicing, &PerfProfile::full())
    }

    /// Allocates a lossless (`r_hold = 1.0`) bank configured with `perf`.
    #[must_use]
    pub fn new_lossless_with_profile(
        sample_rate_hz: f32,
        voicing: &VoicingSpec,
        perf: &PerfProfile,
    ) -> Self {
        Self::build_internal(sample_rate_hz, voicing, true, false, perf)
    }

    /// Updates runtime-tunable Stage 2 `PerfProfile` thresholds with zero heap allocations.
    pub fn set_perf_profile_runtime(&mut self, perf: &PerfProfile) {
        let p = perf.sanitized();
        let old_sav_max = usize::from(self.perf.sav_max_key_idx).min(LONGITUDINAL_MAX_KEY_IDX);
        let new_sav_max = usize::from(p.sav_max_key_idx).min(LONGITUDINAL_MAX_KEY_IDX);
        let pasv_changed =
            (p.passive_res_max_hz - self.perf.passive_res_max_hz).abs() > f32::EPSILON;
        self.perf.low_order_energy_horizon_db = p.low_order_energy_horizon_db;
        self.perf.energy_horizon_db = p.energy_horizon_db;
        self.perf.sav_dormancy_db = p.sav_dormancy_db;
        self.perf.sav_max_key_idx = p.sav_max_key_idx;
        self.perf.passive_res_max_hz = p.passive_res_max_hz;
        if new_sav_max < old_sav_max {
            for k in new_sav_max..old_sav_max {
                if let Some(key) = self.choir_map.key(k) {
                    for s_idx in key.string_range.clone() {
                        self.sav.reset_string(s_idx);
                    }
                }
            }
        }
        if pasv_changed {
            let v = self.last_voicing.clone();
            let specs = build_scale_design(&v);
            self.recompute_passive_res_caps(&v, &specs);
        }
    }

    /// Returns the active Stage 2 [`PerfProfile`].
    #[must_use]
    #[inline]
    pub const fn perf_profile(&self) -> &PerfProfile {
        &self.perf
    }
}
