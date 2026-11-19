use crate::choir_map::ChoirMap;
use crate::{
    voicing_blob_len, voicing_blob_len_with_profile, BridgePortDriver, KeyActionDriver, StringBank,
    VoicingPrepContext,
};
use piano_spec::{build_scale_design, PerfProfile, VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

use core::cell::Cell;

#[derive(Clone)]
struct TestAction {
    hammer_left: [usize; NUM_KEYS],
    hammer_peak_n: [f32; NUM_KEYS],
    u_norm: [f32; NUM_KEYS],
    clear_m: [f32; NUM_KEYS],
    settled_open: bool,
    settled_queries: Cell<usize>,
    tile_calls: usize,
}

impl TestAction {
    fn new(clear_m: f32, settled_open: bool) -> Self {
        Self {
            hammer_left: [0; NUM_KEYS],
            hammer_peak_n: [0.0; NUM_KEYS],
            u_norm: [0.0; NUM_KEYS],
            clear_m: [clear_m; NUM_KEYS],
            settled_open,
            settled_queries: Cell::new(0),
            tile_calls: 0,
        }
    }

    fn strike(&mut self, key_idx: usize, peak_n: f32, samples: usize) {
        self.hammer_left[key_idx] = samples;
        self.hammer_peak_n[key_idx] = peak_n;
    }
}

impl KeyActionDriver for TestAction {
    fn is_hammer_active(&self, key_idx: usize) -> bool {
        self.hammer_left[key_idx] > 0
    }
    fn hammer_compression_norm(&self, key_idx: usize) -> f32 {
        self.u_norm[key_idx]
    }
    fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        self.clear_m[key_idx]
    }
    fn is_damper_settled_open(&self, key_idx: usize, clear_threshold_m: f32) -> Option<f32> {
        self.settled_queries.set(self.settled_queries.get() + 1);
        if self.settled_open && self.clear_m[key_idx] >= clear_threshold_m {
            Some(self.clear_m[key_idx])
        } else {
            None
        }
    }
    fn strike_weights(&self, _: usize) -> [f32; 3] {
        [1.0; 3]
    }
    fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        self.clear_m[key_idx]
    }
    fn step_key_clearance_tile(&mut self, key_idx: usize, t_len: usize, out: &mut [f32; 8]) -> f32 {
        self.tile_calls += 1;
        out[..t_len].fill(self.clear_m[key_idx]);
        self.clear_m[key_idx]
    }
    fn step_hammer_sample(&mut self, key_idx: usize, _: f32, _: f32, _: f32) -> f32 {
        if self.hammer_left[key_idx] == 0 {
            self.u_norm[key_idx] = 0.0;
            return 0.0;
        }
        let rem = self.hammer_left[key_idx];
        self.hammer_left[key_idx] -= 1;
        self.u_norm[key_idx] = if rem > 1 { 0.45 } else { 0.0 };
        self.hammer_peak_n[key_idx]
    }
    fn step_damper_sample(&mut self, _: usize, _: usize, _: f32, _: f32, _: f32, _: f32) -> f32 {
        0.0
    }
}

#[derive(Clone)]
struct SimpleBridge {
    r0: [f32; TOTAL_STRINGS],
    zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
}

impl SimpleBridge {
    fn new(r0: [f32; TOTAL_STRINGS]) -> Self {
        let mut zw = [(0_u8, 1.0_f32, 0_u8, 0.0_f32); TOTAL_STRINGS];
        for (s, entry) in zw.iter_mut().enumerate() {
            *entry = (((s * 8) / TOTAL_STRINGS) as u8, 1.0, 0, 0.0);
        }
        Self { r0, zw }
    }
}

impl BridgePortDriver for SimpleBridge {
    fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        &self.r0
    }
    fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS] {
        &self.zw
    }
    fn step_bridge_sample(
        &mut self,
        a: &[f32; TOTAL_STRINGS],
        _: &[f32; 8],
        b: &mut [f32; TOTAL_STRINGS],
        zone: &mut [f32; 8],
    ) -> (f32, f32) {
        for (dst, &src) in b.iter_mut().zip(a.iter()) {
            *dst = -0.985 * src;
        }
        zone.fill(0.0);
        (0.0, 0.0)
    }
}

#[test]
fn full_profile_matches_default_new_and_voicing_prep_across_profiles() {
    let sr = 48_000.0;
    let voicing = VoicingSpec::default();
    let default_bank = StringBank::new(sr, &voicing);
    let full_bank = StringBank::new_with_profile(sr, &voicing, &PerfProfile::full());
    assert_eq!(default_bank.total_modes(), full_bank.total_modes());
    assert_eq!(default_bank.perf_profile(), &PerfProfile::full());
    assert_eq!(
        voicing_blob_len(sr),
        voicing_blob_len_with_profile(sr, &PerfProfile::full())
    );

    for perf in [
        PerfProfile::full(),
        PerfProfile::balanced(),
        PerfProfile::realtime(),
    ] {
        let mut bank = StringBank::new_bridge_coupled_with_profile(sr, &voicing, &perf);
        let mut prep = VoicingPrepContext::new_with_profile(sr, &perf);
        assert_eq!(prep.blob_len(), voicing_blob_len_with_profile(sr, &perf));

        let v_alt = VoicingSpec::from_globals_with_a4(0.7, 1.5, 1.2, 0.95, 442.0);
        let blob = prep.prepare_voicing(&v_alt);
        assert!(bank.install_precomputed_voicing(&v_alt, blob));

        let mut ref_bank = StringBank::new_bridge_coupled_with_profile(sr, &voicing, &perf);
        ref_bank.update_voicing_in_place(&v_alt);
        assert_eq!(
            ref_bank.bridge_port_resistances(),
            bank.bridge_port_resistances()
        );
        assert_eq!(ref_bank.soa.d_coeff.as_slice(), bank.soa.d_coeff.as_slice());
    }

    let mut prep_switch = VoicingPrepContext::new_with_profile(sr, &PerfProfile::full());
    prep_switch.set_perf_profile(&PerfProfile::realtime());
    assert_eq!(
        prep_switch.blob_len(),
        voicing_blob_len_with_profile(sr, &PerfProfile::realtime())
    );
}

#[test]
fn horiz_max_hz_and_dead_mode_prune_preserve_c8_and_bridge_prefix_invariants() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let full_map = ChoirMap::build_with_profile(&specs, &voicing, &PerfProfile::full());

    let horiz_only = PerfProfile {
        horiz_max_hz: 2_800.0,
        ..PerfProfile::full()
    };
    let h_map = ChoirMap::build_with_profile(&specs, &voicing, &horiz_only);
    assert!(h_map.total_padded_modes() < full_map.total_padded_modes());

    let h_bank = StringBank::new_bridge_coupled_with_profile(48_000.0, &voicing, &horiz_only);
    for (s_idx, entry) in h_bank.choir_map().strings().iter().enumerate() {
        let meta = &h_bank.bridge_ports.meta[s_idx];
        assert!(entry.horiz_modes >= 1);
        assert!(entry.horiz_vecs.len() >= meta.v_br.len());
        assert_eq!(meta.v_br.len(), meta.h_br.len());
        assert!(h_bank.bridge_port_resistances()[s_idx] > 0.0);
    }
    let c8 = h_bank.choir_map().key(87).unwrap().string_range.start;
    assert_eq!(h_bank.choir_map().strings()[c8].horiz_modes, 1);
    assert_eq!(h_bank.choir_map().strings()[c8].horiz_vecs.len(), 1);

    let prune_only = PerfProfile {
        dead_mode_prune_db: -65.0,
        ..PerfProfile::full()
    };
    let p_map = ChoirMap::build_with_profile(&specs, &voicing, &prune_only);
    let prune_50 = PerfProfile {
        dead_mode_prune_db: -50.0,
        ..PerfProfile::full()
    };
    let p50_map = ChoirMap::build_with_profile(&specs, &voicing, &prune_50);
    assert!(
        p_map.active_unpadded_modes() <= full_map.active_unpadded_modes()
            && p50_map.total_padded_modes() < full_map.total_padded_modes(),
        "full={:?} p65={:?} p50={:?}",
        full_map.mode_counts(),
        p_map.mode_counts(),
        p50_map.mode_counts()
    );
    let p_bank = StringBank::new_bridge_coupled_with_profile(48_000.0, &voicing, &prune_only);
    for (s_idx, entry) in p_bank.choir_map().strings().iter().enumerate() {
        let meta = &p_bank.bridge_ports.meta[s_idx];
        assert!(entry.vert_vecs.len() >= meta.v_br.len());
        assert!(entry.horiz_vecs.len() >= meta.h_br.len());
    }
}

#[test]
fn runtime_knobs_energy_horizon_sav_dormancy_sav_max_key_and_passive_res_work() {
    let sr = 48_000.0;
    let voicing = VoicingSpec::default();
    let mut bank = StringBank::new_bridge_coupled(sr, &voicing);
    assert_eq!(
        bank.perf_profile().energy_horizon_sq(),
        1.0e-18,
        "Full energy horizon exact match"
    );
    assert_eq!(
        bank.perf_profile().sav_dormancy_ratio(),
        1.581e-6,
        "Full SAV dormancy exact match"
    );
    assert!(bank.passive_v_vecs.iter().all(|&x| x == u16::MAX));

    let rt = PerfProfile::realtime();
    bank.set_perf_profile_runtime(&rt);
    assert_eq!(bank.perf_profile().sav_max_key_idx, rt.sav_max_key_idx);
    assert_eq!(bank.perf_profile().energy_horizon_db, rt.energy_horizon_db);
    assert!(bank.passive_v_vecs[0] < u16::MAX);
    assert_eq!(bank.passive_v_vecs[TOTAL_STRINGS - 1], 0);

    // Key 30 (>= realtime sav_max_key_idx 26) should keep r_sav == 1.0 when excited.
    bank.apply_excitation_sample(30, [1.0, 1.0, 1.0], 12.0);
    let mut drv = TestAction::new(0.005, true);
    let mut br = SimpleBridge::new(*bank.bridge_port_resistances());
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
    bank.render_subblock_bridge_coupled(&mut drv, &mut br, &mut l, &mut r);
    for s_idx in bank.choir_map().key(30).unwrap().string_range.clone() {
        assert_eq!(bank.sav.r_sav[s_idx], 1.0);
    }

    // Key 10 (< 26) with high energy_horizon_db retires tails faster than Full.
    let mut bank_full = StringBank::new_bridge_coupled(sr, &voicing);
    let mut bank_fast = StringBank::new_bridge_coupled(sr, &voicing);
    bank_fast.set_perf_profile_runtime(&PerfProfile {
        energy_horizon_db: -40.0,
        ..PerfProfile::full()
    });
    bank_full.apply_excitation_sample(10, [1.0, 1.0, 1.0], 1e-4);
    bank_fast.apply_excitation_sample(10, [1.0, 1.0, 1.0], 1e-4);
    let mut drv_a = TestAction::new(0.005, true);
    let mut drv_b = TestAction::new(0.005, true);
    let mut br_a = SimpleBridge::new(*bank_full.bridge_port_resistances());
    let mut br_b = SimpleBridge::new(*bank_fast.bridge_port_resistances());
    bank_full.render_subblock_bridge_coupled(&mut drv_a, &mut br_a, &mut l, &mut r);
    bank_fast.render_subblock_bridge_coupled(&mut drv_b, &mut br_b, &mut l, &mut r);
    let s10 = bank_full.choir_map().key(10).unwrap().string_range.start;
    assert!(bank_fast.ports.tl_v_active[s10] <= bank_full.ports.tl_v_active[s10]);
}

#[test]
fn p0_1_p0_2_and_p1_4_settled_open_skip_and_a_waves_persistence_are_consistent() {
    let sr = 48_000.0;
    let voicing = VoicingSpec::default();
    let mut bank_a = StringBank::new_bridge_coupled(sr, &voicing);
    let mut bank_b = StringBank::new_bridge_coupled(sr, &voicing);

    let mut drv_unsettled = TestAction::new(0.005, false);
    let mut drv_settled = TestAction::new(0.005, true);
    drv_unsettled.strike(48, 18.0, 32);
    drv_settled.strike(48, 18.0, 32);
    let mut br_a = SimpleBridge::new(*bank_a.bridge_port_resistances());
    let mut br_b = SimpleBridge::new(*bank_b.bridge_port_resistances());

    let (mut la, mut ra) = ([0.0_f32; 128], [0.0_f32; 128]);
    let (mut lb, mut rb) = ([0.0_f32; 128], [0.0_f32; 128]);
    for _ in 0..4 {
        bank_a.render_subblock_bridge_coupled(&mut drv_unsettled, &mut br_a, &mut la, &mut ra);
        bank_b.render_subblock_bridge_coupled(&mut drv_settled, &mut br_b, &mut lb, &mut rb);
        let max_diff = la
            .iter()
            .zip(lb.iter())
            .chain(ra.iter().zip(rb.iter()))
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "settled-open skip changed output: {max_diff}"
        );
    }
    assert_eq!(drv_settled.tile_calls, 0);
    assert!(drv_unsettled.tile_calls > 0);
    assert!(drv_settled.settled_queries.get() > 0);

    // P1-4: persisted bridge_ports.a_waves matches fresh compute_a_waves after chunks.
    let mut recomputed = [0.0_f32; TOTAL_STRINGS];
    bank_b
        .bridge_ports
        .compute_a_waves(&bank_b.ports.may_be_nonzero, &mut recomputed);
    for (s_idx, &rec) in recomputed.iter().enumerate() {
        let diff = (bank_b.bridge_ports.a_waves[s_idx] - rec).abs();
        assert!(diff < 1e-6, "P1-4 a_waves mismatch at s={s_idx}: {diff}");
    }

    // P1-4: update_voicing_in_place marks a_waves_dirty and resyncs on next render.
    let v_alt = VoicingSpec::from_globals_with_a4(0.65, 1.0, 1.0, 1.0, 442.0);
    bank_b.update_voicing_in_place(&v_alt);
    assert!(bank_b.bridge_ports.a_waves_dirty);
    bank_b.render_subblock_bridge_coupled(&mut drv_settled, &mut br_b, &mut lb, &mut rb);
    assert!(!bank_b.bridge_ports.a_waves_dirty);
    bank_b
        .bridge_ports
        .compute_a_waves(&bank_b.ports.may_be_nonzero, &mut recomputed);
    for (s_idx, &rec) in recomputed.iter().enumerate() {
        let diff = (bank_b.bridge_ports.a_waves[s_idx] - rec).abs();
        assert!(
            diff < 1e-6,
            "P1-4 post-voicing mismatch at s={s_idx}: {diff}"
        );
    }
}

#[test]
fn p0_2_sav_coupled_damp_updates_and_presets_render_finite_audio() {
    let sr = 48_000.0;
    let voicing = VoicingSpec::default();
    let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);

    for perf in [
        PerfProfile::full(),
        PerfProfile::balanced(),
        PerfProfile::realtime(),
    ] {
        let mut bank = StringBank::new_bridge_coupled_with_profile(sr, &voicing, &perf);
        let mut drv = TestAction::new(0.005, true);
        // Strike bass, tenor (damped), and treble (undamped) simultaneously.
        drv.strike(0, 18.0, 40);
        drv.strike(39, 20.0, 30);
        drv.strike(87, 14.0, 15);
        let mut br = SimpleBridge::new(*bank.bridge_port_resistances());
        bank.render_subblock_bridge_coupled(&mut drv, &mut br, &mut l, &mut r);

        // Engage damper on key 39 (< 52) while SAV is active to exercise sav_coupled_damp_strs.
        drv.settled_open = false;
        drv.clear_m[39] = -1.0e-5;
        bank.render_subblock_bridge_coupled(&mut drv, &mut br, &mut l, &mut r);

        let energy: f64 = l
            .iter()
            .zip(r.iter())
            .map(|(&a, &b)| {
                assert!(a.is_finite() && b.is_finite());
                f64::from(a) * f64::from(a) + f64::from(b) * f64::from(b)
            })
            .sum();
        assert!(energy > 1.0e-12, "silent output under {perf:?}");
    }
}
