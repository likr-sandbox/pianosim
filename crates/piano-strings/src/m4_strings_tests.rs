//! Unit/integration tests for M4 Wave 2A: Item 2b (half-pedal passive resonance demotion)
//! and Item 3 (mode-order-dependent energy horizon).

use piano_simd::F32x4;
use piano_spec::{PerfProfile, VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

use crate::{BridgePortDriver, KeyActionDriver, StringBank};

struct M4TestAction {
    hammer_left: [usize; NUM_KEYS],
    clear_m: [f32; NUM_KEYS],
}

impl M4TestAction {
    fn new(clear_m: f32) -> Self {
        Self {
            hammer_left: [0; NUM_KEYS],
            clear_m: [clear_m; NUM_KEYS],
        }
    }
}

impl KeyActionDriver for M4TestAction {
    fn is_hammer_active(&self, k: usize) -> bool {
        self.hammer_left[k] > 0
    }
    fn damper_clearance_m(&self, k: usize) -> f32 {
        self.clear_m[k]
    }
    fn strike_weights(&self, _: usize) -> [f32; 3] {
        [1.0; 3]
    }
    fn step_key_clearance(&mut self, k: usize) -> f32 {
        self.clear_m[k]
    }
    fn step_key_clearance_tile(&mut self, k: usize, t_len: usize, out: &mut [f32; 8]) -> f32 {
        out[..t_len].fill(self.clear_m[k]);
        self.clear_m[k]
    }
    fn step_hammer_sample(&mut self, k: usize, _: f32, _: f32, _: f32) -> f32 {
        if self.hammer_left[k] > 0 {
            self.hammer_left[k] -= 1;
            10.0
        } else {
            0.0
        }
    }
    fn step_damper_sample(&mut self, _: usize, _: usize, _: f32, _: f32, _: f32, _: f32) -> f32 {
        0.0
    }
}

struct M4TestBridge {
    r0: [f32; TOTAL_STRINGS],
    zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
}

impl M4TestBridge {
    fn new(r0: [f32; TOTAL_STRINGS]) -> Self {
        let mut zw = [(0_u8, 1.0_f32, 0_u8, 0.0_f32); TOTAL_STRINGS];
        for (s, entry) in zw.iter_mut().enumerate() {
            *entry = (((s * 8) / TOTAL_STRINGS) as u8, 1.0, 0, 0.0);
        }
        Self { r0, zw }
    }
}

impl BridgePortDriver for M4TestBridge {
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
            *dst = -0.99 * src;
        }
        zone.fill(0.0);
        (0.0, 0.0)
    }
}

#[test]
fn item3_mode_order_energy_horizon_preserves_modes_5_to_8_under_realtime() {
    let sr = 48_000.0;
    let voicing = VoicingSpec::default();
    let rt = PerfProfile::realtime();
    // Verify Realtime thresholds: low=-108 dB (1.585e-17), high=-84 dB (3.981e-15).
    let low_sq = rt.low_order_energy_horizon_sq();
    let high_sq = rt.energy_horizon_sq();
    assert!(low_sq < 2.0e-17 && high_sq > 3.0e-15);

    // Key 52 (C#5, >= 52) has v_br.len() == 1 (modes 1..=4), so tl_v[0] holds modes 5..=8.
    let mut bank = StringBank::new_bridge_coupled_with_profile(sr, &voicing, &rt);
    let s52 = bank.choir_map().key(52).unwrap().string_range.start;
    let m = bank.bridge_ports.meta[s52].clone();
    assert_eq!(m.v_br.len(), 1);
    assert!(m.tl_v.len() >= 2);

    // Set tl_v[0] (modes 5..=8) and tl_v[1] (modes 9..=12) to u^2+v^2 = 1e-16
    // (above -108 dB low_sq, below -84 dB high_sq).
    let amp = 1.0e-8_f32;
    {
        let tu = bank.bridge_ports.tl_soa.u.as_f32x4_slice_mut();
        tu[m.tl_v.start] = F32x4::splat(amp);
        tu[m.tl_v.start + 1] = F32x4::splat(amp);
    }
    bank.bridge_ports.state[m.beta_v_off].u = F32x4::splat(amp);
    bank.ports.may_be_nonzero[s52] = true;
    bank.ports.tail_nonzero[s52] = true;
    bank.ports.tl_v_active[s52] = m.tl_v.len() as u16;
    bank.ports.tl_h_active[s52] = 0;
    bank.soa_dirty = true;

    let no_hammers = [false; NUM_KEYS];
    bank.flush_bridge_subnormals(&no_hammers);

    // High-order modes (> 8) must be pruned, while modes 5..=8 (tl_v[0]) stay active!
    assert_eq!(bank.ports.tl_v_active[s52], 1);
    assert!(bank.ports.tail_nonzero[s52]);
    let tu = bank.bridge_ports.tl_soa.u.as_f32x4_slice();
    assert_eq!(tu[m.tl_v.start].to_array()[0], amp);
    assert_eq!(tu[m.tl_v.start + 1].to_array()[0], 0.0);

    // Closed damper prefix-only slice at 1e-16 (between -108 dB and -84 dB) stays alive.
    bank.ports.tail_nonzero[s52] = false;
    bank.ports.tl_v_active[s52] = 0;
    let mut closed = [false; NUM_KEYS];
    closed[52] = true;
    bank.flush_bridge_subnormals_with_dampers(&no_hammers, &closed);
    assert!(bank.ports.may_be_nonzero[s52]);
    assert_eq!(bank.bridge_ports.state[m.beta_v_off].u.to_array()[0], amp);
}

#[test]
fn item2b_half_pedal_demotes_decayed_high_band_on_realtime_but_keeps_full_exact() {
    let sr = 48_000.0;
    let voicing = VoicingSpec::default();
    let k_e4 = 40usize;

    for (perf, expect_demote) in [
        (PerfProfile::full(), false),
        (PerfProfile::realtime(), true),
    ] {
        // Case A: low-order modes still ringing above 4e-12 while high-band (> passive_res_max_hz)
        // dropped below 1e-9 -> Realtime demotes struck_active to passive resonance.
        let mut bank = StringBank::new_bridge_coupled_with_profile(sr, &voicing, &perf);
        let s0 = bank.choir_map().key(k_e4).unwrap().string_range.start;
        let m = bank.bridge_ports.meta[s0].clone();
        assert!(m.tl_v.len() >= 4);

        bank.mark_key_possibly_active(k_e4);
        {
            let tu = bank.bridge_ports.tl_soa.u.as_f32x4_slice_mut();
            tu[m.tl_v.start] = F32x4::splat(1.0e-5); // e = 1e-10 (> 4e-12, < graze_lim)
            let pv = usize::from(bank.passive_v_vecs[s0]).min(m.tl_v.len() - 1);
            tu[m.tl_v.start + pv] = F32x4::splat(3.0e-6); // e = 9e-12 (>= 4e-12, < 1e-9!)
        }
        bank.soa_dirty = true;
        bank.bp_dirty = false;

        let mut drv = M4TestAction::new(2.5e-5); // shallow half-pedal (+25 um)
        let mut br = M4TestBridge::new(*bank.bridge_port_resistances());
        let (mut l, mut r) = ([0.0_f32; 128], [0.0_f32; 128]);
        bank.render_subblock_bridge_coupled(&mut drv, &mut br, &mut l, &mut r);

        assert_eq!(
            !bank.ports.struck_active[s0], expect_demote,
            "perf={perf:?}: struck_active={}",
            bank.ports.struck_active[s0]
        );

        // Case B: only high-band (> passive_res_max_hz) has energy in [4e-12, 1e-9):
        // Realtime clears tl_damp_active while Full keeps tl_damp_active true.
        let mut bank_b = StringBank::new_bridge_coupled_with_profile(sr, &voicing, &perf);
        bank_b.mark_key_possibly_active(k_e4);
        {
            let tu = bank_b.bridge_ports.tl_soa.u.as_f32x4_slice_mut();
            let pv = usize::from(bank_b.passive_v_vecs[s0]).min(m.tl_v.len() - 1);
            tu[m.tl_v.start + pv] = F32x4::splat(3.0e-6); // e = 9e-12
        }
        bank_b.soa_dirty = true;
        bank_b.bp_dirty = false;
        bank_b.render_subblock_bridge_coupled(&mut drv, &mut br, &mut l, &mut r);
        assert_eq!(
            !bank_b.ports.tl_damp_active[s0], expect_demote,
            "perf={perf:?}: tl_damp_active={}",
            bank_b.ports.tl_damp_active[s0]
        );
    }
}
