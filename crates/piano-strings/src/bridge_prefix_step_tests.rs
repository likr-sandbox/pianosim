use piano_spec::{VoicingSpec, TOTAL_STRINGS};

use crate::bridge_port::BridgePortDriver;
use crate::contact::KeyActionDriver;
use crate::StringBank;

struct PeelTestAction {
    clear: f32,
    peel: [f32; 88],
    mid_drop_key: Option<(usize, usize)>,
}

impl KeyActionDriver for PeelTestAction {
    fn is_hammer_active(&self, _key_idx: usize) -> bool {
        false
    }
    fn has_mid_block_event(&self, key_idx: usize, chunk_base: usize, sub: usize) -> bool {
        self.mid_drop_key
            .is_some_and(|(k, at)| k == key_idx && at >= chunk_base && at < chunk_base + sub)
    }
    fn apply_sample_events_at(&mut self, abs_sample: usize) -> u128 {
        if let Some((_, at)) = self.mid_drop_key {
            if abs_sample == at {
                self.clear = -3.0e-4;
            }
        }
        0
    }
    fn damper_clearance_m(&self, _key_idx: usize) -> f32 {
        self.clear
    }
    fn strike_weights(&self, _key_idx: usize) -> [f32; 3] {
        [1.0; 3]
    }
    fn step_key_clearance(&mut self, _key_idx: usize) -> f32 {
        self.clear
    }
    fn step_key_clearance_tile(
        &mut self,
        _key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
    ) -> f32 {
        out_clears[..t_len.min(8)].fill(self.clear);
        self.clear
    }
    fn damper_peel_force_n(&self, key_idx: usize) -> f32 {
        self.peel.get(key_idx).copied().unwrap_or(0.0)
    }
    fn step_hammer_sample(&mut self, _: usize, _: f32, _: f32, _: f32) -> f32 {
        0.0
    }
    fn step_damper_sample(
        &mut self,
        _: usize,
        _: usize,
        _: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        if y_clear_m < 0.0 {
            let c = 24.0_f32.min(2.4 / r_port_damp.max(1.0e-6));
            -(c * v_damp_mps) / (1.0 + 0.5 * r_port_damp * c)
        } else {
            0.0
        }
    }
}

struct ZeroBridge {
    r0: [f32; TOTAL_STRINGS],
    zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
}

impl BridgePortDriver for ZeroBridge {
    fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        &self.r0
    }
    fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS] {
        &self.zw
    }
    fn step_bridge_sample(
        &mut self,
        _: &[f32; TOTAL_STRINGS],
        _: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
    ) -> (f32, f32) {
        b_waves_out.fill(0.0);
        zone_out.fill(0.0);
        (0.0, 0.0)
    }
}

#[test]
fn peel_off_excites_prefix_modes_only_leaving_tl_soa_zero_and_tail_nonzero_false() {
    let mut bank = StringBank::new_bridge_coupled(48_000.0, &VoicingSpec::default());
    let mut bridge = ZeroBridge {
        r0: *bank.bridge_port_resistances(),
        zw: [(0, 1.0, 0, 0.0); TOTAL_STRINGS],
    };
    let mut action = PeelTestAction {
        clear: 1.0e-5,
        peel: [0.0; 88],
        mid_drop_key: None,
    };
    action.peel[39] = 1.5e-3;

    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut out_l, &mut out_r);

    let peak_audio = out_l
        .iter()
        .chain(out_r.iter())
        .fold(0.0_f32, |m, &x| m.max(x.abs()));
    assert!(
        peak_audio > 1.0e-8,
        "peel force did not excite prefix audio"
    );

    // C4 (key 39) strings must stay tail_nonzero == false and tl_soa must stay strictly zero.
    for s_idx in bank.choir_map.key(39).unwrap().string_range.clone() {
        assert!(bank.ports.may_be_nonzero[s_idx]);
        assert!(!bank.ports.tail_nonzero[s_idx]);
    }
    let tl_u = bank.bridge_ports.tl_soa.u.as_slice();
    let tl_v = bank.bridge_ports.tl_soa.v.as_slice();
    assert!(tl_u.iter().all(|&x| x == 0.0) && tl_v.iter().all(|&x| x == 0.0));
    // Neighbor key 40 must remain completely idle.
    for s_idx in bank.choir_map.key(40).unwrap().string_range.clone() {
        assert!(!bank.ports.may_be_nonzero[s_idx]);
    }

    // Mid-block damper drop (sample_offset = 37) on may_be_nonzero && !tail_nonzero string
    // must damp prefix modes cleanly while keeping tl_soa strictly zero.
    let e_pre = bank.key_sav_energy_joules(39);
    assert!(e_pre > 0.0);
    action.peel[39] = 0.0;
    action.mid_drop_key = Some((39, 37));
    bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut out_l, &mut out_r);
    action.mid_drop_key = None;
    for _ in 0..48 {
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut out_l, &mut out_r);
    }
    let e_post = bank.key_sav_energy_joules(39);
    assert!(
        e_post < e_pre * 1.0e-4,
        "mid-block damper drop failed on prefix-only string: e_pre={e_pre:.3e}, e_post={e_post:.3e}"
    );
    for s_idx in bank.choir_map.key(39).unwrap().string_range.clone() {
        assert!(!bank.ports.tail_nonzero[s_idx]);
    }
    let tl_u = bank.bridge_ports.tl_soa.u.as_slice();
    let tl_v = bank.bridge_ports.tl_soa.v.as_slice();
    assert!(tl_u.iter().all(|&x| x == 0.0) && tl_v.iter().all(|&x| x == 0.0));
}
