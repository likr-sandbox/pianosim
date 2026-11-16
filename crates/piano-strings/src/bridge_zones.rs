//! 8-zone bridge velocity output path feeding Stage 4 soundboard radiation (`no_std`).

use crate::{BridgePortDriver, KeyActionDriver, StringBank};

impl StringBank {
    /// Steps a subblock with Stage 1 + Stage 3 WDF coupling, recording per-sample 8-zone
    /// bridge velocities (`zone_frames`) and deferring subnormal flush to block end.
    pub fn render_subblock_bridge_zones_no_flush(
        &mut self,
        action: &mut impl KeyActionDriver,
        bridge: &mut impl BridgePortDriver,
        out_l: &mut [f32],
        out_r: &mut [f32],
        zone_frames: &mut [[f32; 8]],
    ) {
        let frames = out_l.len().min(out_r.len()).min(zone_frames.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }
        if zone_frames.len() > frames {
            zone_frames[frames..].fill([0.0; 8]);
        }
        if frames == 0 {
            return;
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
            self.lossless,
            false,
            &mut out_l[..frames],
            &mut out_r[..frames],
            Some(&mut zone_frames[..frames]),
        );
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use alloc::vec;
    use piano_spec::{VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

    use crate::{BridgePortDriver, KeyActionDriver, StringBank};

    struct ZoneTestAction {
        hammer_left: [usize; NUM_KEYS],
        hammer_peak_n: [f32; NUM_KEYS],
    }

    impl ZoneTestAction {
        fn new() -> Self {
            Self {
                hammer_left: [0; NUM_KEYS],
                hammer_peak_n: [0.0; NUM_KEYS],
            }
        }

        fn strike(&mut self, key_idx: usize, peak_n: f32, samples: usize) {
            self.hammer_left[key_idx] = samples;
            self.hammer_peak_n[key_idx] = peak_n;
        }
    }

    impl KeyActionDriver for ZoneTestAction {
        fn is_hammer_active(&self, key_idx: usize) -> bool {
            self.hammer_left[key_idx] > 0
        }

        fn damper_clearance_m(&self, _: usize) -> f32 {
            0.01
        }

        fn strike_weights(&self, _: usize) -> [f32; 3] {
            [1.0; 3]
        }

        fn step_key_clearance(&mut self, _: usize) -> f32 {
            0.01
        }

        fn step_key_clearance_tile(
            &mut self,
            _: usize,
            t_len: usize,
            out: &mut [f32; crate::zoh_kernel::TILE],
        ) -> f32 {
            out[..t_len].fill(0.01);
            0.01
        }

        fn step_hammer_sample(&mut self, key_idx: usize, _: f32, _: f32, _: f32) -> f32 {
            if self.hammer_left[key_idx] == 0 {
                return 0.0;
            }
            self.hammer_left[key_idx] -= 1;
            self.hammer_peak_n[key_idx]
        }

        fn step_damper_sample(
            &mut self,
            _: usize,
            _: usize,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> f32 {
            0.0
        }
    }

    struct ZoneTestBridge {
        r0: [f32; TOTAL_STRINGS],
        zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
        v: [f32; 8],
        x: [f32; 8],
        m: f64,
        k: f64,
        r_damp: f64,
        ts: f64,
    }

    impl ZoneTestBridge {
        fn new(sr: f32, r0: [f32; TOTAL_STRINGS]) -> Self {
            let mut zw = [(0_u8, 1.0_f32, 0_u8, 0.0_f32); TOTAL_STRINGS];
            for (s, entry) in zw.iter_mut().enumerate() {
                *entry = (((s * 8) / TOTAL_STRINGS) as u8, 1.0, 0, 0.0);
            }
            let m = 0.055_f64;
            let w = f64::from(2.0 * PI * 140.0);
            Self {
                r0,
                zw,
                v: [0.0; 8],
                x: [0.0; 8],
                m,
                k: m * w * w,
                r_damp: 18.0,
                ts: 1.0 / f64::from(sr),
            }
        }
    }

    impl BridgePortDriver for ZoneTestBridge {
        fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
            &self.r0
        }

        fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS] {
            &self.zw
        }

        fn step_bridge_sample(
            &mut self,
            a_waves: &[f32; TOTAL_STRINGS],
            zone_str_audio: &[f32; 8],
            b_waves_out: &mut [f32; TOTAL_STRINGS],
            zone_out: &mut [f32; 8],
        ) -> (f32, f32) {
            let (mut rhs_a, mut sum_r0) = ([0.0_f64; 8], [0.0_f64; 8]);
            for (s, &aw) in a_waves.iter().enumerate() {
                let z = usize::from(self.zw[s].0);
                let rs = f64::from(self.r0[s]);
                rhs_a[z] += rs * f64::from(aw);
                sum_r0[z] += rs;
            }
            let mut v_bar = [0.0_f32; 8];
            for z in 0..8 {
                let denom =
                    (2.0 * self.m / self.ts) + self.r_damp + 0.5 * self.ts * self.k + sum_r0[z];
                let rhs = (2.0 * self.m / self.ts) * f64::from(self.v[z])
                    - self.k * f64::from(self.x[z])
                    + rhs_a[z];
                let vb = rhs / denom;
                self.v[z] = (2.0 * vb - f64::from(self.v[z])) as f32;
                self.x[z] = (f64::from(self.x[z]) + self.ts * vb) as f32;
                v_bar[z] = vb as f32;
                zone_out[z] = v_bar[z] + zone_str_audio[z] / 250.0;
            }
            for (s, bw) in b_waves_out.iter_mut().enumerate() {
                *bw = 2.0 * v_bar[usize::from(self.zw[s].0)] - a_waves[s];
            }
            (0.0, 0.0)
        }
    }

    #[test]
    fn render_subblock_bridge_zones_matches_coupled_and_captures_8_zones() {
        const SR: f32 = 48_000.0;
        const SUB: usize = 128;
        const BLOCKS: usize = 4;
        let voicing = VoicingSpec::default();
        let mut bank_coupled = StringBank::new(SR, &voicing);
        let mut bank_zones = StringBank::new(SR, &voicing);
        let r0 = *bank_coupled.bridge_port_resistances();

        let mut act_a = ZoneTestAction::new();
        let mut act_b = ZoneTestAction::new();
        for &(key, peak, dur) in &[(0usize, 18.0_f32, 32usize), (48, 22.0, 24), (87, 15.0, 12)] {
            act_a.strike(key, peak, dur);
            act_b.strike(key, peak, dur);
        }
        let mut br_a = ZoneTestBridge::new(SR, r0);
        let mut br_b = ZoneTestBridge::new(SR, r0);

        let (mut l_a, mut r_a) = (vec![0.0_f32; SUB], vec![0.0_f32; SUB]);
        let (mut l_b, mut r_b) = (vec![0.0_f32; SUB], vec![0.0_f32; SUB]);
        let mut zones = vec![[0.0_f32; 8]; SUB];
        let mut zone_sq = [0.0_f64; 8];
        let mut max_diff = 0.0_f32;
        let no_hammers = [false; NUM_KEYS];

        for _ in 0..BLOCKS {
            bank_coupled
                .render_subblock_bridge_coupled_no_flush(&mut act_a, &mut br_a, &mut l_a, &mut r_a);
            bank_coupled.flush_bridge_subnormals(&no_hammers);
            bank_zones.render_subblock_bridge_zones_no_flush(
                &mut act_b, &mut br_b, &mut l_b, &mut r_b, &mut zones,
            );
            bank_zones.flush_bridge_subnormals(&no_hammers);

            for i in 0..SUB {
                max_diff = max_diff
                    .max((l_a[i] - l_b[i]).abs())
                    .max((r_a[i] - r_b[i]).abs());
                for (z, &zv) in zones[i].iter().enumerate() {
                    assert!(zv.is_finite());
                    zone_sq[z] += f64::from(zv) * f64::from(zv);
                }
            }
        }
        assert!(max_diff == 0.0, "max stereo diff {max_diff}");
        let z0 = usize::from(br_b.zw[bank_zones.choir_map().key(0).unwrap().string_range.start].0);
        let z48 =
            usize::from(br_b.zw[bank_zones.choir_map().key(48).unwrap().string_range.start].0);
        let z87 =
            usize::from(br_b.zw[bank_zones.choir_map().key(87).unwrap().string_range.start].0);
        assert!(z0 != z48 && z48 != z87 && z0 != z87);
        assert!(
            zone_sq[z0] > 1.0e-12 && zone_sq[z48] > 1.0e-12 && zone_sq[z87] > 1.0e-12,
            "unexpected low zone energies: {zone_sq:?}"
        );
        for (z, &sq) in zone_sq.iter().enumerate() {
            if z != z0 && z != z48 && z != z87 {
                assert_eq!(sq, 0.0, "undriven zone {z} leaked energy");
            }
        }
    }

    #[test]
    fn zone_frames_truncation_and_zero_length_zero_tail() {
        let mut bank = StringBank::new(48_000.0, &VoicingSpec::default());
        let r0 = *bank.bridge_port_resistances();
        let mut act = ZoneTestAction::new();
        act.strike(48, 12.0, 8);
        let mut br = ZoneTestBridge::new(48_000.0, r0);

        let (mut out_l, mut out_r) = (vec![9.0_f32; 16], vec![9.0_f32; 16]);
        let mut zf = vec![[9.0_f32; 8]; 8];
        bank.render_subblock_bridge_zones_no_flush(
            &mut act, &mut br, &mut out_l, &mut out_r, &mut zf,
        );
        assert!(out_l[8..].iter().all(|&x| x == 0.0));
        assert!(out_r[8..].iter().all(|&x| x == 0.0));

        let (mut out_l2, mut out_r2) = (vec![9.0_f32; 8], vec![9.0_f32; 8]);
        let mut zf2 = vec![[9.0_f32; 8]; 16];
        bank.render_subblock_bridge_zones_no_flush(
            &mut act,
            &mut br,
            &mut out_l2,
            &mut out_r2,
            &mut zf2,
        );
        assert!(zf2[8..].iter().all(|z| *z == [0.0; 8]));

        bank.render_subblock_bridge_zones_no_flush(
            &mut act,
            &mut br,
            &mut out_l2,
            &mut out_r2,
            &mut [],
        );
        assert!(out_l2.iter().all(|&x| x == 0.0) && out_r2.iter().all(|&x| x == 0.0));
    }
}
