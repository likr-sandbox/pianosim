use piano_spec::{VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

use crate::{BridgePortDriver, KeyActionDriver, StringBank};

struct MockAction {
    clearances: [f32; NUM_KEYS],
    weights: [[f32; 3]; NUM_KEYS],
    hammer_left: [usize; NUM_KEYS],
    hammer_peak_n: [f32; NUM_KEYS],
    damper_cap: f32,
}

impl MockAction {
    fn all_open() -> Self {
        Self::with_damper_cap(0.7)
    }

    fn with_damper_cap(damper_cap: f32) -> Self {
        Self {
            clearances: [0.01; NUM_KEYS],
            weights: [[1.0; 3]; NUM_KEYS],
            hammer_left: [0; NUM_KEYS],
            hammer_peak_n: [0.0; NUM_KEYS],
            damper_cap,
        }
    }

    fn strike(&mut self, k: usize, peak_n: f32, samples: usize) {
        self.hammer_peak_n[k] = peak_n;
        self.hammer_left[k] = samples;
    }
}

impl KeyActionDriver for MockAction {
    fn is_hammer_active(&self, k: usize) -> bool {
        self.hammer_left[k] > 0
    }
    fn damper_clearance_m(&self, k: usize) -> f32 {
        self.clearances[k]
    }
    fn strike_weights(&self, k: usize) -> [f32; 3] {
        self.weights[k]
    }
    fn step_key_clearance(&mut self, k: usize) -> f32 {
        self.clearances[k]
    }
    fn step_key_clearance_tile(
        &mut self,
        k: usize,
        t_len: usize,
        out: &mut [f32; crate::zoh_kernel::TILE],
    ) -> f32 {
        out[..t_len].fill(self.clearances[k]);
        self.clearances[k]
    }
    fn step_hammer_sample(&mut self, k: usize, _: f32, _: f32, _: f32) -> f32 {
        if self.hammer_left[k] == 0 {
            return 0.0;
        }
        self.hammer_left[k] -= 1;
        self.hammer_peak_n[k]
    }
    fn step_damper_sample(
        &mut self,
        _: usize,
        _: usize,
        yd: f32,
        vd: f32,
        rp: f32,
        yc: f32,
    ) -> f32 {
        let pen = ((yd.abs() - yc).max(0.0)) / 1.0e-3;
        if pen <= 0.0 {
            return 0.0;
        }
        let c_raw = 300.0 * pen.min(4.0);
        // Cap well below 2.0/rp (vd_post = 0 node-pinning) in Test 2; 2.4/rp in regression test.
        let c = if rp > 0.0 {
            c_raw.min(self.damper_cap / rp)
        } else {
            c_raw
        };
        (-c * vd) / (1.0 + 0.5 * rp * c)
    }
}

struct MockBridge {
    r0: [f32; TOTAL_STRINGS],
    zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
    v: [f32; 8],
    x: [f32; 8],
    m: [f32; 8],
    k: [f32; 8],
    r_damp: [f32; 8],
    ts: f32,
    max_zone_audio_n: f32,
}

impl MockBridge {
    fn build(
        sr: f32,
        r0: [f32; TOTAL_STRINGS],
        split_8_zones: bool,
        m_kg: f32,
        f_hz: f32,
        r_nsm: f32,
    ) -> Self {
        let mut zw = [(0_u8, 1.0_f32, 0_u8, 0.0_f32); TOTAL_STRINGS];
        if split_8_zones {
            for (s, entry) in zw.iter_mut().enumerate() {
                *entry = (((s * 8) / TOTAL_STRINGS) as u8, 1.0, 0, 0.0);
            }
        }
        let w = 2.0 * core::f32::consts::PI * f_hz;
        Self {
            r0,
            zw,
            v: [0.0; 8],
            x: [0.0; 8],
            m: [m_kg; 8],
            k: [m_kg * w * w; 8],
            r_damp: [r_nsm; 8],
            ts: 1.0 / sr,
            max_zone_audio_n: 0.0,
        }
    }

    fn total_energy_joules(&self) -> f64 {
        let mut e = 0.0_f64;
        for z in 0..8 {
            let vz = f64::from(self.v[z]);
            let xz = f64::from(self.x[z]);
            e += 0.5 * f64::from(self.m[z]) * vz * vz + 0.5 * f64::from(self.k[z]) * xz * xz;
        }
        e
    }
}

impl BridgePortDriver for MockBridge {
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
            let (z0, w0, _, _) = self.zw[s];
            let z = usize::from(z0);
            let rs = f64::from(self.r0[s]) * f64::from(w0);
            rhs_a[z] += rs * f64::from(aw);
            sum_r0[z] += rs;
        }
        let ts = f64::from(self.ts);
        let mut v_bar = [0.0_f32; 8];
        for z in 0..8 {
            let (mz, kz, rz) = (
                f64::from(self.m[z]),
                f64::from(self.k[z]),
                f64::from(self.r_damp[z]),
            );
            let denom = (2.0 * mz / ts) + rz + 0.5 * ts * kz + sum_r0[z];
            let rhs = (2.0 * mz / ts) * f64::from(self.v[z]) - kz * f64::from(self.x[z]) + rhs_a[z];
            let vb = rhs / denom;
            self.v[z] = (2.0 * vb - f64::from(self.v[z])) as f32;
            self.x[z] = (f64::from(self.x[z]) + ts * vb) as f32;
            v_bar[z] = vb as f32;
            self.max_zone_audio_n = self.max_zone_audio_n.max(zone_str_audio[z].abs());
            zone_out[z] = v_bar[z] + zone_str_audio[z] / 250.0;
        }
        for (s, bw) in b_waves_out.iter_mut().enumerate() {
            let z0 = self.zw[s].0;
            *bw = 2.0 * v_bar[usize::from(z0)] - a_waves[s];
        }
        (0.0, 0.0)
    }
}

#[test]
fn wdf_bridge_port_conservation_and_passivity_across_rates() {
    let voicing = VoicingSpec::default();
    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        // Linear lossless check (< 1e-6), reusing lossless StringBank cleanly via reset().
        let mut bank = StringBank::new_lossless(sr, &voicing);
        bank.set_nonlinear_coupling(false, false);
        bank.apply_excitation_sample(15, [1.0, 0.0, 0.0], 0.25);
        bank.apply_excitation_sample(39, [1.0, 0.0, 0.0], 0.25);
        bank.apply_excitation_sample(63, [1.0, 0.0, 0.0], 0.25);
        let mut action = MockAction::all_open();
        let mut bridge =
            MockBridge::build(sr, *bank.bridge_port_resistances(), true, 0.06, 160.0, 0.0);
        let (mut l, mut r) = (vec![0.0_f32; 64], vec![0.0_f32; 64]);
        let e0 = bank.total_sav_energy_joules() + bridge.total_energy_joules();
        for _ in 0..2 {
            bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
        }
        let e1 = bank.total_sav_energy_joules() + bridge.total_energy_joules();
        let rel_lin = (e1 - e0).abs() / e0.max(1e-12);
        assert!(rel_lin < 1.0e-6, "sr={sr} linear rel err {rel_lin}");
        assert!(bridge.total_energy_joules() > 0.0);

        // SAV lossless check (< 1e-5).
        bank.reset();
        bank.set_nonlinear_coupling(true, true);
        bank.apply_excitation_sample(15, [1.0, 0.0, 0.0], 0.45);
        let mut bridge_sav =
            MockBridge::build(sr, *bank.bridge_port_resistances(), true, 0.05, 140.0, 0.0);
        let es0 = bank.total_sav_energy_joules() + bridge_sav.total_energy_joules();
        for _ in 0..2 {
            bank.render_subblock_bridge_coupled(&mut action, &mut bridge_sav, &mut l, &mut r);
        }
        let es1 = bank.total_sav_energy_joules() + bridge_sav.total_energy_joules();
        let rel_sav = (es1 - es0).abs() / es0.max(1e-12);
        assert!(rel_sav < 1.0e-5, "sr={sr} SAV rel err {rel_sav}");

        // Lossy strict passivity check.
        let mut bank_d = StringBank::new(sr, &voicing);
        bank_d.apply_excitation_sample(39, [1.0, 1.0, 1.0], 0.35);
        let mut bridge_d = MockBridge::build(
            sr,
            *bank_d.bridge_port_resistances(),
            false,
            0.04,
            150.0,
            30.0,
        );
        let mut prev_e = bank_d.total_sav_energy_joules() + bridge_d.total_energy_joules();
        for _ in 0..2 {
            bank_d.render_subblock_bridge_coupled(&mut action, &mut bridge_d, &mut l, &mut r);
            assert!(l.iter().chain(r.iter()).all(|x| x.is_finite()));
            let cur_e = bank_d.total_sav_energy_joules() + bridge_d.total_energy_joules();
            assert!(
                cur_e <= prev_e + 1.0e-7 * prev_e.max(1.0),
                "sr={sr} energy grew {prev_e} -> {cur_e}"
            );
            prev_e = cur_e;
        }
    }
}

#[test]
fn sympathetic_resonance_prefers_octave_c5_over_offresonant_cs5_and_damper_kills_it() {
    let sr = 48_000.0_f32;
    let voicing = VoicingSpec {
        unison_detune_scale: 0.0,
        ..VoicingSpec::default()
    };
    let mut bank = StringBank::new(sr, &voicing);
    assert_eq!(
        bank.bridge_link_energy_joules(bank.bridge_port_resistances()),
        0.0
    );
    assert!(bank
        .bridge_port_resistances()
        .iter()
        .all(|&x| x > 0.0 && x.is_finite()));

    let mut action = MockAction::all_open();
    action.strike(39, 12.0, 8);
    let mut bridge = MockBridge::build(
        sr,
        *bank.bridge_port_resistances(),
        false,
        0.03,
        180.0,
        12.0,
    );
    let (mut l, mut r) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    for _ in 0..12 {
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
    }
    assert!(
        bridge.max_zone_audio_n > 0.005 && bridge.max_zone_audio_n.is_finite(),
        "expected finite Newtons in zone_str_audio, got {}",
        bridge.max_zone_audio_n
    );
    let c4_sr = bank.choir_map.key(39).unwrap().string_range.clone();
    let c5_sr = bank.choir_map.key(51).unwrap().string_range.clone();
    let cs5_sr = bank.choir_map.key(52).unwrap().string_range.clone();
    assert!(c4_sr.clone().all(|s| bank.ports.struck_active[s]));
    assert!(c5_sr
        .clone()
        .chain(cs5_sr)
        .all(|s| bank.ports.may_be_nonzero[s] && !bank.ports.struck_active[s]));

    let key_energy = |b: &StringBank, key: usize| -> f64 {
        b.choir_map
            .key(key)
            .unwrap()
            .string_range
            .clone()
            .map(|s| b.string_linear_energy_joules(s))
            .sum()
    };
    let e_c5 = key_energy(&bank, 51);
    let e_cs5 = key_energy(&bank, 52);
    assert!(
        e_c5 > 0.0 && e_c5 > 4.0 * e_cs5,
        "expected C5={e_c5} >> C#5={e_cs5}"
    );

    action.clearances[51] = -0.002;
    action.clearances[39] = -0.002;
    let mut prev_e = bank.total_sav_energy_joules() + bridge.total_energy_joules();
    for _ in 0..32 {
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
        let cur_e = bank.total_sav_energy_joules() + bridge.total_energy_joules();
        assert!(
            cur_e <= prev_e + 1.0e-7 * prev_e.max(1.0),
            "DampedSymp/Struck energy grew {prev_e} -> {cur_e}"
        );
        prev_e = cur_e;
    }
    assert!(c5_sr.clone().all(|s| !bank.ports.struck_active[s]));
    let e_c5_damped = key_energy(&bank, 51);
    assert!(
        e_c5_damped < e_c5 * 0.10,
        "expected C5 damped {e_c5_damped} << {e_c5}"
    );
}

#[test]
fn c4_trichord_bridge_coupling_produces_two_stage_and_phase_dependent_decay() {
    let sr = 48_000.0_f32;
    let voicing = VoicingSpec {
        unison_detune_scale: 16.0,
        ..VoicingSpec::default()
    };
    let mut bank = StringBank::new(sr, &voicing);
    let mut run_decay = |weights: [f32; 3], blocks: usize| -> Vec<f64> {
        bank.reset();
        let mut action = MockAction::all_open();
        let mut bridge = MockBridge::build(
            sr,
            *bank.bridge_port_resistances(),
            false,
            0.004,
            260.0,
            12.0,
        );
        let (mut l, mut r) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
        bank.apply_excitation_sample(39, weights, 0.85);
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
        let mut energies = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
            energies.push(bank.key_sav_energy_joules(39).max(1e-30));
        }
        energies
    };

    let b_in = run_decay([1.0, 1.0, 1.0], 8);
    let b_anti = run_decay([1.0, -1.0, 0.0], 4);
    let dt = 128.0 / 48_000.0;
    let prompt_in = 10.0 * (b_in[3] / b_in[0]).log10() / (3.0 * dt);
    let late_in = 10.0 * (b_in[7] / b_in[5]).log10() / (2.0 * dt);
    let prompt_anti = 10.0 * (b_anti[3] / b_anti[0]).log10() / (3.0 * dt);
    assert!(
        (-prompt_in) > (-late_in) * 1.25,
        "expected faster prompt decay than late tail: prompt={prompt_in:.2} dB/s, late={late_in:.2} dB/s"
    );
    assert!(
        (-prompt_in) > (-prompt_anti) * 1.25,
        "expected coherent prompt decay faster than anti-phase: in={prompt_in:.2} dB/s, anti={prompt_anti:.2} dB/s"
    );
}

struct RigidBridge {
    r0: [f32; TOTAL_STRINGS],
    zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
}

impl BridgePortDriver for RigidBridge {
    fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        &self.r0
    }
    fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS] {
        &self.zw
    }
    fn step_bridge_sample(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        _: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
    ) -> (f32, f32) {
        for (bw, &aw) in b_waves_out.iter_mut().zip(a_waves.iter()) {
            *bw = -aw;
        }
        zone_out.fill(0.0);
        (0.0, 0.0)
    }
}

#[test]
fn dry_c4_note_off_rigid_bridge_2_4_cap_drops_over_60_db_within_250_ms() {
    let sr = 48_000.0_f32;
    let voicing = VoicingSpec::default();
    let mut bank = StringBank::new(sr, &voicing);
    let mut action = MockAction::with_damper_cap(2.4);
    let mut bridge = RigidBridge {
        r0: *bank.bridge_port_resistances(),
        zw: [(0, 1.0, 0, 0.0); TOTAL_STRINGS],
    };
    let (mut l, mut r) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    action.strike(39, 12.0, 12);
    for _ in 0..8 {
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
    }
    let e_pre = bank.key_sav_energy_joules(39);
    assert!(e_pre > 0.0 && e_pre.is_finite());

    // Engage damper (`clearances[39] = -0.002`) for 93 blocks = 11,904 samples (< 250 ms @ 48 kHz),
    // then 49 more blocks (total ~379 ms) for > 60 dB drop even on lossless RigidBridge.
    action.clearances[39] = -0.002;
    for _ in 0..93 {
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
    }
    let e_post_250 = bank.key_sav_energy_joules(39);
    let ratio_250 = e_post_250 / e_pre;
    assert!(
        ratio_250 < 2.0e-5,
        "expected dry C4 NoteOff > 47 dB drop (<2e-5) within 250 ms on RigidBridge, got ratio={ratio_250:.3e}"
    );
    for _ in 0..49 {
        bank.render_subblock_bridge_coupled(&mut action, &mut bridge, &mut l, &mut r);
    }
    let e_post = bank.key_sav_energy_joules(39);
    let ratio = e_post / e_pre;
    assert!(
        ratio < 1.0e-6,
        "expected dry C4 NoteOff > 60 dB drop (<1e-6) within 380 ms on RigidBridge, got ratio={ratio:.3e} (e_pre={e_pre:.3e}, e_post={e_post:.3e})"
    );
}
