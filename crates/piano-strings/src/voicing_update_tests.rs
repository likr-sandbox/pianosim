use alloc::vec;
use core::f64::consts::TAU;
use piano_spec::{
    build_scale_design, effective_string_f0_hz, Polarization, VoicingSpec, NUM_KEYS, TOTAL_STRINGS,
};

use crate::{BridgePortDriver, KeyActionDriver, StringBank};

fn test_voicing_b() -> VoicingSpec {
    let mut v = VoicingSpec {
        hammer_hardness: 0.72,
        unison_detune_scale: 1.6,
        stretch_amount: 0.65,
        string_length_scale: 1.02,
        ..VoicingSpec::default()
    };
    v.note_offsets[48].pitch_cents = 18.0;
    v.note_offsets[48].level_db_offset = 2.0;
    v.note_offsets[48].hammer_hardness_offset = 0.15;
    v.note_offsets[48].inharmonicity_scale = 0.10;
    v.note_offsets[27].pitch_cents = -14.0;
    v.note_offsets[27].level_db_offset = -1.5;
    v
}

fn rms(slice: &[f32]) -> f32 {
    (slice.iter().map(|x| x * x).sum::<f32>() / slice.len().max(1) as f32).sqrt()
}

fn goertzel_power(samples: &[f64], freq_hz: f64, sr: f64) -> f64 {
    let coeff = 2.0 * (TAU * freq_hz / sr).cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in samples {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn peak_freq_hz(samples: &[f64], f_center: f64, sr: f64) -> f64 {
    let d_cents = 0.05;
    let r = 2.0_f64.powf(d_cents / 1200.0);
    let p_lo = goertzel_power(samples, f_center / r, sr);
    let p_mid = goertzel_power(samples, f_center, sr);
    let p_hi = goertzel_power(samples, f_center * r, sr);
    let denom = p_lo - 2.0 * p_mid + p_hi;
    let shift = if denom < 0.0 {
        (0.5 * (p_lo - p_hi) / denom).clamp(-1.0, 1.0) * d_cents
    } else {
        0.0
    };
    f_center * 2.0_f64.powf(shift / 1200.0)
}

struct VAction {
    hammer_left: [usize; NUM_KEYS],
}

impl KeyActionDriver for VAction {
    fn is_hammer_active(&self, k: usize) -> bool {
        self.hammer_left[k] > 0
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
    fn step_key_clearance_tile(&mut self, _: usize, t: usize, out: &mut [f32; 8]) -> f32 {
        out[..t].fill(0.01);
        0.01
    }
    fn step_hammer_sample(&mut self, k: usize, _: f32, _: f32, _: f32) -> f32 {
        if self.hammer_left[k] == 0 {
            return 0.0;
        }
        self.hammer_left[k] -= 1;
        20.0
    }
    fn step_damper_sample(&mut self, _: usize, _: usize, _: f32, _: f32, _: f32, _: f32) -> f32 {
        0.0
    }
}

struct VBridge {
    r0: [f32; TOTAL_STRINGS],
    zw: [(u8, f32, u8, f32); TOTAL_STRINGS],
}

impl BridgePortDriver for VBridge {
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
fn update_voicing_matches_fresh_bank_tuning_and_modes() {
    const N: usize = 32_768;
    let vb = test_voicing_b();
    let mut bank_in_place = StringBank::new(48_000.0, &VoicingSpec::default());
    bank_in_place.update_voicing_in_place(&vb);
    let mut bank_fresh = StringBank::new(48_000.0, &vb);

    for k in [27usize, 48] {
        let s0 = bank_in_place.choir_map().key(k).unwrap().string_range.start;
        let (r_ip, r_fr) = (
            bank_in_place.bridge_port_resistances()[s0],
            bank_fresh.bridge_port_resistances()[s0],
        );
        assert!((r_ip - r_fr).abs() / r_fr.max(1e-12) < 1e-6);
        assert!(
            (bank_in_place.ports.r_port_strike[s0] - bank_fresh.ports.r_port_strike[s0]).abs()
                < 1e-4
        );
    }

    bank_in_place.set_nonlinear_coupling(false, false);
    bank_fresh.set_nonlinear_coupling(false, false);
    bank_in_place.apply_excitation_sample(48, [1.0, 0.0, 0.0], 0.5);
    bank_fresh.apply_excitation_sample(48, [1.0, 0.0, 0.0], 0.5);
    let (mut l_ip, mut r_ip) = (vec![0.0_f32; N], vec![0.0_f32; N]);
    let (mut l_fr, mut r_fr) = (vec![0.0_f32; N], vec![0.0_f32; N]);
    bank_in_place.render_subblock(&mut l_ip, &mut r_ip);
    bank_fresh.render_subblock(&mut l_fr, &mut r_fr);

    let mono: alloc::vec::Vec<f64> = l_ip
        .iter()
        .zip(r_ip.iter())
        .map(|(&l, &r)| f64::from(l + r))
        .collect();
    let specs = build_scale_design(&vb);
    let f_target = effective_string_f0_hz(48, 0, Polarization::Vertical, &specs, &vb);
    let f_meas = peak_freq_hz(&mono, f_target, 48_000.0);
    let cents_err = 1200.0 * (f_meas / f_target).log2();
    assert!(cents_err.abs() < 0.05, "tuned f0 cents err {cents_err}");

    let max_diff = l_ip
        .iter()
        .zip(l_fr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 2e-5, "wave parity max diff {max_diff}");
}

#[test]
fn update_voicing_in_place_is_click_free_and_preserves_live_state_both_modes() {
    let mut vb = test_voicing_b();
    vb.string_length_scale = 1.05;

    // Branch A: soa_dirty = true (Stage 3 bridge-coupled live ringing).
    let mut bank_a = StringBank::new(48_000.0, &VoicingSpec::default());
    let mut act = VAction {
        hammer_left: [0; NUM_KEYS],
    };
    act.hammer_left[27] = 28;
    act.hammer_left[48] = 20;
    let mut br = VBridge {
        r0: *bank_a.bridge_port_resistances(),
        zw: [(0, 1.0, 0, 0.0); TOTAL_STRINGS],
    };
    let (mut l, mut r) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
    for _ in 0..8 {
        bank_a.render_subblock_bridge_coupled_no_flush(&mut act, &mut br, &mut l, &mut r);
    }
    let max_step_a = l
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f32, f32::max);
    let (last_a, rms_before_a) = (l[127], rms(&l));
    let sav_before_a = bank_a.sav.r_sav;
    let (soa_len, bp_len) = (
        bank_a.soa.u.as_slice().len(),
        bank_a.bridge_ports.state.len(),
    );

    bank_a.update_voicing_in_place(&vb);
    assert_eq!(bank_a.sav.r_sav, sav_before_a);
    assert_eq!(bank_a.soa.u.as_slice().len(), soa_len);
    assert_eq!(bank_a.bridge_ports.state.len(), bp_len);
    let s48 = bank_a.choir_map().key(48).unwrap().string_range.start;
    assert!(bank_a.ports.may_be_nonzero[s48] && !bank_a.ports.ports_synced[s48]);

    br.r0 = *bank_a.bridge_port_resistances();
    bank_a.render_subblock_bridge_coupled_no_flush(&mut act, &mut br, &mut l, &mut r);
    let step_a = (l[0] - last_a).abs();
    assert!(l.iter().all(|x| x.is_finite()) && r.iter().all(|x| x.is_finite()));
    assert!(rms(&l) > 0.2 * rms_before_a);
    assert!(
        step_a <= 3.0 * max_step_a.max(1e-5),
        "step_a={step_a} max_before={max_step_a}"
    );

    // Branch B: soa_dirty = false (direct ModalSoa render_subblock + bp_dirty handoff).
    let mut bank_b = StringBank::new(48_000.0, &VoicingSpec::default());
    bank_b.apply_excitation_sample(48, [1.0, 1.0, 1.0], 0.8);
    let (mut lb, mut rb) = (vec![0.0_f32; 256], vec![0.0_f32; 256]);
    bank_b.render_subblock(&mut lb, &mut rb);
    let max_step_b = lb[128..]
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f32, f32::max);
    let (last_b, rms_before_b) = (lb[255], rms(&lb[128..]));
    let sav_before_b = bank_b.sav.r_sav;

    vb.string_length_scale = 1.08;
    bank_b.update_voicing_in_place(&vb);
    assert_eq!(bank_b.sav.r_sav, sav_before_b);
    bank_b.render_subblock(&mut lb, &mut rb);
    let step_b = (lb[0] - last_b).abs();
    assert!(lb.iter().all(|x| x.is_finite()) && rms(&lb) > 0.2 * rms_before_b);
    assert!(
        step_b <= 3.0 * max_step_b.max(1e-5),
        "step_b={step_b} max_before={max_step_b}"
    );
}

#[test]
fn update_voicing_extreme_clamping_and_sample_rates() {
    let mut extreme_lo = VoicingSpec {
        hammer_hardness: -1.0,
        unison_detune_scale: -2.0,
        stretch_amount: -2.0,
        string_length_scale: 0.2,
        ..VoicingSpec::default()
    };
    extreme_lo.note_offsets[0].pitch_cents = -250.0;
    extreme_lo.note_offsets[0].inharmonicity_scale = -2.0;
    extreme_lo.note_offsets[0].level_db_offset = -50.0;
    let mut extreme_hi = VoicingSpec {
        hammer_hardness: 3.0,
        unison_detune_scale: 9.0,
        stretch_amount: 5.0,
        string_length_scale: 4.0,
        ..VoicingSpec::default()
    };
    extreme_hi.note_offsets[87].pitch_cents = 250.0;
    extreme_hi.note_offsets[87].inharmonicity_scale = 9.0;
    extreme_hi.note_offsets[87].level_db_offset = 40.0;

    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        let mut bank = StringBank::new(sr, &VoicingSpec::default());
        bank.apply_excitation_sample(0, [1.0, 0.0, 0.0], 0.4);
        bank.apply_excitation_sample(87, [1.0, 1.0, 1.0], 0.4);
        let (mut l, mut r) = (vec![0.0_f32; 128], vec![0.0_f32; 128]);
        bank.render_subblock(&mut l, &mut r);
        bank.update_voicing_in_place(&extreme_lo);
        bank.render_subblock(&mut l, &mut r);
        bank.update_voicing_in_place(&extreme_hi);
        bank.render_subblock(&mut l, &mut r);
        assert!(bank.soa.d_coeff.as_slice().iter().all(|x| x.is_finite()));
        assert!(bank.soa.s_coeff.as_slice().iter().all(|x| x.is_finite()));
        assert!(bank.bridge_port_resistances().iter().all(|x| x.is_finite()));
        assert!(l.iter().all(|x| x.is_finite()) && r.iter().all(|x| x.is_finite()));
    }
}
