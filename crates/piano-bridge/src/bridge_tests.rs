use core::f64::consts::TAU;

use piano_spec::{CHOIR_LAYOUTS, NUM_KEYS, TOTAL_STRINGS};

use crate::hertz_contact::{hertz_potential_f32, DEFAULT_HERTZ_K3, DEFAULT_HERTZ_K4};
use crate::{
    BridgeBank, HertzBridgeContact, KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES, NUM_RADIATOR_ZONES,
    NUM_REDUCED_MODES, TREBLE_DUPLEX_ZONE_START,
};

const SAMPLE_RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

pub(super) fn wdf_absorbed_power_w(
    r0: &[f32; TOTAL_STRINGS],
    a_waves: &[f32; TOTAL_STRINGS],
    b_waves: &[f32; TOTAL_STRINGS],
) -> f64 {
    let mut power = 0.0_f64;
    for s in 0..TOTAL_STRINGS {
        let a = f64::from(a_waves[s]);
        let b = f64::from(b_waves[s]);
        power += 0.25 * f64::from(r0[s]) * (a * a - b * b);
    }
    power
}

#[test]
fn partition_of_unity_and_zone_invariants() {
    assert_eq!(NUM_BRIDGE_ZONES, 8);
    assert_eq!(NUM_RADIATOR_ZONES, 9);
    assert_eq!(KEYBED_ZONE_IDX, 8);
    assert_eq!(TREBLE_DUPLEX_ZONE_START, 2);

    let bridge = BridgeBank::new(48_000.0);
    let weights = bridge.zone_weights();
    assert_eq!(weights.len(), TOTAL_STRINGS);

    let mut zone_totals = [0.0_f64; NUM_BRIDGE_ZONES];
    let mut prev_center = -1.0_f64;
    for &(z0, w0, z1, w1) in weights {
        assert!(usize::from(z0) < NUM_BRIDGE_ZONES);
        assert!(usize::from(z1) < NUM_BRIDGE_ZONES);
        assert!(usize::from(z0) != KEYBED_ZONE_IDX);
        assert!(usize::from(z1) != KEYBED_ZONE_IDX);
        assert!((0.0..=1.0).contains(&w0));
        assert!((0.0..=1.0).contains(&w1));
        assert!(
            (w0 + w1 - 1.0).abs() <= f32::EPSILON,
            "partition of unity violated: w0={w0}, w1={w1}"
        );
        zone_totals[usize::from(z0)] += f64::from(w0);
        zone_totals[usize::from(z1)] += f64::from(w1);

        let center = f64::from(z0) * f64::from(w0) + f64::from(z1) * f64::from(w1);
        assert!(center + 1.0e-5 >= prev_center);
        prev_center = center;
    }

    for (z, &tot) in zone_totals.iter().enumerate() {
        assert!(tot > 1.0, "bridge zone {z} under-populated: {tot}");
    }
    assert!(prev_center > 6.8, "treble end should approach Zone 7");
    assert_eq!(CHOIR_LAYOUTS.len(), NUM_KEYS);
}

#[test]
fn reduced_poles_and_high_frequency_y_inf_mobility() {
    let mut bridge = BridgeBank::new(48_000.0);
    let poles = bridge.reduced_poles();
    assert_eq!(poles.len(), NUM_REDUCED_MODES);
    assert!(poles[0].freq_hz >= 50.0 && poles[0].freq_hz <= 60.0);
    assert!(poles[NUM_REDUCED_MODES - 1].freq_hz >= 1_050.0);
    for (m, pole) in poles.iter().enumerate() {
        assert!(pole.zeta > 0.0);
        let expected_q = 1.0 / (2.0 * pole.zeta);
        assert!((pole.q_factor - expected_q).abs() < 1.0e-4);
        assert!(pole.modal_mass_kg > 0.0);
        if m < 3 {
            assert!(pole.zone_gains[0].abs() <= 0.22);
            assert!(pole.zone_gains[1].abs() <= 0.22);
        }
    }

    // Verify full spatial rank 8x8 SPD admittance and well-conditioned WDF junction solve.
    let mut l_chol = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    let a_br = bridge.ssm().a_br();
    for i in 0..NUM_BRIDGE_ZONES {
        for j in 0..=i {
            let mut sum = a_br[i][j];
            for (&lik, &ljk) in l_chol[i][..j].iter().zip(&l_chol[j][..j]) {
                sum -= lik * ljk;
            }
            if i == j {
                assert!(sum > 0.0, "a_br Cholesky pivot {i} not SPD: {sum}");
                l_chol[i][j] = sum.sqrt();
            } else {
                l_chol[i][j] = sum / l_chol[j][j];
            }
        }
    }
    let solve = bridge.ssm().solve_8x8();
    for (i, row) in solve.iter().enumerate() {
        assert!(row[i] > 0.0 && row[i] <= 1.0 + 1.0e-9);
    }

    let mut b_zero = [0.0_f32; TOTAL_STRINGS];
    let mut b_audio = [0.0_f32; TOTAL_STRINGS];
    let mut z_out_zero = [0.0_f32; NUM_BRIDGE_ZONES];
    let mut z_out_audio = [0.0_f32; NUM_BRIDGE_ZONES];

    let a_waves = [0.0_f32; TOTAL_STRINGS];
    let mut audio_force = [0.0_f32; NUM_BRIDGE_ZONES];
    audio_force[6] = 2.5;

    let mut clone_b = bridge.clone();
    let _ = bridge.step_bridge_sample(
        &a_waves,
        &[0.0; NUM_BRIDGE_ZONES],
        &mut b_zero,
        &mut z_out_zero,
    );
    let _ = clone_b.step_bridge_sample(&a_waves, &audio_force, &mut b_audio, &mut z_out_audio);

    assert_eq!(
        b_zero, b_audio,
        "zone_str_audio must never enter b_waves_out"
    );
    let expected_dv = bridge.ssm().y_inf()[6] * 2.5;
    assert!(expected_dv > 5.0e-4);
    assert!((z_out_audio[6] - z_out_zero[6] - expected_dv).abs() < 1.0e-7);

    // High-frequency (~7 kHz) string wave still moves treble zone via y_inf (no 1.12 kHz wall).
    let sr = 48_000.0_f64;
    let mut hf_bridge = BridgeBank::new(48_000.0);
    let c7_str = usize::from(CHOIR_LAYOUTS[75].first_string_idx);
    let mut max_zone_v = 0.0_f32;
    for n in 0..240 {
        let mut a = [0.0_f32; TOTAL_STRINGS];
        a[c7_str] = (TAU * 7_000.0 * (n as f64 / sr)).sin() as f32 * 0.4;
        let mut b = [0.0_f32; TOTAL_STRINGS];
        let mut z = [0.0_f32; NUM_BRIDGE_ZONES];
        let _ = hf_bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
        max_zone_v = max_zone_v.max(z[6].abs());
    }
    assert!(max_zone_v > 1.0e-5, "7 kHz mobility stalled: {max_zone_v}");
}

#[test]
fn hertz_contact_potential_and_conservation() {
    assert!(hertz_potential_f32(0.0, DEFAULT_HERTZ_K3, DEFAULT_HERTZ_K4) == 0.0);
    let v_pos = hertz_potential_f32(1.0e-4, DEFAULT_HERTZ_K3, DEFAULT_HERTZ_K4);
    let v_neg = hertz_potential_f32(-1.0e-4, DEFAULT_HERTZ_K3, DEFAULT_HERTZ_K4);
    assert!(v_pos > v_neg && v_neg > 0.0);

    for sr in SAMPLE_RATES {
        let ts = 1.0 / f64::from(sr);
        let mut hertz = HertzBridgeContact::new(sr);
        let e0 = hertz.energy_joules();
        let mut work = 0.0_f64;
        let mut peak_e = 0.0_f64;
        let mut max_abs_err = 0.0_f64;
        for n in 0..1_200 {
            let t = (n as f64) * ts;
            let v_val = 0.08 * (TAU * 220.0 * t).sin();
            let v_zone = [v_val, -0.5 * v_val, 0.3 * v_val, 0.0, 0.0, 0.0, 0.0, 0.0];
            let g = hertz.eval_sav_gradient();
            let f_on_br = hertz.commit_step(&g, &v_zone);
            for z in 0..NUM_BRIDGE_ZONES {
                work -= ts * f_on_br[z] * v_zone[z];
            }
            let e_cur = hertz.energy_joules();
            peak_e = peak_e.max((e_cur - e0).abs());
            max_abs_err = max_abs_err.max((e_cur - e0 - work).abs());
        }
        assert!(peak_e > 1.0e-7, "Hertz peak energy too small: {peak_e}");
        let rel_err = max_abs_err / peak_e;
        assert!(rel_err < 1.0e-11, "Hertz SAV drift at {sr} Hz: {rel_err}");
    }
}

#[test]
fn lossless_energy_conservation_across_sample_rates() {
    let freqs = [89.0_f64, 212.0_f64, 405.0_f64];
    let drive_keys = [18_usize, 32, 48];
    for sr in SAMPLE_RATES {
        let ts = 1.0 / f64::from(sr);
        let steps_drive = (0.045 * f64::from(sr)) as usize;
        let steps_ring = (0.015 * f64::from(sr)) as usize;

        for with_duplex in [false, true] {
            let mut bridge = if with_duplex {
                BridgeBank::lossless_with_duplex_for_test(sr)
            } else {
                BridgeBank::lossless_for_test(sr)
            };
            let r0 = *bridge.port_resistances();
            let e0 = bridge.total_energy_joules();
            let mut e_in_wdf = 0.0_f64;
            let mut max_e = 0.0_f64;

            for n in 0..(steps_drive + steps_ring) {
                let mut a_waves = [0.0_f32; TOTAL_STRINGS];
                if n < steps_drive {
                    let t = (n as f64) * ts;
                    for (&key_idx, &f_hz) in drive_keys.iter().zip(freqs.iter()) {
                        let first = usize::from(CHOIR_LAYOUTS[key_idx].first_string_idx);
                        let count = usize::from(CHOIR_LAYOUTS[key_idx].strings_per_key);
                        let val = (0.25 * (TAU * f_hz * t).sin()) as f32;
                        for slot in &mut a_waves[first..first + count] {
                            *slot = val;
                        }
                    }
                }
                let mut b_waves = [0.0_f32; TOTAL_STRINGS];
                let mut zone_out = [0.0_f32; NUM_BRIDGE_ZONES];
                let _ = bridge.step_bridge_sample(
                    &a_waves,
                    &[0.0; NUM_BRIDGE_ZONES],
                    &mut b_waves,
                    &mut zone_out,
                );
                e_in_wdf += ts * wdf_absorbed_power_w(&r0, &a_waves, &b_waves);
                max_e = max_e.max(bridge.total_energy_joules());
            }

            let de_bridge = bridge.total_energy_joules() - e0;
            let scale = max_e.max(e_in_wdf.abs()).max(1.0e-12);
            let rel_err = (de_bridge - e_in_wdf).abs() / scale;
            let tol = if with_duplex { 5.0e-6 } else { 1.0e-6 };
            assert!(
                rel_err < tol,
                "lossless energy error at {sr} Hz (duplex={with_duplex}): rel_err={rel_err}, de={de_bridge}, ein={e_in_wdf}"
            );
            assert!(max_e > 1.0e-6, "bridge energy stayed too small: {max_e}");
        }
    }
}

#[test]
fn wdf_passivity_across_sample_rates() {
    for sr in SAMPLE_RATES {
        let ts = 1.0 / f64::from(sr);
        let mut bridge = BridgeBank::new(sr);
        let r0 = *bridge.port_resistances();
        let e0 = bridge.total_energy_joules();
        let mut e_in_wdf = 0.0_f64;
        let mut max_e = 0.0_f64;

        let n_steps = (0.06 * f64::from(sr)) as usize;
        for n in 0..n_steps {
            let t = (n as f64) * ts;
            let mut a_waves = [0.0_f32; TOTAL_STRINGS];
            for k in (5..85).step_by(9) {
                let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
                let f = 80.0 + 35.0 * (k as f64);
                a_waves[s] = (0.2 * (TAU * f * t).sin()) as f32;
            }
            let audio_force = [20.0_f32; NUM_BRIDGE_ZONES];
            let mut b_waves = [0.0_f32; TOTAL_STRINGS];
            let mut zone_out = [0.0_f32; NUM_BRIDGE_ZONES];
            let _ = bridge.step_bridge_sample(&a_waves, &audio_force, &mut b_waves, &mut zone_out);

            e_in_wdf += ts * wdf_absorbed_power_w(&r0, &a_waves, &b_waves);
            let cur_e = bridge.total_energy_joules();
            max_e = max_e.max(cur_e);
            let de = cur_e - e0;
            let eps = 1.0e-8 * (max_e.max(e_in_wdf.abs()) + 1.0e-9);
            assert!(
                de <= e_in_wdf + eps,
                "WDF passivity violation at sr={sr}, step={n}: dE={de}, E_in={e_in_wdf}"
            );
        }
        assert!(bridge.dissipated_energy_joules() > 0.0);
    }
}

#[test]
fn trichord_in_phase_vs_anti_phase_absorption_ratio() {
    let key_idx = 48_usize; // A4 trichord
    let layout = CHOIR_LAYOUTS[key_idx];
    assert_eq!(layout.strings_per_key, 3);
    let s0 = usize::from(layout.first_string_idx);

    let sr = 48_000.0_f32;
    let ts = 1.0 / f64::from(sr);
    let f0 = 440.0_f64;
    let mut b_in = BridgeBank::new(sr);
    let mut b_anti = BridgeBank::new(sr);
    let r0 = *b_in.port_resistances();

    let mut e_in_phase = 0.0_f64;
    let mut e_anti_phase = 0.0_f64;
    let n_steps = 4_800_usize;

    for n in 0..n_steps {
        let t = (n as f64) * ts;
        let phase = TAU * f0 * t;

        let mut a_in = [0.0_f32; TOTAL_STRINGS];
        let w = (0.3 * phase.sin()) as f32;
        a_in[s0] = w;
        a_in[s0 + 1] = w;
        a_in[s0 + 2] = w;

        let mut a_anti = [0.0_f32; TOTAL_STRINGS];
        a_anti[s0] = (0.3 * phase.sin()) as f32;
        a_anti[s0 + 1] = (0.3 * (phase + TAU / 3.0).sin()) as f32;
        a_anti[s0 + 2] = (0.3 * (phase * 1.0005 + 2.0 * TAU / 3.0).sin()) as f32;

        let mut bout_in = [0.0_f32; TOTAL_STRINGS];
        let mut bout_anti = [0.0_f32; TOTAL_STRINGS];
        let mut z_out = [0.0_f32; NUM_BRIDGE_ZONES];

        let _ = b_in.step_bridge_sample(&a_in, &[0.0; NUM_BRIDGE_ZONES], &mut bout_in, &mut z_out);
        let _ = b_anti.step_bridge_sample(
            &a_anti,
            &[0.0; NUM_BRIDGE_ZONES],
            &mut bout_anti,
            &mut z_out,
        );

        e_in_phase += ts * wdf_absorbed_power_w(&r0, &a_in, &bout_in);
        e_anti_phase += ts * wdf_absorbed_power_w(&r0, &a_anti, &bout_anti);
    }

    assert!(e_in_phase > 0.0 && e_anti_phase >= 0.0);
    let ratio = e_in_phase / e_anti_phase.max(1.0e-18);
    assert!(
        ratio > 10.0,
        "trichord in/anti absorption ratio too low: ratio={ratio}, in={e_in_phase}, anti={e_anti_phase}"
    );
}

#[test]
fn non_finite_inputs_are_sanitized_without_panic() {
    let mut bridge = BridgeBank::new(f32::NAN);
    assert!((bridge.sample_rate() - 48_000.0).abs() < 1.0e-3);

    let mut bad_r0 = *bridge.port_resistances();
    bad_r0[0] = f32::NAN;
    bad_r0[1] = -5.0;
    bridge.set_port_resistances(&bad_r0);

    let mut a = [0.0_f32; TOTAL_STRINGS];
    a[0] = f32::NAN;
    a[10] = f32::INFINITY;
    let audio = [f32::NEG_INFINITY; NUM_BRIDGE_ZONES];
    let mut b = [0.0_f32; TOTAL_STRINGS];
    let mut z = [0.0_f32; NUM_BRIDGE_ZONES];
    let (l, r) = bridge.step_bridge_sample(&a, &audio, &mut b, &mut z);

    assert!(l.is_finite() && r.is_finite());
    assert!(bridge.total_energy_joules().is_finite());
    assert!(b.iter().all(|v| v.is_finite()));
    assert!(z.iter().all(|v| v.is_finite()));
}

#[test]
fn stage3_subnormal_flush_restores_quiescence_and_idle_bypass() {
    let mut bridge = BridgeBank::new(48_000.0);
    assert!(bridge.is_quiescent());

    // Excite with a small impulse and verify non-quiescence, then step with zero input
    // until SSM/Hertz/Duplex decay below 1e-20 and flush_subnormals restores quiescence.
    let mut a = [0.0_f32; TOTAL_STRINGS];
    a[140] = 1.0e-14;
    let mut b = [0.0_f32; TOTAL_STRINGS];
    let mut z = [0.0_f32; NUM_BRIDGE_ZONES];
    let _ = bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
    assert!(!bridge.is_quiescent());

    let zeros = [0.0_f32; TOTAL_STRINGS];
    for _ in 0..48_000 {
        let _ = bridge.step_bridge_sample(&zeros, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
        bridge.flush_subnormals();
        if bridge.is_quiescent() {
            break;
        }
    }
    assert!(
        bridge.is_quiescent(),
        "BridgeBank failed to return to quiescent idle bypass after decay"
    );
    assert_eq!(bridge.ssm_energy_joules(), 0.0);
    assert_eq!(bridge.duplex_energy_joules(), 0.0);
}

#[test]
fn fixed_end_mode_reflects_inverted_waves_without_coupling_or_state_buildup() {
    let mut bridge = BridgeBank::new(48_000.0);
    assert!(!bridge.is_fixed_end());
    bridge.set_fixed_end(true);
    assert!(bridge.is_fixed_end());

    let layout = CHOIR_LAYOUTS[48]; // A4 trichord
    let s0 = usize::from(layout.first_string_idx);
    let mut a = [0.0_f32; TOTAL_STRINGS];
    a[s0] = 0.35;
    let mut audio = [0.0_f32; NUM_BRIDGE_ZONES];
    audio[4] = 1.25;
    let mut b = [0.0_f32; TOTAL_STRINGS];
    let mut z_out = [0.0_f32; NUM_BRIDGE_ZONES];
    let mut dup_out = [1.0_f32; NUM_BRIDGE_ZONES];

    let (dl, dr) =
        bridge.step_bridge_sample_with_duplex_zones(&a, &audio, &mut b, &mut z_out, &mut dup_out);
    assert_eq!((dl, dr), (0.0, 0.0));
    assert_eq!(dup_out, [0.0_f32; NUM_BRIDGE_ZONES]);
    assert!(bridge.is_quiescent());
    assert!(bridge.total_energy_joules().abs() < 1.0e-16);
    assert_eq!(b[s0], -a[s0]);
    assert_eq!(b[s0 + 1], 0.0, "no inter-string coupling in fixed-end mode");
    assert!(z_out.iter().any(|&v| v > 0.0));

    // Non-finite inputs must remain sanitized without panic.
    a[s0 + 1] = f32::NAN;
    audio[4] = f32::INFINITY;
    let (dl2, dr2) =
        bridge.step_bridge_sample_with_duplex_zones(&a, &audio, &mut b, &mut z_out, &mut dup_out);
    assert_eq!((dl2, dr2), (0.0, 0.0));
    assert!(b.iter().all(|v| v.is_finite()));
    assert!(z_out.iter().all(|v| v.is_finite()));

    bridge.set_fixed_end(false);
    assert!(!bridge.is_fixed_end());
}
