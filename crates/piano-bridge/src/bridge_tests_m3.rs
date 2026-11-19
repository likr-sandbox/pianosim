use core::f64::consts::{FRAC_PI_2, SQRT_2, TAU};

use piano_spec::{PerfProfile, VoicingSpec, CHOIR_LAYOUTS, TOTAL_STRINGS};

use crate::bridge_presets::MAX_Y_INF;
use crate::tests::wdf_absorbed_power_w;
use crate::tests_pbdy::custom_upright_like_preset;
use crate::{AdmittanceSsm, BridgeBank, NUM_BRIDGE_ZONES, NUM_REDUCED_MODES};

const SAMPLE_RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

fn assert_cholesky_spd(a_br: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES]) {
    let mut l_chol = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
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
}

fn delay_energy_j(ts: f64, r0: &[f32; TOTAL_STRINGS], lines: &[[f32; 32]]) -> f64 {
    lines
        .iter()
        .enumerate()
        .map(|(s, line)| {
            let sq: f64 = line.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            0.25 * ts * f64::from(r0[s]) * sq
        })
        .sum()
}

#[test]
fn full_profile_matches_new_with_voicing_bit_for_bit() {
    let voicing = VoicingSpec::default();
    let full = PerfProfile::full();
    for sr in SAMPLE_RATES {
        let mut b_ref = BridgeBank::new_with_voicing(sr, &voicing);
        let mut b_full = BridgeBank::new_with_profile(sr, &voicing, &full);

        assert_eq!(b_full.active_poles(), 10);
        assert!(b_full.is_hertz_nonlinear());
        assert_eq!(b_full.ssm().a_br(), b_ref.ssm().a_br());
        assert_eq!(b_full.ssm().solve_8x8(), b_ref.ssm().solve_8x8());
        assert_eq!(b_full.ssm().solve_abr(), b_ref.ssm().solve_abr());
        assert_eq!(b_full.ssm().solve_phi(), b_ref.ssm().solve_phi());

        for n in 0..512 {
            let t = (n as f64) / f64::from(sr);
            let mut a = [0.0_f32; TOTAL_STRINGS];
            for &k in &[12_usize, 36, 48, 66] {
                let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
                a[s] = (0.22 * (TAU * (75.0 + 25.0 * k as f64) * t).sin()) as f32;
            }
            let audio = [0.4_f32; NUM_BRIDGE_ZONES];
            let (mut br, mut bf) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; TOTAL_STRINGS]);
            let (mut zr, mut zf) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
            let (mut dr, mut df) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
            let lr =
                b_ref.step_bridge_sample_with_duplex_zones(&a, &audio, &mut br, &mut zr, &mut dr);
            let lf =
                b_full.step_bridge_sample_with_duplex_zones(&a, &audio, &mut bf, &mut zf, &mut df);

            assert_eq!(lr.0.to_bits(), lf.0.to_bits());
            assert_eq!(lr.1.to_bits(), lf.1.to_bits());
            for s in 0..TOTAL_STRINGS {
                assert_eq!(br[s].to_bits(), bf[s].to_bits(), "b_waves mismatch at {s}");
            }
            for z in 0..NUM_BRIDGE_ZONES {
                assert_eq!(zr[z].to_bits(), zf[z].to_bits());
                assert_eq!(dr[z].to_bits(), df[z].to_bits());
            }
        }
        assert_eq!(
            b_ref.total_energy_joules().to_bits(),
            b_full.total_energy_joules().to_bits()
        );
        assert_eq!(
            b_ref.dissipated_energy_joules().to_bits(),
            b_full.dissipated_energy_joules().to_bits()
        );
    }
}

#[test]
fn hertz_disabled_skips_nonlinear_sav_preserves_linearity_and_passivity() {
    let mut prof = PerfProfile::full();
    prof.hertz_nonlinear = false;
    let voicing = VoicingSpec::default();
    let mut b1 = BridgeBank::new_with_profile(48_000.0, &voicing, &prof);
    let mut b2 = BridgeBank::new_with_profile(48_000.0, &voicing, &prof);
    assert!(!b1.is_hertz_nonlinear());
    assert!(b1.hertz().is_quiescent());
    assert_eq!(b1.hertz().eval_sav_gradient(), [0.0; NUM_BRIDGE_ZONES]);

    // Verify strict linearity with respect to input wave scaling when Hertz SAV is disabled.
    let s0 = usize::from(CHOIR_LAYOUTS[48].first_string_idx);
    for n in 0..256 {
        let t = (n as f64) / 48_000.0;
        let w = (0.8 * (TAU * 440.0 * t).sin()) as f32;
        let mut a1 = [0.0_f32; TOTAL_STRINGS];
        let mut a2 = [0.0_f32; TOTAL_STRINGS];
        a1[s0] = w;
        a2[s0] = 2.0 * w;
        let (mut bw1, mut bw2) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; TOTAL_STRINGS]);
        let (mut z1, mut z2) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
        let _ = b1.step_bridge_sample(&a1, &[0.0; NUM_BRIDGE_ZONES], &mut bw1, &mut z1);
        let _ = b2.step_bridge_sample(&a2, &[0.0; NUM_BRIDGE_ZONES], &mut bw2, &mut z2);

        assert_eq!(b1.hertz_energy_joules(), 0.0);
        assert_eq!(b1.hertz().potential_energy_joules(), 0.0);
        assert_eq!(b1.hertz().displacements_m(), [0.0; NUM_BRIDGE_ZONES]);
        assert!(b1.hertz().is_quiescent());
        for z in 0..NUM_BRIDGE_ZONES {
            assert!((z2[z] - 2.0 * z1[z]).abs() < 1.0e-6);
        }
    }

    // Disabling Hertz mid-playback resets (x, v_prev, r_sav) and keeps passivity intact.
    let mut active = BridgeBank::new(48_000.0);
    let r0 = *active.port_resistances();
    let ts = 1.0 / 48_000.0;
    let mut e_in = 0.0_f64;
    let mut max_e = 0.0_f64;
    for n in 0..600 {
        if n == 200 {
            active.set_hertz_nonlinear(false);
            assert!(!active.is_hertz_nonlinear());
            assert_eq!(active.hertz_energy_joules(), 0.0);
            assert!(active.hertz().is_quiescent());
        } else if n == 400 {
            active.set_hertz_nonlinear(true);
            assert!(active.is_hertz_nonlinear());
        }
        let t = (n as f64) * ts;
        let mut a = [0.0_f32; TOTAL_STRINGS];
        a[s0] = (0.75 * (TAU * 330.0 * t).sin()) as f32;
        let (mut bw, mut zw) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
        let _ = active.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut bw, &mut zw);
        assert!(bw.iter().all(|v| v.is_finite()) && zw.iter().all(|v| v.is_finite()));
        e_in += ts * wdf_absorbed_power_w(&r0, &a, &bw);
        let cur_e = active.total_energy_joules();
        max_e = max_e.max(cur_e);
        let eps = 1.0e-8 * (max_e.max(e_in.abs()) + 1.0e-9);
        assert!(cur_e <= e_in + eps, "passivity violated at step {n}");
    }
}

#[test]
fn active_poles_fold_conductance_preserve_y_inf_base_and_passivity() {
    let ref_ssm = AdmittanceSsm::new(48_000.0);
    let poles = ref_ssm.poles();

    // Clamping out-of-bounds pole counts into 4..=10.
    assert_eq!(
        AdmittanceSsm::new_with_active_poles(48_000.0, 0).active_poles(),
        4
    );
    assert_eq!(
        AdmittanceSsm::new_with_active_poles(48_000.0, 99).active_poles(),
        10
    );

    for &n_poles in &[10_u8, 6, 4] {
        let ssm = AdmittanceSsm::new_with_active_poles(48_000.0, n_poles);
        assert_eq!(ssm.active_poles(), n_poles);
        assert_eq!(
            ssm.y_inf(),
            ref_ssm.y_inf(),
            "y_inf base must stay unchanged"
        );
        assert_cholesky_spd(ssm.a_br());

        let active = usize::from(n_poles);
        for z in 0..NUM_BRIDGE_ZONES {
            let mut expected = f64::from(ref_ssm.y_inf()[z]);
            for pole in &poles[active..] {
                let denom = f64::from(pole.modal_mass_kg) * TAU * f64::from(pole.freq_hz);
                let g = f64::from(pole.zone_gains[z]);
                expected =
                    (expected + FRAC_PI_2 * SQRT_2 * g * g / denom).min(f64::from(MAX_Y_INF));
            }
            assert!((ssm.y_inf_eff_f64()[z] - expected).abs() < 1.0e-15);
        }
        for m in active..NUM_REDUCED_MODES {
            assert_eq!(ssm.solve_phi()[m], [0.0; NUM_BRIDGE_ZONES]);
        }
    }

    // Exact closed-loop WDF energy conservation across all (sr, hertz, poles) combinations.
    for sr in SAMPLE_RATES {
        let ts = 1.0 / f64::from(sr);
        for &hertz in &[true, false] {
            for &n_poles in &[10_u8, 6, 4] {
                let perf = PerfProfile {
                    hertz_nonlinear: hertz,
                    active_bridge_poles: n_poles,
                    ..PerfProfile::full()
                };
                let mut bridge = BridgeBank::new_with_profile(sr, &VoicingSpec::default(), &perf);
                bridge.set_duplex_gain(0.0);
                let r0 = *bridge.port_resistances();
                let mut lines = vec![[0.0_f32; 32]; TOTAL_STRINGS];
                for (idx, &k) in [18_usize, 44, 67].iter().enumerate() {
                    let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
                    let amp = [0.24_f32, -0.19, 0.16][idx];
                    for (d, slot) in lines[s].iter_mut().enumerate() {
                        *slot = amp * (TAU * (d as f64) / 32.0).sin() as f32;
                    }
                }
                let e0_strings = delay_energy_j(ts, &r0, &lines);
                let mut e_in_loop = 0.0_f64;
                for n in 0..1_200 {
                    let slot = n % 32;
                    let a_waves: [f32; TOTAL_STRINGS] = core::array::from_fn(|s| lines[s][slot]);
                    let (mut b_waves, mut z_out) =
                        ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
                    let _ = bridge.step_bridge_sample(
                        &a_waves,
                        &[0.0; NUM_BRIDGE_ZONES],
                        &mut b_waves,
                        &mut z_out,
                    );
                    assert!(b_waves.iter().all(|v| v.is_finite()));
                    assert!(z_out.iter().all(|v| v.is_finite()));
                    e_in_loop += ts * wdf_absorbed_power_w(&r0, &a_waves, &b_waves);
                    for (s, line) in lines.iter_mut().enumerate() {
                        line[slot] = b_waves[s];
                    }
                }
                let e_strings_end = delay_energy_j(ts, &r0, &lines);
                let (e_stored, e_diss) = (
                    bridge.total_energy_joules(),
                    bridge.dissipated_energy_joules(),
                );
                assert!(e_diss > 0.0);
                assert!(
                    ((e_stored + e_diss) - e_in_loop).abs() / e0_strings < 1.0e-5,
                    "bridge conservation error sr={sr}, hertz={hertz}, poles={n_poles}"
                );
                assert!(
                    ((e_strings_end + e_stored + e_diss) - e0_strings).abs() / e0_strings < 1.0e-5,
                    "coupled conservation error sr={sr}, hertz={hertz}, poles={n_poles}"
                );
            }
        }
    }
}

#[test]
fn reduced_model_smooth_and_runtime_hot_swap_preserve_continuity_and_restore_full() {
    let custom = custom_upright_like_preset();
    let voicing = VoicingSpec::default();

    for &(n_poles, max_jump_ratio) in &[(10_u8, 0.15_f32), (6, 0.50), (4, 0.85)] {
        let perf = PerfProfile {
            hertz_nonlinear: false,
            active_bridge_poles: n_poles,
            ..PerfProfile::full()
        };
        let mut idle = BridgeBank::new_with_profile(48_000.0, &voicing, &perf);
        idle.set_reduced_model_smooth(&custom.y_inf, &custom.reduced_poles);
        assert!(idle.is_quiescent());
        assert_eq!(idle.active_poles(), n_poles);

        let mut hard = BridgeBank::new_with_profile(48_000.0, &voicing, &perf);
        let mut smooth = BridgeBank::new_with_profile(48_000.0, &voicing, &perf);
        let (mut b, mut z_prev) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
        let mut a = [0.0_f32; TOTAL_STRINGS];
        for n in 0..256 {
            let t = (n as f64) / 48_000.0;
            for k in [15_usize, 27, 48, 60] {
                let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
                a[s] = (0.18 * (TAU * (70.0 + 18.0 * (k as f64)) * t).sin()) as f32;
            }
            let _ = hard.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_prev);
            let _ = smooth.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_prev);
        }
        hard.set_reduced_model(&custom.y_inf, &custom.reduced_poles);
        smooth.set_reduced_model_smooth(&custom.y_inf, &custom.reduced_poles);
        assert_eq!(smooth.active_poles(), n_poles);
        assert!(smooth.ssm().q[usize::from(n_poles)..]
            .iter()
            .all(|&v| v == 0.0));
        assert!(smooth.ssm().q_dot[usize::from(n_poles)..]
            .iter()
            .all(|&v| v == 0.0));

        let (mut z_hard, mut z_smooth) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
        let _ = hard.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_hard);
        let _ = smooth.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_smooth);
        let jump_hard: f32 = z_hard.iter().zip(&z_prev).map(|(x, y)| (x - y).abs()).sum();
        let jump_smooth: f32 = z_smooth
            .iter()
            .zip(&z_prev)
            .map(|(x, y)| (x - y).abs())
            .sum();
        assert!(smooth.ssm_energy_joules() > 0.0);
        assert!(
            jump_smooth < jump_hard * max_jump_ratio,
            "poles={n_poles}: jump_smooth={jump_smooth}, jump_hard={jump_hard}"
        );
    }

    // Runtime hot-swap 10 -> 6 -> 4 -> 10 via set_perf_profile_runtime restores exact 10-pole matrices.
    let ref_full = BridgeBank::new(48_000.0);
    let mut bridge = BridgeBank::new(48_000.0);
    let mut a = [0.0_f32; TOTAL_STRINGS];
    let s0 = usize::from(CHOIR_LAYOUTS[30].first_string_idx);
    let (mut b, mut z) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
    for n in 0..120 {
        a[s0] = (0.25 * (TAU * 220.0 * (n as f64) / 48_000.0).sin()) as f32;
        let _ = bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
    }

    for profile in [
        PerfProfile::balanced(),
        PerfProfile::realtime(),
        PerfProfile::full(),
    ] {
        let diss_before = bridge.dissipated_energy_joules();
        bridge.set_perf_profile_runtime(&profile);
        assert_eq!(bridge.active_poles(), profile.active_bridge_poles);
        assert_eq!(bridge.is_hertz_nonlinear(), profile.hertz_nonlinear);
        assert!(bridge.dissipated_energy_joules() >= diss_before);
        let _ = bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
        assert!(b.iter().all(|v| v.is_finite()) && z.iter().all(|v| v.is_finite()));
    }

    assert_eq!(bridge.ssm().a_br(), ref_full.ssm().a_br());
    assert_eq!(bridge.ssm().solve_8x8(), ref_full.ssm().solve_8x8());
    assert_eq!(bridge.ssm().solve_abr(), ref_full.ssm().solve_abr());
    assert_eq!(bridge.ssm().solve_phi(), ref_full.ssm().solve_phi());
    assert_eq!(bridge.ssm().y_inf_eff_f64(), ref_full.ssm().y_inf_eff_f64());

    // Disable nonlinear Hertz (which has zero linear stiffness near x=0) and verify SSM+Duplex decay to quiescence.
    bridge.set_hertz_nonlinear(false);
    let zeros = [0.0_f32; TOTAL_STRINGS];
    for _ in 0..(5 * 48_000) {
        let _ = bridge.step_bridge_sample(&zeros, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
        bridge.flush_subnormals();
        if bridge.is_quiescent() {
            break;
        }
    }
    assert!(
        bridge.is_quiescent(),
        "ssm={}, hertz={}, duplex={}",
        bridge.ssm().is_quiescent(),
        bridge.hertz().is_quiescent(),
        bridge.duplex().is_quiescent()
    );
}

#[test]
fn fixed_end_mode_is_invariant_across_m3_profiles() {
    let voicing = VoicingSpec::default();
    let s0 = usize::from(CHOIR_LAYOUTS[48].first_string_idx);
    let mut a = [0.0_f32; TOTAL_STRINGS];
    a[s0] = 0.35;
    let mut audio = [0.0_f32; NUM_BRIDGE_ZONES];
    audio[4] = 1.25;

    let mut b_ref = [0.0_f32; TOTAL_STRINGS];
    let mut z_ref = [0.0_f32; NUM_BRIDGE_ZONES];
    let mut dup_ref = [1.0_f32; NUM_BRIDGE_ZONES];
    let mut fixed_ref = BridgeBank::new_with_profile(48_000.0, &voicing, &PerfProfile::full());
    fixed_ref.set_fixed_end(true);
    let dl_ref = fixed_ref.step_bridge_sample_with_duplex_zones(
        &a,
        &audio,
        &mut b_ref,
        &mut z_ref,
        &mut dup_ref,
    );
    assert_eq!(dl_ref, (0.0, 0.0));

    for prof in [PerfProfile::balanced(), PerfProfile::realtime()] {
        let mut bridge = BridgeBank::new_with_profile(48_000.0, &voicing, &prof);
        bridge.set_fixed_end(true);
        let (mut b, mut z, mut dup) = (
            [0.0_f32; TOTAL_STRINGS],
            [0.0_f32; NUM_BRIDGE_ZONES],
            [1.0_f32; NUM_BRIDGE_ZONES],
        );
        let dl = bridge.step_bridge_sample_with_duplex_zones(&a, &audio, &mut b, &mut z, &mut dup);
        assert_eq!(dl, (0.0, 0.0));
        assert_eq!(dup, [0.0_f32; NUM_BRIDGE_ZONES]);
        assert!(bridge.is_quiescent());
        assert_eq!(b, b_ref);
        assert_eq!(z, z_ref);
    }
}
