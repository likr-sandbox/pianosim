use core::f64::consts::TAU;

use piano_spec::{BodyPreset, PresetReducedPole, CHOIR_LAYOUTS, TOTAL_STRINGS};

use crate::bridge_presets::DEFAULT_Y_INF;
use crate::tests::wdf_absorbed_power_w;
use crate::{AdmittanceSsm, BridgeBank, NUM_BRIDGE_ZONES, NUM_REDUCED_MODES};

const SAMPLE_RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

fn default_preset_poles(ssm: &AdmittanceSsm) -> [PresetReducedPole; NUM_REDUCED_MODES] {
    core::array::from_fn(|m| {
        let p = ssm.poles()[m];
        PresetReducedPole {
            freq_hz: p.freq_hz,
            zeta: p.zeta,
            modal_mass_kg: p.modal_mass_kg,
            zone_gains: p.zone_gains,
        }
    })
}

pub(super) fn custom_upright_like_preset() -> BodyPreset {
    let y_inf = [
        0.000_21, 0.000_19, 0.000_16, 0.000_14, 0.000_26, 0.000_18, 0.000_24, 0.000_29,
    ];
    let freqs = [
        72.0_f32, 118.0, 175.0, 256.0, 342.0, 468.0, 610.0, 815.0, 1_040.0, 1_290.0,
    ];
    let zetas = [
        0.048_f32, 0.036, 0.028, 0.024, 0.021, 0.018, 0.022, 0.020, 0.019, 0.017,
    ];
    let masses = [2.8_f32, 2.0, 1.1, 0.72, 0.54, 0.44, 0.85, 1.75, 2.3, 2.9];
    let reduced_poles = core::array::from_fn(|m| PresetReducedPole {
        freq_hz: freqs[m],
        zeta: zetas[m],
        modal_mass_kg: masses[m],
        zone_gains: core::array::from_fn(|z| {
            0.25 * (((m + 1) * (z + 1)) as f32 * 0.65).sin() + 0.08
        }),
    });
    BodyPreset {
        y_inf,
        reduced_poles,
        ..BodyPreset::default()
    }
}

fn assert_mat8_close(
    a: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    b: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    tol: f64,
    label: &str,
) {
    for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        for (j, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
            let err = (va - vb).abs();
            assert!(
                err <= tol,
                "{label} ({i}, {j}): lhs={va}, rhs={vb}, err={err}"
            );
        }
    }
}

pub(super) fn assert_ssm_junction_consistent(ssm: &AdmittanceSsm, tol: f64) {
    let (a_br, g_tot, solve) = (ssm.a_br(), ssm.g_total(), ssm.solve_8x8());
    let mut sys = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    for (i, srow) in sys.iter_mut().enumerate() {
        srow[i] = 1.0;
        for (j, cell) in srow.iter_mut().enumerate() {
            for (l, grow) in g_tot.iter().enumerate() {
                *cell += a_br[i][l] * grow[j];
            }
        }
    }
    let (mut prod, mut ident, mut expected_abr) =
        ([[0.0_f64; 8]; 8], [[0.0_f64; 8]; 8], [[0.0_f64; 8]; 8]);
    for i in 0..NUM_BRIDGE_ZONES {
        ident[i][i] = 1.0;
        for j in 0..NUM_BRIDGE_ZONES {
            for k in 0..NUM_BRIDGE_ZONES {
                prod[i][j] += sys[i][k] * solve[k][j];
                expected_abr[i][j] += solve[i][k] * a_br[k][j];
            }
        }
    }
    assert_mat8_close(&prod, &ident, tol, "(I + A_br * G_total) * solve_8x8");
    assert_mat8_close(ssm.solve_abr(), &expected_abr, tol, "solve_abr");

    for (m, (sp_row, pole)) in ssm.solve_phi().iter().zip(ssm.poles().iter()).enumerate() {
        for (i, (&got, inv_row)) in sp_row.iter().zip(solve.iter()).enumerate() {
            let expected: f64 = inv_row
                .iter()
                .zip(pole.zone_gains.iter())
                .map(|(&sij, &g)| sij * f64::from(g))
                .sum();
            assert!((got - expected).abs() <= tol, "solve_phi ({m}, {i})");
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
fn from_reduced_model_concert_grand_matches_new_bit_for_bit() {
    for sr in SAMPLE_RATES {
        let ssm_ref = AdmittanceSsm::new(sr);
        let preset_poles = default_preset_poles(&ssm_ref);
        let ssm_pbdy = AdmittanceSsm::from_reduced_model(sr, &DEFAULT_Y_INF, &preset_poles);

        for z in 0..NUM_BRIDGE_ZONES {
            assert_eq!(ssm_pbdy.y_inf()[z].to_bits(), ssm_ref.y_inf()[z].to_bits());
        }
        for (a, b) in ssm_pbdy.poles().iter().zip(ssm_ref.poles().iter()) {
            assert_eq!(a.freq_hz.to_bits(), b.freq_hz.to_bits());
            assert_eq!(a.zeta.to_bits(), b.zeta.to_bits());
            assert_eq!(a.q_factor.to_bits(), b.q_factor.to_bits());
            assert_eq!(a.modal_mass_kg.to_bits(), b.modal_mass_kg.to_bits());
            for (&ga, &gb) in a.zone_gains.iter().zip(b.zone_gains.iter()) {
                assert_eq!(ga.to_bits(), gb.to_bits());
            }
        }
        for m in 0..NUM_REDUCED_MODES {
            for z in 0..NUM_BRIDGE_ZONES {
                assert_eq!(
                    ssm_pbdy.solve_phi()[m][z].to_bits(),
                    ssm_ref.solve_phi()[m][z].to_bits()
                );
            }
        }
        for i in 0..NUM_BRIDGE_ZONES {
            for j in 0..NUM_BRIDGE_ZONES {
                assert_eq!(
                    ssm_pbdy.a_br()[i][j].to_bits(),
                    ssm_ref.a_br()[i][j].to_bits()
                );
                assert_eq!(
                    ssm_pbdy.g_total()[i][j].to_bits(),
                    ssm_ref.g_total()[i][j].to_bits()
                );
                assert_eq!(
                    ssm_pbdy.solve_8x8()[i][j].to_bits(),
                    ssm_ref.solve_8x8()[i][j].to_bits()
                );
                assert_eq!(
                    ssm_pbdy.solve_abr()[i][j].to_bits(),
                    ssm_ref.solve_abr()[i][j].to_bits()
                );
            }
        }

        let mut bridge_ref = BridgeBank::new(sr);
        let mut bridge_sync = BridgeBank::new(sr);
        let mut a = [0.0_f32; TOTAL_STRINGS];
        a[120] = 0.25;
        let (mut b_tmp, mut z_tmp) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
        let _ =
            bridge_sync.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b_tmp, &mut z_tmp);
        assert!(!bridge_sync.is_quiescent());

        let preset = BodyPreset {
            y_inf: DEFAULT_Y_INF,
            reduced_poles: preset_poles,
            ..BodyPreset::default()
        };
        bridge_sync.set_body_preset(&preset);
        assert!(bridge_sync.is_quiescent());
        for i in 0..NUM_BRIDGE_ZONES {
            for j in 0..NUM_BRIDGE_ZONES {
                assert_eq!(
                    bridge_sync.ssm().solve_8x8()[i][j].to_bits(),
                    bridge_ref.ssm().solve_8x8()[i][j].to_bits()
                );
                assert_eq!(
                    bridge_sync.ssm().solve_abr()[i][j].to_bits(),
                    bridge_ref.ssm().solve_abr()[i][j].to_bits()
                );
            }
        }

        let (mut b_ref, mut b_sync) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; TOTAL_STRINGS]);
        let (mut z_ref, mut z_sync) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
        let dup_ref =
            bridge_ref.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b_ref, &mut z_ref);
        let dup_sync =
            bridge_sync.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b_sync, &mut z_sync);
        assert_eq!(dup_ref, dup_sync);
        assert_eq!(b_ref, b_sync);
        assert_eq!(z_ref, z_sync);
    }
}

#[test]
fn set_reduced_model_and_body_preset_sync_scattering_and_matrices() {
    let mut bridge = BridgeBank::new(48_000.0);
    let default_bridge = BridgeBank::new(48_000.0);
    let mut a = [0.0_f32; TOTAL_STRINGS];
    a[90] = 0.3;
    let (mut b_init, mut z_init) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
    let _ = bridge.step_bridge_sample(&a, &[1.0; NUM_BRIDGE_ZONES], &mut b_init, &mut z_init);
    assert!(!bridge.is_quiescent());
    assert!(bridge.dissipated_energy_joules() > 0.0);

    let custom = custom_upright_like_preset();
    bridge.set_body_preset(&custom);

    assert!(bridge.is_quiescent());
    assert_eq!(bridge.ssm_energy_joules(), 0.0);
    assert_eq!(bridge.duplex_energy_joules(), 0.0);
    assert!(bridge.total_energy_joules().abs() < 1.0e-16);
    assert_eq!(bridge.dissipated_energy_joules(), 0.0);
    assert_eq!(bridge.ssm().y_inf(), &custom.y_inf);
    for (got, want) in bridge
        .reduced_poles()
        .iter()
        .zip(custom.reduced_poles.iter())
    {
        assert_eq!(got.freq_hz, want.freq_hz);
        assert_eq!(got.zeta, want.zeta);
        assert_eq!(got.modal_mass_kg, want.modal_mass_kg);
        assert_eq!(got.zone_gains, want.zone_gains);
    }

    assert_ssm_junction_consistent(bridge.ssm(), 1.0e-12);
    assert_ne!(bridge.ssm().a_br(), default_bridge.ssm().a_br());
    assert_ne!(bridge.ssm().solve_abr(), default_bridge.ssm().solve_abr());

    let a_br = bridge.ssm().a_br();
    let mut l_chol = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    for i in 0..NUM_BRIDGE_ZONES {
        for j in 0..=i {
            let mut sum = a_br[i][j];
            for (&lik, &ljk) in l_chol[i][..j].iter().zip(&l_chol[j][..j]) {
                sum -= lik * ljk;
            }
            if i == j {
                assert!(sum > 0.0, "custom A_br pivot {i} not SPD: {sum}");
                l_chol[i][j] = sum.sqrt();
            } else {
                l_chol[i][j] = sum / l_chol[j][j];
            }
        }
    }

    let mut def_cmp = BridgeBank::new(48_000.0);
    let (mut b_custom, mut b_default) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; TOTAL_STRINGS]);
    let (mut z_custom, mut z_default) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
    let _ = bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b_custom, &mut z_custom);
    let _ =
        def_cmp.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b_default, &mut z_default);
    assert_ne!(b_custom, b_default);
    assert_ne!(z_custom, z_default);

    let mut ssm = AdmittanceSsm::new(48_000.0);
    ssm.set_port_resistances(
        bridge.port_resistances(),
        bridge.zone_weights(),
        bridge.duplex().port_resistances(),
    );
    ssm.set_reduced_model(&custom.y_inf, &custom.reduced_poles);
    assert_ssm_junction_consistent(&ssm, 1.0e-12);
    assert_mat8_close(
        ssm.solve_abr(),
        bridge.ssm().solve_abr(),
        1.0e-15,
        "ssm vs bridge solve_abr",
    );
}

#[test]
fn custom_reduced_model_maintains_wdf_passivity_and_energy_conservation() {
    let custom = custom_upright_like_preset();

    for sr in SAMPLE_RATES {
        let ts = 1.0 / f64::from(sr);
        let mut bridge = BridgeBank::new(sr);
        bridge.set_body_preset(&custom);
        let r0 = *bridge.port_resistances();
        let (mut e_in_wdf, mut max_e) = (0.0_f64, 0.0_f64);
        let n_steps = (0.05 * f64::from(sr)) as usize;
        for n in 0..n_steps {
            let t = (n as f64) * ts;
            let mut a_waves = [0.0_f32; TOTAL_STRINGS];
            for k in (4..84).step_by(8) {
                let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
                a_waves[s] = (0.2 * (TAU * (95.0 + 30.0 * (k as f64)) * t).sin()) as f32;
            }
            let (mut b_waves, mut zone_out) =
                ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
            let _ = bridge.step_bridge_sample(
                &a_waves,
                &[5.0; NUM_BRIDGE_ZONES],
                &mut b_waves,
                &mut zone_out,
            );
            e_in_wdf += ts * wdf_absorbed_power_w(&r0, &a_waves, &b_waves);
            let cur_e = bridge.total_energy_joules();
            max_e = max_e.max(cur_e);
            let eps = 1.0e-8 * (max_e.max(e_in_wdf.abs()) + 1.0e-9);
            assert!(
                cur_e <= e_in_wdf + eps,
                "WDF passivity violation at sr={sr}, step={n}"
            );
        }
        assert!(bridge.dissipated_energy_joules() > 0.0);

        let mut coupled = BridgeBank::new(sr);
        coupled.set_duplex_gain(0.0);
        coupled.set_reduced_model(&custom.y_inf, &custom.reduced_poles);
        let r0_c = *coupled.port_resistances();
        let mut lines = vec![[0.0_f32; 32]; TOTAL_STRINGS];
        for (idx, &k) in [20_usize, 48, 68].iter().enumerate() {
            let s_idx = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
            let amp = [0.25_f32, -0.20, 0.15][idx];
            for (d, slot) in lines[s_idx].iter_mut().enumerate() {
                *slot = amp * (TAU * (d as f64) / 32.0).sin() as f32;
            }
        }
        let e0_strings = delay_energy_j(ts, &r0_c, &lines);
        let mut e_in_loop = 0.0_f64;
        for n in 0..1_200 {
            let slot = n % 32;
            let a_waves: [f32; TOTAL_STRINGS] = core::array::from_fn(|s| lines[s][slot]);
            let (mut b_waves, mut zone_out) =
                ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
            let _ = coupled.step_bridge_sample(
                &a_waves,
                &[0.0; NUM_BRIDGE_ZONES],
                &mut b_waves,
                &mut zone_out,
            );
            e_in_loop += ts * wdf_absorbed_power_w(&r0_c, &a_waves, &b_waves);
            for (s, line) in lines.iter_mut().enumerate() {
                line[slot] = b_waves[s];
            }
        }
        let e_strings_end = delay_energy_j(ts, &r0_c, &lines);
        let (e_stored, e_diss) = (
            coupled.total_energy_joules(),
            coupled.dissipated_energy_joules(),
        );
        assert!(e_diss > 0.0);
        assert!(((e_stored + e_diss) - e_in_loop).abs() / e0_strings < 1.0e-5);
        assert!(((e_strings_end + e_stored + e_diss) - e0_strings).abs() / e0_strings < 1.0e-5);
    }
}
