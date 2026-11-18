use core::f64::consts::TAU;

use piano_spec::{PresetReducedPole, CHOIR_LAYOUTS, TOTAL_STRINGS};

use crate::tests::wdf_absorbed_power_w;
use crate::tests_pbdy::{assert_ssm_junction_consistent, custom_upright_like_preset};
use crate::{AdmittanceSsm, BridgeBank, NUM_BRIDGE_ZONES, NUM_REDUCED_MODES};

#[test]
fn invalid_reduced_model_inputs_are_sanitized_to_passive_bounds() {
    let bad_y_inf = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.25,
        -1.0e-12,
        100.0,
        0.0002,
    ];
    let mut bad_poles = [PresetReducedPole::default(); NUM_REDUCED_MODES];
    bad_poles[0] = PresetReducedPole {
        freq_hz: f32::NAN,
        zeta: -0.5,
        modal_mass_kg: 0.0,
        zone_gains: [
            f32::NAN,
            f32::INFINITY,
            -f32::INFINITY,
            -2.0,
            0.5,
            0.0,
            1e6,
            -1e6,
        ],
    };
    bad_poles[1] = PresetReducedPole {
        freq_hz: -100.0,
        zeta: f32::INFINITY,
        modal_mass_kg: f32::NEG_INFINITY,
        zone_gains: [0.3; NUM_BRIDGE_ZONES],
    };

    let ssm = AdmittanceSsm::from_reduced_model(48_000.0, &bad_y_inf, &bad_poles);
    assert!(ssm.y_inf().iter().all(|&y| y.is_finite() && y > 0.0));
    for pole in ssm.poles() {
        assert!(pole.freq_hz.is_finite() && pole.freq_hz > 0.0);
        assert!(pole.zeta.is_finite() && pole.zeta > 0.0);
        assert!(pole.q_factor.is_finite() && pole.q_factor > 0.0);
        assert!(pole.modal_mass_kg.is_finite() && pole.modal_mass_kg > 0.0);
        assert!(pole.zone_gains.iter().all(|g| g.is_finite()));
    }

    let mut bridge = BridgeBank::new(48_000.0);
    bridge.set_reduced_model(&bad_y_inf, &bad_poles);
    assert_ssm_junction_consistent(bridge.ssm(), 1.0e-10);

    let e0_bad = bridge.total_energy_joules();
    let (r0, ts) = (*bridge.port_resistances(), 1.0 / 48_000.0_f64);
    let (mut e_in, mut max_e) = (0.0_f64, 0.0_f64);
    for n in 0..1_200 {
        let t = (n as f64) * ts;
        let mut a = [0.0_f32; TOTAL_STRINGS];
        for k in (5..85).step_by(9) {
            let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
            a[s] = (0.2 * (TAU * (80.0 + 35.0 * (k as f64)) * t).sin()) as f32;
        }
        let (mut b, mut z) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
        let (l, r) = bridge.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z);
        assert!(l.is_finite() && r.is_finite());
        assert!(b.iter().all(|v| v.is_finite()) && z.iter().all(|v| v.is_finite()));
        e_in += ts * wdf_absorbed_power_w(&r0, &a, &b);
        let de = bridge.total_energy_joules() - e0_bad;
        max_e = max_e.max(de);
        let eps = 1.0e-8 * (max_e.max(e_in.abs()) + 1.0e-9);
        assert!(
            de <= e_in + eps,
            "step={n}, de={de}, e_in={e_in}, max_e={max_e}"
        );
    }
    assert!(bridge.dissipated_energy_joules() >= 0.0);
}

#[test]
fn custom_reduced_model_flush_subnormals_restores_quiescence() {
    let custom = custom_upright_like_preset();
    let mut bridge = BridgeBank::new(48_000.0);
    bridge.set_body_preset(&custom);
    assert!(bridge.is_quiescent());

    let mut a = [0.0_f32; TOTAL_STRINGS];
    a[140] = 1.0e-14;
    let (mut b, mut z) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
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
    assert!(bridge.is_quiescent());
    assert_eq!(bridge.ssm_energy_joules(), 0.0);
    assert_eq!(bridge.duplex_energy_joules(), 0.0);
}

#[test]
fn set_reduced_model_smooth_projects_active_zones_and_keeps_quiescence() {
    let custom = custom_upright_like_preset();
    let mut idle = BridgeBank::new(48_000.0);
    idle.set_reduced_model_smooth(&custom.y_inf, &custom.reduced_poles);
    assert!(idle.is_quiescent());
    assert_eq!(idle.ssm_energy_joules(), 0.0);

    let (mut hard, mut smooth) = (BridgeBank::new(48_000.0), BridgeBank::new(48_000.0));
    let (mut b, mut z_prev) = ([0.0_f32; TOTAL_STRINGS], [0.0_f32; NUM_BRIDGE_ZONES]);
    let mut a = [0.0_f32; TOTAL_STRINGS];
    for n in 0..256 {
        let t = (n as f64) / 48_000.0;
        for k in [15_usize, 22, 27, 31, 60] {
            let s = usize::from(CHOIR_LAYOUTS[k].first_string_idx);
            a[s] = (0.18 * (TAU * (65.0 + 18.0 * (k as f64)) * t).sin()) as f32;
        }
        let _ = hard.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_prev);
        let _ = smooth.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_prev);
    }
    hard.set_reduced_model(&custom.y_inf, &custom.reduced_poles);
    smooth.set_reduced_model_smooth(&custom.y_inf, &custom.reduced_poles);
    assert!(smooth.duplex_energy_joules() > 0.0);
    assert_eq!(hard.duplex_energy_joules(), 0.0);

    let (mut z_hard, mut z_smooth) = ([0.0_f32; NUM_BRIDGE_ZONES], [0.0_f32; NUM_BRIDGE_ZONES]);
    let _ = hard.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_hard);
    let _ = smooth.step_bridge_sample(&a, &[0.0; NUM_BRIDGE_ZONES], &mut b, &mut z_smooth);
    let jump_hard: f32 = z_hard.iter().zip(&z_prev).map(|(x, y)| (x - y).abs()).sum();
    let jump_smooth: f32 = z_smooth
        .iter()
        .zip(&z_prev)
        .map(|(x, y)| (x - y).abs())
        .sum();
    assert!(
        jump_smooth < jump_hard * 0.15,
        "expected smooth body projection jump ({jump_smooth}) << hard reset jump ({jump_hard})"
    );
}
