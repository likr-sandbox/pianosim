//! M3 (`F9`) Stage 1 parameterized speedups & `P1-5` / `P0-2` verification tests.

use alloc::vec::Vec;

use piano_simd::{AdaaPowerLut, AdaaState, PolyPowerApprox};
use piano_spec::{build_scale_design, PedalKind, PerfProfile, VoicingSpec, DAMPED_KEYS, NUM_KEYS};

use crate::agraffe_delay::{hammer_beta_for_key, AgraffeDelayLine};
use crate::borin_k::{BorinKLut, BORIN_LUT_SIZE, BORIN_P_MAX};
use crate::decimation_fir::DecimationFir;
use crate::felt_stulov::{FeltSharedCoeffs, KeyFeltSpec};
use crate::k_method::{
    compute_lambda_for_p, solve_coupled_substep, solve_coupled_substep_lerp, CoupledSubstepInput,
};
use crate::k_method_poly::{solve_borin_poly, solve_coupled_substep_poly};
use crate::pedals::{PedalBank, CLEARANCE_OPEN_M};
use crate::shank_timoshenko::ShankParams;
use crate::tests_common::TestModalString;
use crate::ActionBank;

#[test]
fn action_bank_new_matches_new_with_full_profile_bit_for_bit() {
    let voicing = VoicingSpec::default();
    let mut bank_new = ActionBank::new(48_000.0, &voicing);
    let mut bank_full = ActionBank::new_with_profile(48_000.0, &voicing, &PerfProfile::full());
    assert_eq!(bank_full.perf_profile(), PerfProfile::full());
    assert_eq!(bank_full.lut_table_bytes(), 384 * 1024);

    let mut s_new = TestModalString::new(39, 48_000.0, &voicing);
    let mut s_full = TestModalString::new(39, 48_000.0, &voicing);
    bank_new.note_on(39, 51_602);
    bank_full.note_on(39, 51_602);

    let (mut c_new, mut c_full) = ([0.0_f32; 8], [0.0_f32; 8]);
    let (mut p_new, mut p_full) = ([0.0_f32; 8], [0.0_f32; 8]);
    for i in 0..512 {
        let (yn, vn) = s_new.strike_port();
        let (yf, vf) = s_full.strike_port();
        let fn_ = bank_new.step_hammer_sample(39, yn, vn, s_new.r_port_strike);
        let ff_ = bank_full.step_hammer_sample(39, yf, vf, s_full.r_port_strike);
        assert_eq!(fn_.to_bits(), ff_.to_bits(), "sample {i}");
        s_new.step_forces(fn_, 0.0);
        s_full.step_forces(ff_, 0.0);
        if i == 120 {
            bank_new.note_off(39, 40_000);
            bank_full.note_off(39, 40_000);
        }
        if i % 8 == 0 {
            let rn = bank_new.step_key_clearance_tile_with_peel(39, 8, &mut c_new, &mut p_new);
            let rf = bank_full.step_key_clearance_tile_with_peel(39, 8, &mut c_full, &mut p_full);
            assert_eq!(rn.0.to_bits(), rf.0.to_bits());
            assert_eq!(rn.1, rf.1);
        }
    }
}

#[test]
fn action_oversample_scales_fir_agraffe_and_strike_across_8_4_2() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);

    for &sr in &[48_000.0_f64, 96_000.0] {
        let expected_tau_ms =
            1000.0 * hammer_beta_for_key(39) / piano_spec::key_tuned_f0_hz(39, &specs, &voicing);
        for &os in &[8_u8, 4, 2] {
            let fir = DecimationFir::for_oversample(os);
            assert_eq!(fir.factor(), os);
            assert_eq!(fir.taps(), os * 2);

            let line = AgraffeDelayLine::new_oversampled(39, sr, &voicing, &specs, os);
            assert_eq!(line.oversample(), os);
            let tau_ms = 1000.0 * f64::from(line.delay_substeps()) / (f64::from(os) * sr);
            assert!(
                (tau_ms - expected_tau_ms).abs() < 1e-3,
                "sr={sr} os={os}: tau_ms={tau_ms} expected={expected_tau_ms}"
            );

            let sp = ShankParams::for_key_oversampled(39, sr, os);
            assert!((f64::from(sp.dt_sub) - 1.0 / (f64::from(os) * sr)).abs() < 1e-10);
            let fc = FeltSharedCoeffs::new_oversampled(sr, os);
            assert!(fc.a_fast_sub > 0.0 && fc.a_fast_sub < 1.0);
            let kf = KeyFeltSpec::for_key_oversampled(39, sr, &voicing, os);
            assert!(kf.mem_sub_gain > 0.0);
        }
    }

    let run_peak = |perf: &PerfProfile| -> (f32, [f32; 8]) {
        let mut bank = ActionBank::new_with_profile(48_000.0, &voicing, perf);
        let mut s = TestModalString::new(48, 48_000.0, &voicing);
        bank.note_on(48, 51_602);
        let mut peak = 0.0_f32;
        let mut last_subs = [0.0_f32; 8];
        for _ in 0..256 {
            let (y, v) = s.strike_port();
            let f = bank.step_hammer_sample(48, y, v, s.r_port_strike);
            assert!(f.is_finite() && f >= 0.0);
            if f > peak {
                peak = f;
                last_subs = bank.last_hammer_forces(48).1;
            }
            s.step_forces(f, 0.0);
        }
        (peak, last_subs)
    };

    let (peak8, _) = run_peak(&PerfProfile::full());
    let mut p4 = PerfProfile::full();
    p4.action_oversample = 4;
    let (peak4, subs4) = run_peak(&p4);
    let mut p2 = PerfProfile::full();
    p2.action_oversample = 2;
    let (peak2, subs2) = run_peak(&p2);

    assert!(subs4[4..].iter().all(|&x| x == 0.0), "subs4={subs4:?}");
    assert!(subs2[2..].iter().all(|&x| x == 0.0), "subs2={subs2:?}");
    assert!(
        (peak4 - peak8).abs() / peak8 < 0.08,
        "p4={peak4} p8={peak8}"
    );
    assert!(
        (peak2 - peak8).abs() / peak8 < 0.15,
        "p2={peak2} p8={peak8}"
    );

    // Reconfiguring 2 -> 8 grows agraffe buffer safely without panic or truncation.
    let mut bank = ActionBank::new_with_profile(48_000.0, &voicing, &p2);
    bank.set_perf_profile(&PerfProfile::full());
    assert_eq!(bank.perf_profile().action_oversample, 8);
    bank.note_on(0, 58_000);
    let mut s0 = TestModalString::new(0, 48_000.0, &voicing);
    let mut pk0 = 0.0_f32;
    for _ in 0..400 {
        let (y, v) = s0.strike_port();
        let f = bank.step_hammer_sample(0, y, v, s0.r_port_strike);
        pk0 = pk0.max(f);
        s0.step_forces(f, 0.0);
    }
    assert!(pk0 > 5.0);
}

#[test]
fn poly_borin_and_action_bank_polynomial_adaa_match_lut_within_two_percent() {
    for &p in &[2.1_f32, 2.5, 2.8, 3.2, 3.5] {
        let poly = PolyPowerApprox::new(p);
        assert_eq!(solve_borin_poly(&poly, -0.5, 1.0), 0.0);
        assert_eq!(solve_borin_poly(&poly, 0.0, 1.0), 0.0);
        for &alpha in &[1.0e-3_f32, 0.05, 1.0, 12.0, 100.0] {
            for &pu in &[1.0e-4_f32, 0.05, 0.35, 0.9, 1.5, 6.0] {
                let u = solve_borin_poly(&poly, pu, alpha);
                assert!(u.is_finite() && u >= 0.0);
                let res = (u + alpha * poly.eval_f(u) - pu).abs();
                assert!(
                    res < 1.0e-4 * pu.max(1.0),
                    "p={p} alpha={alpha} pu={pu} u={u} res={res}"
                );
            }
        }
    }

    // Full ActionBank strike parity between LUT and Poly across bass, mid, treble & velocities.
    let voicing = VoicingSpec::default();
    let poly_prof = PerfProfile {
        polynomial_adaa: true,
        ..PerfProfile::full()
    };
    let mut bank_lut = ActionBank::new_with_profile(48_000.0, &voicing, &PerfProfile::full());
    let mut bank_poly = ActionBank::new_with_profile(48_000.0, &voicing, &poly_prof);
    assert_eq!(bank_poly.lut_table_bytes(), 0);

    // Runtime switch: poly-only init keeps polynomial_adaa = true (zero-alloc).
    bank_poly.set_perf_profile_runtime(&PerfProfile::full());
    assert!(bank_poly.perf_profile().polynomial_adaa);
    assert_eq!(bank_poly.lut_table_bytes(), 0);

    // Runtime switch on LUT-initialized bank toggles cleanly between LUT and Poly without alloc.
    bank_lut.set_perf_profile_runtime(&poly_prof);
    assert!(bank_lut.perf_profile().polynomial_adaa);
    bank_lut.set_perf_profile_runtime(&PerfProfile::full());
    assert!(!bank_lut.perf_profile().polynomial_adaa);

    for &k in &[0_usize, 39, 48, 72, 87] {
        for &vel in &[12_000_u16, 40_000, 65_535] {
            bank_lut.reset();
            bank_poly.reset();
            let mut sl = TestModalString::new(k, 48_000.0, &voicing);
            let mut sp = TestModalString::new(k, 48_000.0, &voicing);
            bank_lut.note_on(k, vel);
            bank_poly.note_on(k, vel);

            let (mut pk_l, mut pk_p) = (0.0_f32, 0.0_f32);
            let (mut imp_l, mut imp_p) = (0.0_f64, 0.0_f64);
            let (mut sq_l, mut sq_diff) = (0.0_f64, 0.0_f64);
            for _ in 0..512 {
                let (yl, vl) = sl.strike_port();
                let (yp, vp) = sp.strike_port();
                let fl = bank_lut.step_hammer_sample(k, yl, vl, sl.r_port_strike);
                let fp = bank_poly.step_hammer_sample(k, yp, vp, sp.r_port_strike);
                assert!(fp.is_finite() && fp >= 0.0);
                pk_l = pk_l.max(fl);
                pk_p = pk_p.max(fp);
                imp_l += f64::from(fl);
                imp_p += f64::from(fp);
                sq_l += f64::from(fl) * f64::from(fl);
                sq_diff += f64::from(fl - fp) * f64::from(fl - fp);
                sl.step_forces(fl, 0.0);
                sp.step_forces(fp, 0.0);
            }
            let pk_rel = (pk_l - pk_p).abs() / pk_l.max(1.0e-6);
            let imp_rel = (imp_l - imp_p).abs() / imp_l.max(1.0e-12);
            let rms_rel = (sq_diff / sq_l.max(1.0e-12)).sqrt();
            assert!(
                pk_rel < 0.02 && imp_rel < 0.02 && rms_rel < 0.03,
                "k={k} vel={vel}: pk_l={pk_l} pk_p={pk_p} pk_rel={pk_rel} imp_rel={imp_rel} rms_rel={rms_rel}"
            );
        }
    }

    // Live hardness/voicing updates refresh poly_adaa too.
    let hard = VoicingSpec {
        hammer_hardness: 0.9,
        ..VoicingSpec::default()
    };
    bank_poly.update_voicing_in_place(&hard);
    assert!(
        (bank_poly.keys[48].poly_adaa.exponent() - bank_poly.keys[48].felt_spec.exponent_p).abs()
            < 1e-6
    );
}

#[test]
fn shank_flex_disabled_propagates_across_all_keys_and_matches_rigid_strike() {
    let voicing = VoicingSpec::default();
    let rigid_prof = PerfProfile {
        shank_flex_enabled: false,
        ..PerfProfile::full()
    };
    let mut bank_prof = ActionBank::new_with_profile(48_000.0, &voicing, &rigid_prof);
    let mut bank_manual = ActionBank::new(48_000.0, &voicing);
    for k in 0..NUM_KEYS {
        assert!(!bank_prof.keys[k].hammer.shank.flex_enabled);
        bank_manual.set_shank_flex_enabled(k, false);
    }

    let mut s1 = TestModalString::new(48, 48_000.0, &voicing);
    let mut s2 = TestModalString::new(48, 48_000.0, &voicing);
    bank_prof.note_on(48, 58_000);
    bank_manual.note_on(48, 58_000);
    for _ in 0..256 {
        let (y1, v1) = s1.strike_port();
        let (y2, v2) = s2.strike_port();
        let f1 = bank_prof.step_hammer_sample(48, y1, v1, s1.r_port_strike);
        let f2 = bank_manual.step_hammer_sample(48, y2, v2, s2.r_port_strike);
        assert_eq!(f1.to_bits(), f2.to_bits());
        assert_eq!(bank_prof.keys[48].hammer.shank.eta1, 0.0);
        assert_eq!(bank_prof.keys[48].hammer.shank.v1, 0.0);
        s1.step_forces(f1, 0.0);
        s2.step_forces(f2, 0.0);
    }

    bank_manual.set_perf_profile_runtime(&PerfProfile::full());
    assert!(bank_manual.keys.iter().all(|s| s.hammer.shank.flex_enabled));
}

#[test]
fn damper_control_decim_2_and_4_preserve_half_pedal_and_peel_parity() {
    let run_half_pedal_and_peel = |decim: u8| -> (f32, f32, f32) {
        let mut p = PedalBank::new_with_decim(48_000.0, decim);
        assert_eq!(p.control_decim(), decim);
        let mut clears = [0.0_f32; 8];
        let mut peels = [0.0_f32; 8];

        // Half-pedal settle from open.
        p.note_on(39);
        p.set_pedal(PedalKind::Sustain, 32_768);
        p.note_off(39, 32_768);
        for _ in 0..800 {
            let _ = p.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
        }
        let hp_clear = p.damper_clearance_m(39);

        // Dry release -> settle -> Sustain lift peel pulse.
        p.set_pedal(PedalKind::Sustain, 0);
        for _ in 0..800 {
            let _ = p.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
        }
        p.set_pedal(PedalKind::Sustain, u16::MAX);
        let mut trace = Vec::new();
        for _ in 0..300 {
            let _ = p.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
            trace.extend_from_slice(&peels);
        }
        let peel_peak = trace.iter().copied().fold(0.0_f32, f32::max);
        let max_jump = trace
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);
        assert_eq!(trace[0], 0.0);
        assert_eq!(*trace.last().unwrap(), 0.0);
        (hp_clear, peel_peak, max_jump)
    };

    let (hp1, pk1, _) = run_half_pedal_and_peel(1);
    for decim in [2_u8, 4] {
        let (hp_d, pk_d, jump_d) = run_half_pedal_and_peel(decim);
        assert!(
            (hp_d - hp1).abs() < 2.0e-6,
            "decim={decim} hp={hp_d} vs {hp1}"
        );
        assert!(
            (5.0e-4..=2.0e-3).contains(&pk_d) && (pk_d - pk1).abs() / pk1 < 0.08,
            "decim={decim} pk_d={pk_d} vs pk1={pk1}"
        );
        assert!(jump_d < pk_d * 0.1, "decim={decim} jump={jump_d}");
    }
}

#[test]
fn p1_5_free_flight_and_single_table_fast_paths_match_exact() {
    let p = 2.8_f32;
    let adaa = AdaaPowerLut::new(p, 1.0, 1024);
    let borin = BorinKLut::new(p, BORIN_P_MAX, BORIN_LUT_SIZE);
    let poly = PolyPowerApprox::new(p);
    let lambda = compute_lambda_for_p(p, 2.0e-3, 8.0e-9, 750.0);
    let inp = CoupledSubstepInput {
        delta_prev_m: -1.5e-5,
        p_end_m: -0.5e-5,
        k_loop_end: 8.0e-9,
        delta_scale_m: 2.0e-3,
        f_scale_eff_n: 800.0,
        eps_eff: 0.34,
        a_fast_sub: 0.94,
        lambda_end: lambda,
    };

    let mut a1 = AdaaState::new();
    let mut z1 = 0.25_f32;
    let o1 = solve_coupled_substep(&adaa, &borin, &mut a1, &mut z1, inp);
    assert_eq!(o1.force_n, 0.0);
    assert_eq!(o1.u_norm_mid, 0.0);
    assert_eq!(z1.to_bits(), (0.25_f32 * 0.94_f32).to_bits());

    let mut a2 = AdaaState::new();
    let mut z2 = 0.25_f32;
    let o2 = solve_coupled_substep_lerp(&adaa, &adaa, &borin, &borin, 0.5, &mut a2, &mut z2, inp);
    assert_eq!(o2.force_n, 0.0);
    assert_eq!(z2.to_bits(), z1.to_bits());

    let mut a3 = AdaaState::new();
    let mut z3 = 0.25_f32;
    let o3 = solve_coupled_substep_poly(&poly, &mut a3, &mut z3, inp);
    assert_eq!(o3.force_n, 0.0);
    assert_eq!(z3.to_bits(), z1.to_bits());

    // Verify P1-5 single-table dispatch in ActionBank::step_hammer_sample near p_grid endpoints.
    let voicing = VoicingSpec::default();
    let mut bank_lo = ActionBank::new(48_000.0, &voicing);
    bank_lo.keys[0].felt_spec.exponent_p = crate::p_grid::P_GRID_MIN + 1.0e-7;
    bank_lo.note_on(0, 48_000);
    let mut bank_exact = ActionBank::new(48_000.0, &voicing);
    bank_exact.keys[0].felt_spec.exponent_p = crate::p_grid::P_GRID_MIN;
    bank_exact.note_on(0, 48_000);
    for _ in 0..32 {
        let f_lo = bank_lo.step_hammer_sample(0, 0.0, 0.0, 0.01);
        let f_ex = bank_exact.step_hammer_sample(0, 0.0, 0.0, 0.01);
        assert_eq!(f_lo.to_bits(), f_ex.to_bits());
    }
}

#[test]
fn p0_2_is_damper_settled_open_tracks_state_transitions() {
    let voicing = VoicingSpec::default();
    let mut bank = ActionBank::new(48_000.0, &voicing);
    let thresh = 4.5e-3_f32;

    // Fresh bank: damped keys closed (None), undamped keys open (Some).
    assert_eq!(bank.is_damper_settled_open(39, thresh), None);
    assert_eq!(
        bank.is_damper_settled_open(DAMPED_KEYS, thresh),
        Some(CLEARANCE_OPEN_M)
    );

    // NoteOn lifts immediately to +5mm settled open.
    bank.note_on(39, 40_000);
    assert_eq!(
        bank.is_damper_settled_open(39, thresh),
        Some(CLEARANCE_OPEN_M)
    );

    // NoteOff with Sustain=0 starts settling -> None until fully closed (still None < thresh).
    bank.note_off(39, 40_000);
    assert_eq!(bank.is_damper_settled_open(39, thresh), None);
    let mut clears = [0.0_f32; 8];
    let mut peels = [0.0_f32; 8];
    for _ in 0..800 {
        let _ = bank.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
    }
    assert_eq!(bank.is_damper_settled_open(39, thresh), None);

    // Lift Sustain=MAX: returns None while settling / peeling, then Some once settled open.
    bank.set_pedal(PedalKind::Sustain, u16::MAX);
    assert_eq!(bank.is_damper_settled_open(39, thresh), None);
    for _ in 0..800 {
        let _ = bank.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
    }
    let settled = bank.is_damper_settled_open(39, thresh);
    assert!(settled.is_some_and(|c| c >= thresh), "settled={settled:?}");

    // Half-pedal (+32 um < 4.5 mm) settles below threshold -> None.
    bank.set_pedal(PedalKind::Sustain, 32_768);
    for _ in 0..800 {
        let _ = bank.step_key_clearance_tile_with_peel(39, 8, &mut clears, &mut peels);
    }
    assert_eq!(bank.is_damper_settled_open(39, thresh), None);
}
