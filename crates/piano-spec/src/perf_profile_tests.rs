use super::*;

#[test]
fn perf_preset_roundtrip_and_default() {
    assert_eq!(PerfPreset::default(), PerfPreset::Full);
    assert_eq!(PerfPreset::from_u8(0), Some(PerfPreset::Full));
    assert_eq!(PerfPreset::from_u8(1), Some(PerfPreset::Balanced));
    assert_eq!(PerfPreset::from_u8(2), Some(PerfPreset::Realtime));
    for v in 3..=u8::MAX {
        assert_eq!(PerfPreset::from_u8(v), None);
    }
    assert_eq!(PerfProfile::default(), PerfProfile::full());
    assert_eq!(
        PerfProfile::from_preset(PerfPreset::Full),
        PerfProfile::full()
    );
    assert_eq!(
        PerfProfile::from_preset(PerfPreset::Balanced),
        PerfProfile::balanced()
    );
    assert_eq!(
        PerfProfile::from_preset(PerfPreset::Realtime),
        PerfProfile::realtime()
    );
}

#[test]
fn full_profile_preserves_exact_d2dab9c_baseline_parity() {
    let full = PerfProfile::full();
    assert_eq!(full.sanitized(), full);
    assert!(full.is_full_mode_counts());
    assert!(!full.is_dead_mode_pruning_enabled());
    assert_eq!(
        full.low_order_energy_horizon_sq().to_bits(),
        1.0e-18_f32.to_bits()
    );
    assert_eq!(full.energy_horizon_sq().to_bits(), 1.0e-18_f32.to_bits());
    assert_eq!(full.sav_dormancy_ratio().to_bits(), 1.581e-6_f64.to_bits());
    assert_eq!(full.action_oversample, 8);
    assert!(!full.polynomial_adaa);
    assert!(full.shank_flex_enabled);
    assert_eq!(full.damper_control_decim, 1);
    assert!(full.hertz_nonlinear);
    assert_eq!(full.active_bridge_poles, 10);
    assert_eq!(full.fir_head_taps, 64);
    assert!(!full.fir_block_fft);
    assert_eq!(full.hall_fdn_lines, 8);
    assert_eq!(full.spatial_tail_split_tap, u16::MAX);
    assert_eq!(full.ir_duration_secs, 0.50);
    assert_eq!(full.sav_max_key_idx, 52);
    assert!(!full.bass_phantom_sparse);
}

#[test]
fn balanced_and_realtime_thresholds_convert_accurately() {
    let bal = PerfProfile::balanced();
    assert_eq!(bal.sanitized(), bal);
    assert!(bal.is_dead_mode_pruning_enabled());
    assert!(!bal.is_full_mode_counts());
    assert_eq!(bal.damper_control_decim, 8);
    let bal_low = bal.low_order_energy_horizon_sq();
    let expected_bal_low = 10.0_f64.powf((-112.0 - 60.0) / 10.0) as f32;
    assert!((bal_low - expected_bal_low).abs() <= 1e-6 * expected_bal_low);
    let bal_eh = bal.energy_horizon_sq();
    let expected_bal_eh = 10.0_f64.powf((-96.0 - 60.0) / 10.0) as f32;
    assert!((bal_eh - expected_bal_eh).abs() <= 1e-6 * expected_bal_eh);
    let bal_sav = bal.sav_dormancy_ratio();
    let expected_bal_sav = 0.5 * 10.0_f64.powf(-42.0 / 10.0);
    assert!((bal_sav - expected_bal_sav).abs() <= 1e-12 * expected_bal_sav);

    let rt = PerfProfile::realtime();
    assert_eq!(rt.sanitized(), rt);
    assert!(rt.is_dead_mode_pruning_enabled());
    assert!(!rt.is_full_mode_counts());
    assert_eq!(rt.damper_control_decim, 16);
    let rt_low = rt.low_order_energy_horizon_sq();
    let expected_rt_low = 10.0_f64.powf((-108.0 - 60.0) / 10.0) as f32;
    assert!((rt_low - expected_rt_low).abs() <= 1e-6 * expected_rt_low);
    let rt_eh = rt.energy_horizon_sq();
    let expected_rt_eh = 10.0_f64.powf((-84.0 - 60.0) / 10.0) as f32;
    assert!((rt_eh - expected_rt_eh).abs() <= 1e-6 * expected_rt_eh);
    let rt_sav = rt.sav_dormancy_ratio();
    let expected_rt_sav = 0.5 * 10.0_f64.powf(-38.0 / 10.0);
    assert!((rt_sav - expected_rt_sav).abs() <= 1e-12 * expected_rt_sav);
}

#[test]
fn sanitized_clamps_bounds_replaces_nan_and_snaps_discrete_fields() {
    let dirty = PerfProfile {
        horiz_max_hz: f32::NAN,
        dead_mode_prune_db: -5.0,
        low_order_energy_horizon_db: f32::NAN,
        energy_horizon_db: f32::NAN,
        sav_dormancy_db: -999.0,
        sav_max_key_idx: 87,
        bass_phantom_sparse: true,
        passive_res_max_hz: 100.0,
        ir_duration_secs: 2.0,
        spatial_tail_split_tap: 128,
        fir_head_taps: 40,
        fir_block_fft: true,
        hall_fdn_lines: 14,
        action_oversample: 5,
        polynomial_adaa: true,
        shank_flex_enabled: false,
        damper_control_decim: 3,
        hertz_nonlinear: false,
        active_bridge_poles: 2,
    };
    let clean = dirty.sanitized();
    assert_eq!(clean.horiz_max_hz, 20_000.0);
    assert_eq!(clean.dead_mode_prune_db, -20.0);
    assert_eq!(clean.low_order_energy_horizon_db, -120.0);
    assert_eq!(clean.energy_horizon_db, -120.0);
    assert_eq!(clean.sav_dormancy_db, -80.0);
    assert_eq!(clean.sav_max_key_idx, 52);
    assert_eq!(clean.passive_res_max_hz, 1_000.0);
    assert_eq!(clean.ir_duration_secs, 0.50);
    assert_eq!(clean.spatial_tail_split_tap, 512);
    assert_eq!(clean.fir_head_taps, 32);
    assert_eq!(clean.hall_fdn_lines, 16);
    assert_eq!(clean.action_oversample, 4);
    assert_eq!(clean.damper_control_decim, 2);
    assert_eq!(clean.active_bridge_poles, 4);

    // Verify damper_control_decim snaps to {1, 2, 4, 8, 16}.
    for &(raw, expected) in &[
        (0_u8, 1_u8),
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 4),
        (6, 4),
        (7, 8),
        (8, 8),
        (12, 8),
        (13, 16),
        (16, 16),
        (255, 16),
    ] {
        let p = PerfProfile {
            damper_control_decim: raw,
            ..PerfProfile::full()
        }
        .sanitized();
        assert_eq!(p.damper_control_decim, expected, "raw={raw}");
    }

    // Verify low_order_energy_horizon_db clamps to [-120.0, -40.0].
    let lo_clamp = PerfProfile {
        low_order_energy_horizon_db: -200.0,
        ..PerfProfile::full()
    }
    .sanitized();
    assert_eq!(lo_clamp.low_order_energy_horizon_db, -120.0);
    let hi_clamp = PerfProfile {
        low_order_energy_horizon_db: -10.0,
        ..PerfProfile::full()
    }
    .sanitized();
    assert_eq!(hi_clamp.low_order_energy_horizon_db, -40.0);

    // Check Copy semantics.
    fn assert_copy<T: Copy>() {}
    assert_copy::<PerfPreset>();
    assert_copy::<PerfProfile>();
}
