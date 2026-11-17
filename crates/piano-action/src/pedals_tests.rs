use super::*;

#[test]
fn sustain_curve_hits_landmarks_and_is_strictly_monotonic() {
    let c0 = sustain_clearance_m(0.0, CLEARANCE_PRELOAD_M);
    let chalf = sustain_clearance_m(0.5, CLEARANCE_PRELOAD_M);
    let c1 = sustain_clearance_m(1.0, CLEARANCE_PRELOAD_M);
    assert!((c0 - CLEARANCE_PRELOAD_M).abs() < 1e-9);
    assert!(
        (2.0e-5..=4.5e-5).contains(&chalf),
        "half-pedal clearance={chalf}"
    );
    assert!((chalf - HALF_PEDAL_MID_CLEARANCE_M).abs() < 1e-7);
    assert!((c1 - CLEARANCE_OPEN_M).abs() < 1e-7);

    let mut prev = c0;
    for i in 1..=200 {
        let s = (i as f32) / 200.0;
        let cur = sustain_clearance_m(s, CLEARANCE_PRELOAD_M);
        assert!(cur > prev, "not strictly monotonic at s={s}");
        prev = cur;
    }
}

#[test]
fn sostenuto_latches_only_open_damped_keys_on_crossing() {
    let mut p = PedalBank::new(48_000.0);
    // Hold key 39 (open), release key 40 long ago (closed), undamped key 75.
    p.note_on(39);
    assert!(p.damper_clearance_m(39) > SOSTENUTO_OPEN_MIN_M);
    assert!(p.damper_clearance_m(40) < 0.0);

    p.set_pedal(PedalKind::Sostenuto, 40_000); // crosses >= 0.5
    assert!(p.is_sostenuto_latched(39));
    assert!(!p.is_sostenuto_latched(40));

    // Key struck AFTER sostenuto down must NOT latch.
    p.note_on(41);
    assert!(!p.is_sostenuto_latched(41));

    // Release keys and run smoother for 200 ms: key 39 stays open, key 41 closes.
    p.note_off(39, 50_000);
    p.note_off(41, 50_000);
    for _ in 0..9_600 {
        p.step_key_clearance(39);
        p.step_key_clearance(41);
    }
    assert!((p.damper_clearance_m(39) - CLEARANCE_OPEN_M).abs() < 1e-6);
    assert!(p.damper_clearance_m(41) < -2.0e-4);

    // Release sostenuto (< 0.5) and verify key 39 drops to preload.
    p.set_pedal(PedalKind::Sostenuto, 10_000);
    assert!(!p.is_sostenuto_latched(39));
    for _ in 0..9_600 {
        p.step_key_clearance(39);
    }
    assert!(p.damper_clearance_m(39) < -2.0e-4);
    assert_eq!(p.step_key_clearance(75), CLEARANCE_OPEN_M);
}

#[test]
fn una_corda_weights_match_monochord_bichord_trichord() {
    let mut p = PedalBank::new(48_000.0);
    assert_eq!(p.strike_weights(0), [1.0, 0.0, 0.0]);
    assert_eq!(p.strike_weights(12), [1.0, 1.0, 0.0]);
    assert_eq!(p.strike_weights(48), [1.0, 1.0, 1.0]);

    p.set_pedal(PedalKind::UnaCorda, u16::MAX);
    assert!((p.strike_weights(0)[0] - 0.78).abs() < 1e-6);
    assert_eq!(p.strike_weights(12), [0.0, 1.0, 0.0]);
    assert_eq!(p.strike_weights(48), [0.0, 1.0, 1.0]);
}

#[test]
fn clearance_tile_matches_scalar_tracks_settling_and_handles_edges() {
    let mut p_tile = PedalBank::new(48_000.0);
    let mut p_scalar = PedalBank::new(48_000.0);
    assert!(!p_tile.settling.iter().any(|&s| s));

    p_tile.note_on(24);
    p_scalar.note_on(24);
    assert!(!p_tile.settling[24]);

    p_tile.note_off(24, 32_768);
    p_scalar.note_off(24, 32_768);
    assert!(p_tile.settling[24]);

    let mut clears = [0.0_f32; 8];
    for _ in 0..150 {
        let min_tile = p_tile.step_key_clearance_tile(24, 8, &mut clears);
        let mut min_scalar = f32::INFINITY;
        for &c_tile in &clears {
            let c_scalar = p_scalar.step_key_clearance(24);
            assert_eq!(c_tile.to_bits(), c_scalar.to_bits());
            min_scalar = min_scalar.min(c_scalar);
        }
        assert_eq!(min_tile.to_bits(), min_scalar.to_bits());
    }

    // Converge until settled and verify O(1) settled path snaps to exact preload.
    for _ in 0..2_000 {
        p_tile.step_key_clearance_tile(24, 8, &mut clears);
    }
    assert!(!p_tile.settling[24]);
    let settled_min = p_tile.step_key_clearance_tile(24, 5, &mut clears);
    assert_eq!(settled_min, p_tile.damper_clearance_m(24));
    assert!(clears[..5].iter().all(|&c| c == settled_min));

    // Redundant Sustain 0 does not re-enable settling; moving Sustain does.
    p_tile.set_pedal(PedalKind::Sustain, 0);
    assert!(!p_tile.settling[24]);
    p_tile.set_pedal(PedalKind::Sustain, 52_428); // ~0.8
    assert!(p_tile.settling.iter().all(|&s| s));
    for _ in 0..2_000 {
        p_tile.step_key_clearance_tile(24, 8, &mut clears);
    }
    assert!(!p_tile.settling[24]);

    // Pressing Sostenuto on a settled sustained key latches and lifts it.
    p_tile.set_pedal(PedalKind::Sostenuto, 65_535);
    assert!(p_tile.is_sostenuto_latched(24));
    assert!(p_tile.settling[24]);
    for _ in 0..2_000 {
        p_tile.step_key_clearance_tile(24, 8, &mut clears);
    }
    assert!(!p_tile.settling[24]);
    assert!((p_tile.damper_clearance_m(24) - CLEARANCE_OPEN_M).abs() < 1e-7);

    // Undamped keys and t_len == 0 remain safe.
    assert_eq!(
        p_tile.step_key_clearance_tile(75, 4, &mut clears),
        CLEARANCE_OPEN_M
    );
    assert!(clears[..4].iter().all(|&c| c == CLEARANCE_OPEN_M));
    let _ = p_tile.step_key_clearance_tile(24, 0, &mut clears);
}

#[test]
fn una_corda_felt_mod_softens_stiffness_and_raises_hysteresis() {
    let mut p = PedalBank::new(48_000.0);
    for k in [0_usize, 12, 48, 99] {
        assert_eq!(p.una_corda_felt_mod(k), (1.0, 0.0));
    }

    let mut prev_k = [1.0_f32; 3];
    let mut prev_eps = [0.0_f32; 3];
    for &val in &[16_384_u16, 32_768, 49_152, u16::MAX] {
        p.set_pedal(PedalKind::UnaCorda, val);
        for (idx, &k) in [0_usize, 12, 48].iter().enumerate() {
            let (k_scale, eps_add) = p.una_corda_felt_mod(k);
            assert!(
                k_scale < prev_k[idx],
                "k={k} val={val}: {k_scale} >= {}",
                prev_k[idx]
            );
            assert!(
                eps_add > prev_eps[idx],
                "k={k} val={val}: {eps_add} <= {}",
                prev_eps[idx]
            );
            prev_k[idx] = k_scale;
            prev_eps[idx] = eps_add;
        }
    }
    assert!((prev_k[0] - 0.52).abs() < 1e-6 && (prev_eps[0] - 0.12).abs() < 1e-6);
    assert!((prev_k[1] - 0.25).abs() < 1e-6 && (prev_eps[1] - 0.12).abs() < 1e-6);
    assert!((prev_k[2] - 0.28).abs() < 1e-6 && (prev_eps[2] - 0.12).abs() < 1e-6);
}

#[test]
fn damper_peel_off_fires_smooth_0p5_to_2mn_only_for_played_damped_keys() {
    let mut p = PedalBank::new(48_000.0);

    // 1. Bare Sustain pump on untouched virgin keys must emit zero peel-off pulse.
    p.set_pedal(PedalKind::Sustain, u16::MAX);
    let mut clears = [0.0_f32; 8];
    let mut peels = [0.0_f32; 8];
    for k in 0..DAMPED_KEYS {
        for _ in 0..600 {
            let (_, any_p) = p.step_key_clearance_tile_with_peel(k, 8, &mut clears, &mut peels);
            assert!(!any_p, "virgin key {k} emitted spurious peel pulse");
            assert_eq!(p.damper_peel_force_n(k), 0.0);
        }
    }

    // 2. Play keys (bass k=0, tenor k=39, high damped k=69), release & seat dampers, then lift Sustain.
    p.set_pedal(PedalKind::Sustain, 0);
    for k in 0..DAMPED_KEYS {
        for _ in 0..600 {
            let _ = p.step_key_clearance_tile(k, 8, &mut clears);
        }
    }
    for &k in &[0_usize, 39, 69] {
        p.note_on(k);
        p.note_off(k, 40_000);
        for _ in 0..800 {
            let (_, any_p) = p.step_key_clearance_tile_with_peel(k, 8, &mut clears, &mut peels);
            assert!(!any_p, "downward damper drop should not peel on key {k}");
        }
        assert!(p.damper_clearance_m(k) < -1.0e-4);
    }

    p.set_pedal(PedalKind::Sustain, u16::MAX);
    let mut peaks = [0.0_f32; 3];
    for (idx, &k) in [0_usize, 39, 69].iter().enumerate() {
        let mut trace = Vec::new();
        for _ in 0..400 {
            let (_, _) = p.step_key_clearance_tile_with_peel(k, 8, &mut clears, &mut peels);
            trace.extend_from_slice(&peels);
        }
        let peak = trace.iter().copied().fold(0.0_f32, f32::max);
        assert!(
            (5.0e-4..=2.0e-3).contains(&peak),
            "key {k} peel peak {peak} N out of [0.5 mN, 2.0 mN]"
        );
        assert_eq!(trace[0], 0.0);
        assert_eq!(*trace.last().unwrap(), 0.0);
        let max_jump = trace
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_jump < peak * 0.1,
            "peel pulse not smooth: jump={max_jump}"
        );
        peaks[idx] = peak;
    }
    assert!(
        peaks[0] > peaks[1] && peaks[1] > peaks[2],
        "peaks={peaks:?}"
    );

    // Untouched neighbor k=38 still produces zero peel-off during the same Sustain lift.
    for _ in 0..400 {
        let (_, any_p) = p.step_key_clearance_tile_with_peel(38, 8, &mut clears, &mut peels);
        assert!(!any_p);
    }

    // 3. Reset clears played/armed state.
    p.note_on(24);
    p.note_off(24, 40_000);
    for _ in 0..800 {
        let _ = p.step_key_clearance_tile(24, 8, &mut clears);
    }
    p.reset();
    p.set_pedal(PedalKind::Sustain, u16::MAX);
    for _ in 0..400 {
        let (_, any_p) = p.step_key_clearance_tile_with_peel(24, 8, &mut clears, &mut peels);
        assert!(!any_p);
    }
}
