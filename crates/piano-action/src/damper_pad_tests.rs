use super::*;
use core::f64::consts::TAU;
use piano_spec::key_choir;

#[test]
fn stationary_string_produces_exact_zero_force_even_under_preload() {
    let mut bank = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    for k in [0usize, 15, 39, 48, 69] {
        let str_idx = usize::from(key_choir(k).unwrap().first_string_idx);
        for slot_idx in [str_idx * 2, str_idx * 2 + 1] {
            for _ in 0..32 {
                let f = bank.step_damper_sample(k, slot_idx, 0.0, 0.0, 0.25, -3.0e-4);
                assert_eq!(f, 0.0);
            }
        }
    }
    // Undamped keys 70..88 and out-of-range slot indices always yield 0.0.
    assert_eq!(
        bank.step_damper_sample(70, 352, 1.0e-3, 1.5, 0.5, -3.0e-4),
        0.0
    );
    assert_eq!(
        bank.step_damper_sample(48, DAMPER_PAD_SLOTS, 1.0e-3, 1.5, 0.1, -3.0e-4),
        0.0
    );
}

#[test]
fn damper_is_strictly_passive_and_damps_in_80_to_180_ms_across_rates() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);

    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        let mut bank = DamperPadBank::new(sr, &voicing);
        let ts = 1.0 / f64::from(sr);
        for &k in &[0usize, 15, 39, 48, 69] {
            let str_idx = usize::from(key_choir(k).unwrap().first_string_idx);
            let slot_v = str_idx * 2;
            let p = specs[k];
            let m_wire = p.linear_density_kg_m * p.speaking_length_m;
            let phi_d = 0.38_f64;
            let m_mode = 0.5 * m_wire;
            let omega = TAU * p.f0_et_hz;
            // Coupled ZOH-like 1-mode oscillator + realistic high-mode port impedance.
            let r_port = (2.0 * ts / m_wire * (phi_d * phi_d * 18.0)) as f32;
            let mut u = 1.2_f64; // initial modal velocity (m/s)
            let mut q = 0.0_f64;
            let (s, c) = (omega * ts).sin_cos();
            let mut e_prev = 0.5 * m_mode * u * u;
            let e0 = e_prev;

            let n_steps = (0.35 * f64::from(sr)) as usize;
            let mut t60_est = 0.0_f64;
            for step in 0..n_steps {
                let y_d = (phi_d * q) as f32;
                let v_d = (phi_d * u) as f32;
                let f_damp = bank.step_damper_sample(k, slot_v, y_d, v_d, r_port, -3.0e-4);
                assert!(f_damp.is_finite());

                u += (phi_d * f64::from(f_damp) * ts) / m_mode;
                let u_next = c * u - omega * s * q;
                let q_next = (s / omega) * u + c * q;
                u = u_next;
                q = q_next;

                let e_cur = 0.5 * m_mode * (u * u + omega * omega * q * q);
                assert!(
                    e_cur <= e_prev * (1.0 + 1e-6) + 1e-18,
                    "passivity violation sr={sr} k={k} step={step}: {e_cur} > {e_prev}"
                );
                e_prev = e_cur;
                if t60_est == 0.0 && e_cur <= 1.0e-6 * e0 {
                    t60_est = (step as f64) * ts;
                }
            }
            assert!(
                (0.050..=0.220).contains(&t60_est),
                "sr={sr} k={k} damped T60={t60_est:.4}s out of expected band"
            );
        }
    }
}

#[test]
fn orthogonal_vh_slots_are_isolated_with_zero_cross_contamination() {
    let voicing = VoicingSpec::default();
    let mut isolated_h = DamperPadBank::new(48_000.0, &voicing);
    let mut dual_vh = DamperPadBank::new(48_000.0, &voicing);

    let k = 48;
    let s_idx = usize::from(key_choir(k).unwrap().first_string_idx);
    let slot_v = s_idx * 2;
    let slot_h = s_idx * 2 + 1;

    for step in 0..512 {
        let phase = (step as f32) * 0.055;
        let (yv, vv) = (1.6e-3 * phase.sin(), 1.4 * phase.cos());
        let (yh, vh) = (-7.5e-4 * (phase * 1.13).cos(), 0.85 * (phase * 1.13).sin());

        let _fv = dual_vh.step_damper_sample(k, slot_v, yv, vv, 0.08, -3.0e-4);
        let fh_dual = dual_vh.step_damper_sample(k, slot_h, yh, vh, 0.08, -3.0e-4);
        let fh_ref = isolated_h.step_damper_sample(k, slot_h, yh, vh, 0.08, -3.0e-4);
        assert_eq!(fh_dual.to_bits(), fh_ref.to_bits());
    }
}

#[test]
fn update_voicing_in_place_updates_damper_coeffs_and_matches_fresh_bank() {
    let mut voicing = VoicingSpec::default();
    let mut bank = DamperPadBank::new(48_000.0, &voicing);
    voicing.string_length_scale = 1.15;
    bank.update_voicing_in_place(&voicing);
    let fresh = DamperPadBank::new(48_000.0, &voicing);

    for k in [0_usize, 24, 48, 69] {
        assert!((bank.c_nom_key[k] - fresh.c_nom_key[k]).abs() < 1e-6);
        assert!((bank.f_scale_key[k] - fresh.f_scale_key[k]).abs() < 1e-6);
    }
    assert_ne!(
        bank.c_nom_key[39].to_bits(),
        DamperPadBank::new(48_000.0, &VoicingSpec::default()).c_nom_key[39].to_bits()
    );
}

#[test]
fn smooth_c2_engagement_suppresses_hf_grazing_buzz_and_shallow_preload_gains_match() {
    assert_eq!(smooth_felt_engage(0.0), 0.0);
    assert_eq!(smooth_felt_engage(ENGAGE_ONSET_M), 0.0);
    assert_eq!(smooth_felt_engage(-1.0e-4), 0.0);
    assert!(smooth_felt_engage(1.0) == 1.0);

    // Compare > 6 kHz spectral buzz of C^2 grazing force vs old C^0 linear corner on 261.63 Hz tone.
    let sr = 48_000.0_f64;
    let n_fft = 2_048_usize;
    let mut bank = DamperPadBank::new(sr as f32, &VoicingSpec::default());
    bank.lift_key_open(39);
    let s_idx = usize::from(key_choir(39).unwrap().first_string_idx);
    let (mut f_c2, mut f_c0) = ([0.0_f64; 2_048], [0.0_f64; 2_048]);
    let w0 = TAU * 261.63;
    for i in 0..n_fft {
        let t = (i as f64) / sr;
        let win = 0.5 - 0.5 * (TAU * (i as f64) / (n_fft as f64)).cos();
        let y = (1.2e-4 * (w0 * t).sin()) as f32;
        let v = (1.2e-4 * w0 * (w0 * t).cos()) as f32;
        let fd = bank.step_damper_sample(39, s_idx * 2, y, v, 0.08, 4.0e-5);
        f_c2[i] = f64::from(fd) * win;
        let s_lin = (y - 4.0e-5).max(0.0) + (-y - 4.0e-5).max(0.0);
        let eng_c0 = if s_lin > 0.0 {
            s_lin / (s_lin + ENGAGE_HALF_M)
        } else {
            0.0
        };
        f_c0[i] = f64::from(-bank.c_nom_key[39] * eng_c0 * v) * win;
    }
    let hf_ratio_db = |sig: &[f64]| -> f64 {
        let (mut tot, mut hf) = (0.0_f64, 0.0_f64);
        for k in (1..n_fft / 2).step_by(2) {
            let ph_step = TAU * (k as f64) / (n_fft as f64);
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (n, &x) in sig.iter().enumerate() {
                let (s, c) = (ph_step * (n as f64)).sin_cos();
                re += x * c;
                im -= x * s;
            }
            let p = re * re + im * im;
            tot += p;
            if (k as f64) * sr / (n_fft as f64) >= 6_000.0 {
                hf += p;
            }
        }
        10.0 * ((hf + 1.0e-30) / (tot + 1.0e-30)).log10()
    };
    let db_c2 = hf_ratio_db(&f_c2);
    let db_c0 = hf_ratio_db(&f_c0);
    assert!(
        db_c2 < db_c0 - 25.0,
        "expected >= 25 dB >6kHz buzz reduction: c2={db_c2:.1} dB vs c0={db_c0:.1} dB"
    );

    // Shallow negative preload (yc = -2.44 um, CC64=42) linear gains match step_damper_pair.
    let yc_shallow = -2.44e-6_f32;
    for _ in 0..8 {
        let _ =
            bank.symp_linear_gains_for_chunk(39, s_idx, 0.08, 0.06, yc_shallow, yc_shallow, 128);
    }
    let (gv, gh) = bank
        .symp_linear_gains_if_settled(39, s_idx, 0.08, 0.06, yc_shallow)
        .expect("shallow negative preload should settle linear gains");
    let (fv, fh) = bank.step_damper_pair(
        39, s_idx, 1.0e-8, 0.002, 0.08, -1.0e-8, -0.001, 0.06, yc_shallow,
    );
    assert!((fv - gv * 0.002).abs() <= 1.0e-7);
    assert!((fh - gh * (-0.001)).abs() <= 1.0e-7);

    // Open-gap relaxation decays z identically to 128 contact-free samples once in open gap.
    let mut b_chunk = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    let mut b_step = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    let _ = b_chunk.step_damper_pair(39, s_idx, 1.0e-6, 0.001, 0.08, 1.0e-6, 0.001, 0.06, 5.0e-5);
    let _ = b_step.step_damper_pair(39, s_idx, 1.0e-6, 0.001, 0.08, 1.0e-6, 0.001, 0.06, 5.0e-5);
    b_chunk.relax_open_gap_chunk(s_idx, 128);
    for _ in 0..128 {
        let _ =
            b_step.step_damper_pair(39, s_idx, 1.0e-6, 0.001, 0.08, 1.0e-6, 0.001, 0.06, 5.0e-5);
    }
    assert!((b_chunk.slots[s_idx * 2].z_pos - b_step.slots[s_idx * 2].z_pos).abs() < 1.0e-6);
}

#[test]
fn engage_gates_felt_hysteresis_below_onset_and_across_lin_lim() {
    let mut bank = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    let s_idx = usize::from(key_choir(39).unwrap().first_string_idx);
    let slot_v = s_idx * 2;
    let slot_h = s_idx * 2 + 1;

    // Prime z_pos and z_neg well above 1e-5 via heavy felt compression.
    for _ in 0..64 {
        let _ = bank.step_damper_sample(39, slot_v, 1.5e-3, 1.0, 0.08, -3.0e-4);
        let _ = bank.step_damper_sample(39, slot_v, -1.5e-3, -1.0, 0.08, -3.0e-4);
        let _ = bank.step_damper_sample(39, slot_h, 1.5e-3, 1.0, 0.06, -3.0e-4);
    }
    assert!(bank.slots[slot_v].z_pos > 1.0e-5);
    assert!(bank.slots[slot_v].z_neg > 1.0e-5);

    // 1. s_sum = 2.9 um <= ENGAGE_ONSET_M (3 um) must yield exact 0.0 even with primed hysteresis.
    let f_sub_onset = bank.step_damper_sample(39, slot_v, 4.29e-5, 0.25, 0.08, 4.0e-5);
    assert_eq!(f_sub_onset, 0.0);

    // 2. Smooth O(u^3) onset just above ENGAGE_ONSET_M (du = 1 um vs 2 um -> ~8x force ratio).
    let mut b1 = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    b1.lift_key_open(39);
    let mut b2 = b1.clone();
    let f1 = b1
        .step_damper_sample(39, slot_v, 4.4e-5, 0.1, 0.08, 4.0e-5)
        .abs();
    let f2 = b2
        .step_damper_sample(39, slot_v, 4.5e-5, 0.1, 0.08, 4.0e-5)
        .abs();
    assert!(f1 > 0.0);
    let ratio = f2 / f1;
    assert!(
        (7.0..=9.0).contains(&ratio),
        "expected O(u^3) ratio ~8, got {ratio:.3}"
    );

    // 3. y_clear_m = -1.0e-6 (2*|yc| = 2 um <= 3 um) yields exact (0.0, 0.0) in both paths.
    let (pv, ph) =
        bank.step_damper_pair(39, s_idx, 1.0e-7, 0.05, 0.08, -1.0e-7, -0.05, 0.06, -1.0e-6);
    assert_eq!((pv, ph), (0.0, 0.0));
    let (gv, gh) = bank
        .symp_linear_gains_for_chunk(39, s_idx, 0.08, 0.06, -1.0e-6, -1.0e-6, 128)
        .expect("y_clear_m = -1e-6 <= MIN_CLOSED_LINEAR_CLEAR_M");
    assert_eq!((gv, gh), (0.0, 0.0));

    // 4. Continuity across lin_lim = 8.0e-6 at y_clear_m = -1.0e-5.
    let yc = -1.0e-5_f32;
    let mut ba = DamperPadBank::new(48_000.0, &VoicingSpec::default());
    for _ in 0..512 {
        let _ = ba.step_damper_sample(39, slot_v, 0.0, 0.0, 0.08, yc);
    }
    let mut bb = ba.clone();
    let f_below = ba.step_damper_sample(39, slot_v, 7.99e-6, 0.05, 0.08, yc);
    let f_above = bb.step_damper_sample(39, slot_v, 8.01e-6, 0.05, 0.08, yc);
    let rel_jump = (f_below - f_above).abs() / f_below.abs().max(1.0e-12);
    assert!(
        rel_jump < 1.0e-3,
        "lin_lim discontinuity: f_below={f_below}, f_above={f_above}, rel={rel_jump}"
    );
}
