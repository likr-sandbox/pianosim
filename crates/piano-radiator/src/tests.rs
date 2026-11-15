use alloc::vec;
use alloc::vec::Vec;
use piano_bridge::AdmittanceSsm;
use piano_simd::RealFft;
use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES,
    PRESET_REDUCED_POLES,
};

use crate::math::{
    acos_f64, asin_f64, atan2_f64, atan_f64, exp_f64, invert_10x10, ln_f64, powf_f64, sin_cos_f64,
    sqrt_f64, tan_f64, PI,
};
use crate::nupols_tests::test_preset;
use crate::RadiatorBank;

fn fft_mag_sq(signal: &[f32], fft_len: usize) -> Vec<f64> {
    let mut padded = vec![0.0_f32; fft_len];
    let copy_n = signal.len().min(fft_len);
    padded[..copy_n].copy_from_slice(&signal[..copy_n]);
    let mut fft = RealFft::new(fft_len);
    let spec_len = fft.spectrum_len();
    let mut re = vec![0.0_f32; spec_len];
    let mut im = vec![0.0_f32; spec_len];
    fft.forward_split(&padded, &mut re, &mut im);
    (0..spec_len)
        .map(|k| (re[k] as f64).powi(2) + (im[k] as f64).powi(2))
        .collect()
}

fn band_mean_power(mag_sq: &[f64], sr: f64, fft_len: usize, f_lo: f64, f_hi: f64) -> f64 {
    let k0 = ((f_lo * fft_len as f64 / sr).ceil() as usize).max(1);
    let k1 = ((f_hi * fft_len as f64 / sr).floor() as usize).min(mag_sq.len() - 1);
    assert!(k1 >= k0, "empty FFT band [{f_lo}, {f_hi}] at sr={sr}");
    let sum: f64 = mag_sq[k0..=k1].iter().sum();
    sum / ((k1 - k0 + 1) as f64)
}

#[test]
fn no_std_math_matches_std_accuracy() {
    for &x in &[0.0, 0.25, 0.5, 1.0, 2.5, 17.3, 1.0e-7, 1.0e4] {
        assert!((sqrt_f64(x) - x.sqrt()).abs() <= 1.0e-13 * x.sqrt().max(1.0));
    }
    for i in -40..=40 {
        let x = (i as f64) * 0.23;
        let (s, c) = sin_cos_f64(x);
        assert!((s - x.sin()).abs() < 1.0e-13, "sin({x})");
        assert!((c - x.cos()).abs() < 1.0e-13, "cos({x})");
        assert!((atan_f64(x) - x.atan()).abs() < 1.0e-13, "atan({x})");
    }
    for &x in &[0.01, 0.2, 0.5, 1.1, 1.45] {
        assert!((tan_f64(x) - x.tan()).abs() < 1.0e-12, "tan({x})");
    }
    for &x in &[-0.999, -0.5, 0.0, 0.4, 0.999] {
        assert!((asin_f64(x) - x.asin()).abs() < 1.0e-12, "asin({x})");
        assert!((acos_f64(x) - x.acos()).abs() < 1.0e-12, "acos({x})");
    }
    for &(y, x) in &[
        (-0.5, 0.8),
        (0.5, 0.8),
        (0.3, -0.9),
        (-0.7, -0.2),
        (1.0, 0.0),
    ] {
        assert!((atan2_f64(y, x) - y.atan2(x)).abs() < 1.0e-12);
    }
    for &x in &[-20.0, -3.5, -0.1, 0.0, 0.45, 7.2] {
        assert!((exp_f64(x) - x.exp()).abs() <= 1.0e-12 * x.exp().max(1.0));
    }
    for &x in &[1.0e-5, 0.3, 1.0, 2.72, 96000.0] {
        assert!((ln_f64(x) - x.ln()).abs() < 1.0e-12);
        assert!((powf_f64(x, 0.37) - x.powf(0.37)).abs() <= 1.0e-11 * x.powf(0.37).max(1.0));
    }
    let mut a = [[0.0_f64; 10]; 10];
    for r in 0..10 {
        a[r][r] = 4.0 + (r as f64) * 0.2;
        if r + 1 < 10 {
            a[r][r + 1] = 0.7;
            a[r + 1][r] = 0.7;
        }
    }
    let inv = invert_10x10(a).expect("SPD 10x10 must invert");
    for (r, row) in a.iter().enumerate() {
        for c in 0..10 {
            let dot: f64 = row.iter().zip(inv.iter()).map(|(&ak, ik)| ak * ik[c]).sum();
            let target = if r == c { 1.0 } else { 0.0 };
            assert!((dot - target).abs() < 1.0e-12);
        }
    }
}

#[test]
fn lid_state_hf_ordering_closed_lt_half_lt_open() {
    let sr = 48_000.0;
    let fft_len = 32_768;
    let preset = test_preset(160);
    let perspectives = [
        ListenerPerspective::Player,
        ListenerPerspective::CloseStereo,
        ListenerPerspective::ConcertHall,
        ListenerPerspective::BinauralHrtf,
    ];

    for &persp in &perspectives {
        let ir_closed = RadiatorBank::synthesize_ir_for_test(&preset, sr, persp, LidState::Closed);
        let ir_half = RadiatorBank::synthesize_ir_for_test(&preset, sr, persp, LidState::Half);
        let ir_open = RadiatorBank::synthesize_ir_for_test(&preset, sr, persp, LidState::Open);

        let hf_energy = |irs: &[[Vec<f32>; 2]; PRESET_RADIATOR_ZONES]| -> f64 {
            let mut sum = 0.0_f64;
            for pair in irs {
                for ch_ir in pair {
                    let m = fft_mag_sq(ch_ir, fft_len);
                    sum += band_mean_power(&m, sr, fft_len, 2_000.0, 14_000.0);
                }
            }
            sum
        };

        let e_closed = hf_energy(&ir_closed);
        let e_half = hf_energy(&ir_half);
        let e_open = hf_energy(&ir_open);
        let db_half_closed = 10.0 * (e_half / e_closed).log10();
        let db_open_half = 10.0 * (e_open / e_half).log10();
        eprintln!(
            "{persp:?} >2kHz HF diff: Half/Closed = +{db_half_closed:.2} dB, Open/Half = +{db_open_half:.2} dB"
        );
        assert!(
            db_half_closed > 2.5 && db_open_half > 2.0,
            "HF > 2kHz ordering failed for {persp:?}: Closed={e_closed}, Half={e_half}, Open={e_open}"
        );
    }
}

fn measure_onset_index(ir: &[f32]) -> usize {
    let peak = ir.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
    let thresh = peak * 0.20;
    ir.iter().position(|&s| s.abs() >= thresh).unwrap_or(0)
}

#[test]
fn binaural_hrtf_opposite_sign_itd_and_ild_bass_vs_treble() {
    let sr = 48_000.0;
    let preset = test_preset(120);
    let irs = RadiatorBank::synthesize_ir_for_test(
        &preset,
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
    );

    // Zone 0: left bass (`x = -0.58 m`) -> L leads (`t_L - t_R < 0`) and L louder (`ILD > 0 dB`).
    let onset_0_l = measure_onset_index(&irs[0][0]) as isize;
    let onset_0_r = measure_onset_index(&irs[0][1]) as isize;
    let itd_0_samples = onset_0_l - onset_0_r;

    let energy = |v: &[f32]| -> f64 { v.iter().map(|&s| (s as f64) * (s as f64)).sum() };
    let ild_0_db = 10.0 * (energy(&irs[0][0]) / energy(&irs[0][1])).log10();

    // Zone 7: right treble (`x = +0.57 m`) -> R leads (`t_L - t_R > 0`) and R louder (`ILD < 0 dB`).
    let onset_7_l = measure_onset_index(&irs[7][0]) as isize;
    let onset_7_r = measure_onset_index(&irs[7][1]) as isize;
    let itd_7_samples = onset_7_l - onset_7_r;
    let ild_7_db = 10.0 * (energy(&irs[7][0]) / energy(&irs[7][1])).log10();

    eprintln!(
        "Binaural Zone 0: ITD={itd_0_samples} smp, ILD={ild_0_db:.2} dB | Zone 7: ITD={itd_7_samples} smp, ILD={ild_7_db:.2} dB"
    );

    assert!(
        itd_0_samples <= -6,
        "Zone 0 L should lead R cleanly, got {itd_0_samples}"
    );
    assert!(
        itd_7_samples >= 6,
        "Zone 7 R should lead L cleanly, got {itd_7_samples}"
    );
    assert!(
        ild_0_db > 1.5,
        "Zone 0 L should be louder than R, got {ild_0_db:.2} dB"
    );
    assert!(
        ild_7_db < -1.5,
        "Zone 7 R should be louder than L, got {ild_7_db:.2} dB"
    );
}

#[test]
fn sample_rate_consistency_44100_48000_96000_within_1db() {
    let preset = test_preset(240);
    let rates = [
        (44_100.0, 32_768_usize),
        (48_000.0, 32_768),
        (96_000.0, 65_536),
    ];
    let bands = [
        (125.0, 250.0),
        (250.0, 500.0),
        (500.0, 1_000.0),
        (1_000.0, 2_000.0),
        (2_000.0, 4_000.0),
        (4_000.0, 8_000.0),
        (100.0, 8_000.0),
    ];

    let mut spectra = Vec::new();
    for &(sr, fft_len) in &rates {
        let irs = RadiatorBank::synthesize_ir_for_test(
            &preset,
            sr,
            ListenerPerspective::Player,
            LidState::Open,
        );
        let mut avg_mag = vec![0.0_f64; fft_len / 2 + 1];
        for pair in &irs[..PRESET_BRIDGE_ZONES] {
            for ch_ir in pair {
                let m = fft_mag_sq(ch_ir, fft_len);
                for (dst, &src) in avg_mag.iter_mut().zip(m.iter()) {
                    *dst += src;
                }
            }
        }
        spectra.push((sr, fft_len, avg_mag));
    }

    let ref_entry = &spectra[1]; // 48 kHz reference
    for entry in &spectra {
        for &(f_lo, f_hi) in &bands {
            let p_sr = band_mean_power(&entry.2, entry.0, entry.1, f_lo, f_hi);
            let p_ref = band_mean_power(&ref_entry.2, ref_entry.0, ref_entry.1, f_lo, f_hi);
            let diff_db = 10.0 * (p_sr / p_ref).log10();
            assert!(
                diff_db.abs() <= 1.0,
                "Rate {} Hz band [{f_lo}..{f_hi}] deviated {diff_db:.3} dB (> 1.0 dB) vs 48 kHz",
                entry.0
            );
        }
    }
}

#[test]
fn stage3_pole_cancellation_eliminates_reduced_pole_peaking() {
    let sr = 48_000.0;
    let ssm = AdmittanceSsm::new(sr as f32);
    let mut preset = test_preset(60);
    preset.y_inf = *ssm.y_inf();
    for (p, ssm_pole) in ssm.poles().iter().enumerate() {
        preset.reduced_poles[p].freq_hz = ssm_pole.freq_hz;
        // Amplify prominence slightly to stress-test cancellation sharpness.
        preset.reduced_poles[p].zeta = ssm_pole.zeta.min(0.012);
        preset.reduced_poles[p].modal_mass_kg = ssm_pole.modal_mass_kg;
        preset.reduced_poles[p].zone_gains = ssm_pole.zone_gains;
    }

    // Compare cancelled IR cascade `H_total(z) * Y_red(z) * Y_inf^{-1}` against
    // the exact pole-free Stage 4 target `H_rad(z)` across all 10 pole frequencies.
    let ir_cancelled = RadiatorBank::synthesize_ir_for_test(
        &preset,
        sr,
        ListenerPerspective::Player,
        LidState::Open,
    );
    let mut no_pole_preset = preset.clone();
    for pole in &mut no_pole_preset.reduced_poles {
        pole.modal_mass_kg = 0.0;
    }
    let ir_target = RadiatorBank::synthesize_ir_for_test(
        &no_pole_preset,
        sr,
        ListenerPerspective::Player,
        LidState::Open,
    );

    // Simulate driving trapezoidal `Y_red(z) * Y_inf^{-1}` with impulse on Zone 2
    // and convolving its output with `ir_cancelled` over `N = 24,000` samples.
    let n_len = ir_cancelled[0][0].len();
    let drive_zone = 2_usize;
    let y_red_impulse = simulate_y_red_norm_impulse(&preset, sr, drive_zone, n_len);

    let mut cascade_l = vec![0.0_f64; n_len];
    for bz in 0..PRESET_BRIDGE_ZONES {
        let h_l = &ir_cancelled[bz][0];
        let v_z = &y_red_impulse[bz];
        for n in 0..n_len {
            let x = v_z[n];
            if x != 0.0 {
                for k in 0..(n_len - n) {
                    cascade_l[n + k] += x * (h_l[k] as f64);
                }
            }
        }
    }
    let cascade_f32: Vec<f32> = cascade_l.into_iter().map(|x| x as f32).collect();

    let fft_len = 32_768;
    let m_casc = fft_mag_sq(&cascade_f32, fft_len);
    let m_targ = fft_mag_sq(&ir_target[drive_zone][0], fft_len);
    let m_y_red = fft_mag_sq(
        &y_red_impulse[drive_zone]
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<_>>(),
        fft_len,
    );

    let mut max_excess_db = 0.0_f64;
    for pole in &preset.reduced_poles {
        let bin = ((pole.freq_hz as f64) * fft_len as f64 / sr).round() as usize;
        if bin > 0 && bin < fft_len / 2 {
            let diff_db = 10.0 * (m_casc[bin] / m_targ[bin].max(1e-30)).log10();
            let raw_pole_db = 10.0 * m_y_red[bin].log10();
            max_excess_db = max_excess_db.max(diff_db.abs());
            assert!(
                diff_db.abs() <= 1.0,
                "Pole at {:.1} Hz failed to cancel (raw={raw_pole_db:.2} dB, diff={diff_db:.3} dB)",
                pole.freq_hz
            );
        }
    }
    eprintln!("Stage 3 10-pole cancellation max residual across poles = {max_excess_db:.4} dB");
}

fn simulate_y_red_norm_impulse(
    preset: &BodyPreset,
    sr: f64,
    impulse_zone: usize,
    n_len: usize,
) -> [Vec<f64>; PRESET_BRIDGE_ZONES] {
    let h = 0.5 / sr;
    let mut eta = [0.0_f64; PRESET_REDUCED_POLES];
    let mut prev_eta_dot = [0.0_f64; PRESET_REDUCED_POLES];
    let mut prev_f = [0.0_f64; PRESET_BRIDGE_ZONES];
    let mut out: [Vec<f64>; PRESET_BRIDGE_ZONES] =
        core::array::from_fn(|_| Vec::with_capacity(n_len));

    let inv_y0 = 1.0 / (preset.y_inf[impulse_zone] as f64);
    for n in 0..n_len {
        let mut f = [0.0_f64; PRESET_BRIDGE_ZONES];
        if n == 0 {
            f[impulse_zone] = inv_y0;
        }
        let mut v_modes = [0.0_f64; PRESET_BRIDGE_ZONES];
        for m in 0..PRESET_REDUCED_POLES {
            let pole = &preset.reduced_poles[m];
            let omega = 2.0 * PI * (pole.freq_hz as f64);
            let mass = pole.modal_mass_kg as f64;
            let damp = 2.0 * (pole.zeta as f64) * omega * mass;
            let stiff = omega * omega * mass;
            let lhs = mass + h * damp + h * h * stiff;
            let mut f_proj = 0.0_f64;
            for z in 0..PRESET_BRIDGE_ZONES {
                f_proj += (pole.zone_gains[z] as f64) * (f[z] + prev_f[z]);
            }
            let rhs = h * f_proj + (mass - h * damp - h * h * stiff) * prev_eta_dot[m]
                - 2.0 * h * stiff * eta[m];
            let eta_dot = rhs / lhs;
            eta[m] += h * (eta_dot + prev_eta_dot[m]);
            prev_eta_dot[m] = eta_dot;
            for (v_z, &g_z) in v_modes.iter_mut().zip(pole.zone_gains.iter()) {
                *v_z += (g_z as f64) * eta_dot;
            }
        }
        for z in 0..PRESET_BRIDGE_ZONES {
            out[z].push((preset.y_inf[z] as f64) * f[z] + v_modes[z]);
            prev_f[z] = f[z];
        }
    }
    out
}

#[test]
fn ir_synth_completes_under_40ms_on_1500_modes() {
    let preset = test_preset(1_500);
    let bytes = preset.encode();
    let mut bank = RadiatorBank::from_preset_bytes(&bytes, 48_000.0).expect("preset decode");

    let mut scratch =
        crate::ir_synth::IrSynthScratch::new(crate::ir_synth::ir_len_for_sample_rate(48_000.0));
    let mut best_synth_ms = f64::INFINITY;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        crate::ir_synth::synthesize_irs(
            &preset,
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            48_000.0,
            &mut scratch,
        );
        let ms = t0.elapsed().as_secs_f64() * 1_000.0;
        if ms < best_synth_ms {
            best_synth_ms = ms;
        }
    }

    let mut best_ms = f64::INFINITY;
    for iter in 0..5 {
        let persp = if iter % 2 == 0 {
            ListenerPerspective::BinauralHrtf
        } else {
            ListenerPerspective::CloseStereo
        };
        let lid = if iter % 2 == 0 {
            LidState::Half
        } else {
            LidState::Open
        };
        let t0 = std::time::Instant::now();
        bank.reconfigure_space(persp, lid);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        if elapsed_ms < best_ms {
            best_ms = elapsed_ms;
        }
    }
    eprintln!(
        "1500-mode 0.5s 18-path IR synth = {best_synth_ms:.2} ms | full reconfigure_space = {best_ms:.2} ms"
    );
    #[cfg(not(debug_assertions))]
    assert!(
        best_synth_ms < 40.0 && best_ms < 40.0,
        "1500-mode IR synth took {best_synth_ms:.2} ms / reconfigure {best_ms:.2} ms (>= 40.0 ms)"
    );

    // Measure steady-state 128-sample runtime throughput (1024 blocks = 131,072 samples).
    let block = [[0.01_f32; PRESET_RADIATOR_ZONES]; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    let mut timings_us = Vec::with_capacity(1_024);
    for _ in 0..1_024 {
        let t0 = std::time::Instant::now();
        bank.process_block(&block, &mut out_l, &mut out_r);
        timings_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    timings_us.sort_by(|a, b| a.total_cmp(b));
    let mean_us = timings_us.iter().sum::<f64>() / (timings_us.len() as f64);
    let p99_us = timings_us[(timings_us.len() * 99) / 100];
    eprintln!("RadiatorBank 128-sample runtime @ 48 kHz: mean={mean_us:.1} us, p99={p99_us:.1} us");
}

#[test]
fn safety_limiter_clamps_and_sanitizes_non_finite() {
    let mut preset = BodyPreset {
        y_inf: [1.0e-4; PRESET_BRIDGE_ZONES],
        direct_zone_gain: [1.0; PRESET_RADIATOR_ZONES],
        zone_pos_m: test_preset(0).zone_pos_m,
        reduced_poles: test_preset(0).reduced_poles,
        modes: Vec::new(),
    };
    preset
        .reduced_poles
        .iter_mut()
        .for_each(|p| p.modal_mass_kg = 0.0);

    let mut bank = RadiatorBank::from_preset(&preset, 48_000.0);
    let mut frames = [[0.0_f32; PRESET_RADIATOR_ZONES]; 8];
    frames[0][3] = f32::NAN;
    frames[1][3] = f32::INFINITY;
    frames[2][3] = f32::NEG_INFINITY;
    frames[3][3] = 500.0;
    frames[4][3] = -500.0;
    frames[5][3] = 0.1;

    let mut out_l = [0.0_f32; 8];
    let mut out_r = [0.0_f32; 8];
    bank.process_block(&frames, &mut out_l, &mut out_r);

    for i in 0..8 {
        assert!(out_l[i].is_finite() && out_l[i].abs() <= 1.0);
        assert!(out_r[i].is_finite() && out_r[i].abs() <= 1.0);
    }
    assert_eq!(out_l[0], 0.0);
    assert_eq!(out_l[1], 0.0);
    assert_eq!(out_l[2], 0.0);
    assert!(out_l[5].abs() > 0.01);
}
