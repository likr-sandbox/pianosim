//! Hammer strike, ADAA/oversampling aliasing, Timoshenko shank, and Stulov memory tests.

use piano_spec::VoicingSpec;

use crate::math::pow_nonneg_f64;
use crate::tests_common::{
    measure_contact_duration_ms, spectral_centroid_and_band, TestModalString,
};
use crate::ActionBank;

#[test]
fn contact_durations_match_bass_mid_treble_physical_targets() {
    let (dur_a0_51, peak_a0_51) = measure_contact_duration_ms(0, 51_602, 48_000.0);
    let (dur_a4_51, peak_a4_51) = measure_contact_duration_ms(48, 51_602, 48_000.0);
    let (dur_c8_51, peak_c8_51) = measure_contact_duration_ms(87, 51_602, 48_000.0);
    let (dur_a0, peak_a0) = measure_contact_duration_ms(0, 38_000, 48_000.0);
    let (dur_a4, peak_a4) = measure_contact_duration_ms(48, 38_000, 48_000.0);
    let (dur_c8, peak_c8) = measure_contact_duration_ms(87, 38_000, 48_000.0);

    assert!(peak_a0_51 > 1.0 && peak_a4_51 > 1.0 && peak_c8_51 > 0.5);
    assert!(
        (2.0..=5.0).contains(&dur_a0_51),
        "A0(51602) dur={dur_a0_51} ms"
    );
    assert!(
        (0.85..=2.4).contains(&dur_a4_51),
        "A4(51602) dur={dur_a4_51} ms"
    );
    assert!(
        (0.06..=0.45).contains(&dur_c8_51),
        "C8(51602) dur={dur_c8_51} ms"
    );
    assert!(
        dur_a0_51 > dur_a4_51 && dur_a4_51 > dur_c8_51,
        "{dur_a0_51} > {dur_a4_51} > {dur_c8_51}"
    );

    assert!(peak_a0 > 1.0 && peak_a4 > 1.0 && peak_c8 > 0.5);
    assert!((2.0..=5.0).contains(&dur_a0), "A0 contact dur={dur_a0} ms");
    assert!((0.85..=2.4).contains(&dur_a4), "A4 contact dur={dur_a4} ms");
    assert!(
        (0.06..=0.45).contains(&dur_c8),
        "C8 contact dur={dur_c8} ms"
    );
    assert!(
        dur_a0 > dur_a4 && dur_a4 > dur_c8,
        "{dur_a0} > {dur_a4} > {dur_c8}"
    );
}

#[test]
fn velocity_sweep_is_strictly_monotonic_in_peak_rms_and_brightness() {
    let voicing = VoicingSpec::default();
    let vels = [6_000_u16, 20_000, 38_000, 51_602, 65_535];
    let mut prev_peak = 0.0_f32;
    let mut prev_rms = 0.0_f32;
    let mut prev_centroid = 0.0_f32;

    for &vel in &vels {
        let mut bank = ActionBank::new(48_000.0, &voicing);
        let mut s = TestModalString::new(48, 48_000.0, &voicing);
        bank.note_on(48, vel);

        let mut pulse = vec![0.0_f32; 512];
        let mut peak = 0.0_f32;
        let mut str_sq = 0.0_f64;
        for out in &mut pulse {
            let (y, v) = s.strike_port();
            let f = bank.step_hammer_sample(48, y, v, s.r_port_strike);
            *out = f;
            peak = peak.max(f);
            s.step_forces(f, 0.0);
            let (_, v_next) = s.strike_port();
            str_sq += f64::from(v_next) * f64::from(v_next);
        }
        let rms = (str_sq / 512.0).sqrt() as f32;
        let (centroid, _) = spectral_centroid_and_band(&pulse, 48_000.0, 100.0, 20_000.0);

        assert!(peak > prev_peak, "vel={vel} peak={peak} <= {prev_peak}");
        assert!(rms > prev_rms, "vel={vel} rms={rms} <= {prev_rms}");
        assert!(
            centroid > prev_centroid,
            "vel={vel} centroid={centroid} <= {prev_centroid}"
        );
        prev_peak = peak;
        prev_rms = rms;
        prev_centroid = centroid;
    }
}

#[test]
fn adaa_and_8x_decimation_suppresses_nyquist_aliasing_on_fff_treble_strike() {
    let voicing = VoicingSpec::default();
    let mut bank = ActionBank::new(48_000.0, &voicing);
    let mut s = TestModalString::new(72, 48_000.0, &voicing);
    bank.note_on(72, 65_535);

    let n = 512;
    let mut adaa_pulse = vec![0.0_f32; n];
    for slot in &mut adaa_pulse {
        let (y, v) = s.strike_port();
        let f = bank.step_hammer_sample(72, y, v, s.r_port_strike);
        *slot = f;
        s.step_forces(f, 0.0);
    }

    // Compare host-rate ADAA+decimation highband against non-oversampled naive evaluation
    // of the same key-72 physical felt stiffness & mass.
    let spec72 = crate::felt_stulov::KeyFeltSpec::for_key(72, 48_000.0, &voicing);
    let m_h72 =
        f64::from(crate::shank_timoshenko::ShankParams::for_key(72, 48_000.0).hammer_mass_kg);
    let mut s_naive = TestModalString::new(72, 48_000.0, &voicing);
    let mut y_h = -2.0e-5_f64;
    let mut v_h = 4.6_f64;
    let dt = 1.0 / 48_000.0;
    let mut naive_pulse = vec![0.0_f32; n];
    for slot in &mut naive_pulse {
        let (y_s, _) = s_naive.strike_port();
        let d = (y_h - f64::from(y_s)).max(0.0);
        let f = (f64::from(spec72.k0_si) * pow_nonneg_f64(d, f64::from(spec72.exponent_p))) as f32;
        *slot = f.min(600.0);
        let a = -f64::from(*slot) / m_h72;
        y_h += v_h * dt + 0.5 * a * dt * dt;
        v_h += a * dt;
        s_naive.step_forces(*slot, 0.0);
    }

    let (_, adaa_low) = spectral_centroid_and_band(&adaa_pulse, 48_000.0, 100.0, 6_000.0);
    let (_, adaa_nyq) = spectral_centroid_and_band(&adaa_pulse, 48_000.0, 17_000.0, 23_800.0);
    let (_, naive_low) = spectral_centroid_and_band(&naive_pulse, 48_000.0, 100.0, 6_000.0);
    let (_, naive_nyq) = spectral_centroid_and_band(&naive_pulse, 48_000.0, 17_000.0, 23_800.0);

    let adaa_ratio = adaa_nyq / adaa_low.max(1e-12);
    let naive_ratio = naive_nyq / naive_low.max(1e-12);
    assert!(
        adaa_ratio < 1.0e-3,
        "adaa nyquist ratio too high: {adaa_ratio}"
    );
    assert!(
        adaa_ratio * 20.0 < naive_ratio,
        "adaa_ratio={adaa_ratio} vs naive_ratio={naive_ratio}"
    );
}

#[test]
fn timoshenko_shank_induces_1p5_to_3p5khz_modulation_vs_rigid() {
    let voicing = VoicingSpec::default();
    let run = |flex: bool| -> (Vec<f32>, f32) {
        let mut bank = ActionBank::new(48_000.0, &voicing);
        bank.set_shank_flex_enabled(48, flex);
        let mut s = TestModalString::new(48, 48_000.0, &voicing);
        bank.note_on(48, 58_000);
        let mut f_vec = vec![0.0_f32; 512];
        let mut peak = 0.0_f32;
        for out in &mut f_vec {
            let (y, v) = s.strike_port();
            let f = bank.step_hammer_sample(48, y, v, s.r_port_strike);
            *out = f;
            peak = peak.max(f);
            s.step_forces(f, 0.0);
        }
        (f_vec, peak)
    };

    let (flex_f, flex_peak) = run(true);
    let (rigid_f, rigid_peak) = run(false);
    let diff: f32 = flex_f
        .iter()
        .zip(rigid_f.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(diff > 0.2, "flex vs rigid difference too small: {diff}");
    assert!(
        (flex_peak - rigid_peak).abs() > 0.05,
        "flex={flex_peak}, rigid={rigid_peak}"
    );

    let (_, band_flex) = spectral_centroid_and_band(&flex_f, 48_000.0, 1_500.0, 3_500.0);
    let (_, band_rigid) = spectral_centroid_and_band(&rigid_f, 48_000.0, 1_500.0, 3_500.0);
    assert!(
        (band_flex - band_rigid).abs() > 1e-2 * band_rigid,
        "band_flex={band_flex}, band_rigid={band_rigid}"
    );
}

#[test]
fn stulov_12hz_ribattuto_memory_evolves_persists_across_note_off_and_clears_on_reset() {
    let run_ribattuto = |mem_enabled: bool| -> (Vec<f32>, Vec<f32>, ActionBank) {
        let voicing = VoicingSpec::default();
        let mut bank = ActionBank::new(48_000.0, &voicing);
        bank.set_memory_enabled(mem_enabled);
        let mut s = TestModalString::new(48, 48_000.0, &voicing);

        let mut peaks = Vec::new();
        let mut centroids = Vec::new();
        // 12 Hz => 4000 host samples per strike (500 active + 3500 idle decay)
        for strike in 0..6 {
            bank.note_on(48, 42_000);
            let mut pulse = vec![0.0_f32; 512];
            let mut peak = 0.0_f32;
            for (i, slot) in pulse.iter_mut().enumerate() {
                let (y, v) = s.strike_port();
                let f = bank.step_hammer_sample(48, y, v, s.r_port_strike);
                *slot = f;
                peak = peak.max(f);
                s.step_forces(f, 0.0);
                if i == 240 {
                    bank.note_off(48, 32_768);
                }
            }
            if strike < 5 {
                bank.decay_idle_memory(4_000 - 512);
                for _ in 0..(4_000 - 512) {
                    s.step_forces(0.0, 0.0);
                }
            }
            let (c, _) = spectral_centroid_and_band(&pulse, 48_000.0, 100.0, 15_000.0);
            peaks.push(peak);
            centroids.push(c);
        }
        (peaks, centroids, bank)
    };

    let (mem_peaks, mem_cents, mut bank_mem) = run_ribattuto(true);
    let (nomem_peaks, nomem_cents, _) = run_ribattuto(false);

    // Memory-enabled restrike must diverge positively in peak force and brightness.
    assert!(
        mem_peaks[5] > mem_peaks[0] * 1.05,
        "mem peaks: first={} last={}",
        mem_peaks[0],
        mem_peaks[5]
    );
    assert!(
        mem_cents[5] > mem_cents[0],
        "mem centroid: first={} last={}",
        mem_cents[0],
        mem_cents[5]
    );
    assert!(
        mem_peaks[5] > nomem_peaks[5] * 1.05,
        "mem={} vs nomem={}",
        mem_peaks[5],
        nomem_peaks[5]
    );
    assert!(
        mem_cents[5] > nomem_cents[5],
        "mem_c={} vs nomem_c={}",
        mem_cents[5],
        nomem_cents[5]
    );

    // Slow memory persists across note_off and is cleared only by reset().
    bank_mem.note_off(48, 65_535);
    assert!(bank_mem.felt_memory(48) > 0.05);
    bank_mem.reset();
    assert_eq!(bank_mem.felt_memory(48), 0.0);
}

#[test]
fn agraffe_delay_induces_multi_peak_reflections_at_48khz_and_96khz() {
    use piano_spec::{build_scale_design, key_tuned_f0_hz, CHOIR_LAYOUTS};

    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);

    for &(key_idx, label) in &[(39_usize, "C4"), (63_usize, "C6")] {
        let f0 = key_tuned_f0_hz(key_idx, &specs, &voicing);
        let tau_agg_ms = 1000.0 * crate::agraffe_delay::hammer_beta_for_key(key_idx) / f0;

        for &sr in &[48_000.0_f32, 96_000.0] {
            let mut bank = ActionBank::new(sr, &voicing);
            assert_eq!(bank.hammer_compression_norm(key_idx), 0.0);
            assert_eq!(bank.last_hammer_forces(key_idx), (0.0, [0.0; 8]));
            assert_eq!(bank.last_hammer_forces(99), (0.0, [0.0; 8]));

            let mut s = TestModalString::new(key_idx, f64::from(sr), &voicing);
            bank.note_on(key_idx, 48_000);

            let n_host = if sr > 60_000.0 { 1024 } else { 512 };
            let mut sub_f = Vec::with_capacity(n_host * 8);
            let mut max_u = 0.0_f32;
            for _ in 0..n_host {
                let (y, v) = s.strike_port();
                let f = bank.step_hammer_sample(key_idx, y, v, s.r_port_strike);
                let (last_host, subs) = bank.last_hammer_forces(key_idx);
                assert_eq!(last_host, f);
                assert!(subs.iter().all(|&x| x.is_finite() && x >= 0.0));
                sub_f.extend_from_slice(&subs);
                max_u = max_u.max(bank.hammer_compression_norm(key_idx));
                s.step_forces(f, 0.0);
            }
            assert!(max_u > 0.05 && max_u <= 1.0, "{label}@{sr}: max_u={max_u}");
            assert_eq!(bank.hammer_compression_norm(key_idx), 0.0);

            let pk = sub_f.iter().copied().fold(0.0_f32, f32::max);
            assert!(pk > 5.0, "{label}@{sr}: peak={pk}");
            let sr_sub = f64::from(sr) * 8.0;
            let min_sep = ((0.6 * tau_agg_ms * 1.0e-3) * sr_sub).max(4.0) as usize;
            let mut peaks = Vec::new();
            for i in 2..sub_f.len().saturating_sub(2) {
                if sub_f[i] > 0.15 * pk
                    && sub_f[i] >= sub_f[i - 1]
                    && sub_f[i] >= sub_f[i + 1]
                    && sub_f[i] > sub_f[i - 2]
                    && sub_f[i] > sub_f[i + 2]
                    && peaks.last().is_none_or(|&p: &usize| i - p >= min_sep)
                {
                    peaks.push(i);
                }
            }
            assert!(
                peaks.len() >= 2,
                "{label}@{sr}: expected >=2 agraffe peaks (tau_agg={tau_agg_ms:.3}ms), got {:?}",
                peaks
            );
            let dt_ms = 1000.0 * ((peaks[1] - peaks[0]) as f64) / sr_sub;
            assert!(
                dt_ms >= 0.55 * tau_agg_ms && dt_ms <= 2.2 * tau_agg_ms,
                "{label}@{sr}: inter-peak {dt_ms:.3}ms vs tau_agg={tau_agg_ms:.3}ms"
            );
        }
    }

    // Verify agraffe delay creates genuine multi-peak ripples on a non-reflecting resistive port.
    for &sr in &[48_000.0_f32, 96_000.0] {
        let k = 63_usize;
        let f0 = key_tuned_f0_hz(k, &specs, &voicing);
        let n_str = f64::from(CHOIR_LAYOUTS[k].strings_per_key.max(1));
        let m_str = specs[k].linear_density_kg_m * specs[k].speaking_length_m;
        let z_choir = n_str * 2.0 * f0 * m_str;
        let r_port = (1.0 / (2.0 * z_choir)) as f32;

        let run_resistive = |enable_agg: bool| -> Vec<f32> {
            let mut bank = ActionBank::new(sr, &voicing);
            if !enable_agg {
                bank.set_agraffe_delay_samples(k, 1.0, 0.0);
            }
            bank.note_on(k, 48_000);
            let mut y_s = 0.0_f32;
            let dt = 1.0 / sr;
            let n_host = if sr > 60_000.0 { 256 } else { 128 };
            let mut out = Vec::with_capacity(n_host * 8);
            for _ in 0..n_host {
                let f = bank.step_hammer_sample(k, y_s, 0.0, r_port);
                y_s += f * r_port * dt;
                out.extend_from_slice(&bank.last_hammer_forces(k).1);
            }
            out
        };

        let s_on = run_resistive(true);
        let s_off = run_resistive(false);
        let max_diff = s_on
            .iter()
            .zip(s_off.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1.0,
            "C6@{sr}: expected agraffe reflection delta > 1.0 N on resistive port, got {max_diff}"
        );
    }
}

#[test]
fn a0_at_a4_392hz_and_96khz_never_overflows_agraffe_buffer() {
    let mut low_voicing = VoicingSpec {
        a4_freq_hz: piano_spec::A4_FREQ_MIN_HZ as f32,
        stretch_amount: 2.0,
        ..VoicingSpec::default()
    };
    low_voicing.note_offsets[0].pitch_cents = -100.0;

    let mut bank = ActionBank::new(96_000.0, &low_voicing);
    let mut s = TestModalString::new(0, 96_000.0, &low_voicing);
    bank.note_on(0, 65_535);

    let mut peak = 0.0_f32;
    for _ in 0..1_200 {
        let (y, v) = s.strike_port();
        let f = bank.step_hammer_sample(0, y, v, s.r_port_strike);
        assert!(f.is_finite() && f >= 0.0);
        peak = peak.max(f);
        s.step_forces(f, 0.0);
    }
    assert!(peak > 5.0, "A0 peak={peak}");

    let high_voicing = VoicingSpec {
        a4_freq_hz: piano_spec::A4_FREQ_MAX_HZ as f32,
        ..VoicingSpec::default()
    };
    bank.update_voicing_in_place(&high_voicing);
    bank.update_voicing_in_place(&low_voicing);
    bank.reset();
    assert_eq!(bank.last_hammer_forces(0), (0.0, [0.0; 8]));
}
