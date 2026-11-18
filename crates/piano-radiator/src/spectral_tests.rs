use alloc::vec;
use alloc::vec::Vec;
use piano_simd::RealFft;
use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES,
};

use crate::modal_prepare::rad_tilt_gain;
use crate::{RadiatorBank, CONCERT_GRAND_PRESET_BYTES};

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
    let sum: f64 = mag_sq[k0..=k1].iter().sum();
    sum / ((k1 - k0 + 1) as f64)
}

#[test]
fn rad_tilt_gain_and_per_zone_octave_band_spectral_slope() {
    assert!(rad_tilt_gain(48.0) >= 0.94 && rad_tilt_gain(80.0) >= 0.85);
    assert!(rad_tilt_gain(48.0) > rad_tilt_gain(160.0));
    assert!(rad_tilt_gain(160.0) > rad_tilt_gain(1_000.0));
    assert!(rad_tilt_gain(1_000.0) > rad_tilt_gain(8_000.0));
    let drop_db = 20.0 * (rad_tilt_gain(8_000.0) / rad_tilt_gain(160.0)).log10();
    assert!((-33.0..=-26.0).contains(&drop_db), "drop_db={drop_db:.2}");

    let sr = 48_000.0;
    let fft_len = 32_768_usize;
    let mut modal_preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid preset");
    modal_preset.direct_zone_gain = [0.0; PRESET_RADIATOR_ZONES];
    for m in &mut modal_preset.modes {
        for zd in &mut m.zone_drive[..PRESET_BRIDGE_ZONES] {
            *zd *= 10.0;
        }
    }
    let irs = RadiatorBank::synthesize_ir_for_test(
        &modal_preset,
        sr,
        ListenerPerspective::Player,
        LidState::Open,
    );
    let octs = [
        (80.0, 125.0),
        (125.0, 250.0),
        (250.0, 500.0),
        (500.0, 1_000.0),
        (1_000.0, 2_000.0),
        (2_000.0, 4_000.0),
        (4_000.0, 8_000.0),
    ];
    for (z, pair) in irs.iter().enumerate().take(PRESET_BRIDGE_ZONES) {
        let mono: Vec<f32> = pair[0]
            .iter()
            .zip(pair[1].iter())
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        let mag = fft_mag_sq(&mono, fft_len);
        let bdb: Vec<f64> = octs
            .iter()
            .map(|&(a, b)| 10.0 * band_mean_power(&mag, sr, fft_len, a, b).max(1e-30).log10())
            .collect();
        let mx = octs
            .iter()
            .map(|&(a, b)| (a * b).sqrt().log2())
            .sum::<f64>()
            / 7.0;
        let my = bdb.iter().sum::<f64>() / 7.0;
        let (num, den) = octs
            .iter()
            .zip(bdb.iter())
            .fold((0.0, 0.0), |(n, d), (&(a, b), &y)| {
                let dx = (a * b).sqrt().log2() - mx;
                (n + dx * (y - my), d + dx * dx)
            });
        let slope = num / den;
        let lf_db = 10.0
            * band_mean_power(&mag, sr, fft_len, 45.0, 90.0)
                .max(1e-30)
                .log10()
            - bdb[1];
        let mid_db = 10.0
            * band_mean_power(&mag, sr, fft_len, 300.0, 700.0)
                .max(1e-30)
                .log10()
            - 0.5 * (bdb[1] + bdb[4]);
        eprintln!(
            "Zone {z}: modal slope={slope:+.2} dB/oct, <90Hz={lf_db:+.2} dB, 300-700Hz={mid_db:+.2} dB"
        );
        assert!(
            (-4.2..=-0.3).contains(&slope),
            "z{z} slope={slope:+.2} must tilt downward"
        );
        if z <= 2 {
            assert!(
                (-3.8..=-2.4).contains(&slope),
                "z{z} bass/tenor slope={slope:+.2}"
            );
        }
        assert!(lf_db >= -6.0, "z{z} <90Hz sag={lf_db:+.2} dB");
        assert!(mid_db <= 5.5, "z{z} 300-700Hz hump={mid_db:+.2} dB");
    }

    // Also verify all 8 zones tilt cleanly downward once M1's `stat_treble_focus = 1.85` shelf is removed.
    for m in &mut modal_preset.modes {
        if m.freq_hz > 1_850.0 {
            for zd in &mut m.zone_drive[3..PRESET_BRIDGE_ZONES] {
                *zd /= 1.85;
            }
        }
    }
    let irs_unbiased = RadiatorBank::synthesize_ir_for_test(
        &modal_preset,
        sr,
        ListenerPerspective::Player,
        LidState::Open,
    );
    for (z, pair) in irs_unbiased.iter().enumerate().take(PRESET_BRIDGE_ZONES) {
        let mono: Vec<f32> = pair[0]
            .iter()
            .zip(pair[1].iter())
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        let mag = fft_mag_sq(&mono, fft_len);
        let bdb: Vec<f64> = octs
            .iter()
            .map(|&(a, b)| 10.0 * band_mean_power(&mag, sr, fft_len, a, b).max(1e-30).log10())
            .collect();
        let mx = octs
            .iter()
            .map(|&(a, b)| (a * b).sqrt().log2())
            .sum::<f64>()
            / 7.0;
        let my = bdb.iter().sum::<f64>() / 7.0;
        let (num, den) = octs
            .iter()
            .zip(bdb.iter())
            .fold((0.0, 0.0), |(n, d), (&(a, b), &y)| {
                let dx = (a * b).sqrt().log2() - mx;
                (n + dx * (y - my), d + dx * dx)
            });
        let slope = num / den;
        assert!(
            (-4.2..=-1.0).contains(&slope),
            "unbiased z{z} slope={slope:+.2} out of [-4.2, -1.0]"
        );
    }
}
