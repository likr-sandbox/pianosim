//! Zero-dependency spectral analysis utilities for scenario acoustic verification.

use std::f64::consts::TAU;

use piano_simd::{Complex, RealFft};

/// Stereo downmix to monaural signal.
#[must_use]
pub fn mono_mix(l: &[f32], r: &[f32]) -> Vec<f32> {
    l.iter()
        .zip(r.iter())
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect()
}

/// Computes root-mean-square (RMS) level over a slice.
#[must_use]
pub fn slice_rms(samples: &[f32]) -> f64 {
    let sum_sq: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / (samples.len().max(1) as f64)).sqrt()
}

/// Computes maximum absolute peak over a slice.
#[must_use]
pub fn peak_abs(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0_f32, |m, &x| m.max(x.abs()))
}

/// Computes Hann-windowed forward real FFT spectrum.
#[must_use]
pub fn hann_windowed_spectrum(samples: &[f32]) -> Vec<Complex> {
    let n = samples.len();
    assert!(n.is_power_of_two(), "FFT size must be power of two");
    let mut windowed = vec![0.0_f32; n];
    let denom = (n - 1).max(1) as f64;
    for (i, (&s, dst)) in samples.iter().zip(windowed.iter_mut()).enumerate() {
        let w = 0.5 * (1.0 - (TAU * (i as f64) / denom).cos());
        *dst = (f64::from(s) * w) as f32;
    }
    let mut rfft = RealFft::new(n);
    let mut spec = vec![Complex::ZERO; rfft.spectrum_len()];
    rfft.forward(&windowed, &mut spec);
    spec
}

/// Computes spectral centroid in Hertz.
#[must_use]
pub fn spectral_centroid_hz(samples: &[f32], sr: f64) -> f64 {
    let spec = hann_windowed_spectrum(samples);
    let bin_hz = sr / (samples.len() as f64);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (k, c) in spec.iter().enumerate().skip(1) {
        let mag = f64::from(c.norm_sq()).sqrt();
        num += (k as f64) * bin_hz * mag;
        den += mag;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Sums power across a frequency band `[lo_hz, hi_hz]`.
#[allow(dead_code)]
#[must_use]
pub fn band_power(spec: &[Complex], n_fft: usize, sr: f64, lo_hz: f64, hi_hz: f64) -> f64 {
    let bin_hz = sr / (n_fft as f64);
    let lo_bin = ((lo_hz / bin_hz).floor() as usize).max(1);
    let hi_bin = ((hi_hz / bin_hz).ceil() as usize).min(spec.len().saturating_sub(1));
    if lo_bin <= hi_bin && hi_bin < spec.len() {
        spec[lo_bin..=hi_bin]
            .iter()
            .map(|c| f64::from(c.norm_sq()))
            .sum()
    } else {
        0.0
    }
}

/// Blackman-Harris 4-term minimum sidelobe window.
#[must_use]
pub fn blackman_harris_window(samples: &[f32]) -> Vec<f64> {
    let n = samples.len();
    let denom = (n - 1).max(1) as f64;
    samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = TAU * (i as f64) / denom;
            let w =
                0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
            f64::from(x) * w
        })
        .collect()
}

/// Goertzel algorithm single-frequency power calculation.
#[must_use]
pub fn goertzel_power_f64(win: &[f64], freq_hz: f64, sr: f64) -> f64 {
    let coeff = 2.0 * (TAU * freq_hz / sr).cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in win {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(1.0e-30)
}

/// High-resolution fundamental frequency estimator via Goertzel and parabolic peak refinement.
#[must_use]
pub fn measure_f0_hz(mono: &[f32], f_nominal_hz: f64, sr: f64) -> f64 {
    let win = blackman_harris_window(mono);
    let mut best_f = f_nominal_hz;
    let mut best_p = 0.0_f64;
    for step in -200..=200 {
        let f_try = f_nominal_hz * 2.0_f64.powf((step as f64 * 0.5) / 1200.0);
        let p = goertzel_power_f64(&win, f_try, sr);
        if p > best_p {
            best_p = p;
            best_f = f_try;
        }
    }
    let dc = 0.5_f64;
    let r = 2.0_f64.powf(dc / 1200.0);
    let p_lo = goertzel_power_f64(&win, best_f / r, sr);
    let p_mid = goertzel_power_f64(&win, best_f, sr);
    let p_hi = goertzel_power_f64(&win, best_f * r, sr);
    let denom_q = p_lo - 2.0 * p_mid + p_hi;
    let shift = if denom_q < 0.0 {
        (0.5 * (p_lo - p_hi) / denom_q).clamp(-1.0, 1.0) * dc
    } else {
        0.0
    };
    best_f * 2.0_f64.powf(shift / 1200.0)
}

/// Ordinary least squares linear regression slope.
#[must_use]
pub fn ols_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for &(x, y) in points {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Calculates logarithmic decay slope in dB per second.
#[must_use]
pub fn decay_slope_db_per_s(
    mono: &[f32],
    sr: f64,
    t_start_s: f64,
    t_end_s: f64,
    win_s: f64,
) -> f64 {
    let win = (win_s * sr).round() as usize;
    let start_idx = (t_start_s * sr).round() as usize;
    let end_idx = ((t_end_s * sr).round() as usize).min(mono.len());
    let mut pts = Vec::new();
    let mut pos = start_idx;
    while pos + win <= end_idx {
        let rms = slice_rms(&mono[pos..pos + win]);
        let t_mid = (pos as f64 + 0.5 * (win as f64)) / sr;
        pts.push((t_mid, 20.0 * rms.max(1e-12).log10()));
        pos += win;
    }
    if pts.len() >= 3 {
        -ols_slope(&pts)
    } else {
        0.0
    }
}
