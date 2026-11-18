//! Analytical binaural HRTF synthesis: Woodworth spherical-head ITD,
//! Brown-Duda bilinear ILD head-shadow filter, and pinna elevation cues.

use crate::math::{
    abs_f64, clamp_f64, exp_f64, max_f64, powf_f64, round_f64, sin_cos_f64, sinc_f64, sqrt_f64,
    tan_f64, FRAC_PI_2, PI,
};

/// Average adult human head radius `[m]` for Woodworth / Brown-Duda models.
pub(crate) const HEAD_RADIUS_M: f64 = 0.0875;
/// Speed of sound in dry air at 20 deg C `[m/s]`.
pub(crate) const SPEED_OF_SOUND_MPS: f64 = 343.0;

/// Direct-Form II Transposed biquad filter (`y[n] = b0 x[n] + ...`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Biquad {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Biquad {
    #[inline]
    pub(crate) const fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    #[inline]
    pub(crate) fn is_identity(&self) -> bool {
        self.b0 == 1.0 && self.b1 == 0.0 && self.b2 == 0.0 && self.a1 == 0.0 && self.a2 == 0.0
    }

    /// First-order shelf `(1 + alpha * s / omega0) / (1 + s / omega0)` discretized
    /// via bilinear transform with exact frequency prewarping at `omega0`.
    pub(crate) fn first_order_head_shelf(omega0: f64, alpha: f64, sample_rate: f64) -> Self {
        let clamped_sr = max_f64(sample_rate, 8_000.0);
        let w_norm = clamp_f64(omega0 / (2.0 * clamped_sr), 1.0e-5, FRAC_PI_2 * 0.90);
        let gamma = tan_f64(w_norm);
        let a0 = gamma + 1.0;
        let inv_a0 = 1.0 / a0;
        Self {
            b0: ((gamma + alpha) * inv_a0) as f32,
            b1: ((gamma - alpha) * inv_a0) as f32,
            b2: 0.0,
            a1: ((gamma - 1.0) * inv_a0) as f32,
            a2: 0.0,
        }
    }

    /// Second-order bilinear lowpass filter (`cutoff_hz`, resonance `q`).
    pub(crate) fn lowpass_2nd(cutoff_hz: f64, q: f64, sample_rate: f64) -> Self {
        let sr = max_f64(sample_rate, 8_000.0);
        if cutoff_hz >= 0.47 * sr {
            return Self::identity();
        }
        let fc = clamp_f64(cutoff_hz, 20.0, 0.46 * sr);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w, cos_w) = sin_cos_f64(w0);
        let alpha = sin_w / (2.0 * max_f64(q, 0.25));
        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;
        let b1 = 1.0 - cos_w;
        Self {
            b0: (0.5 * b1 * inv_a0) as f32,
            b1: (b1 * inv_a0) as f32,
            b2: (0.5 * b1 * inv_a0) as f32,
            a1: (-2.0 * cos_w * inv_a0) as f32,
            a2: ((1.0 - alpha) * inv_a0) as f32,
        }
    }

    /// Second-order bilinear highpass filter (`cutoff_hz`, resonance `q`).
    pub(crate) fn highpass_2nd(cutoff_hz: f64, q: f64, sample_rate: f64) -> Self {
        let sr = max_f64(sample_rate, 8_000.0);
        if cutoff_hz <= 0.5 {
            return Self::identity();
        }
        let fc = clamp_f64(cutoff_hz, 20.0, 0.46 * sr);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w, cos_w) = sin_cos_f64(w0);
        let alpha = sin_w / (2.0 * max_f64(q, 0.25));
        let a0 = 1.0 + alpha;
        let inv_a0 = 1.0 / a0;
        let one_plus_cos = 1.0 + cos_w;
        Self {
            b0: (0.5 * one_plus_cos * inv_a0) as f32,
            b1: (-one_plus_cos * inv_a0) as f32,
            b2: (0.5 * one_plus_cos * inv_a0) as f32,
            a1: (-2.0 * cos_w * inv_a0) as f32,
            a2: ((1.0 - alpha) * inv_a0) as f32,
        }
    }

    /// Audio-EQ-Cookbook low-shelf filter.
    pub(crate) fn low_shelf(freq_hz: f64, gain_db: f64, slope: f64, sample_rate: f64) -> Self {
        let sr = max_f64(sample_rate, 8_000.0);
        if abs_f64(gain_db) < 1.0e-3 || freq_hz >= 0.47 * sr {
            return Self::identity();
        }
        let fc = clamp_f64(freq_hz, 40.0, 0.46 * sr);
        let a_lin = powf_f64(10.0, gain_db / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w, cos_w) = sin_cos_f64(w0);
        let s_clamped = clamp_f64(slope, 0.1, 2.0);
        let inside = max_f64(0.0, (a_lin + 1.0 / a_lin) * (1.0 / s_clamped - 1.0) + 2.0);
        let alpha = 0.5 * sin_w * sqrt_f64(inside);
        let two_sqrt_a_alpha = 2.0 * sqrt_f64(a_lin) * alpha;
        let a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w + two_sqrt_a_alpha;
        let inv_a0 = 1.0 / a0;
        Self {
            b0: (a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w + two_sqrt_a_alpha) * inv_a0)
                as f32,
            b1: (2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w) * inv_a0) as f32,
            b2: (a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w - two_sqrt_a_alpha) * inv_a0)
                as f32,
            a1: (-2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w) * inv_a0) as f32,
            a2: (((a_lin + 1.0) + (a_lin - 1.0) * cos_w - two_sqrt_a_alpha) * inv_a0) as f32,
        }
    }

    /// Audio-EQ-Cookbook high-shelf filter.
    pub(crate) fn high_shelf(freq_hz: f64, gain_db: f64, slope: f64, sample_rate: f64) -> Self {
        let sr = max_f64(sample_rate, 8_000.0);
        if abs_f64(gain_db) < 1.0e-3 || freq_hz >= 0.47 * sr {
            return Self::identity();
        }
        let fc = clamp_f64(freq_hz, 40.0, 0.46 * sr);
        let a_lin = powf_f64(10.0, gain_db / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w, cos_w) = sin_cos_f64(w0);
        let s_clamped = clamp_f64(slope, 0.1, 2.0);
        let inside = max_f64(0.0, (a_lin + 1.0 / a_lin) * (1.0 / s_clamped - 1.0) + 2.0);
        let alpha = 0.5 * sin_w * sqrt_f64(inside);
        let two_sqrt_a_alpha = 2.0 * sqrt_f64(a_lin) * alpha;
        let a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w + two_sqrt_a_alpha;
        let inv_a0 = 1.0 / a0;
        Self {
            b0: (a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w + two_sqrt_a_alpha) * inv_a0)
                as f32,
            b1: (-2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w) * inv_a0) as f32,
            b2: (a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w - two_sqrt_a_alpha) * inv_a0)
                as f32,
            a1: (2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w) * inv_a0) as f32,
            a2: (((a_lin + 1.0) - (a_lin - 1.0) * cos_w - two_sqrt_a_alpha) * inv_a0) as f32,
        }
    }

    /// Audio-EQ-Cookbook peaking EQ / parametric notch filter.
    pub(crate) fn peaking_eq(freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) -> Self {
        let sr = max_f64(sample_rate, 8_000.0);
        if abs_f64(gain_db) < 1.0e-3 || freq_hz >= 0.47 * sr {
            return Self::identity();
        }
        let fc = clamp_f64(freq_hz, 40.0, 0.46 * sr);
        let a_lin = powf_f64(10.0, gain_db / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let (sin_w, cos_w) = sin_cos_f64(w0);
        let alpha = sin_w / (2.0 * max_f64(q, 0.25));
        let a0 = 1.0 + alpha / a_lin;
        let inv_a0 = 1.0 / a0;
        Self {
            b0: ((1.0 + alpha * a_lin) * inv_a0) as f32,
            b1: (-2.0 * cos_w * inv_a0) as f32,
            b2: ((1.0 - alpha * a_lin) * inv_a0) as f32,
            a1: (-2.0 * cos_w * inv_a0) as f32,
            a2: ((1.0 - alpha / a_lin) * inv_a0) as f32,
        }
    }

    /// Causal in-place cascade of up to 6 biquads in a single memory pass.
    pub(crate) fn process_chain_in_place(chain: &[Self], buf: &mut [f32]) {
        let mut coeffs = [(1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64); 6];
        let mut z1 = [0.0_f64; 6];
        let mut z2 = [0.0_f64; 6];
        let mut active = 0_usize;
        for f in chain {
            if !f.is_identity() && active < 6 {
                coeffs[active] = (
                    f.b0 as f64,
                    f.b1 as f64,
                    f.b2 as f64,
                    f.a1 as f64,
                    f.a2 as f64,
                );
                active += 1;
            }
        }
        if active == 0 {
            return;
        }
        for sample in buf.iter_mut() {
            let mut x = *sample as f64;
            let mut i = 0;
            while i < active {
                let (b0, b1, b2, a1, a2) = coeffs[i];
                let y = b0 * x + z1[i];
                z1[i] = b1 * x - a1 * y + z2[i];
                z2[i] = b2 * x - a2 * y;
                x = y;
                i += 1;
            }
            *sample = x as f32;
        }
    }
}

/// Stateful Direct-Form II Transposed biquad processor for real-time blocks.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadState {
    #[inline]
    pub(crate) fn process(&mut self, c: &Biquad, x: f32) -> f32 {
        let y = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * y + self.z2;
        self.z2 = c.b2 * x - c.a2 * y;
        y
    }

    #[inline]
    pub(crate) fn flush_denormals(&mut self) {
        if self.z1.abs() < 1.0e-20 {
            self.z1 = 0.0;
        }
        if self.z2.abs() < 1.0e-20 {
            self.z2 = 0.0;
        }
    }

    #[inline]
    pub(crate) fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Per-ear spatial delay and binaural HRTF filter chain for one zone.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BinauralEarParams {
    pub delay_samples: f64,
    pub head_shadow: Biquad,
    pub concha: Biquad,
    pub pinna_notch: Biquad,
}

/// Full left/right binaural cue parameters derived from 3D azimuth & elevation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BinauralPairParams {
    pub left: BinauralEarParams,
    pub right: BinauralEarParams,
}

/// Compute Woodworth spherical-head ITD magnitude in seconds for angle magnitude `|theta|`.
#[inline]
pub(crate) fn woodworth_itd_seconds(abs_azimuth_rad: f64, elevation_rad: f64) -> f64 {
    let abs_az = clamp_f64(abs_azimuth_rad, 0.0, PI);
    let (_, cos_el) = sin_cos_f64(clamp_f64(elevation_rad, -FRAC_PI_2, FRAC_PI_2));
    let arc = if abs_az <= FRAC_PI_2 {
        let (sin_az, _) = sin_cos_f64(abs_az);
        abs_az + sin_az
    } else {
        let back = PI - abs_az;
        let (sin_b, _) = sin_cos_f64(back);
        back + sin_b
    };
    (HEAD_RADIUS_M / SPEED_OF_SOUND_MPS) * arc * max_f64(0.15, cos_el)
}

/// Compute analytical Woodworth ITD + Brown-Duda ILD + pinna notch for `(azimuth, elevation)`.
///
/// Convention: `azimuth_rad < 0` is to the listener's LEFT (`L` arrives earlier and louder),
/// `azimuth_rad > 0` is to the listener's RIGHT (`R` arrives earlier and louder).
pub(crate) fn compute_binaural_params(
    azimuth_rad: f64,
    elevation_rad: f64,
    sample_rate: f64,
) -> BinauralPairParams {
    let abs_az = abs_f64(azimuth_rad);
    let itd_mag_s = woodworth_itd_seconds(abs_az, elevation_rad);

    let (delay_l_s, delay_r_s) = if azimuth_rad < 0.0 {
        (0.0, itd_mag_s)
    } else {
        (itd_mag_s, 0.0)
    };

    let (sin_az, _) = sin_cos_f64(azimuth_rad);
    let (sin_el, cos_el) = sin_cos_f64(clamp_f64(elevation_rad, -FRAC_PI_2, FRAC_PI_2));
    // Cosine of angle from ear outward normal: `sin(-az)` for left ear, `sin(+az)` for right ear.
    let ear_proj_l = -sin_az * max_f64(0.2, cos_el);
    let ear_proj_r = sin_az * max_f64(0.2, cos_el);

    let omega0 = 2.0 * SPEED_OF_SOUND_MPS / HEAD_RADIUS_M;
    // Asymmetric Brown-Duda shelf depth (~ +4.5 dB ipsilateral, -9.5 dB contralateral).
    let alpha_l = exp_f64(0.90 * ear_proj_l);
    let alpha_r = exp_f64(0.90 * ear_proj_r);

    let head_shadow_l = Biquad::first_order_head_shelf(omega0, alpha_l, sample_rate);
    let head_shadow_r = Biquad::first_order_head_shelf(omega0, alpha_r, sample_rate);

    let concha_gain_l = 2.2 + 0.8 * ear_proj_l;
    let concha_gain_r = 2.2 + 0.8 * ear_proj_r;
    let concha_l = Biquad::peaking_eq(3_300.0, 1.8, concha_gain_l, sample_rate);
    let concha_r = Biquad::peaking_eq(3_300.0, 1.8, concha_gain_r, sample_rate);

    // Pinna N1 notch rises from ~6.5 kHz below ear level toward ~9.0 kHz overhead.
    let notch_hz = 7_100.0 + 1_900.0 * sin_el;
    let notch_l = Biquad::peaking_eq(notch_hz, 3.2, -5.5, sample_rate);
    let notch_r = Biquad::peaking_eq(notch_hz * 1.03, 3.2, -5.5, sample_rate);

    BinauralPairParams {
        left: BinauralEarParams {
            delay_samples: delay_l_s * sample_rate,
            head_shadow: head_shadow_l,
            concha: concha_l,
            pinna_notch: notch_l,
        },
        right: BinauralEarParams {
            delay_samples: delay_r_s * sample_rate,
            head_shadow: head_shadow_r,
            concha: concha_r,
            pinna_notch: notch_r,
        },
    }
}

/// Apply a strictly causal fractional sample delay (`>= 0`) in-place via `scratch`.
pub(crate) fn apply_causal_fractional_delay(
    buf: &mut [f32],
    scratch: &mut [f32],
    delay_samples: f64,
) {
    if delay_samples <= 1.0e-5 || buf.is_empty() {
        return;
    }
    let n_len = buf.len().min(scratch.len());
    scratch[..n_len].copy_from_slice(&buf[..n_len]);

    if delay_samples <= 1.0 {
        let frac = delay_samples as f32;
        let one_minus = 1.0 - frac;
        buf[0] = one_minus * scratch[0];
        let mut n = 1;
        while n < n_len {
            buf[n] = one_minus * scratch[n] + frac * scratch[n - 1];
            n += 1;
        }
        return;
    }

    let d_int = round_f64(delay_samples) as isize;
    let frac = delay_samples - (d_int as f64);
    let radius: isize = if d_int >= 5 { 4 } else { d_int - 1 };
    let mut weights = [0.0_f64; 9];
    let taps = (2 * radius + 1) as usize;
    let r_f = (radius as f64) + 1.0;
    let mut k = -radius;
    let mut sum_w = 0.0_f64;
    let mut idx = 0;
    while k <= radius {
        let u = ((k + radius) as f64) + 1.0;
        let win = 0.42 - 0.5 * sin_cos_f64((2.0 * PI * u) / (2.0 * r_f)).1
            + 0.08 * sin_cos_f64((4.0 * PI * u) / (2.0 * r_f)).1;
        let w = sinc_f64((k as f64) - frac) * max_f64(0.0, win);
        weights[idx] = w;
        sum_w += w;
        k += 1;
        idx += 1;
    }
    let inv_sum = if abs_f64(sum_w) > 1.0e-12 {
        1.0 / sum_w
    } else {
        1.0
    };
    let mut w_f32 = [0.0_f32; 9];
    let mut i = 0;
    while i < taps {
        w_f32[i] = (weights[i] * inv_sum) as f32;
        i += 1;
    }

    let first_full = ((d_int + radius) as usize).min(n_len);
    let mut n = 0;
    while n < first_full {
        let mut acc = 0.0_f32;
        let mut t = 0;
        while t < taps {
            let offset = d_int + (t as isize - radius);
            let src_idx = (n as isize) - offset;
            if src_idx >= 0 && (src_idx as usize) < n_len {
                acc += w_f32[t] * scratch[src_idx as usize];
            }
            t += 1;
        }
        buf[n] = acc;
        n += 1;
    }
    let start_offset = (d_int - radius) as usize;
    let end_offset = (d_int + radius) as usize;
    while n < n_len {
        let lo = n - end_offset;
        let hi = n - start_offset;
        let src_win = &scratch[lo..=hi];
        let mut acc = 0.0_f32;
        let mut t = 0;
        while t < taps {
            acc += w_f32[t] * src_win[taps - 1 - t];
            t += 1;
        }
        buf[n] = acc;
        n += 1;
    }
}
