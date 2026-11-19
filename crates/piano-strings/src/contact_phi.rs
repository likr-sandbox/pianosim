//! Spatial windowing (`strike_phi`, `damper_phi`) and hammer/damper physical geometry (`no_std`).

use core::f64::consts::PI;

use piano_spec::{hammer_beta, DAMPED_KEYS, NUM_KEYS};

use crate::modal_soa::{exp_f64, sin_cos_f64, sqrt_f64};

pub(crate) const HORIZ_STRIKE_SCALE: f64 = 0.65;
pub(crate) const HORIZ_DAMP_SCALE: f64 = 0.85;
pub(crate) const DAMPER_CLEAR_THRESHOLD_M: f32 = 4.5e-3;

const DAMPER_W_BASS_M: f64 = 0.080;
const DAMPER_W_TREB_M: f64 = 0.055;
const DAMPER_K_SPATIAL: f64 = 1.0;
const DAMPER_VISC_CUTOFF_HZ: f64 = 2_600.0;

#[inline]
pub(crate) fn hammer_contact_width_bounds_m(key_idx: usize) -> (f64, f64) {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let r = x / 0.28;
    let w_wide = (0.072 / (1.0 + r * r * r)).max(0.0028);
    let w_narrow = 0.38 * w_wide;
    (w_narrow, w_wide)
}

#[inline]
pub(crate) fn hammer_contact_width_m(key_idx: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let r = x / 0.26;
    (0.11 / (1.0 + r * r * r)).max(0.0035)
}

#[inline]
pub(crate) fn hammer_contact_time_s(key_idx: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    (0.0032 * exp_f64(-0.35 * x - 2.75 * x * x * x)).max(0.00014)
}

#[inline]
pub(crate) fn strike_phi_with_width(
    key_idx: usize,
    partial_index: usize,
    speaking_length_m: f64,
    width_m: f64,
) -> f64 {
    let jf = partial_index as f64;
    let beta = hammer_beta(key_idx);
    let sin_term = sin_cos_f64(jf * PI * beta).0;
    let u = if speaking_length_m > 0.0 {
        jf * width_m / speaking_length_m
    } else {
        0.0
    };
    let window = if (1.0 - u).abs() < 1.0e-4 {
        core::f64::consts::FRAC_PI_4 - (PI / 8.0) * (u - 1.0)
    } else {
        sin_cos_f64(0.5 * PI * u).1 / (1.0 - u * u)
    };
    sin_term * window
}

#[inline]
pub(crate) fn strike_phi(key_idx: usize, partial_index: usize, speaking_length_m: f64) -> f64 {
    strike_phi_with_width(
        key_idx,
        partial_index,
        speaking_length_m,
        hammer_contact_width_m(key_idx),
    )
}

#[inline]
pub(crate) fn damper_phi(
    key_idx: usize,
    partial_index: usize,
    speaking_length_m: f64,
    freq_hz: f64,
) -> f64 {
    let jf = partial_index as f64;
    let x = (key_idx.min(DAMPED_KEYS - 1) as f64) / ((DAMPED_KEYS - 1) as f64);
    let w_felt_m = DAMPER_W_BASS_M + (DAMPER_W_TREB_M - DAMPER_W_BASS_M) * x;
    let w_d = if speaking_length_m > 0.0 {
        (w_felt_m / speaking_length_m).clamp(0.03, 0.22)
    } else {
        0.082
    };
    let pw = jf * PI * w_d;
    let sinc_w = sin_cos_f64(pw).0 / pw;
    let cos_2xd = sin_cos_f64(2.0 * jf * PI * 0.125).1;
    let env_rms = sqrt_f64((0.5 - 0.5 * cos_2xd * sinc_w).max(0.0));
    let ramp = 1.0 + 0.44 * (1.0 - exp_f64(-0.48 * ((partial_index - 1) as f64)));
    let u_sp = DAMPER_K_SPATIAL * jf * w_d;
    let spatial_lpf = 1.0 / sqrt_f64(1.0 + u_sp * u_sp);
    let fr = freq_hz / DAMPER_VISC_CUTOFF_HZ;
    let visc_lpf = 1.0 / sqrt_f64(1.0 + fr * fr);
    env_rms * ramp * spatial_lpf * visc_lpf
}
