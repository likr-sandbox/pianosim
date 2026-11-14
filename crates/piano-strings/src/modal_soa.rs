//! 64-byte aligned Structure-of-Arrays (SoA) modal resonator buffers and
//! Exact Coupled-Form Incremental ZOH pole initialization.
//!
//! Hot L2-resident arrays (`u`, `v`, `d_coeff`, `s_coeff`, `c_bridge`) consume
//! ~512 KB for ~26,184 modes (`~6,546` `F32x4` vectors). Stereo pan is applied
//! once per string after summing `v_next * c_bridge`.

use core::f64::consts::{FRAC_2_PI, LN_10, LOG2_E, PI, TAU};
use piano_simd::AlignedBuf;
use piano_spec::{
    effective_string_f0_hz, partial_frequency_hz, Polarization, StringPhysicalParams, VoicingSpec,
    NUM_KEYS,
};

use crate::choir_map::{ChoirMap, StringMapEntry};

const LN_1000: f64 = 6.907_755_278_982_137;
const PIO2_HI: f64 = 1.570_796_326_734_125_6;
const PIO2_LO: f64 = 6.077_100_506_506_192e-11;
const LN2_HI: f64 = 0.693_147_180_485_539_1;
const LN2_LO: f64 = 7.440_614_790_965_583e-11;
pub(crate) const NYQUIST_CUTOFF_RATIO: f64 = 0.49;
pub(crate) const C_VERT: f32 = 1.0;
pub(crate) const C_HORIZ: f32 = 0.18;

const SIN_COEFFS: [f64; 8] = [
    1.0,
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5_040.0,
    1.0 / 362_880.0,
    -1.0 / 39_916_800.0,
    1.0 / 6_227_020_800.0,
    -1.0 / 1_307_674_368_000.0,
];
const COS_COEFFS: [f64; 8] = [
    1.0,
    -0.5,
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40_320.0,
    -1.0 / 3_628_800.0,
    1.0 / 479_001_600.0,
    -1.0 / 87_178_291_200.0,
];
const EXP_COEFFS: [f64; 14] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5_040.0,
    1.0 / 40_320.0,
    1.0 / 362_880.0,
    1.0 / 3_628_800.0,
    1.0 / 39_916_800.0,
    1.0 / 479_001_600.0,
    1.0 / 6_227_020_800.0,
];
const ONE_MINUS_EXP_NEG_DIV_ALPHA: [f64; 6] =
    [1.0, -0.5, 1.0 / 6.0, -1.0 / 24.0, 1.0 / 120.0, -1.0 / 720.0];

#[inline]
fn eval_horner(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    let mut i = coeffs.len();
    while i > 0 {
        i -= 1;
        acc = acc * x + coeffs[i];
    }
    acc
}

#[inline]
fn round_f64(x: f64) -> i64 {
    if x >= 0.0 {
        (x + 0.5) as i64
    } else {
        (x - 0.5) as i64
    }
}

/// Pure-Rust `no_std` `f64` `(sin(angle), cos(angle))` using `eval_horner`.
pub(crate) fn sin_cos_f64(angle: f64) -> (f64, f64) {
    if !angle.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let q = round_f64(angle * FRAC_2_PI);
    let qf = q as f64;
    let r = (angle - qf * PIO2_HI) - qf * PIO2_LO;
    let r2 = r * r;
    let s = r * eval_horner(r2, &SIN_COEFFS);
    let c = eval_horner(r2, &COS_COEFFS);
    match q.rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Pure-Rust `no_std` `f64` `exp(y)` using `eval_horner`.
pub(crate) fn exp_f64(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y == 0.0 {
        return 1.0;
    }
    if y > 709.78 {
        return f64::INFINITY;
    }
    if y < -745.13 {
        return 0.0;
    }
    let k = round_f64(y * LOG2_E);
    let kf = k as f64;
    let r = (y - kf * LN2_HI) - kf * LN2_LO;
    let poly = eval_horner(r, &EXP_COEFFS);
    if k < -1022 {
        let sub_scale = f64::from_bits(((k + 1077) as u64) << 52);
        let tiny_scale = f64::from_bits((1023_u64 - 55) << 52);
        (poly * sub_scale) * tiny_scale
    } else {
        poly * f64::from_bits(((k + 1023) as u64) << 52)
    }
}

#[inline]
fn one_minus_exp_neg_f64(alpha: f64) -> f64 {
    if alpha <= 0.0 {
        0.0
    } else if alpha < 1.0e-3 {
        alpha * eval_horner(alpha, &ONE_MINUS_EXP_NEG_DIV_ALPHA)
    } else {
        1.0 - exp_f64(-alpha)
    }
}

/// Pure-Rust `no_std` `f64` `sqrt(x)` via IEEE-754 seed and 6 Newton-Raphson steps.
#[inline]
pub(crate) fn sqrt_f64(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    let (scaled, post_scale) = if x < f64::MIN_POSITIVE {
        (x * 18_014_398_509_481_984.0, 1.0_f64 / 134_217_728.0)
    } else {
        (x, 1.0_f64)
    };
    let mut y = f64::from_bits((scaled.to_bits() >> 1) + 0x1ff8_0000_0000_0000);
    for _ in 0..6 {
        y = 0.5 * (y + scaled / y);
    }
    y * post_scale
}

/// 64-byte aligned Structure-of-Arrays resonator state and pole coefficients.
#[derive(Clone, Debug)]
pub struct ModalSoa {
    /// Coupled-form real state `u` (hot L2 array).
    pub u: AlignedBuf,
    /// Coupled-form imaginary state `v` (hot L2 array).
    pub v: AlignedBuf,
    /// Active pole offset `d = 1 - r * cos(theta)` (hot L2 array).
    pub d_coeff: AlignedBuf,
    /// Active pole imaginary `s = r * sin(theta)` (hot L2 array).
    pub s_coeff: AlignedBuf,
    /// Bridge coupling weight `c_bridge` per mode (hot L2 array).
    pub c_bridge: AlignedBuf,
    /// Excitation coupling weight `b_exc` per mode (for test impulse helpers).
    pub b_exc: AlignedBuf,
}

impl ModalSoa {
    /// Allocates and initializes permanent undamped SoA arrays for `sample_rate`, `specs`, `voicing`, and `map`.
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
    ) -> Self {
        Self::build_with_lossless(sample_rate, voicing, specs, map, false)
    }

    /// Allocates and initializes SoA mode tables with optional lossless (`r_hold = 1.0`) poles.
    #[must_use]
    pub(crate) fn build_with_lossless(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        lossless: bool,
    ) -> Self {
        let len = map.total_padded_modes();
        let mut soa = Self {
            u: AlignedBuf::zeroed(len),
            v: AlignedBuf::zeroed(len),
            d_coeff: AlignedBuf::filled(len, 1.0),
            s_coeff: AlignedBuf::zeroed(len),
            c_bridge: AlignedBuf::zeroed(len),
            b_exc: AlignedBuf::zeroed(len),
        };
        for entry in map.strings() {
            populate_string_modes(&mut soa, entry, sample_rate, voicing, specs, lossless);
        }
        soa
    }

    /// Clears `(u, v)` back to zero resting state.
    pub fn reset_resting(&mut self) {
        self.u.clear();
        self.v.clear();
    }

    #[inline]
    fn write_slot(&mut self, slot: usize, poles: (f32, f32), c_bridge: f32, b_exc: f32) {
        self.d_coeff.as_mut_slice()[slot] = poles.0;
        self.s_coeff.as_mut_slice()[slot] = poles.1;
        self.c_bridge.as_mut_slice()[slot] = c_bridge;
        self.b_exc.as_mut_slice()[slot] = b_exc;
    }
}

struct PolParams {
    base_slot: usize,
    mode_count: usize,
    key_idx: usize,
    f0_hz: f64,
    b_inharm: f64,
    sample_rate: f64,
    t60_v_base: f64,
    hold_mult: f64,
    c_bridge: f32,
    exc_scale: f64,
    lossless: bool,
}

#[inline]
pub(crate) fn t60_vert_fundamental(key_idx: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    30.0 * exp_f64(-0.45 * x - 4.35 * x * x + 1.50 * x * x * x)
}

#[inline]
fn strike_mode_shape(key_idx: usize, partial_index: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let beta = 0.137 - 0.042 * x;
    let strike_sin = sin_cos_f64((partial_index as f64) * PI * beta).0.abs();
    let slope = 0.12 + 1.80 * x;
    (0.35 + 0.65 * strike_sin) / (1.0 + slope * ((partial_index - 1) as f64))
}

#[inline]
pub(crate) fn coupled_pole_f32(
    freq_hz: f64,
    sample_rate: f64,
    t60_hold: f64,
) -> Option<(f32, f32)> {
    if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz >= NYQUIST_CUTOFF_RATIO * sample_rate {
        return None;
    }
    let theta = TAU * freq_hz / sample_rate;
    let (sin_t, cos_t) = sin_cos_f64(theta);
    let (one_minus_rh, r_hold) = if t60_hold.is_infinite() && t60_hold.is_sign_positive() {
        (0.0, 1.0)
    } else {
        let one_minus = one_minus_exp_neg_f64(LN_1000 / (t60_hold * sample_rate));
        (one_minus, 1.0 - one_minus)
    };
    let s_f32 = (r_hold * sin_t) as f32;
    let s_ratio = (f64::from(s_f32) / r_hold).clamp(-1.0, 1.0);
    let cos_mag = sqrt_f64((1.0 - s_ratio * s_ratio).max(0.0));
    let d_f32 = if cos_t >= 0.0 {
        (one_minus_rh + r_hold * (s_ratio * s_ratio) / (1.0 + cos_mag)) as f32
    } else {
        (1.0 + r_hold * cos_mag) as f32
    };
    Some((d_f32, s_f32))
}

fn populate_string_modes(
    soa: &mut ModalSoa,
    entry: &StringMapEntry,
    sample_rate: f64,
    voicing: &VoicingSpec,
    specs: &[StringPhysicalParams; NUM_KEYS],
    lossless: bool,
) {
    let k = entry.key_idx as usize;
    let s_choir = entry.string_in_choir as usize;
    let spec = specs[k];
    let b_inharm = spec.inharmonicity_b;
    let t60_v_base = t60_vert_fundamental(k);

    let f0_vert = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
    let f0_horiz = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
    let level_gain = exp_f64(f64::from(voicing.effective_level_db_offset(k)) * (LN_10 / 20.0));
    let nyquist = NYQUIST_CUTOFF_RATIO * sample_rate;

    let mut energy_sq = 0.0_f64;
    for j in 1..=entry.vert_modes {
        let f_j = partial_frequency_hz(f0_vert, b_inharm, j);
        if f_j > 0.0 && f_j < nyquist {
            let w = strike_mode_shape(k, j) * f64::from(C_VERT);
            energy_sq += w * w;
        }
    }
    for j in 1..=entry.horiz_modes {
        let f_j = partial_frequency_hz(f0_horiz, b_inharm, j);
        if f_j > 0.0 && f_j < nyquist {
            let w = 0.85 * strike_mode_shape(k, j) * f64::from(C_HORIZ);
            energy_sq += w * w;
        }
    }
    let norm = if energy_sq > 0.0 {
        0.14 * level_gain / sqrt_f64(f64::from(entry.choir_size) * energy_sq)
    } else {
        0.0
    };

    populate_transverse_pol(
        soa,
        &PolParams {
            base_slot: entry.vert_vecs.start * 4,
            mode_count: entry.vert_modes,
            key_idx: k,
            f0_hz: f0_vert,
            b_inharm,
            sample_rate,
            t60_v_base,
            hold_mult: 1.0,
            c_bridge: C_VERT,
            exc_scale: norm,
            lossless,
        },
    );
    populate_transverse_pol(
        soa,
        &PolParams {
            base_slot: entry.horiz_vecs.start * 4,
            mode_count: entry.horiz_modes,
            key_idx: k,
            f0_hz: f0_horiz,
            b_inharm,
            sample_rate,
            t60_v_base,
            hold_mult: 2.8,
            c_bridge: C_HORIZ,
            exc_scale: 0.85 * norm,
            lossless,
        },
    );

    let long_base = entry.long_vecs.start * 4;
    for m in 1..=entry.long_modes {
        let f_init = crate::longitudinal::long_init_freq_hz(spec.f_long_hz, m);
        let t60_l = if lossless {
            f64::INFINITY
        } else {
            crate::longitudinal::t60_longitudinal(m)
        };
        if let Some(poles) = coupled_pole_f32(f_init, sample_rate, t60_l) {
            let c_l = crate::longitudinal::long_bridge_weight(k, f0_vert, spec.f_long_hz, m);
            soa.write_slot(long_base + (m - 1), poles, c_l, 0.0);
        }
    }
}

fn populate_transverse_pol(soa: &mut ModalSoa, p: &PolParams) {
    for j in 1..=p.mode_count {
        let f_j = partial_frequency_hz(p.f0_hz, p.b_inharm, j);
        let t60_h = if p.lossless {
            f64::INFINITY
        } else {
            let f_norm = f_j / 20_000.0;
            let loss_scale = 1.0 + 0.05 * ((j - 1) as f64) + 1.5 * f_norm * f_norm;
            (p.t60_v_base / loss_scale).max(0.25) * p.hold_mult
        };

        if let Some(poles) = coupled_pole_f32(f_j, p.sample_rate, t60_h) {
            let b = (strike_mode_shape(p.key_idx, j) * p.exc_scale) as f32;
            let c_signed = if j % 2 == 1 { p.c_bridge } else { -p.c_bridge };
            soa.write_slot(p.base_slot + (j - 1), poles, c_signed, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use piano_spec::build_scale_design;

    #[test]
    fn nostd_math_matches_std_and_poles_preserve_padding_invariants() {
        for i in -32..=32 {
            let a = f64::from(i) * 0.413_789_123;
            let (s, c) = sin_cos_f64(a);
            assert!((s - a.sin()).abs() < 1e-14 && (c - a.cos()).abs() < 1e-14);
        }
        for &y in &[-25.0_f64, -1.5, -1e-6, 0.0, 1.5, 10.0] {
            assert!((exp_f64(y) - y.exp()).abs() <= 1e-13 * y.exp().max(1.0));
        }
        for &alpha in &[1.0e-7_f64, 2.4e-6, 5.0e-4, 2.0e-2] {
            assert!((one_minus_exp_neg_f64(alpha) - (1.0 - (-alpha).exp())).abs() < 1e-15);
        }

        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);
        let map = ChoirMap::build_with_specs(&specs, &voicing);
        let soa = ModalSoa::build(96_000.0, &voicing, &specs, &map);
        assert_eq!(soa.u.as_ptr() as usize % 64, 0);
        assert_eq!(soa.d_coeff.as_ptr() as usize % 64, 0);

        let a0 = &map.strings()[0];
        for slot in (a0.vert_vecs.start * 4 + a0.vert_modes)..(a0.vert_vecs.end * 4) {
            assert_eq!(
                (
                    soa.d_coeff[slot],
                    soa.s_coeff[slot],
                    soa.c_bridge[slot],
                    soa.b_exc[slot]
                ),
                (1.0, 0.0, 0.0, 0.0)
            );
        }
        assert!(soa.d_coeff[a0.vert_vecs.start * 4] > 0.0);
        assert!(soa.d_coeff[a0.vert_vecs.start * 4] < 1.0e-3);
        assert!(soa.s_coeff[a0.vert_vecs.start * 4] > 0.0);

        for &sr in &[44_100.0_f64, 48_000.0, 96_000.0] {
            for &freq in &[27.5_f64, 65.406, 130.813, 200.0, 261.626, 440.0] {
                let (d, s) =
                    coupled_pole_f32(freq, sr, f64::INFINITY).expect("valid lossless pole");
                let err = ((1.0 - f64::from(d)).powi(2) + f64::from(s).powi(2) - 1.0).abs();
                let limit = if freq <= 262.0 { 1.0e-10 } else { 5.0e-10 };
                assert!(err <= limit, "sr={sr} freq={freq} err={err:e}");
            }
        }
    }
}
