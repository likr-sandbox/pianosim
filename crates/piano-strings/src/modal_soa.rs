//! 64-byte aligned Structure-of-Arrays (SoA) modal resonator buffers and
//! Exact Coupled-Form Incremental ZOH pole initialization.
//!
//! Hot L2-resident arrays (`u`, `v`, `d_coeff`, `s_coeff`, `c_bridge`) consume
//! ~512 KB for ~26,184 modes (`~6,546` `F32x4` vectors). Stereo pan is applied
//! once per string after summing `v_next * c_bridge`.

use core::f64::consts::{LN_10, PI, TAU};
use piano_simd::AlignedBuf;
use piano_spec::{
    count_longitudinal_partials, count_transverse_partials, effective_string_f0_hz,
    partial_frequency_hz, Polarization, StringPhysicalParams, VoicingSpec, MAX_AUDIBLE_FREQ_HZ,
    NUM_KEYS,
};

use crate::choir_map::{ChoirMap, StringMapEntry};
#[allow(unused_imports)]
pub(crate) use crate::damping_law::{
    t60_transverse_partial, t60_vert_fundamental, t60_wire_fundamental,
};
#[allow(unused_imports)]
pub(crate) use crate::math::{exp_f64, one_minus_exp_neg_f64, sin_cos_f64, sqrt_f64};

const LN_1000: f64 = 6.907_755_278_982_137;
pub(crate) const NYQUIST_CUTOFF_RATIO: f64 = 0.49;
pub(crate) const C_VERT: f32 = 1.0;
pub(crate) const C_HORIZ: f32 = 0.18;

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
        Self::build_with_mode(sample_rate, voicing, specs, map, lossless, false)
    }

    /// Allocates and initializes SoA mode tables choosing between standalone string damping and
    /// intrinsic wire/air damping (`bridge_coupled = true`) when coupled to Stage 3 `BridgeBank`.
    #[must_use]
    pub(crate) fn build_with_mode(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        lossless: bool,
        bridge_coupled: bool,
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
            populate_string_modes_with_mode(
                &mut soa,
                entry,
                sample_rate,
                voicing,
                specs,
                lossless,
                bridge_coupled,
            );
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
    slot_cap: usize,
    key_idx: usize,
    f0_hz: f64,
    b_inharm: f64,
    sample_rate: f64,
    t60_v_base: f64,
    hold_mult: f64,
    c_bridge: f32,
    exc_scale: f64,
    lossless: bool,
    bridge_coupled: bool,
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

pub(crate) fn populate_string_modes_with_mode(
    soa: &mut ModalSoa,
    entry: &StringMapEntry,
    sample_rate: f64,
    voicing: &VoicingSpec,
    specs: &[StringPhysicalParams; NUM_KEYS],
    lossless: bool,
    bridge_coupled: bool,
) {
    let k = entry.key_idx as usize;
    let s_choir = entry.string_in_choir as usize;
    let spec = specs[k];
    let b_inharm = spec.inharmonicity_b;
    let t60_v_base = if bridge_coupled {
        t60_wire_fundamental(k)
    } else {
        t60_vert_fundamental(k)
    };
    let h_mult = 2.8;

    let f0_vert = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
    let f0_horiz = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
    let level_gain = exp_f64(f64::from(voicing.effective_level_db_offset(k)) * (LN_10 / 20.0));
    let nyquist = NYQUIST_CUTOFF_RATIO * sample_rate;
    let (v_cap, h_cap, l_cap) = (
        entry.vert_vecs.len() * 4,
        entry.horiz_vecs.len() * 4,
        entry.long_vecs.len() * 4,
    );
    let v_modes = count_transverse_partials(f0_vert, b_inharm, MAX_AUDIBLE_FREQ_HZ).min(v_cap);
    let h_modes = count_transverse_partials(f0_horiz, b_inharm, MAX_AUDIBLE_FREQ_HZ).min(h_cap);
    let l_modes = count_longitudinal_partials(k, spec.f_long_hz, MAX_AUDIBLE_FREQ_HZ).min(l_cap);

    let mut energy_sq = 0.0_f64;
    for j in 1..=v_modes {
        let f_j = partial_frequency_hz(f0_vert, b_inharm, j);
        if f_j > 0.0 && f_j < nyquist {
            let w = strike_mode_shape(k, j) * f64::from(C_VERT);
            energy_sq += w * w;
        }
    }
    for j in 1..=h_modes {
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
            mode_count: v_modes,
            slot_cap: v_cap,
            key_idx: k,
            f0_hz: f0_vert,
            b_inharm,
            sample_rate,
            t60_v_base,
            hold_mult: 1.0,
            c_bridge: C_VERT,
            exc_scale: norm,
            lossless,
            bridge_coupled,
        },
    );
    populate_transverse_pol(
        soa,
        &PolParams {
            base_slot: entry.horiz_vecs.start * 4,
            mode_count: h_modes,
            slot_cap: h_cap,
            key_idx: k,
            f0_hz: f0_horiz,
            b_inharm,
            sample_rate,
            t60_v_base,
            hold_mult: h_mult,
            c_bridge: C_HORIZ,
            exc_scale: 0.85 * norm,
            lossless,
            bridge_coupled,
        },
    );

    let long_base = entry.long_vecs.start * 4;
    for m in 1..=l_modes {
        let f_init = crate::longitudinal::long_init_freq_hz(k, spec.f_long_hz, m);
        let t60_l = if lossless {
            f64::INFINITY
        } else {
            crate::longitudinal::t60_longitudinal(m)
        };
        if let Some(poles) = coupled_pole_f32(f_init, sample_rate, t60_l) {
            let c_l = crate::longitudinal::long_bridge_weight(k, f0_vert, spec.f_long_hz, m);
            soa.write_slot(long_base + (m - 1), poles, c_l, 0.0);
        } else {
            soa.write_slot(long_base + (m - 1), (1.0, 0.0), 0.0, 0.0);
        }
    }
    for slot_off in l_modes..l_cap {
        soa.write_slot(long_base + slot_off, (1.0, 0.0), 0.0, 0.0);
    }
}

fn populate_transverse_pol(soa: &mut ModalSoa, p: &PolParams) {
    for j in 1..=p.mode_count {
        let f_j = partial_frequency_hz(p.f0_hz, p.b_inharm, j);
        let t60_h = if p.lossless {
            f64::INFINITY
        } else {
            t60_transverse_partial(p.t60_v_base, f_j, j, p.hold_mult, p.bridge_coupled)
        };

        if let Some(poles) = coupled_pole_f32(f_j, p.sample_rate, t60_h) {
            let b = (strike_mode_shape(p.key_idx, j) * p.exc_scale) as f32;
            let c_signed = if j % 2 == 1 { p.c_bridge } else { -p.c_bridge };
            soa.write_slot(p.base_slot + (j - 1), poles, c_signed, b);
        } else {
            soa.write_slot(p.base_slot + (j - 1), (1.0, 0.0), 0.0, 0.0);
        }
    }
    for slot_off in p.mode_count..p.slot_cap {
        soa.write_slot(p.base_slot + slot_off, (1.0, 0.0), 0.0, 0.0);
    }
}

#[cfg(test)]
#[path = "modal_soa_tests.rs"]
mod tests;
