//! Mode filtering, low/high band split, pre-decay padding, and dynamic cutoff
//! estimation for `ModalWorkspace`.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::F32x4;
use piano_spec::{BodyPreset, PRESET_RADIATOR_ZONES};

use crate::math::{clamp_f64, exp_f64, ln_f64, max_f64, powf_f64, sin_cos_f64, sqrt_f64, PI};
use crate::mic_perspectives::ZonePathPair;
use crate::modal_synth::{path_index, NUM_IR_PATHS, TILE};
use crate::subrate_upsample::{DECIM, PAD_LEFT};

/// Global scalar gain applied after `2 * zeta` Q-factor modal-sum normalization.
pub(crate) const MODAL_BODY_BLEND: f64 = 2.6;
const RAD_TILT_REF_HZ: f64 = 160.0;
const RAD_TILT_MIN_HZ: f64 = 48.0;
const RAD_TILT_EXP: f64 = 0.90;

/// Sample-rate-invariant radiation admittance tilt targeting `-3..-4 dB/oct`
/// downward modal spectral slope without `< 90 Hz` bass sag.
#[inline]
pub(crate) fn rad_tilt_gain(freq: f64) -> f64 {
    let f_eff = max_f64(freq, RAD_TILT_MIN_HZ);
    let r = f_eff / RAD_TILT_REF_HZ;
    powf_f64(1.0 + r * r, -0.5 * RAD_TILT_EXP)
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedMode {
    pub(crate) c: f64,
    pub(crate) s: f64,
    pub(crate) decay_step: f64,
    pub(crate) seed_cos: F32x4,
    pub(crate) seed_sin: F32x4,
    pub(crate) rc4: f32,
    pub(crate) rs4: f32,
    pub(crate) rc_tile: f64,
    pub(crate) rs_tile: f64,
    pub(crate) n_end: usize,
    pub(crate) weights: [f32; NUM_IR_PATHS],
}

pub(crate) fn prepare_modes_split(
    preset: &BodyPreset,
    paths: &[ZonePathPair; PRESET_RADIATOR_ZONES],
    sr: f64,
    ir_len: usize,
    sub_padded: usize,
    low_out: &mut Vec<PreparedMode>,
    high_out: &mut Vec<PreparedMode>,
) {
    low_out.clear();
    high_out.clear();

    let ts = 1.0 / sr;
    let nyq_limit = 0.49 * sr;
    let rolloff_start = 0.42 * sr;
    let low_band_max_hz = 0.05 * sr;
    let mut global_peak = 0.0_f64;

    for m in &preset.modes {
        let freq = m.freq_hz as f64;
        let zeta = m.zeta as f64;
        if !freq.is_finite()
            || !zeta.is_finite()
            || freq <= 0.5
            || freq >= nyq_limit
            || zeta <= 1.0e-6
            || zeta >= 0.99
        {
            continue;
        }

        let nyq_scale = if freq > rolloff_start {
            clamp_f64((nyq_limit - freq) / (nyq_limit - rolloff_start), 0.0, 1.0)
        } else {
            1.0
        };

        let omega = 2.0 * PI * freq;
        let sqrt_1mz2 = sqrt_f64(max_f64(1.0e-8, 1.0 - zeta * zeta));
        let tilt = rad_tilt_gain(freq);
        let amp = (MODAL_BODY_BLEND * 2.0 * zeta * ts * omega / sqrt_1mz2) * nyq_scale * tilt;

        let mut weights = [0.0_f32; NUM_IR_PATHS];
        let mut max_w = 0.0_f64;
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            let baffle_eff = if z < 8 {
                freq / sqrt_f64(freq * freq + 15.0 * 15.0)
            } else {
                1.0
            };
            let zd = (m.zone_drive[z] as f64) * baffle_eff;
            if zd.is_finite() && zd != 0.0 {
                let wl = zd * dot5(&m.rad_coeffs, &paths[z].left.mode_proj);
                let wr = zd * dot5(&m.rad_coeffs, &paths[z].right.mode_proj);
                weights[path_index(z, 0)] = wl as f32;
                weights[path_index(z, 1)] = wr as f32;
                let awl = if wl < 0.0 { -wl } else { wl };
                let awr = if wr < 0.0 { -wr } else { wr };
                max_w = max_f64(max_w, max_f64(awl, awr));
            }
            z += 1;
        }
        if max_w < 1.0e-12 {
            continue;
        }
        global_peak = max_f64(global_peak, amp * max_w);

        let is_low = freq <= low_band_max_hz && zeta < 0.25;
        let (step_mul, pre_steps, max_len) = if is_low {
            (DECIM as f64, PAD_LEFT * DECIM, sub_padded)
        } else {
            (1.0, 0usize, ir_len)
        };

        let damp_rate = zeta * omega;
        let omega_d = omega * sqrt_1mz2;
        let theta = omega_d * ts * step_mul;
        let r = exp_f64(-damp_rate * ts * step_mul);

        let mode = build_prepared_mode(
            amp,
            r,
            theta,
            damp_rate * ts * (pre_steps as f64),
            omega_d * ts * (pre_steps as f64),
            max_len,
            pre_steps / DECIM,
            weights,
        );
        if is_low {
            low_out.push(mode);
        } else {
            high_out.push(mode);
        }
    }

    let min_thresh = global_peak * 3.981_071_7e-6; // -108 dB
    if min_thresh > 0.0 {
        refine_cutoffs(low_out, min_thresh, PAD_LEFT);
        refine_cutoffs(high_out, min_thresh, 0);
    }

    low_out.sort_unstable_by_key(|b| core::cmp::Reverse(b.n_end));
    high_out.sort_unstable_by_key(|b| core::cmp::Reverse(b.n_end));
}

#[allow(clippy::too_many_arguments)]
fn build_prepared_mode(
    amp: f64,
    r: f64,
    theta: f64,
    neg_damp: f64,
    neg_phase: f64,
    max_len: usize,
    pre_coarse: usize,
    weights: [f32; NUM_IR_PATHS],
) -> PreparedMode {
    let (c_init, s_init) = if pre_coarse > 0 {
        let r_neg = exp_f64(neg_damp);
        let (sin_neg, cos_neg) = sin_cos_f64(-neg_phase);
        (amp * r_neg * cos_neg, amp * r_neg * sin_neg)
    } else {
        (amp, 0.0)
    };

    let (sin_t, cos_t) = sin_cos_f64(theta);
    let rc1 = r * cos_t;
    let rs1 = r * sin_t;
    let r2 = r * r;
    let r3 = r2 * r;
    let (sin_2t, cos_2t) = sin_cos_f64(2.0 * theta);
    let (sin_3t, cos_3t) = sin_cos_f64(3.0 * theta);
    let seed_cos = F32x4::from_array([1.0, rc1 as f32, (r2 * cos_2t) as f32, (r3 * cos_3t) as f32]);
    let seed_sin = F32x4::from_array([0.0, rs1 as f32, (r2 * sin_2t) as f32, (r3 * sin_3t) as f32]);
    let r4 = r2 * r2;
    let (sin_4t, cos_4t) = sin_cos_f64(4.0 * theta);
    let rc4 = (r4 * cos_4t) as f32;
    let rs4 = (r4 * sin_4t) as f32;
    let mut r_tile = r4;
    let mut p2 = 0;
    while p2 < 6 {
        r_tile *= r_tile;
        p2 += 1;
    }
    let (sin_tile, cos_tile) = sin_cos_f64((TILE as f64) * theta);
    let decay_step = -ln_f64(max_f64(r, 1.0e-12));
    let n_est = if decay_step > 1.0e-9 {
        ((12.434 / decay_step) as usize).saturating_add(1 + pre_coarse + 8)
    } else {
        max_len
    };

    PreparedMode {
        c: c_init,
        s: s_init,
        decay_step,
        seed_cos,
        seed_sin,
        rc4,
        rs4,
        rc_tile: r_tile * cos_tile,
        rs_tile: r_tile * sin_tile,
        n_end: n_est.clamp(1, max_len),
        weights,
    }
}

fn refine_cutoffs(modes: &mut [PreparedMode], min_thresh: f64, pre_offset: usize) {
    for m in modes.iter_mut() {
        let mut mw = 0.0_f32;
        for &w in &m.weights {
            mw = mw.max(w.abs());
        }
        let m_peak =
            (m.seed_cos.to_array()[0] as f64) * sqrt_f64(m.c * m.c + m.s * m.s) * (mw as f64);
        if m_peak <= min_thresh {
            m.n_end = 1;
        } else if m.decay_step > 1.0e-9 {
            let need = ((ln_f64(m_peak / min_thresh) / m.decay_step) as usize)
                .saturating_add(1 + pre_offset + 8);
            m.n_end = m.n_end.min(need.max(1));
        }
    }
}

#[inline]
fn dot5(a: &[f32; 5], b: &[f32; 5]) -> f64 {
    (a[0] as f64) * (b[0] as f64)
        + (a[1] as f64) * (b[1] as f64)
        + (a[2] as f64) * (b[2] as f64)
        + (a[3] as f64) * (b[3] as f64)
        + (a[4] as f64) * (b[4] as f64)
}
