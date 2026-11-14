//! Precomputed key modal tables, voice states, and 64-byte aligned SIMD resonator bank.

use core::f64::consts::{FRAC_PI_2, PI, TAU};

use piano_simd::{AlignedBuf, F32x4};

/// Number of simultaneous polyphonic voices in the M1 modal synth.
pub const NUM_VOICES: usize = 16;
/// Number of harmonic partials per voice (packed into one `F32x4` vector).
pub const PARTIALS_PER_VOICE: usize = 4;
/// Total modal resonators (`NUM_VOICES * PARTIALS_PER_VOICE`).
pub const TOTAL_MODES: usize = NUM_VOICES * PARTIALS_PER_VOICE;

pub(crate) const MIDI_KEY_COUNT: usize = 128;
const LN_1000: f64 = 6.907_755_278_982_137;

#[derive(Copy, Clone, Debug)]
pub(crate) struct KeyModalParams {
    pub two_cos_w: [f32; PARTIALS_PER_VOICE],
    pub sin_w: [f32; PARTIALS_PER_VOICE],
    pub r_hold: [f32; PARTIALS_PER_VOICE],
    pub r_damp: [f32; PARTIALS_PER_VOICE],
    pub gain_l: [f32; PARTIALS_PER_VOICE],
    pub gain_r: [f32; PARTIALS_PER_VOICE],
    pub pulse_len: u16,
    pub pulse_init_sin: f32,
    pub pulse_init_cos: f32,
    pub pulse_step_sin: f32,
    pub pulse_step_cos: f32,
    pub pulse_inv_sum: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct VoiceState {
    pub active: bool,
    pub key_down: bool,
    pub sostenuto_latched: bool,
    pub midi_key: u8,
    pub age: u64,
    pub pulse_remaining: u16,
    pub pulse_sin: f32,
    pub pulse_cos: f32,
    pub pulse_step_sin: f32,
    pub pulse_step_cos: f32,
    pub pulse_scale: f32,
}

impl VoiceState {
    #[inline]
    pub fn next_excitation(&mut self) -> f32 {
        if self.pulse_remaining == 0 {
            return 0.0;
        }
        let out = self.pulse_sin * self.pulse_scale;
        let next_s = self.pulse_sin * self.pulse_step_cos + self.pulse_cos * self.pulse_step_sin;
        let next_c = self.pulse_cos * self.pulse_step_cos - self.pulse_sin * self.pulse_step_sin;
        self.pulse_sin = next_s;
        self.pulse_cos = next_c;
        self.pulse_remaining -= 1;
        out
    }
}

/// 64-mode (`16 x F32x4`) Structure-of-Arrays resonator bank backed by `AlignedBuf`.
#[derive(Clone, Debug)]
pub(crate) struct ModalBank {
    pub q0: AlignedBuf,
    pub q1: AlignedBuf,
    pub a1: AlignedBuf,
    pub a2: AlignedBuf,
    pub b_gain: AlignedBuf,
    pub gain_l: AlignedBuf,
    pub gain_r: AlignedBuf,
    pub two_cos_w: AlignedBuf,
    pub r_hold: AlignedBuf,
    pub r_damp: AlignedBuf,
}

impl ModalBank {
    pub fn new() -> Self {
        Self {
            q0: AlignedBuf::new(TOTAL_MODES),
            q1: AlignedBuf::new(TOTAL_MODES),
            a1: AlignedBuf::new(TOTAL_MODES),
            a2: AlignedBuf::new(TOTAL_MODES),
            b_gain: AlignedBuf::new(TOTAL_MODES),
            gain_l: AlignedBuf::new(TOTAL_MODES),
            gain_r: AlignedBuf::new(TOTAL_MODES),
            two_cos_w: AlignedBuf::new(TOTAL_MODES),
            r_hold: AlignedBuf::new(TOTAL_MODES),
            r_damp: AlignedBuf::new(TOTAL_MODES),
        }
    }

    pub fn clear_all(&mut self) {
        self.q0.clear();
        self.q1.clear();
        self.a1.clear();
        self.a2.clear();
        self.b_gain.clear();
        self.gain_l.clear();
        self.gain_r.clear();
        self.two_cos_w.clear();
        self.r_hold.clear();
        self.r_damp.clear();
    }

    pub fn clear_voice_state(&mut self, voice_idx: usize) {
        let zero = F32x4::zero();
        self.q0.store_f32x4(voice_idx, zero);
        self.q1.store_f32x4(voice_idx, zero);
        self.b_gain.store_f32x4(voice_idx, zero);
    }
}

pub(crate) fn build_key_table(sample_rate: f64) -> [KeyModalParams; MIDI_KEY_COUNT] {
    let mut table = [KeyModalParams {
        two_cos_w: [0.0; PARTIALS_PER_VOICE],
        sin_w: [0.0; PARTIALS_PER_VOICE],
        r_hold: [0.0; PARTIALS_PER_VOICE],
        r_damp: [0.0; PARTIALS_PER_VOICE],
        gain_l: [0.0; PARTIALS_PER_VOICE],
        gain_r: [0.0; PARTIALS_PER_VOICE],
        pulse_len: 1,
        pulse_init_sin: 1.0,
        pulse_init_cos: 0.0,
        pulse_step_sin: 0.0,
        pulse_step_cos: 1.0,
        pulse_inv_sum: 1.0,
    }; MIDI_KEY_COUNT];

    let max_pulse = ((0.0022 * sample_rate).round() as usize).max(1);
    let nyquist_guard = 0.45 * sample_rate;

    for (k, entry) in table.iter_mut().enumerate() {
        let f0 = 440.0 * 2.0_f64.powf((k as f64 - 69.0) / 12.0);
        let key_norm = ((k as f64 - 21.0) / 87.0).clamp(0.0, 1.0);
        let pan_angle = (0.22 + 0.56 * key_norm) * FRAC_PI_2;
        let pan_l = pan_angle.cos() as f32;
        let pan_r = pan_angle.sin() as f32;

        let t60_fund = 7.0 - 5.0 * key_norm;
        let t60_damp_base = (0.14 - 0.06 * key_norm).max(0.06);

        for p in 0..PARTIALS_PER_VOICE {
            let fp = (p + 1) as f64 * f0;
            if fp <= 0.0 || fp >= nyquist_guard {
                continue;
            }
            let omega = TAU * fp / sample_rate;
            let (sw, cw) = omega.sin_cos();
            let t60_hold = t60_fund / (1.0 + 0.65 * p as f64);
            let t60_damp = t60_damp_base / (1.0 + 0.30 * p as f64);

            entry.two_cos_w[p] = (2.0 * cw) as f32;
            entry.sin_w[p] = sw as f32;
            entry.r_hold[p] = (-LN_1000 / (t60_hold * sample_rate)).exp() as f32;
            entry.r_damp[p] = (-LN_1000 / (t60_damp * sample_rate)).exp() as f32;
            entry.gain_l[p] = pan_l;
            entry.gain_r[p] = pan_r;
        }

        let pulse_len = ((0.20 * sample_rate / f0).round() as usize).clamp(1, max_pulse);
        entry.pulse_len = pulse_len as u16;
        if pulse_len == 1 {
            entry.pulse_init_sin = 1.0;
            entry.pulse_init_cos = 0.0;
            entry.pulse_step_sin = 0.0;
            entry.pulse_step_cos = 1.0;
            entry.pulse_inv_sum = 1.0;
        } else {
            let step = PI / (pulse_len as f64);
            let mut sum = 0.0_f64;
            for n in 0..pulse_len {
                sum += ((n as f64 + 0.5) * step).sin();
            }
            let (s0, c0) = (0.5 * step).sin_cos();
            let (ss, cs) = step.sin_cos();
            entry.pulse_init_sin = s0 as f32;
            entry.pulse_init_cos = c0 as f32;
            entry.pulse_step_sin = ss as f32;
            entry.pulse_step_cos = cs as f32;
            entry.pulse_inv_sum = (1.0 / sum) as f32;
        }
    }

    table
}
