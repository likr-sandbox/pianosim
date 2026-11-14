//! Polyphonic SIMD-accelerated ZOH modal synthesizer core for M1 verification.

use piano_simd::F32x4;
use piano_spec::{u16_to_unit_f32, EventKind, PedalKind, SampleEvent};

use crate::event_queue::{is_sorted_by_offset, EventQueue};
use crate::modal_bank::{
    build_key_table, KeyModalParams, ModalBank, VoiceState, MIDI_KEY_COUNT, NUM_VOICES,
    PARTIALS_PER_VOICE,
};
use crate::telemetry::{compute_block_telemetry, EngineTelemetry};

const PARTIAL_WEIGHTS: [f32; PARTIALS_PER_VOICE] = [1.0, 0.42, 0.20, 0.10];
const VOICE_SILENCE_THRESHOLD: f32 = 1.0e-5;
const SUBNORMAL_FLUSH_THRESHOLD: f32 = 1.0e-20;

/// Core synthesizer state separated from scratch event queue for clean borrowing.
#[derive(Clone, Debug)]
struct EngineCore {
    sample_rate: f32,
    key_table: [KeyModalParams; MIDI_KEY_COUNT],
    voices: [VoiceState; NUM_VOICES],
    bank: ModalBank,
    sustain: f32,
    una_corda: f32,
    sostenuto_down: bool,
    trigger_seq: u64,
}

/// Zero-allocation polyphonic piano test engine for the M1 vertical slice.
#[derive(Clone, Debug)]
pub struct PianoEngine {
    core: EngineCore,
    scratch_events: EventQueue,
    telemetry: EngineTelemetry,
}

impl PianoEngine {
    /// Allocates and precomputes all modal tuning tables for `sample_rate` once.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let safe_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        Self {
            core: EngineCore {
                sample_rate: safe_rate,
                key_table: build_key_table(f64::from(safe_rate)),
                voices: [VoiceState::default(); NUM_VOICES],
                bank: ModalBank::new(),
                sustain: 0.0,
                una_corda: 0.0,
                sostenuto_down: false,
                trigger_seq: 0,
            },
            scratch_events: EventQueue::new(),
            telemetry: EngineTelemetry::default(),
        }
    }

    /// Returns the configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.core.sample_rate
    }

    /// Returns the telemetry captured during the most recent `render_block` call.
    #[must_use]
    #[inline]
    pub const fn telemetry(&self) -> EngineTelemetry {
        self.telemetry
    }

    /// Returns the count of currently active ringing voices.
    #[must_use]
    pub fn active_voices(&self) -> u32 {
        self.core.active_voice_count()
    }

    /// Resets all voices, pedal states, and modal buffers without reallocating.
    pub fn reset(&mut self) {
        self.core.voices = [VoiceState::default(); NUM_VOICES];
        self.core.bank.clear_all();
        self.core.sustain = 0.0;
        self.core.una_corda = 0.0;
        self.core.sostenuto_down = false;
        self.telemetry = EngineTelemetry::default();
    }

    /// Processes sample-accurate `events` and renders stereo audio into `out_l`/`out_r`
    /// with zero heap allocations.
    pub fn render_block(&mut self, events: &[SampleEvent], out_l: &mut [f32], out_r: &mut [f32]) {
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }

        let out_l = &mut out_l[..frames];
        let out_r = &mut out_r[..frames];

        if is_sorted_by_offset(events) {
            self.core.render_event_slice(events, out_l, out_r);
        } else {
            self.scratch_events.copy_sorted_from(events);
            self.core
                .render_event_slice(self.scratch_events.as_slice(), out_l, out_r);
        }

        self.core.retire_silent_voices();
        self.telemetry = compute_block_telemetry(self.core.active_voice_count(), out_l, out_r);
    }
}

impl EngineCore {
    fn active_voice_count(&self) -> u32 {
        self.voices.iter().filter(|v| v.active).count() as u32
    }

    fn render_event_slice(&mut self, events: &[SampleEvent], out_l: &mut [f32], out_r: &mut [f32]) {
        let frames = out_l.len();
        let mut cursor = 0_usize;
        for &ev in events {
            let target = (ev.sample_offset as usize).min(frames);
            if target > cursor {
                self.render_subblock(&mut out_l[cursor..target], &mut out_r[cursor..target]);
                cursor = target;
            }
            self.apply_event(ev.kind);
        }
        if cursor < frames {
            self.render_subblock(&mut out_l[cursor..frames], &mut out_r[cursor..frames]);
        }
    }

    fn render_subblock(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        let frames = out_l.len();
        let q0_vecs = self.bank.q0.as_padded_f32x4_slice_mut();
        let q1_vecs = self.bank.q1.as_padded_f32x4_slice_mut();
        let a1_vecs = self.bank.a1.as_padded_f32x4_slice();
        let a2_vecs = self.bank.a2.as_padded_f32x4_slice();
        let b_vecs = self.bank.b_gain.as_padded_f32x4_slice();
        let gl_vecs = self.bank.gain_l.as_padded_f32x4_slice();
        let gr_vecs = self.bank.gain_r.as_padded_f32x4_slice();

        for i in 0..frames {
            let mut acc_l = F32x4::zero();
            let mut acc_r = F32x4::zero();

            for v in 0..NUM_VOICES {
                let exc = self.voices[v].next_excitation();
                let cur = q0_vecs[v];
                let prev = q1_vecs[v];
                let next =
                    a1_vecs[v].mul_add(cur, b_vecs[v] * F32x4::splat(exc)) - a2_vecs[v] * prev;
                q1_vecs[v] = cur;
                q0_vecs[v] = next;

                acc_l = next.mul_add(gl_vecs[v], acc_l);
                acc_r = next.mul_add(gr_vecs[v], acc_r);
            }

            out_l[i] = soft_limit(acc_l.reduce_sum());
            out_r[i] = soft_limit(acc_r.reduce_sum());
        }
    }

    fn apply_event(&mut self, kind: EventKind) {
        match kind {
            EventKind::NoteOn {
                midi_key,
                velocity_u16,
            } => {
                if velocity_u16 == 0 {
                    self.note_off(midi_key);
                } else {
                    self.note_on(midi_key, velocity_u16);
                }
            }
            EventKind::NoteOff { midi_key, .. } => self.note_off(midi_key),
            EventKind::Pedal { kind, value_u16 } => self.set_pedal(kind, value_u16),
            EventKind::AllNotesOff => self.all_notes_off(),
        }
    }

    fn note_on(&mut self, midi_key: u8, velocity_u16: u16) {
        let key = midi_key.min(127);
        let (voice_idx, retrigger) = self.allocate_voice(key);
        self.trigger_seq = self.trigger_seq.wrapping_add(1);

        let params = self.key_table[key as usize];
        let v_unit = u16_to_unit_f32(velocity_u16).max(1.0 / 65_535.0);
        let amp = 0.28 * (0.08 + 0.92 * v_unit * v_unit) * (1.0 - 0.25 * self.una_corda);
        let bright = (0.35 + 0.65 * v_unit) * (1.0 - 0.35 * self.una_corda);
        let weights = [
            PARTIAL_WEIGHTS[0],
            PARTIAL_WEIGHTS[1] * bright,
            PARTIAL_WEIGHTS[2] * bright * bright,
            PARTIAL_WEIGHTS[3] * bright * bright * bright,
        ];

        let sin_w = F32x4::from_array(params.sin_w);
        let b_vec = sin_w * F32x4::from_array(weights) * F32x4::splat(amp);

        if !retrigger {
            self.bank.clear_voice_state(voice_idx);
            self.bank
                .two_cos_w
                .store_f32x4(voice_idx, F32x4::from_array(params.two_cos_w));
            self.bank
                .r_hold
                .store_f32x4(voice_idx, F32x4::from_array(params.r_hold));
            self.bank
                .r_damp
                .store_f32x4(voice_idx, F32x4::from_array(params.r_damp));
            self.bank
                .gain_l
                .store_f32x4(voice_idx, F32x4::from_array(params.gain_l));
            self.bank
                .gain_r
                .store_f32x4(voice_idx, F32x4::from_array(params.gain_r));
        }

        self.bank.b_gain.store_f32x4(voice_idx, b_vec);
        self.voices[voice_idx] = VoiceState {
            active: true,
            key_down: true,
            sostenuto_latched: self.voices[voice_idx].sostenuto_latched && retrigger,
            midi_key: key,
            age: self.trigger_seq,
            pulse_remaining: params.pulse_len,
            pulse_sin: params.pulse_init_sin,
            pulse_cos: params.pulse_init_cos,
            pulse_step_sin: params.pulse_step_sin,
            pulse_step_cos: params.pulse_step_cos,
            pulse_scale: params.pulse_inv_sum,
        };
        self.refresh_voice_damping(voice_idx);
    }

    fn note_off(&mut self, midi_key: u8) {
        let key = midi_key.min(127);
        for v in 0..NUM_VOICES {
            if self.voices[v].active && self.voices[v].key_down && self.voices[v].midi_key == key {
                self.voices[v].key_down = false;
                self.refresh_voice_damping(v);
            }
        }
    }

    fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        let unit = u16_to_unit_f32(value_u16);
        match kind {
            PedalKind::Sustain => {
                self.sustain = unit;
                for v in 0..NUM_VOICES {
                    if self.voices[v].active {
                        self.refresh_voice_damping(v);
                    }
                }
            }
            PedalKind::UnaCorda => {
                self.una_corda = unit;
            }
            PedalKind::Sostenuto => {
                let down = unit >= 0.5;
                if down && !self.sostenuto_down {
                    for v in 0..NUM_VOICES {
                        if self.voices[v].active && self.voices[v].key_down {
                            self.voices[v].sostenuto_latched = true;
                        }
                    }
                } else if !down && self.sostenuto_down {
                    for v in 0..NUM_VOICES {
                        if self.voices[v].sostenuto_latched {
                            self.voices[v].sostenuto_latched = false;
                            if self.voices[v].active {
                                self.refresh_voice_damping(v);
                            }
                        }
                    }
                }
                self.sostenuto_down = down;
            }
        }
    }

    fn all_notes_off(&mut self) {
        self.sustain = 0.0;
        self.sostenuto_down = false;
        for v in 0..NUM_VOICES {
            if self.voices[v].active {
                self.voices[v].key_down = false;
                self.voices[v].sostenuto_latched = false;
                self.refresh_voice_damping(v);
            }
        }
    }

    fn allocate_voice(&self, midi_key: u8) -> (usize, bool) {
        for (idx, v) in self.voices.iter().enumerate() {
            if v.active && v.midi_key == midi_key {
                return (idx, true);
            }
        }
        for (idx, v) in self.voices.iter().enumerate() {
            if !v.active {
                return (idx, false);
            }
        }
        let mut best_idx = 0_usize;
        let mut best_released = !self.voices[0].key_down;
        let mut best_age = self.voices[0].age;
        for (idx, v) in self.voices.iter().enumerate().skip(1) {
            let released = !v.key_down;
            if (released && !best_released) || (released == best_released && v.age < best_age) {
                best_idx = idx;
                best_released = released;
                best_age = v.age;
            }
        }
        (best_idx, false)
    }

    fn refresh_voice_damping(&mut self, voice_idx: usize) {
        let v = self.voices[voice_idx];
        let r_hold = self.bank.r_hold.load_f32x4(voice_idx);
        let r_damp = self.bank.r_damp.load_f32x4(voice_idx);
        let two_cos_w = self.bank.two_cos_w.load_f32x4(voice_idx);

        let hold_amount = if v.key_down || v.sostenuto_latched {
            1.0_f32
        } else {
            self.sustain
        };
        let r = r_damp + (r_hold - r_damp) * F32x4::splat(hold_amount);
        self.bank.a1.store_f32x4(voice_idx, two_cos_w * r);
        self.bank.a2.store_f32x4(voice_idx, r * r);
    }

    fn retire_silent_voices(&mut self) {
        for v in 0..NUM_VOICES {
            if !self.voices[v].active {
                continue;
            }
            let q0 = self.bank.q0.load_f32x4(v).abs();
            let q1 = self.bank.q1.load_f32x4(v).abs();
            let peak_arr = q0.max(q1).to_array();
            let max_lane = peak_arr[0]
                .max(peak_arr[1])
                .max(peak_arr[2])
                .max(peak_arr[3]);

            let damped = !self.voices[v].key_down
                && !self.voices[v].sostenuto_latched
                && self.sustain < 0.99;
            if damped && self.voices[v].pulse_remaining == 0 && max_lane < VOICE_SILENCE_THRESHOLD {
                self.voices[v].active = false;
                self.bank.clear_voice_state(v);
                continue;
            }

            if max_lane < SUBNORMAL_FLUSH_THRESHOLD && self.voices[v].pulse_remaining == 0 {
                self.bank.q0.store_f32x4(v, F32x4::zero());
                self.bank.q1.store_f32x4(v, F32x4::zero());
            }
        }
    }
}

#[inline]
fn soft_limit(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let xc = x.clamp(-3.0, 3.0);
    let x2 = xc * xc;
    ((xc * (27.0 + x2)) / (27.0 + 9.0 * x2)).clamp(-1.0, 1.0)
}
