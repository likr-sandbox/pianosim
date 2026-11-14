//! Real-time 88-key / 230-string / ~26k-mode piano synthesizer engine core.

use piano_spec::{
    midi_to_key, u16_to_unit_f32, EventKind, PedalKind, SampleEvent, VoicingSpec, CHOIR_LAYOUTS,
    DAMPED_KEYS, NUM_KEYS,
};
use piano_strings::StringBank;

use crate::event_queue::{is_sorted_by_offset, EventQueue};
use crate::telemetry::{compute_block_telemetry, EngineTelemetry};

const KEY_SILENCE_RMS_THRESHOLD: f32 = 1.0e-4;

/// Core synthesizer state separated from scratch event queue for clean borrowing.
#[derive(Clone, Debug)]
struct EngineCore {
    bank: StringBank,
    key_down: [bool; NUM_KEYS],
    sostenuto_latched: [bool; NUM_KEYS],
    ringing: [bool; NUM_KEYS],
    sustain: f32,
    una_corda: f32,
    sostenuto_down: bool,
}

/// Zero-allocation 88-key polyphonic piano engine backed by [`StringBank`].
#[derive(Clone, Debug)]
pub struct PianoEngine {
    core: EngineCore,
    scratch_events: EventQueue,
    telemetry: EngineTelemetry,
}

impl PianoEngine {
    /// Allocates and precomputes all 88-key / 230-string tables with default voicing.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_voicing(sample_rate, &VoicingSpec::default())
    }

    /// Allocates and precomputes all 88-key / 230-string tables for `voicing`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self {
            core: EngineCore {
                bank: StringBank::new(sample_rate, voicing),
                key_down: [false; NUM_KEYS],
                sostenuto_latched: [false; NUM_KEYS],
                ringing: [false; NUM_KEYS],
                sustain: 0.0,
                una_corda: 0.0,
                sostenuto_down: false,
            },
            scratch_events: EventQueue::new(),
            telemetry: EngineTelemetry::default(),
        }
    }

    /// Returns the configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.core.bank.sample_rate()
    }

    /// Returns total SIMD4-padded modal slots driven on every sample (`~26,184`).
    #[must_use]
    #[inline]
    pub const fn total_modes(&self) -> usize {
        self.core.bank.total_modes()
    }

    /// Returns the telemetry captured during the most recent `render_block` call.
    #[must_use]
    #[inline]
    pub const fn telemetry(&self) -> EngineTelemetry {
        self.telemetry
    }

    /// Returns the count of currently active ringing keys.
    #[must_use]
    pub fn active_voices(&self) -> u32 {
        self.core.active_voice_count()
    }

    /// Resets all 88 keys, pedal states, and modal buffers without reallocating.
    pub fn reset(&mut self) {
        self.core.all_notes_off();
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

        self.core.retire_silent_keys();
        self.telemetry = compute_block_telemetry(self.core.active_voice_count(), out_l, out_r);
    }
}

impl EngineCore {
    fn active_voice_count(&self) -> u32 {
        self.ringing.iter().filter(|&&r| r).count() as u32
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
        self.bank.render_subblock(out_l, out_r);
        for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
            *l = soft_limit(*l);
            *r = soft_limit(*r);
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
        let Some(k) = midi_to_key(midi_key) else {
            return;
        };
        self.key_down[k] = true;
        self.ringing[k] = true;
        self.bank.set_key_hold(k, 1.0);

        let u = self.una_corda;
        let weights = match CHOIR_LAYOUTS[k].strings_per_key {
            3 => [1.0 - u, 1.0, 1.0],
            2 => [1.0 - u, 1.0, 0.0],
            _ => [1.0 - 0.25 * u, 0.0, 0.0],
        };

        let v = u16_to_unit_f32(velocity_u16).max(1.0 / 65_535.0);
        let amp = 0.08 + 0.92 * v * v;
        self.bank.trigger_key_pulse(k, weights, amp);
    }

    fn note_off(&mut self, midi_key: u8) {
        let Some(k) = midi_to_key(midi_key) else {
            return;
        };
        self.key_down[k] = false;
        let eff_hold = if self.sostenuto_latched[k] {
            1.0
        } else {
            self.sustain
        };
        self.bank.set_key_hold(k, eff_hold);
    }

    fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        let unit = u16_to_unit_f32(value_u16);
        match kind {
            PedalKind::Sustain => {
                self.sustain = unit;
                for k in 0..DAMPED_KEYS {
                    if !self.key_down[k] && !self.sostenuto_latched[k] {
                        self.bank.set_key_hold(k, unit);
                    }
                }
            }
            PedalKind::UnaCorda => {
                self.una_corda = unit;
            }
            PedalKind::Sostenuto => {
                let down = unit >= 0.5;
                if down && !self.sostenuto_down {
                    for k in 0..NUM_KEYS {
                        if self.key_down[k] {
                            self.sostenuto_latched[k] = true;
                        }
                    }
                } else if !down && self.sostenuto_down {
                    for k in 0..NUM_KEYS {
                        if self.sostenuto_latched[k] {
                            self.sostenuto_latched[k] = false;
                            if !self.key_down[k] {
                                self.bank.set_key_hold(k, self.sustain);
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
        self.una_corda = 0.0;
        self.sostenuto_down = false;
        self.key_down.fill(false);
        self.sostenuto_latched.fill(false);
        self.ringing.fill(false);
        self.bank.reset();
    }

    fn retire_silent_keys(&mut self) {
        for k in 0..NUM_KEYS {
            if !self.ringing[k] || self.key_down[k] || self.sostenuto_latched[k] {
                continue;
            }
            if k < DAMPED_KEYS && self.sustain >= 0.99 {
                continue;
            }
            let (v_rms, h_rms) = self.bank.measure_key_polarization_rms(k);
            if v_rms.max(h_rms) < KEY_SILENCE_RMS_THRESHOLD {
                self.ringing[k] = false;
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
