//! Real-time 88-key / 230-string / ~26k-mode physical piano engine core.

use piano_action::ActionBank;
use piano_spec::{
    midi_to_key, EventKind, PedalKind, SampleEvent, VoicingSpec, DAMPED_KEYS, NUM_KEYS,
};
use piano_strings::{KeyActionDriver, StringBank};

use crate::event_queue::{is_sorted_by_offset, EventQueue};
use crate::telemetry::{compute_block_telemetry, EngineTelemetry};

const KEY_SILENCE_RMS_THRESHOLD: f32 = 1.0e-4;
const DAMPER_ACTIVE_HOLD_CLEARANCE_M: f32 = 0.5e-3;

struct ActionDriver<'a>(&'a mut ActionBank);

impl KeyActionDriver for ActionDriver<'_> {
    #[inline]
    fn is_hammer_active(&self, key_idx: usize) -> bool {
        self.0.is_hammer_active(key_idx)
    }

    #[inline]
    fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        self.0.damper_clearance_m(key_idx)
    }

    #[inline]
    fn strike_weights(&self, key_idx: usize) -> [f32; 3] {
        self.0.strike_weights(key_idx)
    }

    #[inline]
    fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        self.0.step_key_clearance(key_idx)
    }

    #[inline]
    fn step_key_clearance_tile(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
    ) -> f32 {
        self.0.step_key_clearance_tile(key_idx, t_len, out_clears)
    }

    #[inline]
    fn step_hammer_sample(
        &mut self,
        key_idx: usize,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32 {
        self.0
            .step_hammer_sample(key_idx, y_str_m, v_str_mps, r_port_strike)
    }

    #[inline]
    fn step_damper_sample(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        self.0.step_damper_sample(
            key_idx,
            slot_idx,
            y_damp_m,
            v_damp_mps,
            r_port_damp,
            y_clear_m,
        )
    }
}

/// Core synthesizer state separated from scratch event queue for clean borrowing.
#[derive(Clone, Debug)]
struct EngineCore {
    action: ActionBank,
    bank: StringBank,
    key_down: [bool; NUM_KEYS],
    ringing: [bool; NUM_KEYS],
}

/// Zero-allocation 88-key polyphonic piano engine coupling Stage 1 [`ActionBank`]
/// with Stage 2 [`StringBank`].
#[derive(Clone, Debug)]
pub struct PianoEngine {
    core: EngineCore,
    scratch_events: EventQueue,
    telemetry: EngineTelemetry,
}

impl PianoEngine {
    /// Allocates and precomputes all 88-key action and string tables with default voicing.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_voicing(sample_rate, &VoicingSpec::default())
    }

    /// Allocates and precomputes all 88-key action and string tables for `voicing`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        Self {
            core: EngineCore {
                action: ActionBank::new(sample_rate, voicing),
                bank: StringBank::new(sample_rate, voicing),
                key_down: [false; NUM_KEYS],
                ringing: [false; NUM_KEYS],
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

    /// Resets all 88 keys, pedals, hammers, dampers, and modal buffers without reallocating.
    pub fn reset(&mut self) {
        self.core.all_notes_off();
        self.telemetry = EngineTelemetry::default();
    }

    #[cfg(test)]
    pub(crate) fn set_felt_memory_enabled(&mut self, enabled: bool) {
        self.core.action.set_memory_enabled(enabled);
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
        let mut driver = ActionDriver(&mut self.action);
        self.bank.render_subblock_coupled(&mut driver, out_l, out_r);
        self.action.decay_idle_memory(out_l.len());
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
                    self.note_off(midi_key, 0);
                } else {
                    self.note_on(midi_key, velocity_u16);
                }
            }
            EventKind::NoteOff {
                midi_key,
                release_vel_u16,
            } => self.note_off(midi_key, release_vel_u16),
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
        self.action.note_on(k, velocity_u16);
        self.bank.mark_key_possibly_active(k);
    }

    fn note_off(&mut self, midi_key: u8, release_vel_u16: u16) {
        let Some(k) = midi_to_key(midi_key) else {
            return;
        };
        self.key_down[k] = false;
        self.action.note_off(k, release_vel_u16);
    }

    fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        self.action.set_pedal(kind, value_u16);
    }

    fn all_notes_off(&mut self) {
        self.key_down.fill(false);
        self.ringing.fill(false);
        self.action.reset();
        self.bank.reset();
    }

    fn retire_silent_keys(&mut self) {
        for k in 0..NUM_KEYS {
            if !self.ringing[k] || self.key_down[k] || self.action.is_hammer_active(k) {
                continue;
            }
            if k < DAMPED_KEYS && self.action.damper_clearance_m(k) > DAMPER_ACTIVE_HOLD_CLEARANCE_M
            {
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
