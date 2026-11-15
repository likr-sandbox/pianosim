//! Real-time 88-key / 230-string / ~26k-mode physical piano engine core.

use piano_action::ActionBank;
use piano_bridge::BridgeBank;
use piano_spec::{
    midi_to_key, EventKind, PedalKind, SampleEvent, VoicingSpec, DAMPED_KEYS, NUM_KEYS,
    TOTAL_STRINGS,
};
use piano_strings::{BridgePortDriver, KeyActionDriver, StringBank};

use crate::event_queue::{is_sorted_by_offset, EventQueue};
use crate::telemetry::{compute_block_telemetry, EngineTelemetry};

const KEY_SILENCE_RMS_THRESHOLD: f32 = 1.0e-4;
const DAMPER_ACTIVE_HOLD_CLEARANCE_M: f32 = 0.5e-3;

struct ActionDriver<'a> {
    action: &'a mut ActionBank,
    key_down: &'a mut [bool; NUM_KEYS],
    ringing: &'a mut [bool; NUM_KEYS],
    mid_events: &'a [SampleEvent],
    seg_base: usize,
    ev_cursor: usize,
}

impl KeyActionDriver for ActionDriver<'_> {
    #[inline]
    fn is_hammer_active(&self, key_idx: usize) -> bool {
        self.action.is_hammer_active(key_idx)
    }
    #[inline]
    fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        self.action.damper_clearance_m(key_idx)
    }
    #[inline]
    fn strike_weights(&self, key_idx: usize) -> [f32; 3] {
        self.action.strike_weights(key_idx)
    }
    #[inline]
    fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        self.action.step_key_clearance(key_idx)
    }
    #[inline]
    fn step_key_clearance_tile(&mut self, key_idx: usize, t_len: usize, out: &mut [f32; 8]) -> f32 {
        self.action.step_key_clearance_tile(key_idx, t_len, out)
    }
    #[inline]
    fn has_mid_block_event(&self, key_idx: usize, chunk_base: usize, sub: usize) -> bool {
        let lo = self.seg_base + chunk_base;
        let hi = lo + sub;
        for &ev in &self.mid_events[self.ev_cursor..] {
            let off = ev.sample_offset as usize;
            if off >= hi {
                break;
            }
            if off >= lo {
                let k_opt = match ev.kind {
                    EventKind::NoteOn { midi_key, .. } | EventKind::NoteOff { midi_key, .. } => {
                        midi_to_key(midi_key)
                    }
                    _ => None,
                };
                if k_opt == Some(key_idx) {
                    return true;
                }
            }
        }
        false
    }
    #[inline]
    fn apply_sample_events_at(&mut self, sample_idx: usize) -> u128 {
        let target = self.seg_base + sample_idx;
        let mut note_on_mask = 0_u128;
        while self.ev_cursor < self.mid_events.len() {
            let ev = self.mid_events[self.ev_cursor];
            if (ev.sample_offset as usize) > target {
                break;
            }
            self.ev_cursor += 1;
            match ev.kind {
                EventKind::NoteOn {
                    midi_key,
                    velocity_u16,
                } => {
                    if let Some(k) = midi_to_key(midi_key) {
                        if velocity_u16 == 0 {
                            self.key_down[k] = false;
                            self.action.note_off(k, 0);
                        } else {
                            self.key_down[k] = true;
                            self.ringing[k] = true;
                            self.action.note_on(k, velocity_u16);
                            note_on_mask |= 1_u128 << k;
                        }
                    }
                }
                EventKind::NoteOff {
                    midi_key,
                    release_vel_u16,
                } => {
                    if let Some(k) = midi_to_key(midi_key) {
                        self.key_down[k] = false;
                        self.action.note_off(k, release_vel_u16);
                    }
                }
                _ => {}
            }
        }
        note_on_mask
    }
    #[inline]
    fn step_hammer_sample(&mut self, key_idx: usize, y_s: f32, v_s: f32, r_h: f32) -> f32 {
        self.action.step_hammer_sample(key_idx, y_s, v_s, r_h)
    }
    #[inline]
    fn step_damper_sample(
        &mut self,
        key_idx: usize,
        pad_slot: usize,
        y_d: f32,
        v_d: f32,
        r_d: f32,
        y_clear: f32,
    ) -> f32 {
        self.action
            .step_damper_sample(key_idx, pad_slot, y_d, v_d, r_d, y_clear)
    }
}

struct BridgeDriver<'a>(&'a mut BridgeBank);

impl BridgePortDriver for BridgeDriver<'_> {
    #[inline]
    fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        self.0.port_resistances()
    }

    #[inline]
    fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS] {
        self.0.zone_weights()
    }

    #[inline]
    fn step_bridge_sample(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
    ) -> (f32, f32) {
        self.0
            .step_bridge_sample(a_waves, zone_str_audio, b_waves_out, zone_out)
    }
}

/// Core synthesizer state separated from scratch event queue for clean borrowing.
#[derive(Clone, Debug)]
struct EngineCore {
    action: ActionBank,
    bank: StringBank,
    bridge: BridgeBank,
    key_down: [bool; NUM_KEYS],
    ringing: [bool; NUM_KEYS],
}

/// Zero-allocation 88-key polyphonic piano engine coupling Stage 1 [`ActionBank`],
/// Stage 2 [`StringBank`], and Stage 3 [`BridgeBank`].
#[derive(Clone, Debug)]
pub struct PianoEngine {
    core: EngineCore,
    scratch_events: EventQueue,
    telemetry: EngineTelemetry,
}

impl PianoEngine {
    /// Allocates and precomputes all 88-key action, string, and bridge tables with default voicing.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_voicing(sample_rate, &VoicingSpec::default())
    }

    /// Allocates and precomputes all 88-key action, string, and bridge tables for `voicing`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let action = ActionBank::new(sample_rate, voicing);
        let bank = StringBank::new(sample_rate, voicing);
        let mut bridge = BridgeBank::new(bank.sample_rate());
        bridge.set_port_resistances(bank.bridge_port_resistances());
        Self {
            core: EngineCore {
                action,
                bank,
                bridge,
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

    #[cfg(test)]
    pub(crate) fn set_nonlinear_coupling(
        &mut self,
        longitudinal_enabled: bool,
        sav_phantom_enabled: bool,
    ) {
        self.core
            .bank
            .set_nonlinear_coupling(longitudinal_enabled, sav_phantom_enabled);
    }

    #[cfg(test)]
    pub(crate) fn total_sav_energy_joules(&self) -> f64 {
        self.core.bank.total_sav_energy_joules()
            + self
                .core
                .bank
                .bridge_link_energy_joules(self.core.bridge.port_resistances())
            + self.core.bridge.total_energy_joules()
    }

    #[cfg(test)]
    pub(crate) fn string_linear_energy_joules(&self, s_idx: usize) -> f64 {
        self.core.bank.string_linear_energy_joules(s_idx)
    }

    #[cfg(test)]
    pub(crate) fn key_sav_energy_joules(&self, key_idx: usize) -> f64 {
        self.core.bank.key_sav_energy_joules(key_idx)
    }

    #[cfg(test)]
    pub(crate) fn key_nonlinear_energy_joules(&self, key_idx: usize) -> f64 {
        self.core.bank.key_nonlinear_energy_joules(key_idx)
    }

    #[cfg(test)]
    pub(crate) fn is_any_hammer_active(&self) -> bool {
        (0..NUM_KEYS).any(|k| self.core.action.is_hammer_active(k))
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
        let (mut cursor, mut idx) = (0_usize, 0_usize);
        while cursor < frames {
            while idx < events.len() && (events[idx].sample_offset as usize) <= cursor {
                self.apply_event(events[idx].kind);
                idx += 1;
            }
            let mid_start = idx;
            let mut seg_end = frames;
            while idx < events.len() {
                let off = (events[idx].sample_offset as usize).min(frames);
                if off >= frames {
                    break;
                }
                if matches!(
                    events[idx].kind,
                    EventKind::Pedal { .. } | EventKind::AllNotesOff
                ) {
                    seg_end = off;
                    break;
                }
                idx += 1;
            }
            if seg_end > cursor {
                self.render_segment(
                    cursor,
                    &events[mid_start..idx],
                    &mut out_l[cursor..seg_end],
                    &mut out_r[cursor..seg_end],
                );
                cursor = seg_end;
            }
        }
        while idx < events.len() {
            self.apply_event(events[idx].kind);
            idx += 1;
        }
        let hammers = core::array::from_fn(|k| self.action.is_hammer_active(k));
        self.bank.flush_bridge_subnormals(&hammers);
    }

    fn render_segment(
        &mut self,
        seg_base: usize,
        mid_events: &[SampleEvent],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let mut action_drv = ActionDriver {
            action: &mut self.action,
            key_down: &mut self.key_down,
            ringing: &mut self.ringing,
            mid_events,
            seg_base,
            ev_cursor: 0,
        };
        let mut bridge_drv = BridgeDriver(&mut self.bridge);
        self.bank.render_subblock_bridge_coupled_no_flush(
            &mut action_drv,
            &mut bridge_drv,
            out_l,
            out_r,
        );
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
        self.bridge.reset();
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
