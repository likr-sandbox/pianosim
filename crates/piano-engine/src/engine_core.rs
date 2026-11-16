//! Core synthesizer state and segment rendering pipeline.

use piano_action::ActionBank;
use piano_bridge::BridgeBank;
use piano_radiator::RadiatorBank;
use piano_spec::{
    midi_to_key, EventKind, PedalKind, SampleEvent, VoicingSpec, DAMPED_KEYS, NUM_KEYS,
};
use piano_strings::StringBank;

use crate::drivers::{ActionDriver, BridgeDriver};
use crate::param_smooth::{apply_raised_cosine_crossfade, SpaceCrossfadeGuard};

const KEY_SILENCE_RMS_THRESHOLD: f32 = 2.5e-4;
const DAMPER_ACTIVE_HOLD_CLEARANCE_M: f32 = 0.5e-3;
const RENDER_CHUNK: usize = 128;

#[derive(Clone)]
pub(crate) struct EngineScratch {
    pub zone8: [[f32; 8]; RENDER_CHUNK],
    pub zone9: [[f32; 9]; RENDER_CHUNK],
    pub bridge_l: [f32; RENDER_CHUNK],
    pub bridge_r: [f32; RENDER_CHUNK],
    pub temp_out_l: [f32; RENDER_CHUNK],
    pub temp_out_r: [f32; RENDER_CHUNK],
    pub temp_in_l: [f32; RENDER_CHUNK],
    pub temp_in_r: [f32; RENDER_CHUNK],
}

impl core::fmt::Debug for EngineScratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("EngineScratch { ... }")
    }
}

impl Default for EngineScratch {
    fn default() -> Self {
        Self {
            zone8: [[0.0; 8]; RENDER_CHUNK],
            zone9: [[0.0; 9]; RENDER_CHUNK],
            bridge_l: [0.0; RENDER_CHUNK],
            bridge_r: [0.0; RENDER_CHUNK],
            temp_out_l: [0.0; RENDER_CHUNK],
            temp_out_r: [0.0; RENDER_CHUNK],
            temp_in_l: [0.0; RENDER_CHUNK],
            temp_in_r: [0.0; RENDER_CHUNK],
        }
    }
}

/// Output audio tap selection (Stage 4 acoustic soundboard/room radiator vs Stage 3 direct bridge velocity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioTap {
    /// Stage 4 acoustic soundboard & room radiation via NUPOLS partitioned convolution.
    AcousticRadiator,
    /// Stage 3 direct bridge velocity tap for physical model diagnostics and legacy stage tests.
    BridgeDirect,
}

#[inline]
pub(crate) fn soft_limit(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let xc = x.clamp(-3.0, 3.0);
    let x2 = xc * xc;
    ((xc * (27.0 + x2)) / (27.0 + 9.0 * x2)).clamp(-1.0, 1.0)
}

#[derive(Clone, Debug)]
pub(crate) struct EngineCore {
    pub action: ActionBank,
    pub bank: StringBank,
    pub bridge: BridgeBank,
    pub radiators: [RadiatorBank; 2],
    pub scratch: EngineScratch,
    pub key_down: [bool; NUM_KEYS],
    pub ringing: [bool; NUM_KEYS],
}

impl EngineCore {
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let action = ActionBank::new(sample_rate, voicing);
        let bank = StringBank::new(sample_rate, voicing);
        let mut bridge = BridgeBank::new(bank.sample_rate());
        bridge.set_port_resistances(bank.bridge_port_resistances());
        let rad0 = RadiatorBank::new_concert_grand(f64::from(sample_rate));
        let rad1 = rad0.clone();
        Self {
            action,
            bank,
            bridge,
            radiators: [rad0, rad1],
            scratch: EngineScratch::default(),
            key_down: [false; NUM_KEYS],
            ringing: [false; NUM_KEYS],
        }
    }

    #[inline]
    pub fn active_voice_count(&self) -> u32 {
        self.ringing.iter().filter(|&&r| r).count() as u32
    }

    pub fn render_event_slice(
        &mut self,
        events: &[SampleEvent],
        out_l: &mut [f32],
        out_r: &mut [f32],
        space_guard: &mut SpaceCrossfadeGuard,
        tap: AudioTap,
    ) {
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
                    space_guard,
                    tap,
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
        space_guard: &mut SpaceCrossfadeGuard,
        tap: AudioTap,
    ) {
        let seg_len = out_l.len();
        let mut seg_cursor = 0;
        let mut ev_cursor = 0;

        while seg_cursor < seg_len {
            let chunk_len = (seg_len - seg_cursor).min(RENDER_CHUNK);
            let mut action_drv = ActionDriver {
                action: &mut self.action,
                key_down: &mut self.key_down,
                ringing: &mut self.ringing,
                mid_events,
                seg_base: seg_base + seg_cursor,
                ev_cursor,
            };
            let mut bridge_drv = BridgeDriver(&mut self.bridge);

            self.bank.render_subblock_bridge_zones_no_flush(
                &mut action_drv,
                &mut bridge_drv,
                &mut self.scratch.bridge_l[..chunk_len],
                &mut self.scratch.bridge_r[..chunk_len],
                &mut self.scratch.zone8[..chunk_len],
            );
            ev_cursor = action_drv.ev_cursor;

            for i in 0..chunk_len {
                let kb_force = self.action.step_keybed_force_sample();
                let z = self.scratch.zone8[i];
                self.scratch.zone9[i] = [z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], kb_force];
            }

            let active_idx = space_guard.active_radiator_idx();
            let dest_l = &mut out_l[seg_cursor..seg_cursor + chunk_len];
            let dest_r = &mut out_r[seg_cursor..seg_cursor + chunk_len];

            if tap == AudioTap::BridgeDirect {
                for i in 0..chunk_len {
                    dest_l[i] = soft_limit(self.scratch.bridge_l[i]);
                    dest_r[i] = soft_limit(self.scratch.bridge_r[i]);
                }
            } else if !space_guard.is_crossfading() {
                self.radiators[active_idx].process_block(
                    &self.scratch.zone9[..chunk_len],
                    dest_l,
                    dest_r,
                );
            } else {
                let fade_in_idx = 1 - active_idx;
                let start_smp = space_guard.elapsed_samples();
                let total_smp = space_guard.total_samples();
                self.radiators[active_idx].process_block(
                    &self.scratch.zone9[..chunk_len],
                    &mut self.scratch.temp_out_l[..chunk_len],
                    &mut self.scratch.temp_out_r[..chunk_len],
                );
                self.radiators[fade_in_idx].process_block(
                    &self.scratch.zone9[..chunk_len],
                    &mut self.scratch.temp_in_l[..chunk_len],
                    &mut self.scratch.temp_in_r[..chunk_len],
                );
                apply_raised_cosine_crossfade(
                    &self.scratch.temp_out_l[..chunk_len],
                    &self.scratch.temp_out_r[..chunk_len],
                    &self.scratch.temp_in_l[..chunk_len],
                    &self.scratch.temp_in_r[..chunk_len],
                    dest_l,
                    dest_r,
                    start_smp,
                    total_smp,
                );
                space_guard.step_block(chunk_len);
            }

            seg_cursor += chunk_len;
        }

        self.action.decay_idle_memory(seg_len);
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

    pub fn note_on(&mut self, midi_key: u8, velocity_u16: u16) {
        let Some(k) = midi_to_key(midi_key) else {
            return;
        };
        self.key_down[k] = true;
        self.ringing[k] = true;
        self.action.note_on(k, velocity_u16);
        self.bank.mark_key_possibly_active(k);
    }

    pub fn note_off(&mut self, midi_key: u8, release_vel_u16: u16) {
        let Some(k) = midi_to_key(midi_key) else {
            return;
        };
        self.key_down[k] = false;
        self.action.note_off(k, release_vel_u16);
    }

    pub fn set_pedal(&mut self, kind: PedalKind, value_u16: u16) {
        self.action.set_pedal(kind, value_u16);
    }

    pub fn all_notes_off(&mut self) {
        self.key_down.fill(false);
        self.ringing.fill(false);
        self.action.reset();
        self.bank.reset();
        self.bridge.reset();
        self.radiators[0].reset();
        self.radiators[1].reset();
    }

    pub fn retire_silent_keys(&mut self) {
        if self.action.sustain() > 0.0 {
            return;
        }
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
                self.bank.retire_key_struck_contact(k);
            }
        }
    }
}
