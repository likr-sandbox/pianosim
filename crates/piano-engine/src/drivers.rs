//! Action and Bridge drivers for real-time string coupling.

use piano_action::ActionBank;
use piano_bridge::BridgeBank;
use piano_spec::{midi_to_key, EventKind, SampleEvent, NUM_KEYS, TOTAL_STRINGS};
use piano_strings::{BridgePortDriver, KeyActionDriver};

pub(crate) struct ActionDriver<'a> {
    pub action: &'a mut ActionBank,
    pub key_down: &'a mut [bool; NUM_KEYS],
    pub ringing: &'a mut [bool; NUM_KEYS],
    pub mid_events: &'a [SampleEvent],
    pub seg_base: usize,
    pub ev_cursor: usize,
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

pub(crate) struct BridgeDriver<'a>(pub &'a mut BridgeBank);

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

    #[inline]
    fn step_bridge_sample_with_duplex_zones(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
        dup_zone_out: &mut [f32; 8],
    ) -> (f32, f32) {
        self.0.step_bridge_sample_with_duplex_zones(
            a_waves,
            zone_str_audio,
            b_waves_out,
            zone_out,
            dup_zone_out,
        )
    }
}
