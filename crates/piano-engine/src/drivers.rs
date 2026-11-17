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
    fn damper_peel_force_n(&self, key_idx: usize) -> f32 {
        self.action.damper_peel_force_n(key_idx)
    }

    #[inline]
    fn step_key_clearance_tile_with_peel(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out: &mut [f32; 8],
        out_peel_n: &mut [f32; 8],
    ) -> (f32, bool) {
        self.action
            .step_key_clearance_tile_with_peel(key_idx, t_len, out, out_peel_n)
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

    #[inline]
    fn step_damper_pair(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        y_dv: f32,
        v_dv: f32,
        r_dv: f32,
        y_dh: f32,
        v_dh: f32,
        r_dh: f32,
        y_clear: f32,
    ) -> (f32, f32) {
        self.action.step_damper_pair(
            key_idx, str_idx, y_dv, v_dv, r_dv, y_dh, v_dh, r_dh, y_clear,
        )
    }

    #[inline]
    fn symp_linear_gains_if_settled(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> Option<(f32, f32)> {
        self.action
            .symp_linear_gains_if_settled(key_idx, str_idx, r_dv, r_dh, y_clear_m)
    }

    #[inline]
    fn symp_linear_gains_for_chunk(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        r_dv: f32,
        r_dh: f32,
        yc_mid: f32,
        sub: usize,
    ) -> Option<(f32, f32)> {
        self.action
            .symp_linear_gains_for_chunk(key_idx, str_idx, r_dv, r_dh, yc_mid, sub)
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

#[cfg(test)]
mod tests {
    use piano_spec::{midi_to_key, u7_to_u16, EventKind, PedalKind, SampleEvent, CHOIR_LAYOUTS};

    use crate::m4_tests::{note_off, note_on};
    use crate::PianoEngine;

    #[test]
    fn sustain_pump_on_virgin_keys_emits_zero_while_played_damped_key_peels_cleanly() {
        let mut eng = PianoEngine::new(48_000.0);
        let (mut out_l, mut out_r) = ([0.0_f32; 128], [0.0_f32; 128]);

        // 1. Pump Sustain 0 -> 1 -> 0 -> 1 on virgin engine: zero energy / zero audio.
        for (i, val) in [u16::MAX, 0, u16::MAX, 0].iter().enumerate() {
            let ev = [SampleEvent {
                sample_offset: 0,
                kind: EventKind::Pedal {
                    kind: PedalKind::Sustain,
                    value_u16: *val,
                },
            }];
            for b in 0..32 {
                let slice = if b == 0 { &ev[..] } else { &[] };
                eng.render_block_bridge(slice, &mut out_l, &mut out_r);
                for (&l, &r) in out_l.iter().zip(out_r.iter()) {
                    assert_eq!(l, 0.0, "virgin sustain pump block {i}/{b} produced audio");
                    assert_eq!(r, 0.0, "virgin sustain pump block {i}/{b} produced audio");
                }
            }
        }

        // 2. Strike C4 (60) at pp (audible > v_esc), release and let it damp to rest, then lift Sustain.
        eng.render_block_bridge(&[note_on(60, u7_to_u16(1), 0)], &mut out_l, &mut out_r);
        for _ in 0..20 {
            eng.render_block_bridge(&[], &mut out_l, &mut out_r);
        }
        eng.render_block_bridge(&[note_off(60, 48_000, 0)], &mut out_l, &mut out_r);
        for _ in 0..1_200 {
            eng.render_block_bridge(&[], &mut out_l, &mut out_r);
        }
        let k_c4 = midi_to_key(60).unwrap();
        let layout_c4 = CHOIR_LAYOUTS[k_c4];
        let s_c4 = usize::from(layout_c4.first_string_idx);
        let e_before: f64 = (s_c4..s_c4 + usize::from(layout_c4.strings_per_key))
            .map(|s| eng.string_linear_energy_joules(s))
            .sum();

        // Lift Sustain: C4 damper peels off, producing subtle prefix excitation while neighbor C#4 stays 0.0.
        let mut peel_peak = 0.0_f32;
        for b in 0..30 {
            let ev = if b == 0 {
                &[SampleEvent {
                    sample_offset: 0,
                    kind: EventKind::Pedal {
                        kind: PedalKind::Sustain,
                        value_u16: u16::MAX,
                    },
                }][..]
            } else {
                &[]
            };
            eng.render_block_bridge(ev, &mut out_l, &mut out_r);
            for (&l, &r) in out_l.iter().zip(out_r.iter()) {
                assert!(l.is_finite() && r.is_finite());
                peel_peak = peel_peak.max(l.abs()).max(r.abs());
            }
        }
        let e_after: f64 = (s_c4..s_c4 + usize::from(layout_c4.strings_per_key))
            .map(|s| eng.string_linear_energy_joules(s))
            .sum();
        assert!(
            e_after > e_before + 1.0e-12 && peel_peak > 0.0,
            "expected peel-off bloom on damped C4: e_before={e_before:.3e}, e_after={e_after:.3e}, peel_peak={peel_peak:.3e}"
        );
        let k_cs4 = midi_to_key(61).unwrap();
        let l_cs4 = CHOIR_LAYOUTS[k_cs4];
        let s_cs4 = usize::from(l_cs4.first_string_idx);
        let e_cs4: f64 = (s_cs4..s_cs4 + usize::from(l_cs4.strings_per_key))
            .map(|s| eng.string_linear_energy_joules(s))
            .sum();
        assert!(
            e_cs4 < e_after * 1.0e-2,
            "virgin neighbor C#4 should not wake: {e_cs4:.3e}"
        );
    }

    #[test]
    fn note_off_zone8_release_shock_and_rapid_restrike_radiate_cleanly() {
        let mut eng = PianoEngine::new(48_000.0);
        let mut out_l = vec![0.0_f32; 1_536];
        let mut out_r = vec![0.0_f32; 1_536];
        // Direct NoteOff release shock through Stage 4 Zone 8 radiator path.
        eng.render_block(&[note_off(60, u7_to_u16(127), 0)], &mut out_l, &mut out_r);
        let peak = out_l
            .iter()
            .chain(out_r.iter())
            .fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(
            peak > 1.0e-5,
            "expected audible Zone 8 release shock, got {peak}"
        );

        // Rapid re-strikes on C4 remain finite and click-free.
        eng.reset();
        let mut prev = 0.0_f32;
        let mut max_delta = 0.0_f32;
        for _ in 0..8 {
            eng.render_block(
                &[note_on(60, u16::MAX, 0)],
                &mut out_l[..128],
                &mut out_r[..128],
            );
            for &x in &out_l[..128] {
                assert!(x.is_finite());
                max_delta = max_delta.max((x - prev).abs());
                prev = x;
            }
            for _ in 0..6 {
                eng.render_block(&[], &mut out_l[..128], &mut out_r[..128]);
                for &x in &out_l[..128] {
                    assert!(x.is_finite());
                    max_delta = max_delta.max((x - prev).abs());
                    prev = x;
                }
            }
        }
        assert!(
            max_delta < 0.35,
            "rapid re-strike discontinuity: {max_delta}"
        );
    }
}
