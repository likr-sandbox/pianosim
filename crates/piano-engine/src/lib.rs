//! Zero-allocation 4-stage real-time piano synthesizer engine.

pub mod engine;
pub mod event_queue;
pub mod telemetry;

pub use engine::PianoEngine;
pub use event_queue::{EventQueue, MAX_EVENTS};
pub use piano_spec;
pub use telemetry::EngineTelemetry;

#[cfg(test)]
mod tests {
    use core::f64::consts::TAU;

    use piano_spec::{
        build_scale_design, key_tuned_f0_hz, partial_frequency_hz, u7_to_u16, EventKind, PedalKind,
        SampleEvent, VoicingSpec,
    };

    use super::*;

    fn rms(slice: &[f32]) -> f32 {
        let sum_sq: f64 = slice.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        (sum_sq / (slice.len().max(1) as f64)).sqrt() as f32
    }

    fn peak_abs(slice: &[f32]) -> f32 {
        slice.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()))
    }

    fn estimate_peak_freq_hz(samples: &[f32], sample_rate: f64, f_min: f64, f_max: f64) -> f64 {
        let mut best_freq = f_min;
        let mut best_mag = -1.0_f64;

        let mut f = f_min;
        while f <= f_max {
            let mag = goertzel_mag_sq(samples, sample_rate, f);
            if mag > best_mag {
                best_mag = mag;
                best_freq = f;
            }
            f += 0.5;
        }

        let start = (best_freq - 1.0).max(f_min);
        let end = (best_freq + 1.0).min(f_max);
        let mut fine = start;
        while fine <= end {
            let mag = goertzel_mag_sq(samples, sample_rate, fine);
            if mag > best_mag {
                best_mag = mag;
                best_freq = fine;
            }
            fine += 0.01;
        }
        best_freq
    }

    fn goertzel_mag_sq(samples: &[f32], sample_rate: f64, freq_hz: f64) -> f64 {
        let w = TAU * freq_hz / sample_rate;
        let coeff = 2.0 * w.cos();
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;
        for &x in samples {
            let s0 = f64::from(x) + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - coeff * s1 * s2
    }

    fn render_chunks(
        engine: &mut PianoEngine,
        total_frames: usize,
        block_size: usize,
        first_events: &[SampleEvent],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out_l = vec![0.0_f32; total_frames];
        let mut out_r = vec![0.0_f32; total_frames];
        let mut offset = 0_usize;
        let mut first = true;
        while offset < total_frames {
            let end = (offset + block_size).min(total_frames);
            let evs = if first { first_events } else { &[] };
            first = false;
            engine.render_block(evs, &mut out_l[offset..end], &mut out_r[offset..end]);
            offset = end;
        }
        (out_l, out_r)
    }

    #[test]
    fn a4_pitch_accuracy_440hz_and_mode_count() {
        let mut engine = PianoEngine::new(48_000.0);
        assert!(
            engine.total_modes() >= 25_000,
            "expected >= 25,000 SIMD modes, got {}",
            engine.total_modes()
        );

        let on = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 69,
                velocity_u16: u7_to_u16(100),
            },
        }];

        let (out_l, out_r) = render_chunks(&mut engine, 24_000, 128, &on);
        for (&l, &r) in out_l.iter().zip(out_r.iter()) {
            assert!(l.is_finite() && r.is_finite());
            assert!(l.abs() <= 1.0 && r.abs() <= 1.0);
        }

        let mono: Vec<f32> = out_l
            .iter()
            .zip(out_r.iter())
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        let mono_rms = rms(&mono);
        let peak = peak_abs(&out_l).max(peak_abs(&out_r));
        assert!(
            mono_rms > 0.01,
            "expected audible A4 rms > 0.01, got {mono_rms}"
        );
        assert!(peak < 0.8, "expected A4 peak < 0.8, got {peak}");

        let peak_hz = estimate_peak_freq_hz(&mono[1024..17408], 48_000.0, 410.0, 470.0);
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);
        let expected_a4 = partial_frequency_hz(440.0, specs[48].inharmonicity_b, 1);
        assert!(
            (peak_hz - expected_a4).abs() < 0.5 && (peak_hz - 440.0).abs() < 0.6,
            "expected A4 near {expected_a4:.2} Hz (~440 Hz), got {peak_hz}"
        );
    }

    #[test]
    fn distinct_stereo_panning_for_bass_and_treble() {
        let mut engine = PianoEngine::new(48_000.0);
        let bass_on = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 36,
                velocity_u16: u7_to_u16(110),
            },
        }];
        let (bass_l, bass_r) = render_chunks(&mut engine, 4096, 128, &bass_on);
        assert!(
            rms(&bass_l) > rms(&bass_r) * 1.15,
            "expected bass C2 panned left: L={}, R={}",
            rms(&bass_l),
            rms(&bass_r)
        );

        engine.reset();
        let treble_on = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 96,
                velocity_u16: u7_to_u16(110),
            },
        }];
        let (treble_l, treble_r) = render_chunks(&mut engine, 4096, 128, &treble_on);
        assert!(
            rms(&treble_r) > rms(&treble_l) * 1.15,
            "expected treble C7 panned right: L={}, R={}",
            rms(&treble_l),
            rms(&treble_r)
        );
        assert!(
            bass_l
                .iter()
                .zip(bass_r.iter())
                .any(|(l, r)| (l - r).abs() > 1e-4),
            "expected distinct L/R samples"
        );
    }

    #[test]
    fn sustain_pedal_holds_note_longer_than_damper_release() {
        let sr = 48_000.0_f32;
        let note_on = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 60,
                velocity_u16: u7_to_u16(105),
            },
        }];
        let note_off = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOff {
                midi_key: 60,
                release_vel_u16: 0,
            },
        }];

        let mut dry = PianoEngine::new(sr);
        let _ = render_chunks(&mut dry, 4800, 128, &note_on);
        let (dry_tail_l, _) = render_chunks(&mut dry, 19_200, 128, &note_off);

        let mut sus = PianoEngine::new(sr);
        let sus_on = [
            SampleEvent {
                sample_offset: 0,
                kind: EventKind::Pedal {
                    kind: PedalKind::Sustain,
                    value_u16: u16::MAX,
                },
            },
            note_on[0],
        ];
        let _ = render_chunks(&mut sus, 4800, 128, &sus_on);
        let (sus_tail_l, _) = render_chunks(&mut sus, 19_200, 128, &note_off);

        let dry_late_rms = rms(&dry_tail_l[14_400..]);
        let sus_late_rms = rms(&sus_tail_l[14_400..]);
        assert!(
            dry_late_rms < 1e-3,
            "expected fast damper decay in tail, got {dry_late_rms}"
        );
        assert!(
            sus_late_rms > dry_late_rms * 20.0,
            "expected sustain tail ({sus_late_rms}) >> dry release tail ({dry_late_rms})"
        );
        assert_eq!(dry.active_voices(), 0);
        assert!(sus.active_voices() >= 1);
    }

    #[test]
    fn sample_accurate_offset_splitting_and_unsorted_events() {
        let mut engine = PianoEngine::new(48_000.0);
        let mut out_l = [0.0_f32; 256];
        let mut out_r = [0.0_f32; 256];

        let events = [
            SampleEvent {
                sample_offset: 192,
                kind: EventKind::NoteOff {
                    midi_key: 69,
                    release_vel_u16: 0,
                },
            },
            SampleEvent {
                sample_offset: 64,
                kind: EventKind::NoteOn {
                    midi_key: 69,
                    velocity_u16: u7_to_u16(127),
                },
            },
        ];

        engine.render_block(&events, &mut out_l, &mut out_r);
        assert!(
            out_l[..64].iter().all(|&x| x == 0.0),
            "samples before offset 64 must remain strictly zero"
        );
        assert!(
            out_l[64..192].iter().any(|&x| x.abs() > 1e-4),
            "samples from offset 64 must contain attack energy"
        );
    }

    #[test]
    fn full_88_key_polyphony_sostenuto_una_corda_and_panic() {
        let mut engine = PianoEngine::new(48_000.0);
        let mut q = EventQueue::new();

        assert!(q.push(SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::UnaCorda,
                value_u16: u16::MAX,
            },
        }));
        for i in 0..18_u8 {
            assert!(q.push(SampleEvent {
                sample_offset: u32::from(i) * 4,
                kind: EventKind::NoteOn {
                    midi_key: 36 + i * 3,
                    velocity_u16: u16::MAX,
                },
            }));
        }
        q.sort_by_offset();

        let (out_l, out_r) = render_chunks(&mut engine, 8192, 256, q.as_slice());
        assert_eq!(engine.active_voices(), 18);
        for (&l, &r) in out_l.iter().zip(out_r.iter()) {
            assert!(l.is_finite() && r.is_finite());
            assert!(l.abs() <= 1.0 && r.abs() <= 1.0);
        }

        // AllNotesOff must immediately stop even undamped treble notes.
        let panic = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::AllNotesOff,
        }];
        let (post_l, post_r) = render_chunks(&mut engine, 256, 128, &panic);
        assert_eq!(engine.active_voices(), 0);
        assert!(post_l.iter().chain(post_r.iter()).all(|&x| x == 0.0));

        // Sostenuto latches currently held notes only.
        engine.reset();
        let mut sos_ev = EventQueue::new();
        sos_ev.push(SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 60,
                velocity_u16: u16::MAX,
            },
        });
        sos_ev.push(SampleEvent {
            sample_offset: 32,
            kind: EventKind::Pedal {
                kind: PedalKind::Sostenuto,
                value_u16: u16::MAX,
            },
        });
        sos_ev.push(SampleEvent {
            sample_offset: 64,
            kind: EventKind::NoteOn {
                midi_key: 72,
                velocity_u16: u16::MAX,
            },
        });
        sos_ev.push(SampleEvent {
            sample_offset: 96,
            kind: EventKind::NoteOff {
                midi_key: 60,
                release_vel_u16: 0,
            },
        });
        sos_ev.push(SampleEvent {
            sample_offset: 96,
            kind: EventKind::NoteOff {
                midi_key: 72,
                release_vel_u16: 0,
            },
        });
        let _ = render_chunks(&mut engine, 24_000, 128, sos_ev.as_slice());
        assert_eq!(engine.active_voices(), 1);
    }

    #[test]
    fn high_treble_c8_has_strong_attack_and_railsback_pitch() {
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);
        let f0_c8 = key_tuned_f0_hz(87, &specs, &voicing);
        let expected_c8 = partial_frequency_hz(f0_c8, specs[87].inharmonicity_b, 1);

        let mut engine = PianoEngine::new_with_voicing(48_000.0, &voicing);
        let on = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 108,
                velocity_u16: u16::MAX,
            },
        }];
        let (out_l, out_r) = render_chunks(&mut engine, 8192, 128, &on);
        let mono: Vec<f32> = out_l
            .iter()
            .zip(out_r.iter())
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        assert!(rms(&mono[..2048]) > 0.01, "expected audible C8 attack");
        let peak_hz = estimate_peak_freq_hz(&mono[64..4096], 48_000.0, 3500.0, 5000.0);
        assert!(
            (peak_hz - expected_c8).abs() < 2.0,
            "expected C8 near {expected_c8} Hz, got {peak_hz}"
        );
        let telem = engine.telemetry().to_array();
        assert!(telem[0] >= 1.0 && telem[1] > 0.0 && telem[2] > 0.0 && telem[3] > 0.0);
        assert_eq!(&telem[4..], &[0.0, 0.0, 0.0, 0.0]);
    }
}
