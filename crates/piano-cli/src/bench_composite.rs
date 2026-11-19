//! Balasubramaniam (DAFx 2026) Composite Stress Workload and Scenario Runner.
//!
//! Evaluates the engine under heavy concurrent acoustic stress:
//! - 10-12 voice sustained chord across bass, tenor, and treble registers
//! - Continuous rapid 32nd-note rolling arpeggios cycling across octaves
//! - Continuous triangular CC64 sustain pedal pumping (half-pedal clearance sweep)
//! - Live Voicing Dirty Mask modification (smooth hammer hardness sweep)

use std::hint::black_box;
use std::time::Instant;

use piano_engine::{PerfProfile, PianoEngine};
use piano_radiator::{
    ir_len_for_sample_rate_secs, synthesize_irs_with_duration, BodyPreset, IrSynthScratch,
    LidState, ListenerPerspective, RadiatorBank, CONCERT_GRAND_PRESET_BYTES, PRESET_RADIATOR_ZONES,
};
use piano_spec::{u7_to_u16, EventKind, PedalKind, SampleEvent, VoicingSpec};

use crate::bench_rt::{compute_percentiles, Percentiles};

pub const WARMUP_BLOCKS: usize = 128;
pub const TIMED_BLOCKS: usize = 1_000;

pub const CHORD_KEYS: [u8; 12] = [36, 43, 48, 52, 55, 60, 64, 67, 71, 72, 76, 79];
pub const ARPEGGIO_KEYS: [u8; 8] = [48, 55, 60, 64, 67, 72, 76, 84];
pub const SUSTAIN_20_KEYS: [u8; 20] = [
    36, 40, 43, 47, 48, 52, 55, 59, 60, 64, 67, 71, 72, 74, 76, 79, 81, 83, 84, 88,
];

pub const COMPOSITE_CHORD_KEYS: [u8; 10] = [28, 36, 43, 48, 55, 60, 64, 67, 72, 76];
pub const COMPOSITE_ARP_KEYS: [u8; 16] = [
    48, 52, 55, 60, 64, 67, 72, 76, 79, 84, 88, 84, 79, 76, 72, 67,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Workload {
    Chord12Arp,
    Sustain20,
    BalasubramaniamComposite,
}

#[derive(Copy, Clone, Debug)]
pub struct BenchScenario {
    pub label: &'static str,
    pub frames: usize,
    pub sample_rate: f32,
    pub workload: Workload,
    pub enforce_gate: bool,
    pub gate_budget_us: Option<f64>,
}

#[derive(Copy, Clone, Debug)]
pub struct IrBenchStats {
    pub min_ms: f64,
    pub p50_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Default)]
pub struct BalasubramaniamComposite {
    step_cursor: usize,
    prev_arp_key: u8,
}

impl BalasubramaniamComposite {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            step_cursor: 0,
            prev_arp_key: 0,
        }
    }

    pub fn init(&mut self, engine: &mut PianoEngine, out_l: &mut [f32], out_r: &mut [f32]) {
        let noop = SampleEvent {
            sample_offset: 0,
            kind: EventKind::AllNotesOff,
        };
        let mut init_events = [noop; 11];
        init_events[0] = SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        };
        for (i, &key) in COMPOSITE_CHORD_KEYS.iter().enumerate() {
            init_events[i + 1] = SampleEvent {
                sample_offset: 0,
                kind: EventKind::NoteOn {
                    midi_key: key,
                    velocity_u16: u7_to_u16(100),
                },
            };
        }
        engine.render_block(&init_events, out_l, out_r);
        self.prev_arp_key = COMPOSITE_ARP_KEYS[0];
    }

    pub fn step(
        &mut self,
        engine: &mut PianoEngine,
        block: usize,
        frames: usize,
        events: &mut [SampleEvent; 3],
    ) -> usize {
        let mut count = 0;

        // 1. Continuous sustain pedal pumping: musical repedaling sweep every 128 blocks (~0.34 s)
        if block.is_multiple_of(16) {
            let phase = (block % 128) as f32 / 128.0;
            let tri = if phase < 0.80 {
                1.0_f32
            } else if phase < 0.90 {
                1.0 - (phase - 0.80) / 0.10
            } else {
                (phase - 0.90) / 0.10
            };
            events[count] = SampleEvent {
                sample_offset: 0,
                kind: EventKind::Pedal {
                    kind: PedalKind::Sustain,
                    value_u16: (tri * 65535.0) as u16,
                },
            };
            count += 1;
        }

        // 2. Rapid 32nd-note rolling arpeggios (cycling octaves)
        if block.is_multiple_of(8) {
            let next_key = COMPOSITE_ARP_KEYS[self.step_cursor % COMPOSITE_ARP_KEYS.len()];
            events[count] = SampleEvent {
                sample_offset: 0,
                kind: EventKind::NoteOff {
                    midi_key: self.prev_arp_key,
                    release_vel_u16: u7_to_u16(64),
                },
            };
            count += 1;
            events[count] = SampleEvent {
                sample_offset: (frames / 2) as u32,
                kind: EventKind::NoteOn {
                    midi_key: next_key,
                    velocity_u16: u7_to_u16(104),
                },
            };
            count += 1;
            self.prev_arp_key = next_key;
            self.step_cursor = self.step_cursor.wrapping_add(1);
        }

        // 3. Live Voicing Dirty Mask modification (smooth hammer hardness sweep < 1 us)
        if block.is_multiple_of(16) {
            let hardness = 0.35 + 0.30 * ((block as f32 * 0.02).sin() * 0.5 + 0.5);
            engine.set_hammer_hardness(hardness);
        }

        count
    }
}

pub fn measure_scenario(scenario: &BenchScenario, profile: PerfProfile) -> Percentiles {
    let mut engine =
        PianoEngine::new_with_profile(scenario.sample_rate, VoicingSpec::default(), profile);
    let mut out_l = vec![0.0_f32; scenario.frames];
    let mut out_r = vec![0.0_f32; scenario.frames];
    let mut timings_us = vec![0.0_f64; TIMED_BLOCKS];
    let noop = SampleEvent {
        sample_offset: 0,
        kind: EventKind::AllNotesOff,
    };
    let mut composite = BalasubramaniamComposite::new();

    match scenario.workload {
        Workload::BalasubramaniamComposite => {
            composite.init(&mut engine, &mut out_l, &mut out_r);
        }
        Workload::Chord12Arp | Workload::Sustain20 => {
            let mut init_events = [noop; 21];
            init_events[0] = SampleEvent {
                sample_offset: 0,
                kind: EventKind::Pedal {
                    kind: PedalKind::Sustain,
                    value_u16: u16::MAX,
                },
            };
            let keys = if scenario.workload == Workload::Chord12Arp {
                &CHORD_KEYS[..]
            } else {
                &SUSTAIN_20_KEYS[..]
            };
            for (idx, &key) in keys.iter().enumerate() {
                init_events[idx + 1] = SampleEvent {
                    sample_offset: 0,
                    kind: EventKind::NoteOn {
                        midi_key: key,
                        velocity_u16: u7_to_u16(100),
                    },
                };
            }
            engine.render_block(&init_events[..keys.len() + 1], &mut out_l, &mut out_r);
        }
    }

    let mut step_events = [noop; 3];
    let half_block = (scenario.frames / 2) as u32;
    let mut step_cursor = 0_usize;
    let warmup_blocks = if scenario.frames <= 64 {
        256
    } else {
        WARMUP_BLOCKS
    };
    let total_blocks = warmup_blocks + TIMED_BLOCKS;

    for block in 0..total_blocks {
        let n_evs = match scenario.workload {
            Workload::BalasubramaniamComposite => {
                composite.step(&mut engine, block, scenario.frames, &mut step_events)
            }
            Workload::Chord12Arp => {
                if block % 4 == 0 {
                    let prev = ARPEGGIO_KEYS[step_cursor % ARPEGGIO_KEYS.len()];
                    let next = ARPEGGIO_KEYS[(step_cursor + 1) % ARPEGGIO_KEYS.len()];
                    step_cursor = step_cursor.wrapping_add(1);
                    step_events[0] = SampleEvent {
                        sample_offset: 0,
                        kind: EventKind::NoteOff {
                            midi_key: prev,
                            release_vel_u16: u7_to_u16(64),
                        },
                    };
                    step_events[1] = SampleEvent {
                        sample_offset: half_block,
                        kind: EventKind::NoteOn {
                            midi_key: next,
                            velocity_u16: u7_to_u16(112),
                        },
                    };
                    2
                } else {
                    0
                }
            }
            Workload::Sustain20 => {
                if block % 8 == 0 {
                    let key = SUSTAIN_20_KEYS[step_cursor % SUSTAIN_20_KEYS.len()];
                    step_cursor = step_cursor.wrapping_add(1);
                    step_events[0] = SampleEvent {
                        sample_offset: half_block,
                        kind: EventKind::NoteOn {
                            midi_key: key,
                            velocity_u16: u7_to_u16(96),
                        },
                    };
                    1
                } else {
                    0
                }
            }
        };

        let events = &step_events[..n_evs];
        if block < warmup_blocks {
            engine.render_block(events, &mut out_l, &mut out_r);
            black_box(&out_l);
            black_box(&out_r);
        } else {
            let t0 = Instant::now();
            engine.render_block(events, &mut out_l, &mut out_r);
            black_box(&out_l);
            black_box(&out_r);
            timings_us[block - warmup_blocks] = t0.elapsed().as_secs_f64() * 1_000_000.0;
        }
    }

    black_box(engine.telemetry());
    compute_percentiles(&mut timings_us)
}

pub fn measure_radiator_block(profile: PerfProfile) -> Percentiles {
    let mut bank = RadiatorBank::new_with_profile(
        48_000.0,
        ListenerPerspective::Player,
        LidState::Open,
        &profile,
    );
    let block = [[0.01_f32; PRESET_RADIATOR_ZONES]; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];

    for _ in 0..WARMUP_BLOCKS {
        bank.process_block(&block, &mut out_l, &mut out_r);
    }
    let mut timings_us = vec![0.0_f64; TIMED_BLOCKS];
    for slot in &mut timings_us {
        let t0 = Instant::now();
        bank.process_block(&block, &mut out_l, &mut out_r);
        black_box(&out_l);
        black_box(&out_r);
        *slot = t0.elapsed().as_secs_f64() * 1_000_000.0;
    }
    compute_percentiles(&mut timings_us)
}

pub fn measure_ir_synthesis(profile: PerfProfile) -> (IrBenchStats, IrBenchStats, usize) {
    let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    let mode_count = preset.modes.len();
    let ir_dur = f64::from(profile.ir_duration_secs);
    let mut bank = RadiatorBank::new_with_profile(
        48_000.0,
        ListenerPerspective::Player,
        LidState::Open,
        &profile,
    );
    let mut scratch = IrSynthScratch::new(ir_len_for_sample_rate_secs(48_000.0, ir_dur));

    for _ in 0..2 {
        synthesize_irs_with_duration(
            &preset,
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            48_000.0,
            ir_dur,
            &mut scratch,
        );
        bank.reconfigure_space(ListenerPerspective::BinauralHrtf, LidState::Half);
    }

    let mut synth_timings = vec![0.0_f64; 10];
    for slot in &mut synth_timings {
        let t0 = Instant::now();
        synthesize_irs_with_duration(
            &preset,
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            48_000.0,
            ir_dur,
            &mut scratch,
        );
        *slot = t0.elapsed().as_secs_f64() * 1_000.0;
        black_box(&scratch.ir_flat[..16]);
    }

    let mut reconfig_timings = vec![0.0_f64; 10];
    for (i, slot) in reconfig_timings.iter_mut().enumerate() {
        let (persp, lid) = match i % 3 {
            0 => (ListenerPerspective::CloseStereo, LidState::Open),
            1 => (ListenerPerspective::ConcertHall, LidState::Closed),
            _ => (ListenerPerspective::BinauralHrtf, LidState::Half),
        };
        let t0 = Instant::now();
        bank.reconfigure_space(persp, lid);
        *slot = t0.elapsed().as_secs_f64() * 1_000.0;
    }

    let to_stats = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        let n = v.len();
        IrBenchStats {
            min_ms: v[0],
            p50_ms: v[n / 2],
            mean_ms: v.iter().sum::<f64>() / n as f64,
            max_ms: v[n - 1],
        }
    };
    (
        to_stats(synth_timings),
        to_stats(reconfig_timings),
        mode_count,
    )
}
