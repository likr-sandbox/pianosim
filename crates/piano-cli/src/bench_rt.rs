//! Real-time DSP block budget benchmark (`piano-cli bench`).

use std::hint::black_box;
use std::time::Instant;

use piano_engine::PianoEngine;
use piano_spec::{u7_to_u16, EventKind, PedalKind, SampleEvent};

const WARMUP_BLOCKS: usize = 64;
const TIMED_BLOCKS: usize = 1_000;
const D1_WARN_RATIO: f64 = 0.70;
const CHORD_KEYS: [u8; 12] = [36, 43, 48, 52, 55, 60, 64, 67, 71, 72, 76, 79];
const ARPEGGIO_KEYS: [u8; 8] = [48, 55, 60, 64, 67, 72, 76, 84];

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

#[derive(Copy, Clone, Debug)]
struct BenchScenario {
    label: &'static str,
    frames: usize,
    sample_rate: f32,
    enforce_gate: bool,
}

const SCENARIOS: [BenchScenario; 3] = [
    BenchScenario {
        label: "128 frames @ 48,000 Hz",
        frames: 128,
        sample_rate: 48_000.0,
        enforce_gate: true,
    },
    BenchScenario {
        label: "64 frames @ 48,000 Hz",
        frames: 64,
        sample_rate: 48_000.0,
        enforce_gate: false,
    },
    BenchScenario {
        label: "128 frames @ 96,000 Hz",
        frames: 128,
        sample_rate: 96_000.0,
        enforce_gate: false,
    },
];

/// Computes nearest-rank `p50`, `p95`, `p99`, and `max` over a non-empty slice.
pub fn compute_percentiles(samples: &mut [f64]) -> Percentiles {
    assert!(
        !samples.is_empty(),
        "percentile input slice must be non-empty"
    );
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    let at = |q: f64| -> f64 {
        let rank = (q * n as f64).ceil() as usize;
        samples[rank.saturating_sub(1).min(n - 1)]
    };
    Percentiles {
        p50: at(0.50),
        p95: at(0.95),
        p99: at(0.99),
        max: samples[n - 1],
    }
}

fn measure_scenario(scenario: BenchScenario) -> Percentiles {
    let mut engine = PianoEngine::new(scenario.sample_rate);
    let mut out_l = vec![0.0_f32; scenario.frames];
    let mut out_r = vec![0.0_f32; scenario.frames];
    let mut timings_us = vec![0.0_f64; TIMED_BLOCKS];

    // Pre-allocate initial sustain + 12-note chord events outside loops.
    let noop = SampleEvent {
        sample_offset: 0,
        kind: EventKind::AllNotesOff,
    };
    let mut init_events = [noop; 13];
    init_events[0] = SampleEvent {
        sample_offset: 0,
        kind: EventKind::Pedal {
            kind: PedalKind::Sustain,
            value_u16: u16::MAX,
        },
    };
    for (idx, &key) in CHORD_KEYS.iter().enumerate() {
        init_events[idx + 1] = SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: key,
                velocity_u16: u7_to_u16(104),
            },
        };
    }
    engine.render_block(&init_events, &mut out_l, &mut out_r);

    let mut arp_events = [noop; 2];
    let half_block = (scenario.frames / 2) as u32;
    let mut arp_cursor = 0_usize;

    let total_blocks = WARMUP_BLOCKS + TIMED_BLOCKS;
    for block in 0..total_blocks {
        let block_events: &[SampleEvent] = if block % 4 == 0 {
            let prev_key = ARPEGGIO_KEYS[arp_cursor % ARPEGGIO_KEYS.len()];
            let next_key = ARPEGGIO_KEYS[(arp_cursor + 1) % ARPEGGIO_KEYS.len()];
            arp_cursor = arp_cursor.wrapping_add(1);
            arp_events[0] = SampleEvent {
                sample_offset: 0,
                kind: EventKind::NoteOff {
                    midi_key: prev_key,
                    release_vel_u16: u7_to_u16(64),
                },
            };
            arp_events[1] = SampleEvent {
                sample_offset: half_block,
                kind: EventKind::NoteOn {
                    midi_key: next_key,
                    velocity_u16: u7_to_u16(112),
                },
            };
            &arp_events
        } else {
            &[]
        };

        if block < WARMUP_BLOCKS {
            engine.render_block(block_events, &mut out_l, &mut out_r);
            black_box(&out_l);
            black_box(&out_r);
        } else {
            let t0 = Instant::now();
            engine.render_block(block_events, &mut out_l, &mut out_r);
            black_box(&out_l);
            black_box(&out_r);
            timings_us[block - WARMUP_BLOCKS] = t0.elapsed().as_secs_f64() * 1_000_000.0;
        }
    }

    black_box(engine.telemetry());
    compute_percentiles(&mut timings_us)
}

/// Executes all real-time DSP block-budget scenarios and enforces M2 D1 gate.
pub fn run_bench() -> Result<(), String> {
    println!(
        "piano-cli bench: 88 keys / {} modes (warmup={} blocks, timed={} blocks, sustain+12-chord+arp)",
        PianoEngine::new(48_000.0).total_modes(),
        WARMUP_BLOCKS,
        TIMED_BLOCKS
    );

    let mut gate_failure: Option<String> = None;

    for scenario in SCENARIOS {
        let budget_us = f64::from(scenario.frames as u32) / f64::from(scenario.sample_rate) * 1e6;
        let pct = measure_scenario(scenario);
        let pct_of = |us: f64| 100.0 * us / budget_us;

        println!(
            "  {:<24} budget={:>7.1} us | p50={:>6.1} us ({:>4.1}%) | p95={:>6.1} us ({:>4.1}%) | p99={:>6.1} us ({:>4.1}%) | max={:>6.1} us ({:>4.1}%)",
            scenario.label,
            budget_us,
            pct.p50,
            pct_of(pct.p50),
            pct.p95,
            pct_of(pct.p95),
            pct.p99,
            pct_of(pct.p99),
            pct.max,
            pct_of(pct.max),
        );

        if scenario.enforce_gate {
            let warn_us = budget_us * D1_WARN_RATIO;
            if pct.p99 > budget_us {
                gate_failure = Some(format!(
                    "real-time gate failed for {}: p99 {:.1} us exceeds 100% budget ({:.1} us)",
                    scenario.label, pct.p99, budget_us
                ));
            } else if pct.p99 > warn_us {
                eprintln!(
                    "WARNING (D1 >70% budget): {} p99 {:.1} us > {:.1} us warning threshold",
                    scenario.label, pct.p99, warn_us
                );
            }
        }
    }

    match gate_failure {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::compute_percentiles;

    #[test]
    fn percentile_calculation_matches_nearest_rank() {
        let mut data: Vec<f64> = (1..=100).rev().map(f64::from).collect();
        let pct = compute_percentiles(&mut data);
        assert_eq!(pct.p50, 50.0);
        assert_eq!(pct.p95, 95.0);
        assert_eq!(pct.p99, 99.0);
        assert_eq!(pct.max, 100.0);
    }
}
