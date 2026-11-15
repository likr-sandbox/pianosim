//! Milestone 3 closed-loop Stage 1 <-> Stage 2 acoustic gate verification suite.

use core::f64::consts::TAU;

use piano_simd::{Complex, RealFft};
use piano_spec::{u7_to_u16, EventKind, PedalKind, SampleEvent};

use crate::{EventQueue, PianoEngine};

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

pub(crate) fn render_chunks(
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

pub(crate) fn mono_mix(l: &[f32], r: &[f32]) -> Vec<f32> {
    l.iter()
        .zip(r.iter())
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect()
}

pub(crate) fn slice_rms(samples: &[f32]) -> f64 {
    let sum_sq: f64 = samples.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / (samples.len().max(1) as f64)).sqrt()
}

pub(crate) fn hann_windowed_spectrum(samples: &[f32]) -> Vec<Complex> {
    let n = samples.len();
    assert!(n.is_power_of_two());
    let mut windowed = vec![0.0_f32; n];
    let denom = (n - 1).max(1) as f64;
    for (i, (&s, dst)) in samples.iter().zip(windowed.iter_mut()).enumerate() {
        let w = 0.5 * (1.0 - (TAU * (i as f64) / denom).cos());
        *dst = (f64::from(s) * w) as f32;
    }
    let mut rfft = RealFft::new(n);
    let mut spec = vec![Complex::ZERO; rfft.spectrum_len()];
    rfft.forward(&windowed, &mut spec);
    spec
}

pub(crate) fn spectral_centroid_hz(samples: &[f32], sr: f64) -> f64 {
    let spec = hann_windowed_spectrum(samples);
    let bin_hz = sr / (samples.len() as f64);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (k, c) in spec.iter().enumerate().skip(1) {
        let mag = f64::from(c.norm_sq()).sqrt();
        num += (k as f64) * bin_hz * mag;
        den += mag;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

pub(crate) fn band_power(spec: &[Complex], n_fft: usize, sr: f64, lo_hz: f64, hi_hz: f64) -> f64 {
    let bin_hz = sr / (n_fft as f64);
    let lo_bin = ((lo_hz / bin_hz).floor() as usize).max(1);
    let hi_bin = ((hi_hz / bin_hz).ceil() as usize).min(spec.len() - 1);
    spec[lo_bin..=hi_bin]
        .iter()
        .map(|c| f64::from(c.norm_sq()))
        .sum()
}

fn peak_bin_power(spec: &[Complex], n_fft: usize, sr: f64, lo_hz: f64, hi_hz: f64) -> (f64, f64) {
    let bin_hz = sr / (n_fft as f64);
    let lo_bin = ((lo_hz / bin_hz).floor() as usize).max(1);
    let hi_bin = ((hi_hz / bin_hz).ceil() as usize).min(spec.len() - 1);
    let mut best_k = lo_bin;
    let mut best_p = -1.0_f64;
    for (offset, c) in spec[lo_bin..=hi_bin].iter().enumerate() {
        let p = f64::from(c.norm_sq());
        if p > best_p {
            best_p = p;
            best_k = lo_bin + offset;
        }
    }
    ((best_k as f64) * bin_hz, best_p)
}

#[test]
fn gate1_velocity_sweep_dynamic_range_centroid_and_aliasing() {
    let velocities: [u16; 8] = [2_048, 6_000, 12_000, 20_000, 30_000, 42_000, 54_000, 65_535];
    let mut engine = PianoEngine::new(SR);

    for &midi_key in &[36_u8, 60, 69, 96] {
        let mut prev_rms = 0.0_f64;
        let mut prev_cent = 0.0_f64;
        let mut pp_rms = 0.0_f64;
        let mut fff_rms = 0.0_f64;

        for (idx, &vel) in velocities.iter().enumerate() {
            engine.reset();
            let ev = [SampleEvent {
                sample_offset: 0,
                kind: EventKind::NoteOn {
                    midi_key,
                    velocity_u16: vel,
                },
            }];
            let (l, r) = render_chunks(&mut engine, 8_192, 128, &ev);
            let mono = mono_mix(&l, &r);
            let rms = slice_rms(&mono[..4_096]);
            let cent = spectral_centroid_hz(&mono[..4_096], SR64);
            assert!(rms.is_finite() && cent.is_finite() && rms > 0.0);
            if idx == 0 {
                pp_rms = rms;
            }
            if idx + 1 == velocities.len() {
                fff_rms = rms;
            }
            assert!(
                rms > prev_rms,
                "key={midi_key} vel={vel}: non-monotonic rms {rms} <= {prev_rms}"
            );
            assert!(
                cent > prev_cent,
                "key={midi_key} vel={vel}: non-monotonic centroid {cent} <= {prev_cent}"
            );
            prev_rms = rms;
            prev_cent = cent;
        }

        let dyn_db = 20.0 * (fff_rms / pp_rms).log10();
        assert!(
            dyn_db >= 24.0,
            "key={midi_key}: dynamic range {dyn_db:.2} dB < 24.0 dB (pp={pp_rms}, fff={fff_rms})"
        );
    }

    // Ultrasonic aliasing check on C2 (36) and C3 (48) at fff (65_535) @ 48 kHz.
    for &(midi_key, f0_guess) in &[(36_u8, 65.41_f64), (48_u8, 130.81_f64)] {
        engine.reset();
        let ev = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key,
                velocity_u16: u16::MAX,
            },
        }];
        let (l, r) = render_chunks(&mut engine, 16_384, 128, &ev);
        let mono = mono_mix(&l, &r);
        let spec = hann_windowed_spectrum(&mono[..16_384]);
        let (_, fund_p) = peak_bin_power(&spec, 16_384, SR64, f0_guess * 0.85, f0_guess * 1.15);
        let (_, alias_p) = peak_bin_power(&spec, 16_384, SR64, 18_000.0, 23_900.0);
        let snr_db = 10.0 * (fund_p / alias_p.max(1e-30)).log10();
        assert!(
            snr_db >= 55.0,
            "key={midi_key}: >18kHz aliasing SNR {snr_db:.2} dB < 55.0 dB"
        );
    }

    // A4 @ u7_to_u16(100) pitch accuracy in [435, 445] Hz.
    engine.reset();
    let a4 = [SampleEvent {
        sample_offset: 0,
        kind: EventKind::NoteOn {
            midi_key: 69,
            velocity_u16: u7_to_u16(100),
        },
    }];
    let (l, r) = render_chunks(&mut engine, 16_384, 128, &a4);
    let mono = mono_mix(&l, &r);
    let spec = hann_windowed_spectrum(&mono);
    let (peak_hz, _) = peak_bin_power(&spec, 16_384, SR64, 100.0, 5_000.0);
    assert!(
        (435.0..=445.0).contains(&peak_hz),
        "A4 peak expected in [435, 445] Hz, got {peak_hz}"
    );
}

#[test]
fn gate2_ribattuto_12hz_felt_memory_evolves_across_strikes() {
    const STRIKES: usize = 12;
    const STRIKE_PERIOD: usize = 4_000; // 12 Hz @ 48 kHz

    for &midi_key in &[69_u8, 72] {
        let run_series = |memory_enabled: bool| -> (Vec<f64>, Vec<f64>) {
            let mut engine = PianoEngine::new(SR);
            engine.set_felt_memory_enabled(memory_enabled);
            let mut rms_list = Vec::with_capacity(STRIKES);
            let mut cent_list = Vec::with_capacity(STRIKES);

            for s in 0..STRIKES {
                let on = [SampleEvent {
                    sample_offset: 0,
                    kind: EventKind::NoteOn {
                        midi_key,
                        velocity_u16: 52_000,
                    },
                }];
                let (l_atk, r_atk) = render_chunks(&mut engine, 2_048, 128, &on);
                let mono = mono_mix(&l_atk, &r_atk);
                rms_list.push(slice_rms(&mono));
                cent_list.push(spectral_centroid_hz(&mono, SR64));

                let off = [SampleEvent {
                    sample_offset: 0,
                    kind: EventKind::NoteOff {
                        midi_key,
                        release_vel_u16: 48_000,
                    },
                }];
                let _ = render_chunks(&mut engine, STRIKE_PERIOD - 2_048, 128, &off);
                let _ = s;
            }
            (rms_list, cent_list)
        };

        let (rms_mem, cent_mem) = run_series(true);
        let (rms_nomem, cent_nomem) = run_series(false);

        let cent_drift_mem = (cent_mem[STRIKES - 1] - cent_mem[0]).abs() / cent_mem[0].max(1.0);
        assert!(
            cent_drift_mem > 0.005,
            "key={midi_key}: expected strike centroid evolution with memory, drift={cent_drift_mem:.4}"
        );

        let last_rms_diff =
            (rms_mem[STRIKES - 1] - rms_nomem[STRIKES - 1]).abs() / rms_nomem[STRIKES - 1];
        let last_cent_diff =
            (cent_mem[STRIKES - 1] - cent_nomem[STRIKES - 1]).abs() / cent_nomem[STRIKES - 1];
        assert!(
            last_rms_diff.max(last_cent_diff) > 0.03,
            "key={midi_key}: expected >3% memory divergence at strike 12 (rms_diff={last_rms_diff:.4}, cent_diff={last_cent_diff:.4})"
        );
    }
}

#[test]
fn gate3_continuous_half_pedal_repedaling_and_16_note_cluster() {
    // 1. Half-pedal: Sustain 0.0 vs 0.5 vs 1.0 after NoteOff on C3 (48) and C4 (60).
    for &(midi_key, f0_hz) in &[(48_u8, 130.81_f64), (60_u8, 261.63_f64)] {
        let run_pedal = |sustain_u16: u16| -> (f64, f64, f64) {
            let mut engine = PianoEngine::new(SR);
            let setup = [
                SampleEvent {
                    sample_offset: 0,
                    kind: EventKind::Pedal {
                        kind: PedalKind::Sustain,
                        value_u16: sustain_u16,
                    },
                },
                SampleEvent {
                    sample_offset: 0,
                    kind: EventKind::NoteOn {
                        midi_key,
                        velocity_u16: 56_000,
                    },
                },
            ];
            let _ = render_chunks(&mut engine, 4_800, 128, &setup);
            let off = [SampleEvent {
                sample_offset: 0,
                kind: EventKind::NoteOff {
                    midi_key,
                    release_vel_u16: 48_000,
                },
            }];
            let _ = render_chunks(&mut engine, 9_600, 128, &off); // 200 ms transition
            let (tail_l, tail_r) = render_chunks(&mut engine, 8_192, 128, &[]);
            let mono = mono_mix(&tail_l, &tail_r);
            let total_rms = slice_rms(&mono);
            let spec = hann_windowed_spectrum(&mono);
            let f1_p = band_power(&spec, 8_192, SR64, f0_hz * 0.85, f0_hz * 1.15);
            let hi_p = band_power(&spec, 8_192, SR64, f0_hz * 3.8, 12_000.0);
            (total_rms, f1_p, hi_p)
        };

        let (rms_0, _, _) = run_pedal(0);
        let (rms_half, half_f1, half_hi) = run_pedal(32_768);
        let (rms_full, full_f1, full_hi) = run_pedal(u16::MAX);

        assert!(
            rms_0 < rms_half && rms_half < rms_full,
            "key={midi_key}: expected rms_0 ({rms_0}) < rms_half ({rms_half}) < rms_full ({rms_full})"
        );
        let f1_loss_db = 10.0 * (full_f1 / half_f1.max(1e-30)).log10();
        let hi_loss_db = 10.0 * (full_hi / half_hi.max(1e-30)).log10();
        assert!(
            hi_loss_db - f1_loss_db >= 6.0,
            "key={midi_key}: half-pedal high partial selectivity expected >= 6 dB (hi_loss={hi_loss_db:.2} dB, f1_loss={f1_loss_db:.2} dB)"
        );
    }

    // 2. Repedaling on C3 (48): sustain 1.0 -> NoteOff -> dip 0.25 for 70 ms -> return 1.0.
    let mut engine = PianoEngine::new(SR);
    let init = [
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: u16::MAX,
            },
        },
        SampleEvent {
            sample_offset: 0,
            kind: EventKind::NoteOn {
                midi_key: 48,
                velocity_u16: 56_000,
            },
        },
    ];
    let _ = render_chunks(&mut engine, 4_800, 128, &init);
    let off = [SampleEvent {
        sample_offset: 0,
        kind: EventKind::NoteOff {
            midi_key: 48,
            release_vel_u16: 48_000,
        },
    }];
    let (pre_l, pre_r) = render_chunks(&mut engine, 8_192, 128, &off);
    let pre_mono = mono_mix(&pre_l, &pre_r);
    let pre_spec = hann_windowed_spectrum(&pre_mono);
    let pre_f1 = band_power(&pre_spec, 8_192, SR64, 110.0, 150.0);
    let pre_hi = band_power(&pre_spec, 8_192, SR64, 1_200.0, 12_000.0);

    let dip = [SampleEvent {
        sample_offset: 0,
        kind: EventKind::Pedal {
            kind: PedalKind::Sustain,
            value_u16: 16_384,
        },
    }];
    let _ = render_chunks(&mut engine, 3_360, 128, &dip); // 70 ms
    let catch = [SampleEvent {
        sample_offset: 0,
        kind: EventKind::Pedal {
            kind: PedalKind::Sustain,
            value_u16: u16::MAX,
        },
    }];
    let _ = render_chunks(&mut engine, 2_400, 128, &catch);
    let (post_l, post_r) = render_chunks(&mut engine, 8_192, 128, &[]);
    let post_mono = mono_mix(&post_l, &post_r);
    let post_spec = hann_windowed_spectrum(&post_mono);
    let post_f1 = band_power(&post_spec, 8_192, SR64, 110.0, 150.0);
    let post_hi = band_power(&post_spec, 8_192, SR64, 1_200.0, 12_000.0);

    let f1_drop_db = 10.0 * (pre_f1 / post_f1.max(1e-30)).log10();
    let hi_drop_db = 10.0 * (pre_hi / post_hi.max(1e-30)).log10();
    assert!(
        post_f1 > 1e-8,
        "expected C3 fundamental still ringing after repedaling"
    );
    assert!(
        hi_drop_db - f1_drop_db >= 6.0,
        "repedaling upper partials (>1.2kHz) drop {hi_drop_db:.2} dB vs fundamental {f1_drop_db:.2} dB (<6 dB diff)"
    );

    // 3. 16-note simultaneous fff cluster across A0..C8 + rapid pedal sweep: finite & bounded.
    engine.reset();
    let mut q = EventQueue::new();
    for i in 0..16_u8 {
        assert!(q.push(SampleEvent {
            sample_offset: u32::from(i) * 4,
            kind: EventKind::NoteOn {
                midi_key: 21 + i * 5,
                velocity_u16: u16::MAX,
            },
        }));
    }
    let (mut l, mut r) = render_chunks(&mut engine, 2_048, 128, q.as_slice());
    let mut out_l = vec![0.0_f32; 128];
    let mut out_r = vec![0.0_f32; 128];
    for step in 0..40_usize {
        let sus = if step % 2 == 0 { u16::MAX } else { 0 };
        let evs = [SampleEvent {
            sample_offset: 0,
            kind: EventKind::Pedal {
                kind: PedalKind::Sustain,
                value_u16: sus,
            },
        }];
        engine.render_block(&evs, &mut out_l, &mut out_r);
        l.extend_from_slice(&out_l);
        r.extend_from_slice(&out_r);
    }
    for (&sl, &sr) in l.iter().zip(r.iter()) {
        assert!(sl.is_finite() && sr.is_finite());
        assert!(sl.abs() <= 1.0 && sr.abs() <= 1.0);
    }
}
