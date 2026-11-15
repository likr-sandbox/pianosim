//! Milestone 5 Stage 3 soundboard/bridge coupled acoustic & multi-rate passivity gate tests.

use piano_spec::{
    build_scale_design, key_tuned_f0_hz, midi_to_key, partial_frequency_hz, EventKind, PedalKind,
    SampleEvent, VoicingSpec, CHOIR_LAYOUTS,
};

use crate::m3_tests::{
    band_power, hann_windowed_spectrum, mono_mix, slice_rms, spectral_centroid_hz,
};
use crate::m4_tests::{note_off, note_on};
use crate::PianoEngine;

const SR: f32 = 48_000.0;
const SR64: f64 = 48_000.0;

fn pedal_ev(kind: PedalKind, value_u16: u16, sample_offset: u32) -> SampleEvent {
    SampleEvent {
        sample_offset,
        kind: EventKind::Pedal { kind, value_u16 },
    }
}

fn render_timeline(
    engine: &mut PianoEngine,
    total_frames: usize,
    block_size: usize,
    events: &[(usize, SampleEvent)],
) -> Vec<f32> {
    let mut out_l = vec![0.0_f32; total_frames];
    let mut out_r = vec![0.0_f32; total_frames];
    let mut block_evs = Vec::new();
    let mut offset = 0_usize;
    while offset < total_frames {
        let end = (offset + block_size).min(total_frames);
        block_evs.clear();
        for &(abs_pos, mut ev) in events {
            if abs_pos >= offset && abs_pos < end {
                ev.sample_offset = (abs_pos - offset) as u32;
                block_evs.push(ev);
            }
        }
        engine.render_block(&block_evs, &mut out_l[offset..end], &mut out_r[offset..end]);
        offset = end;
    }
    let mono = mono_mix(&out_l, &out_r);
    for (i, (&l, &r)) in out_l.iter().zip(out_r.iter()).enumerate() {
        assert!(
            l.is_finite() && r.is_finite(),
            "non-finite audio output at sample {i}: l={l}, r={r}"
        );
    }
    mono
}

fn ols_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for &(x, y) in points {
        let dx = x - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

fn decay_slope_db_per_s(mono: &[f32], sr: f64, t_start_s: f64, t_end_s: f64, win_s: f64) -> f64 {
    let win = (win_s * sr).round() as usize;
    let start_idx = (t_start_s * sr).round() as usize;
    let end_idx = ((t_end_s * sr).round() as usize).min(mono.len());
    let mut pts = Vec::new();
    let mut pos = start_idx;
    while pos + win <= end_idx {
        let rms = slice_rms(&mono[pos..pos + win]);
        let t_mid = (pos as f64 + 0.5 * (win as f64)) / sr;
        pts.push((t_mid, 20.0 * rms.max(1e-12).log10()));
        pos += win;
    }
    assert!(pts.len() >= 3, "insufficient RMS windows for decay slope");
    -ols_slope(&pts)
}

fn assert_two_stage_decay(label: &str, mono: &[f32], late_end_s: f64) {
    let prompt = decay_slope_db_per_s(mono, SR64, 0.05, 0.45, 0.05);
    let late = decay_slope_db_per_s(mono, SR64, 1.40, late_end_s, 0.10);
    let ratio = prompt / late.max(1e-9);
    eprintln!("M5 Gate1 {label}: prompt={prompt:.3} dB/s, late={late:.3} dB/s, ratio={ratio:.2}x");
    assert!(
        prompt > 0.0 && late > 0.0 && ratio >= 2.0,
        "Gate 1 failed on {label}: prompt={prompt:.3} dB/s, late={late:.3} dB/s, ratio={ratio:.2}x < 2.0x"
    );
}

fn estimate_post_attack_t60_s(mono: &[f32], sr: f64) -> f64 {
    let ref_s = (0.05 * sr).round() as usize;
    let ref_e = (0.10 * sr).round() as usize;
    let ref_db = 20.0 * slice_rms(&mono[ref_s..ref_e]).max(1e-12).log10();
    let win = (0.05 * sr).round() as usize;
    let mut pts = Vec::new();
    let mut started = false;
    let mut pos = ref_e;
    while pos + win <= mono.len() {
        let rel_db = 20.0 * slice_rms(&mono[pos..pos + win]).max(1e-12).log10() - ref_db;
        let t_mid = (pos as f64 + 0.5 * (win as f64)) / sr;
        if !started && rel_db <= -5.0 {
            started = true;
        }
        if started {
            pts.push((t_mid, rel_db));
            if pts.len() >= 6 && rel_db <= -30.0 {
                break;
            }
        }
        pos += win;
    }
    assert!(pts.len() >= 4, "insufficient post-attack decay points");
    let decay_rate = -ols_slope(&pts);
    assert!(
        decay_rate > 0.0,
        "expected positive decay rate for T60 estimation, got {decay_rate:.4}"
    );
    60.0 / decay_rate
}

fn choir_energy_joules(engine: &PianoEngine, midi_key: u8) -> f64 {
    let k = midi_to_key(midi_key).expect("valid piano MIDI key");
    let layout = CHOIR_LAYOUTS[k];
    let first = usize::from(layout.first_string_idx);
    let count = usize::from(layout.strings_per_key);
    (first..first + count)
        .map(|s| engine.string_linear_energy_joules(s))
        .sum()
}

fn render_note_hold(midi_key: u8, vel_u16: u16, sustain_u16: u16, secs: f64) -> Vec<f32> {
    let mut engine = PianoEngine::new(SR);
    let total = (secs * SR64).round() as usize;
    let evs = [
        (0, pedal_ev(PedalKind::Sustain, sustain_u16, 0)),
        (0, note_on(midi_key, vel_u16, 0)),
    ];
    render_timeline(&mut engine, total, 128, &evs)
}

#[test]
fn m5_gate1_two_stage_decay_c3_a4() {
    let c3 = render_note_hold(48, 48_000, 0, 3.1);
    let a4 = render_note_hold(69, 48_000, 0, 3.1);
    assert_two_stage_decay("C3", &c3, 3.0);
    assert_two_stage_decay("A4", &a4, 3.0);
}

#[test]
fn m5_gate1_c4_and_gate2_lehtonen_sustain_cluster() {
    let c4_dry = render_note_hold(60, 48_000, 0, 3.8);
    assert_two_stage_decay("C4", &c4_dry, 3.6);
    let c4_sus = render_note_hold(60, 48_000, u16::MAX, 3.8);

    // Lehtonen et al. (JASA 2007 §II.C) harmonic-notched background resonance residual (1.0..3.7 s).
    let n_fft = 16_384_usize;
    let hop = 8_192_usize;
    let spec_len = n_fft / 2 + 1;
    let bin_hz = SR64 / (n_fft as f64);
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let k_c4 = midi_to_key(60).expect("C4");
    let f0 = key_tuned_f0_hz(k_c4, &specs, &voicing);
    let b = specs[k_c4].inharmonicity_b;

    let mut partials = Vec::new();
    for j in 1..128_usize {
        let fj = partial_frequency_hz(f0, b, j);
        if fj > 16_500.0 {
            break;
        }
        partials.push((fj, (0.012 * fj).max(15.0)));
    }
    let mut in_band = vec![false; spec_len];
    let mut keep_res = vec![false; spec_len];
    for bin in 1..spec_len {
        let f_bin = (bin as f64) * bin_hz;
        if (20.0..=16_000.0).contains(&f_bin) {
            in_band[bin] = true;
            keep_res[bin] = partials.iter().all(|&(fj, hw)| (f_bin - fj).abs() > hw);
        }
    }

    let s1 = (1.0 * SR64).round() as usize;
    let s_end = (3.7 * SR64).round() as usize;
    let sum_bins = |mono: &[f32]| -> (f64, f64) {
        let (mut tot, mut res) = (0.0_f64, 0.0_f64);
        let mut pos = s1;
        while pos + n_fft <= s_end {
            let spec = hann_windowed_spectrum(&mono[pos..pos + n_fft]);
            for (bin, c) in spec.iter().enumerate() {
                if in_band[bin] {
                    let p = f64::from(c.norm_sq());
                    tot += p;
                    if keep_res[bin] {
                        res += p;
                    }
                }
            }
            pos += hop;
        }
        (tot, res)
    };

    let (e_tot_dry, e_res_dry) = sum_bins(&c4_dry);
    let (e_tot_sus, e_res_sus) = sum_bins(&c4_sus);
    let noiseless_db = 10.0 * (e_res_sus / e_res_dry.max(1e-30)).log10();
    let e_floor = 1.0e-3 * e_tot_dry;
    let floor_db = 10.0 * ((e_res_sus + e_floor) / (e_res_dry + e_floor)).log10();

    let s_late = (2.8 * SR64).round() as usize;
    let late_rms_dry = slice_rms(&c4_dry[s_late..s_end]);
    let late_rms_sus = slice_rms(&c4_sus[s_late..s_end]);
    let late_delta_db = 20.0 * (late_rms_sus / late_rms_dry.max(1e-30)).log10();
    let t60_dry = estimate_post_attack_t60_s(&c4_dry[..s_end], SR64);
    let t60_sus = estimate_post_attack_t60_s(&c4_sus[..s_end], SR64);
    let t60_ext_pct = 100.0 * (t60_sus - t60_dry) / t60_dry;

    eprintln!(
        "M5 Gate2 C4 Lehtonen: noiseless={noiseless_db:+.2} dB, floor_ref={floor_db:+.2} dB, late_2.8..3.7s={late_delta_db:+.2} dB, T60 {t60_dry:.2}->{t60_sus:.2}s ({t60_ext_pct:+.1}%)"
    );
    assert!(
        (8.0..40.0).contains(&noiseless_db),
        "Gate 2 noiseless residual out of [8.0, 40.0) dB: {noiseless_db:+.2} dB"
    );
    assert!(
        (7.5..=14.5).contains(&floor_db),
        "Gate 2 floor-referenced residual out of +7.5..+14.5 dB: {floor_db:+.2} dB"
    );
    assert!(e_res_sus < e_tot_sus, "cluster exceeded foreground total");
    assert!(
        late_delta_db > 2.0,
        "Gate 2 2.8..3.7s broadband RMS delta <= +2.0 dB: {late_delta_db:+.2} dB"
    );
    assert!(
        t60_ext_pct >= 15.0,
        "Gate 2 C4 T60 extension < +15%: {t60_ext_pct:+.1}%"
    );
}

#[test]
fn m5_gate2_c2_c6_sustain_stability_bounds() {
    for (label, midi) in [("C2", 36_u8), ("C6", 84_u8)] {
        let mut eng = PianoEngine::new(SR);
        let total = (2.0 * SR64).round() as usize;
        let settle_min = (0.05 * SR64).round() as usize;
        let mut mono = Vec::with_capacity(total);
        let (mut out_l, mut out_r) = ([0.0_f32; 128], [0.0_f32; 128]);
        let evs = [
            pedal_ev(PedalKind::Sustain, u16::MAX, 0),
            note_on(midi, 60_000, 0),
        ];
        let mut h_prev: Option<f64> = None;
        let mut peak = 0.0_f64;
        let mut pos = 0_usize;
        while pos < total {
            let n = (total - pos).min(128);
            let block_evs = if pos == 0 { &evs[..] } else { &[] };
            eng.render_block(block_evs, &mut out_l[..n], &mut out_r[..n]);
            for i in 0..n {
                let (l, r) = (out_l[i], out_r[i]);
                assert!(l.is_finite() && r.is_finite(), "{label} non-finite");
                let m = 0.5 * (l + r);
                peak = peak.max(f64::from(l.abs()).max(f64::from(r.abs())));
                mono.push(m);
            }
            pos += n;
            if pos >= settle_min && !eng.is_any_hammer_active() {
                let h_cur = eng.total_sav_energy_joules();
                assert!(h_cur.is_finite(), "{label} non-finite energy");
                if let Some(prev) = h_prev {
                    assert!(
                        h_cur <= prev + 1.0e-12,
                        "{label} energy grew at frame {pos}: {h_cur:.12e} > {prev:.12e}"
                    );
                }
                h_prev = Some(h_cur);
            }
        }
        assert!(h_prev.is_some(), "{label} hammer did not finish in 2.0s");
        let early = slice_rms(&mono[(0.05 * SR64) as usize..(0.55 * SR64) as usize]);
        let late = slice_rms(&mono[(1.45 * SR64) as usize..total]);
        eprintln!(
            "M5 Gate2 {label} ff+sustain: peak={peak:.4}, early_rms={early:.4}, late_rms={late:.4}"
        );
        assert!(peak <= 1.0, "{label} peak > 1.0: {peak:.4}");
        assert!(
            late < early,
            "{label} late_rms >= early_rms: {late} >= {early}"
        );
    }
}

#[test]
fn m5_gate3a_silent_key_sympathetic_resonance() {
    let mut probe = PianoEngine::new(SR);
    let _ = render_timeline(&mut probe, 2_400, 128, &[(0, note_on(60, 16, 0))]);
    assert_eq!(choir_energy_joules(&probe, 60), 0.0, "vel=16 hit string");

    let run_3a = |silent_key: Option<u8>| -> f64 {
        let mut eng = PianoEngine::new(SR);
        let mut evs = Vec::new();
        if let Some(k) = silent_key {
            evs.push((0, note_on(k, 16, 0)));
        }
        evs.push((960, note_on(48, 58_000, 0)));
        evs.push((960 + 5_760, note_off(48, 48_000, 0)));
        let total = (1.20 * SR64).round() as usize;
        let mono = render_timeline(&mut eng, total, 128, &evs);
        slice_rms(&mono[(0.60 * SR64).round() as usize..total])
    };
    let (rms_none, rms_c4, rms_cs4) = (run_3a(None), run_3a(Some(60)), run_3a(Some(61)));
    let vs_none = 20.0 * (rms_c4 / rms_none.max(1e-30)).log10();
    let vs_cs4 = 20.0 * (rms_c4 / rms_cs4.max(1e-30)).log10();
    eprintln!("M5 Gate3A Silent C4 vs none={vs_none:+.2} dB, vs C#4={vs_cs4:+.2} dB");
    assert!(vs_none > 20.0, "Gate 3A vs none <= +20 dB: {vs_none:+.2}");
    assert!(vs_cs4 > 12.0, "Gate 3A vs C#4 <= +12 dB: {vs_cs4:+.2}");
}

#[test]
fn m5_gate3b_una_corda_softening_and_unstruck_coupling() {
    for (label, midi) in [("C4", 60_u8), ("A4", 69_u8)] {
        for vel in [48_000_u16, 60_000] {
            let measure = |uc: u16| -> (f64, f64) {
                let mut eng = PianoEngine::new(SR);
                let strike = 960_usize;
                let total = strike + 4_096;
                let evs = [
                    (0, pedal_ev(PedalKind::UnaCorda, uc, 0)),
                    (strike, note_on(midi, vel, 0)),
                ];
                let mono = render_timeline(&mut eng, total, 128, &evs);
                let cent = spectral_centroid_hz(&mono[strike..total], SR64);
                let first_s =
                    usize::from(CHOIR_LAYOUTS[midi_to_key(midi).unwrap()].first_string_idx);
                (cent, eng.string_linear_energy_joules(first_s))
            };
            let (c_norm, e0_norm) = measure(0);
            let (c_uc, e0_uc) = measure(u16::MAX);
            eprintln!(
                "M5 Gate3B {label} vel={vel}: centroid {c_norm:.1}->{c_uc:.1} Hz ({:.1}%), s0 {e0_norm:.3e}/{e0_uc:.3e}",
                100.0 * (c_uc / c_norm - 1.0)
            );
            assert!(
                c_uc < c_norm * 0.96,
                "Gate 3B {label} vel={vel}: centroid {c_uc:.1} >= 0.96 * {c_norm:.1}"
            );
            assert!(
                e0_uc > 1.0e-7,
                "Gate 3B {label} vel={vel}: unstruck s0 energy {e0_uc:.3e} <= 1e-7"
            );
        }
    }
}

#[test]
fn m5_gate3c_sostenuto_selective_retention() {
    let mut eng = PianoEngine::new(SR);
    let settle_end = (0.70 * SR64).round() as usize;
    let evs = [
        (0, note_on(60, 52_000, 0)),
        (2_400, pedal_ev(PedalKind::Sostenuto, u16::MAX, 0)),
        (4_800, note_on(64, 52_000, 0)),
        (9_600, note_off(60, 48_000, 0)),
        (9_600, note_off(64, 48_000, 0)),
    ];
    let mono = render_timeline(&mut eng, settle_end, 128, &evs);
    let spec = hann_windowed_spectrum(&mono[settle_end - 16_384..settle_end]);
    let p_c4 = band_power(&spec, 16_384, SR64, 245.0, 280.0);
    let p_e4 = band_power(&spec, 16_384, SR64, 310.0, 350.0);
    let audio_sep_db = 10.0 * (p_c4 / p_e4.max(1e-30)).log10();
    let string_sep_db =
        10.0 * (choir_energy_joules(&eng, 60) / choir_energy_joules(&eng, 64).max(1e-30)).log10();
    eprintln!(
        "M5 Gate3C Sostenuto: audio C4/E4={audio_sep_db:+.2} dB, string={string_sep_db:+.2} dB"
    );
    assert!(
        audio_sep_db > 20.0 && string_sep_db > 20.0,
        "Gate 3C Sostenuto separation <= +20 dB (audio={audio_sep_db:+.2}, string={string_sep_db:+.2})"
    );
}

#[test]
fn m5_gate4_multirate_closed_loop_passivity() {
    for sr in [44_100.0_f32, 48_000.0_f32, 96_000.0_f32] {
        let mut eng = PianoEngine::new(sr);
        let (mut out_l, mut out_r) = ([0.0_f32; 128], [0.0_f32; 128]);
        let init_evs = [
            note_on(21, u16::MAX, 0),
            note_on(36, u16::MAX, 0),
            note_on(48, u16::MAX, 0),
        ];
        eng.render_block(&init_evs, &mut out_l, &mut out_r);
        let min_blocks = ((f64::from(sr) * 0.05) as usize).div_ceil(128);
        let mut b = 1_usize;
        while eng.is_any_hammer_active() || b < min_blocks {
            eng.render_block(&[], &mut out_l, &mut out_r);
            for (&l, &r) in out_l.iter().zip(out_r.iter()) {
                assert!(l.is_finite() && r.is_finite() && l.abs() <= 1.0 && r.abs() <= 1.0);
            }
            b += 1;
            assert!(b < 200, "hammer failed to release cleanly at sr={sr}");
        }
        let h_start = eng.total_sav_energy_joules();
        assert!(h_start.is_finite() && h_start > 0.0);
        let mut h_prev = h_start;
        let step_blocks = ((f64::from(sr) * 0.15) as usize).div_ceil(128);
        for step in 0..step_blocks {
            eng.render_block(&[], &mut out_l, &mut out_r);
            for (&l, &r) in out_l.iter().zip(out_r.iter()) {
                assert!(l.is_finite() && r.is_finite() && l.abs() <= 1.0 && r.abs() <= 1.0);
            }
            let h_cur = eng.total_sav_energy_joules();
            assert!(
                h_cur.is_finite() && h_cur <= h_prev + 1.0e-12,
                "Gate 4 free passivity violation at sr={sr}, step={step}: {h_cur:.12e} > {h_prev:.12e}"
            );
            h_prev = h_cur;
        }
        let off_evs = [
            note_off(21, 48_000, 0),
            note_off(36, 48_000, 0),
            note_off(48, 48_000, 0),
        ];
        let damp_blocks = ((f64::from(sr) * 0.25) as usize).div_ceil(128);
        for step in 0..damp_blocks {
            let evs = if step == 0 { &off_evs[..] } else { &[] };
            eng.render_block(evs, &mut out_l, &mut out_r);
            for (&l, &r) in out_l.iter().zip(out_r.iter()) {
                assert!(l.is_finite() && r.is_finite() && l.abs() <= 1.0 && r.abs() <= 1.0);
            }
            let h_cur = eng.total_sav_energy_joules();
            assert!(
                h_cur.is_finite() && h_cur <= h_prev + 1.0e-12,
                "Gate 4 damped passivity violation at sr={sr}, step={step}: {h_cur:.12e} > {h_prev:.12e}"
            );
            h_prev = h_cur;
        }
        eprintln!(
            "M5 Gate4 sr={sr:.0}: h_start={h_start:.6e} J -> h_end={h_prev:.6e} J (ratio={:.4})",
            h_prev / h_start
        );
        assert!(
            h_prev < h_start * 0.5,
            "Gate 4 damped energy did not decay meaningfully at sr={sr}: {h_prev:.6e} vs {h_start:.6e}"
        );
    }
}
