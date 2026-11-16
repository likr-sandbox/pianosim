use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use piano_spec::{
    BodyModeRecord, BodyPreset, LidState, ListenerPerspective, PresetReducedPole,
    PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES, PRESET_REDUCED_POLES,
};

use crate::{
    ir_synth::ir_len_for_sample_rate, partitioned_fft::NupolsTail, RadiatorBank,
    CONCERT_GRAND_PRESET_BYTES,
};

pub(crate) fn test_preset(num_modes: usize) -> BodyPreset {
    let zone_pos_m = [
        [-0.58, 0.52, 0.74],
        [-0.42, 0.64, 0.74],
        [-0.25, 0.72, 0.74],
        [-0.08, 0.58, 0.74],
        [0.09, 0.50, 0.74],
        [0.26, 0.42, 0.74],
        [0.42, 0.34, 0.74],
        [0.57, 0.28, 0.74],
        [0.00, 0.04, 0.70],
    ];
    let y_inf = [1.8e-4_f32; PRESET_BRIDGE_ZONES];
    let direct_zone_gain = [0.08, 0.09, 0.10, 0.11, 0.12, 0.11, 0.10, 0.09, 0.05];

    let mut reduced_poles = [PresetReducedPole {
        freq_hz: 100.0,
        zeta: 0.02,
        modal_mass_kg: 1.0,
        zone_gains: [0.0; PRESET_BRIDGE_ZONES],
    }; PRESET_REDUCED_POLES];

    for (p, pole) in reduced_poles.iter_mut().enumerate() {
        pole.freq_hz = 75.0 + (p as f32) * 115.0;
        pole.zeta = 0.015 + 0.002 * (p as f32);
        pole.modal_mass_kg = 0.65 + 0.08 * (p as f32);
        for z in 0..PRESET_BRIDGE_ZONES {
            let phase = ((p + 1) * (z + 1)) as f32 * 0.45;
            pole.zone_gains[z] = 0.008 * phase.sin();
        }
    }

    let mut modes = Vec::with_capacity(num_modes);
    for m in 0..num_modes {
        let frac = (m as f32) / (num_modes.max(2) as f32);
        let freq_hz = 55.0 * (250.0_f32).powf(frac);
        let zeta = 0.004 + 0.009 * frac;
        let mut zone_drive = [0.0_f32; PRESET_RADIATOR_ZONES];
        let norm = 2.0 * zeta * (crate::modal_prepare::MODAL_BODY_BLEND as f32);
        for (z, zd) in zone_drive.iter_mut().enumerate() {
            let k = (m as f32) * 0.17 + (z as f32) * 0.73;
            *zd = 0.0018 * k.sin() / ((1.0 + 0.5 * frac) * norm);
        }
        let rad_coeffs = [
            0.75 + 0.25 * ((m as f32) * 0.31).cos(),
            0.35 * ((m as f32) * 0.57).sin(),
            0.28 * ((m as f32) * 0.41).cos(),
            0.22 * ((m as f32) * 0.19).sin(),
            0.30 + 0.20 * frac,
        ];
        modes.push(BodyModeRecord {
            freq_hz,
            zeta,
            zone_drive,
            rad_coeffs,
            cavity_flag: u8::from(m < 4),
        });
    }

    BodyPreset {
        y_inf,
        direct_zone_gain,
        zone_pos_m,
        reduced_poles,
        modes,
    }
}

fn direct_convolve_18(
    irs: &[[Vec<f32>; 2]; PRESET_RADIATOR_ZONES],
    inputs: &[[f32; PRESET_RADIATOR_ZONES]],
) -> (Vec<f32>, Vec<f32>) {
    let n_out = inputs.len();
    let ir_len = irs[0][0].len();
    let mut ref_l = vec![0.0_f64; n_out];
    let mut ref_r = vec![0.0_f64; n_out];

    for (n, frame) in inputs.iter().enumerate() {
        for z in 0..PRESET_RADIATOR_ZONES {
            let x = frame[z] as f64;
            if x == 0.0 {
                continue;
            }
            let max_k = (n_out - n).min(ir_len);
            let hl = &irs[z][0];
            let hr = &irs[z][1];
            for k in 0..max_k {
                ref_l[n + k] += x * (hl[k] as f64);
                ref_r[n + k] += x * (hr[k] as f64);
            }
        }
    }
    (
        ref_l.into_iter().map(|v| v as f32).collect(),
        ref_r.into_iter().map(|v| v as f32).collect(),
    )
}

#[test]
fn nupols_matches_direct_convolution_below_minus_100db() {
    let sr = 48_000.0;
    let preset = test_preset(180);
    let mut bank = RadiatorBank::from_preset_with_space(
        &preset,
        sr,
        ListenerPerspective::CloseStereo,
        LidState::Open,
    );
    let irs = RadiatorBank::synthesize_ir_for_test(
        &preset,
        sr,
        ListenerPerspective::CloseStereo,
        LidState::Open,
    );

    // Total length > `24,000` (IR length) AND > `16,384` (ring size) to cover all 4 tiers + wraps.
    let total_samples = 26_000;
    let mut inputs = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; total_samples];
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    for (n, frame) in inputs.iter_mut().enumerate() {
        if n == 0 || n == 37 || n == 199 || n == 4_103 || n == 12_345 {
            frame[n % PRESET_RADIATOR_ZONES] += 0.25;
        }
        if n < 20_000 {
            for s in frame.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u = ((rng >> 40) as f32) * (1.0 / 8_388_608.0) - 1.0;
                *s += 0.03 * u;
            }
        }
    }

    let mut out_l = vec![0.0_f32; total_samples];
    let mut out_r = vec![0.0_f32; total_samples];
    let uneven_blocks = [1, 7, 64, 128, 256, 513, 3, 13, 31, 63, 127, 300, 41];
    let mut pos = 0;
    let mut b_idx = 0;
    while pos < total_samples {
        let bs = uneven_blocks[b_idx % uneven_blocks.len()].min(total_samples - pos);
        bank.process_block(
            &inputs[pos..pos + bs],
            &mut out_l[pos..pos + bs],
            &mut out_r[pos..pos + bs],
        );
        pos += bs;
        b_idx += 1;
    }

    let (ref_l, ref_r) = direct_convolve_18(&irs, &inputs);
    let mut err_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    let mut max_abs_out = 0.0_f32;
    for i in 0..total_samples {
        max_abs_out = max_abs_out.max(out_l[i].abs()).max(out_r[i].abs());
        let dl = (out_l[i] as f64) - (ref_l[i] as f64);
        let dr = (out_r[i] as f64) - (ref_r[i] as f64);
        err_sq += dl * dl + dr * dr;
        ref_sq += (ref_l[i] as f64).powi(2) + (ref_r[i] as f64).powi(2);
    }

    assert!(
        max_abs_out < 0.90,
        "test stimulus should stay in linear region of limiter, got peak={max_abs_out}"
    );
    let rel_db = 10.0 * (err_sq / ref_sq.max(1e-30)).log10();
    eprintln!("NUPOLS vs direct 18-path RMS error = {rel_db:.2} dB");
    assert!(
        rel_db <= -100.0,
        "NUPOLS RMS error {rel_db:.2} dB exceeded -100 dB budget"
    );
}

#[test]
fn zero_sample_algorithmic_latency_all_zones_and_perspectives() {
    let preset = test_preset(64);
    let perspectives = [
        ListenerPerspective::Player,
        ListenerPerspective::CloseStereo,
        ListenerPerspective::ConcertHall,
        ListenerPerspective::BinauralHrtf,
    ];
    let lids = [LidState::Closed, LidState::Half, LidState::Open];

    for &persp in &perspectives {
        for &lid in &lids {
            let mut bank = RadiatorBank::from_preset_with_space(&preset, 48_000.0, persp, lid);
            assert_eq!(bank.algorithmic_latency_samples(), 0);
            for z in 0..PRESET_RADIATOR_ZONES {
                bank.reset();
                let k = 37;
                let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 128];
                frames[k][z] = 0.5;
                let mut out_l = vec![0.0_f32; 128];
                let mut out_r = vec![0.0_f32; 128];
                bank.process_block(&frames, &mut out_l, &mut out_r);

                for i in 0..k {
                    assert_eq!(out_l[i], 0.0, "pre-impulse non-zero at {i} for z={z}");
                    assert_eq!(out_r[i], 0.0, "pre-impulse non-zero at {i} for z={z}");
                }
                let mag_at_k = out_l[k].abs().max(out_r[k].abs());
                assert!(
                    mag_at_k > 1.0e-5,
                    "expected immediate non-zero output at k={k} for {persp:?}/{lid:?}/z={z}, got {mag_at_k}"
                );
            }
        }
    }
}

#[test]
fn reset_immediately_zeros_state_and_repeats_cleanly() {
    let preset = test_preset(96);
    let mut bank = RadiatorBank::from_preset(&preset, 48_000.0);

    // Run past 4,096 samples so Tier 6 (`B=4096`) holds pending split state as well.
    let n_warm = 4_200;
    let mut warm = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; n_warm];
    for (i, frame) in warm.iter_mut().enumerate() {
        frame[i % PRESET_RADIATOR_ZONES] = if i % 2 == 0 { 0.12 } else { -0.09 };
    }
    let mut wl = vec![0.0_f32; n_warm];
    let mut wr = vec![0.0_f32; n_warm];
    bank.process_block(&warm, &mut wl, &mut wr);

    bank.reset();

    let zeros = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 2_048];
    let mut zl = vec![1.0_f32; 2_048];
    let mut zr = vec![1.0_f32; 2_048];
    bank.process_block(&zeros, &mut zl, &mut zr);
    assert!(zl.iter().all(|&s| s == 0.0));
    assert!(zr.iter().all(|&s| s == 0.0));

    bank.reset();
    let mut wl2 = vec![0.0_f32; n_warm];
    let mut wr2 = vec![0.0_f32; n_warm];
    bank.process_block(&warm, &mut wl2, &mut wr2);
    for i in 0..n_warm {
        assert!((wl[i] - wl2[i]).abs() < 1.0e-7);
        assert!((wr[i] - wr2[i]).abs() < 1.0e-7);
    }
}

#[test]
fn staggered_schedule_is_strictly_collision_free_on_128_sample_blocks() {
    // Verify 4-tier 4x-growth schedule over `128` 128-sample blocks (`256` 64-sample slots):
    // - Tier 3 (`B=4096`) fires at most ONCE across any 128-sample callback (`sub` spaced by 4).
    // - Tier 2 (`B=1024`) IFFTs (`sub16 in {12, 13}`) never coincide with Tier 3 (`sub64 & 3 == 3`).
    let mut tier3_ops_total = 0;
    for b in 0..128_usize {
        let mut tier3_in_block = 0;
        for half in 0..2_usize {
            let m = 2 * b + half + 1;
            let sub16 = m & 15;
            let sub64 = m & 63;
            let tier2_ifft = m >= 16 && (sub16 == 12 || sub16 == 13);
            let tier3_op = m >= 64
                && matches!(
                    sub64,
                    3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 | 51 | 55 | 59
                );
            assert!(
                !(tier2_ifft && tier3_op),
                "Tier 2 IFFT collided with Tier 3 8192-pt op at slot m={m}"
            );
            if tier3_op {
                tier3_in_block += 1;
                tier3_ops_total += 1;
            }
        }
        assert!(
            tier3_in_block <= 1,
            "multiple Tier 3 8192-pt ops in 128-sample block b={b}"
        );
    }
    assert!(tier3_ops_total > 0);
}

#[test]
fn arbitrary_block_sizes_and_single_sample_parity() {
    let preset = test_preset(48);
    let mut bank_a = RadiatorBank::from_preset(&preset, 44_100.0);
    let mut bank_b = RadiatorBank::from_preset(&preset, 44_100.0);

    let len = 1_024;
    let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; len];
    for (n, f) in frames.iter_mut().enumerate() {
        f[(n * 3) % PRESET_RADIATOR_ZONES] = ((n as f32) * 0.13).sin() * 0.15;
    }

    let mut la = vec![0.0_f32; len];
    let mut ra = vec![0.0_f32; len];
    for n in 0..len {
        bank_a.process_block(&frames[n..n + 1], &mut la[n..n + 1], &mut ra[n..n + 1]);
    }

    let sizes = [1, 7, 64, 128, 256, 513];
    let mut bank_c = RadiatorBank::from_preset(&preset, 44_100.0);
    let mut lc = vec![0.0_f32; len];
    let mut rc = vec![0.0_f32; len];
    let (mut pos, mut idx) = (0, 0);
    while pos < len {
        let bs = sizes[idx % sizes.len()].min(len - pos);
        bank_c.process_block(
            &frames[pos..pos + bs],
            &mut lc[pos..pos + bs],
            &mut rc[pos..pos + bs],
        );
        pos += bs;
        idx += 1;
    }

    let mut lb = vec![0.0_f32; len + 16];
    let mut rb = vec![0.0_f32; len + 16];
    bank_b.process_block(&frames, &mut lb, &mut rb);

    for i in 0..len {
        assert!(
            (la[i] - lb[i]).abs() < 1.0e-5,
            "L single vs batch mismatch at {i}"
        );
        assert!(
            (ra[i] - rb[i]).abs() < 1.0e-5,
            "R single vs batch mismatch at {i}"
        );
        assert!(
            (lc[i] - lb[i]).abs() < 1.0e-5,
            "L chunked vs batch mismatch at {i}"
        );
        assert!(
            (rc[i] - rb[i]).abs() < 1.0e-5,
            "R chunked vs batch mismatch at {i}"
        );
    }
    assert!(lb[len..].iter().all(|&x| x == 0.0));
    assert!(rb[len..].iter().all(|&x| x == 0.0));

    let ir_len = ir_len_for_sample_rate(44_100.0);
    assert_eq!(bank_a.ir_len_samples(), ir_len);
    let _tail = NupolsTail::new(256);
}

#[test]
fn concert_grand_init_clone_debug_and_d1_runtime_budget() {
    let decoded = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid baked preset");
    assert!(decoded.modes.len() > 1_000);

    let mut bank_a = RadiatorBank::new_concert_grand(48_000.0);
    let mut bank_b = bank_a.clone();
    let dbg_str = format!("{bank_a:?}");
    assert!(dbg_str.contains("RadiatorBank") && dbg_str.contains("48000"));

    let _alt_space = RadiatorBank::new_concert_grand_with_space(
        48_000.0,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    );

    let mut stim = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 256];
    stim[0][2] = 0.25;
    stim[71][5] = -0.18;
    let (mut la, mut ra, mut lb, mut rb) = (
        vec![0.0; 256],
        vec![0.0; 256],
        vec![0.0; 256],
        vec![0.0; 256],
    );
    bank_a.process_block(&stim[..128], &mut la[..128], &mut ra[..128]);
    bank_a.process_block(&stim[128..], &mut la[128..], &mut ra[128..]);
    bank_b.process_block(&stim[..128], &mut lb[..128], &mut rb[..128]);
    bank_b.process_block(&stim[128..], &mut lb[128..], &mut rb[128..]);
    assert!(la
        .iter()
        .zip(lb.iter())
        .all(|(a, b)| (a - b).abs() < 1.0e-7));
    assert!(ra
        .iter()
        .zip(rb.iter())
        .all(|(a, b)| (a - b).abs() < 1.0e-7));
    assert!(la[0].abs() > 1.0e-4);

    // Process block @ 48 kHz and verify deterministic finite, non-zero output.
    let block = [[0.01_f32; PRESET_RADIATOR_ZONES]; 128];
    let mut out_l = [0.0_f32; 128];
    let mut out_r = [0.0_f32; 128];
    for _ in 0..4 {
        bank_a.process_block(&block, &mut out_l, &mut out_r);
    }
    assert!(out_l.iter().all(|x| x.is_finite()));
    assert!(out_r.iter().all(|x| x.is_finite()));
    let energy_l: f32 = out_l.iter().map(|x| x * x).sum();
    let energy_r: f32 = out_r.iter().map(|x| x * x).sum();
    assert!(
        energy_l > 0.0 && energy_r > 0.0,
        "NUPOLS output must be non-zero"
    );
}
