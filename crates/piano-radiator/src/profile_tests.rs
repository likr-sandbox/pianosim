//! Stage 4 M3 `PerfProfile` tests: `Full` M2 parity, `ir_duration_secs`,
//! `spatial_tail_split_tap` collapse, `fir_head_taps` + `[active..64)` sub-tier,
//! `fir_block_fft` 32-sample block FFT equivalence, and `SpaceBlob` v2 streaming.

use alloc::vec;
use alloc::vec::Vec;
use piano_spec::{LidState, ListenerPerspective, PerfProfile, PRESET_RADIATOR_ZONES};

use crate::nupols_tests::test_preset;
use crate::{
    ir_len_for_sample_rate, ir_len_for_sample_rate_secs, space_blob_len_for_profile,
    space_blob_len_for_sample_rate, RadiatorBank, SpacePrepContext,
};

fn build_stimulus(total_samples: usize) -> Vec<[f32; PRESET_RADIATOR_ZONES]> {
    let mut inputs = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; total_samples];
    let mut rng: u64 = 0x517c_c1b7_2722_0a95;
    for (n, frame) in inputs.iter_mut().enumerate() {
        if n == 0 || n == 37 || n == 199 || n == 4_103 {
            frame[n % PRESET_RADIATOR_ZONES] += 0.25;
        }
        if n < 10_000 {
            for s in frame.iter_mut() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let u = ((rng >> 40) as f32) * (1.0 / 8_388_608.0) - 1.0;
                *s += 0.025 * u;
            }
        }
    }
    inputs
}

fn render_with_uneven_blocks(
    bank: &mut RadiatorBank,
    inputs: &[[f32; PRESET_RADIATOR_ZONES]],
) -> (Vec<f32>, Vec<f32>) {
    let total = inputs.len();
    let mut out_l = vec![0.0_f32; total];
    let mut out_r = vec![0.0_f32; total];
    let blocks = [1, 7, 17, 32, 64, 128, 256, 41];
    let mut pos = 0;
    let mut b_idx = 0;
    while pos < total {
        let bs = blocks[b_idx % blocks.len()].min(total - pos);
        bank.process_block(
            &inputs[pos..pos + bs],
            &mut out_l[pos..pos + bs],
            &mut out_r[pos..pos + bs],
        );
        pos += bs;
        b_idx += 1;
    }
    (out_l, out_r)
}

fn rel_error_db(a_l: &[f32], a_r: &[f32], b_l: &[f32], b_r: &[f32]) -> f64 {
    let mut err_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    for i in 0..a_l.len().min(b_l.len()) {
        let dl = f64::from(a_l[i]) - f64::from(b_l[i]);
        let dr = f64::from(a_r[i]) - f64::from(b_r[i]);
        err_sq += dl * dl + dr * dr;
        ref_sq += f64::from(b_l[i]).powi(2) + f64::from(b_r[i]).powi(2);
    }
    10.0 * (err_sq / ref_sq.max(1.0e-30)).log10()
}

#[test]
fn full_profile_matches_default_m2_radiator_bit_exact() {
    let sr = 48_000.0;
    let preset = test_preset(96);
    let mut default_bank = RadiatorBank::from_preset_with_space(
        &preset,
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Half,
    );
    let mut full_bank = RadiatorBank::from_preset_with_profile(
        &preset,
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Half,
        &PerfProfile::full(),
    );

    assert_eq!(full_bank.algorithmic_latency_samples(), 0);
    assert_eq!(full_bank.head_taps(), 64);
    assert_eq!(full_bank.fdn_lines(), 8);
    assert_eq!(full_bank.ir_len_samples(), ir_len_for_sample_rate(sr));

    let inputs = build_stimulus(2_048);
    let (dl, dr) = render_with_uneven_blocks(&mut default_bank, &inputs);
    let (fl, fr) = render_with_uneven_blocks(&mut full_bank, &inputs);
    assert_eq!(dl, fl);
    assert_eq!(dr, fr);
}

#[test]
fn fir_head_taps_32_and_16_sub_tier_preserve_full_ir_and_itd_below_minus_100db() {
    let sr = 48_000.0;
    let preset = test_preset(80);
    let mut full_bank = RadiatorBank::from_preset_with_profile(
        &preset,
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
        &PerfProfile::full(),
    );
    let inputs = build_stimulus(16_000);
    let (ref_l, ref_r) = render_with_uneven_blocks(&mut full_bank, &inputs);

    for taps in [32_u8, 16] {
        let mut prof = PerfProfile::full();
        prof.fir_head_taps = taps;
        let mut bank = RadiatorBank::from_preset_with_profile(
            &preset,
            sr,
            ListenerPerspective::BinauralHrtf,
            LidState::Open,
            &prof,
        );
        assert_eq!(bank.algorithmic_latency_samples(), 0);
        assert_eq!(bank.head_taps(), usize::from(taps));
        let (out_l, out_r) = render_with_uneven_blocks(&mut bank, &inputs);
        let db = rel_error_db(&out_l, &out_r, &ref_l, &ref_r);
        assert!(
            db <= -100.0,
            "fir_head_taps={taps} + sub-tier deviated {db:.2} dB (> -100 dB)"
        );
    }
}

#[test]
fn fir_block_fft_matches_direct_fir_below_minus_100db_shifted_by_32_samples() {
    let sr = 48_000.0;
    let preset = test_preset(80);
    let mut full_bank = RadiatorBank::from_preset_with_profile(
        &preset,
        sr,
        ListenerPerspective::CloseStereo,
        LidState::Open,
        &PerfProfile::full(),
    );
    let mut fft_prof = PerfProfile::full();
    fft_prof.fir_block_fft = true;
    let mut fft_bank = RadiatorBank::from_preset_with_profile(
        &preset,
        sr,
        ListenerPerspective::CloseStereo,
        LidState::Open,
        &fft_prof,
    );

    assert_eq!(fft_bank.algorithmic_latency_samples(), 32);

    let inputs = build_stimulus(18_000);
    let (ref_l, ref_r) = render_with_uneven_blocks(&mut full_bank, &inputs);
    let (fft_l, fft_r) = render_with_uneven_blocks(&mut fft_bank, &inputs);

    for i in 0..32 {
        assert_eq!(fft_l[i], 0.0, "expected 32-sample silence prefix at {i}");
        assert_eq!(fft_r[i], 0.0, "expected 32-sample silence prefix at {i}");
    }
    let len = inputs.len() - 32;
    let db = rel_error_db(&fft_l[32..], &fft_r[32..], &ref_l[..len], &ref_r[..len]);
    assert!(
        db <= -100.0,
        "fir_block_fft shifted by 32 samples exceeded -100 dB: {db:.2} dB"
    );
}

#[test]
fn ir_duration_and_spatial_split_profiles_and_space_blob_v2_roundtrip() {
    let sr = 48_000.0;
    assert_eq!(ir_len_for_sample_rate_secs(sr, 0.25), 12_000);
    assert_eq!(ir_len_for_sample_rate_secs(sr, 0.125), 6_000);
    assert_eq!(ir_len_for_sample_rate_secs(sr, 0.50), 24_000);

    let full_len = space_blob_len_for_sample_rate(sr);
    let bal_len = space_blob_len_for_profile(sr, &PerfProfile::balanced());
    let rt_len = space_blob_len_for_profile(sr, &PerfProfile::realtime());
    assert!(bal_len < full_len);
    assert!(rt_len < bal_len);

    let preset = test_preset(64);
    let mut bank = RadiatorBank::from_preset(&preset, sr);
    let mut prep = SpacePrepContext::new_from_preset(preset.clone(), sr);

    let profiles = [
        PerfProfile::balanced(),
        PerfProfile::realtime(),
        PerfProfile {
            fir_head_taps: 16,
            ..PerfProfile::realtime()
        },
        PerfProfile {
            fir_block_fft: true,
            ..PerfProfile::balanced()
        },
        PerfProfile::full(),
    ];

    let frames = build_stimulus(512);
    for prof in &profiles {
        let mut ref_bank = RadiatorBank::from_preset_with_profile(
            &preset,
            sr,
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            prof,
        );
        ref_bank.set_fdn_lines(bank.fdn_lines());

        prep.set_perf_profile(prof);
        let blob = prep.prepare_space(ListenerPerspective::BinauralHrtf, LidState::Half);
        assert_eq!(blob.len(), space_blob_len_for_profile(sr, prof));
        assert!(blob.len() <= full_len);
        assert!(bank.install_precomputed_space(
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            blob,
        ));
        bank.reset();

        assert_eq!(
            bank.ir_len_samples(),
            ir_len_for_sample_rate_secs(sr, f64::from(prof.ir_duration_secs))
        );
        assert_eq!(
            bank.algorithmic_latency_samples(),
            if prof.fir_block_fft { 32 } else { 0 }
        );

        let (rl, rr) = render_with_uneven_blocks(&mut ref_bank, &frames);
        let (bl, br) = render_with_uneven_blocks(&mut bank, &frames);
        assert_eq!(rl, bl, "L mismatch for profile {prof:?}");
        assert_eq!(rr, br, "R mismatch for profile {prof:?}");

        // Also verify chunked streaming for non-Full profiles.
        let mut chunk_bank = RadiatorBank::from_preset(&preset, sr);
        chunk_bank.set_fdn_lines(bank.fdn_lines());
        let total = blob.len();
        let mut off = 0usize;
        while off < total {
            let n = 4_099.min(total - off);
            assert!(chunk_bank.install_precomputed_space_chunk(
                ListenerPerspective::BinauralHrtf,
                LidState::Half,
                off,
                &blob[off..off + n],
                total,
            ));
            off += n;
        }
        let (cl, cr) = render_with_uneven_blocks(&mut chunk_bank, &frames);
        assert_eq!(rl, cl, "chunked L mismatch for profile {prof:?}");
        assert_eq!(rr, cr, "chunked R mismatch for profile {prof:?}");
    }
}

#[test]
fn set_perf_profile_switches_all_stage4_knobs_cleanly() {
    let sr = 48_000.0;
    let mut bank = RadiatorBank::new_concert_grand(sr);
    assert_eq!(bank.fdn_lines(), 8);
    assert_eq!(bank.ir_len_samples(), 24_000);

    let bal = PerfProfile::balanced();
    bank.set_perf_profile(&bal);
    assert_eq!(bank.fdn_lines(), bal.hall_fdn_lines);
    assert_eq!(bank.head_taps(), usize::from(bal.fir_head_taps));
    assert_eq!(
        bank.ir_len_samples(),
        ir_len_for_sample_rate_secs(sr, f64::from(bal.ir_duration_secs))
    );
    assert_eq!(bank.algorithmic_latency_samples(), 0);

    let rt = PerfProfile::realtime();
    bank.set_perf_profile(&rt);
    assert_eq!(bank.fdn_lines(), rt.hall_fdn_lines);
    assert_eq!(bank.head_taps(), usize::from(rt.fir_head_taps));
    assert_eq!(
        bank.ir_len_samples(),
        ir_len_for_sample_rate_secs(sr, f64::from(rt.ir_duration_secs))
    );
    assert_eq!(bank.algorithmic_latency_samples(), 32);

    let frames = build_stimulus(256);
    let (mut l, mut r) = ([0.0_f32; 256], [0.0_f32; 256]);
    bank.process_block(&frames, &mut l, &mut r);
    assert!(l.iter().all(|x| x.is_finite()));
    assert!(r.iter().all(|x| x.is_finite()));
    assert!(l.iter().any(|&x| x.abs() > 1.0e-5));
}

#[test]
fn collapsed_spatial_tail_preserves_energy_and_binaural_decorrelation() {
    let sr = 48_000.0;
    let total = 8_192_usize;
    let mut stim = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; total];
    stim[0][1] = 0.25;
    stim[0][2] = 0.15;

    let mut full = RadiatorBank::new_with_profile(
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
        &PerfProfile::full(),
    );
    let mut rt = RadiatorBank::new_with_profile(
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
        &PerfProfile::realtime(),
    );

    let (mut fl, mut fr) = (vec![0.0_f32; total], vec![0.0_f32; total]);
    let (mut rl, mut rr) = (vec![0.0_f32; total], vec![0.0_f32; total]);
    full.process_block(&stim, &mut fl, &mut fr);
    rt.process_block(&stim, &mut rl, &mut rr);

    // Early [lat..split_tap) must match Full closely (exact 9-zone MIMO + sub-tier).
    let lat = rt.algorithmic_latency_samples();
    let split = usize::from(PerfProfile::realtime().spatial_tail_split_tap);
    let early_db = rel_error_db(
        &rl[lat..split],
        &rr[lat..split],
        &fl[..split - lat],
        &fr[..split - lat],
    );
    assert!(
        early_db <= -90.0,
        "early MIMO region [{lat}..{split}) deviated {early_db:.2} dB"
    );

    // Late collapsed tail [split..6000) must stay energetic and preserve L/R decorrelation.
    let e_full: f64 = fl[split..6000]
        .iter()
        .chain(fr[split..6000].iter())
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let e_rt: f64 = rl[split..6000]
        .iter()
        .chain(rr[split..6000].iter())
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let ratio_db = 10.0 * (e_rt / e_full.max(1.0e-30)).log10();
    assert!(
        (-6.0..=6.0).contains(&ratio_db),
        "collapsed tail energy ratio {ratio_db:.2} dB out of [-6, +6] dB"
    );

    let mut cross = 0.0_f64;
    let mut el = 0.0_f64;
    let mut er = 0.0_f64;
    for i in split..6000 {
        let lv = f64::from(rl[i]);
        let rv = f64::from(rr[i]);
        cross += lv * rv;
        el += lv * lv;
        er += rv * rv;
    }
    let corr = (cross / (el * er).sqrt().max(1.0e-30)).abs();
    assert!(
        corr < 0.70,
        "collapsed binaural tail L/R correlation {corr:.3} too high"
    );
}

#[test]
fn causal_collapse_has_zero_wraparound_tail_and_no_partition_step_clicks() {
    let sr = 48_000.0;
    let total = 16_384_usize;
    let mut stim = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; total];
    for (i, frame) in stim.iter_mut().enumerate() {
        let t = (i as f32) / 48_000.0;
        let s = 0.05 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
        frame[1] = s;
        frame[2] = 0.7 * s;
    }
    let mut bank = RadiatorBank::new_with_profile(
        sr,
        ListenerPerspective::Player,
        LidState::Open,
        &PerfProfile::balanced(),
    );
    let (mut ol, mut or) = (vec![0.0_f32; total], vec![0.0_f32; total]);
    bank.process_block(&stim, &mut ol, &mut or);
    // Steady sine response after initial transient must have tiny 3rd difference (no 1024-sample partition clicks).
    let max_d3 = ol[4096..]
        .windows(4)
        .map(|w| (w[3] - 3.0 * w[2] + 3.0 * w[1] - w[0]).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_d3 < 1.0e-3,
        "expected smooth collapsed sine without partition clicks, got max_d3={max_d3}"
    );
}
