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

    bank.set_perf_profile(&PerfProfile::balanced());
    assert_eq!(bank.fdn_lines(), 4);
    assert_eq!(bank.head_taps(), 32);
    assert_eq!(bank.ir_len_samples(), 12_000);
    assert_eq!(bank.algorithmic_latency_samples(), 0);

    bank.set_perf_profile(&PerfProfile::realtime());
    assert_eq!(bank.fdn_lines(), 4);
    assert_eq!(bank.head_taps(), 32);
    assert_eq!(bank.ir_len_samples(), 6_000);

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

    // Early [0..1024) must match Full closely (exact 9-zone MIMO + sub-tier).
    let early_db = rel_error_db(&rl[..1024], &rr[..1024], &fl[..1024], &fr[..1024]);
    assert!(
        early_db <= -90.0,
        "early MIMO region [0..1024) deviated {early_db:.2} dB"
    );

    // Late collapsed tail [1024..6000) must stay energetic and preserve L/R decorrelation.
    let e_full: f64 = fl[1024..6000]
        .iter()
        .chain(fr[1024..6000].iter())
        .map(|&v| f64::from(v).powi(2))
        .sum();
    let e_rt: f64 = rl[1024..6000]
        .iter()
        .chain(rr[1024..6000].iter())
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
    for i in 1024..6000 {
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
