use alloc::vec;
use piano_simd::{F32x4, RealFft};
use piano_spec::{LidState, ListenerPerspective, PRESET_RADIATOR_ZONES};

use crate::hall_fdn::HallFdn;
use crate::hall_hadamard::{hadamard_16, hadamard_4, hadamard_8};
use crate::nupols_tests::test_preset;
use crate::RadiatorBank;

#[test]
fn hadamard_4_is_strictly_orthogonal_and_energy_preserving() {
    let mut cols = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        let mut basis = [0.0_f32; 4];
        basis[c] = 1.0;
        cols[c] = hadamard_4(F32x4::from_array(basis)).to_array();
    }
    for i in 0..4 {
        for j in 0..4 {
            let dot: f32 = (0..4).map(|r| cols[i][r] * cols[j][r]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1.0e-6,
                "H4^T H4 [{i}][{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn hadamard_8_is_strictly_orthogonal_and_energy_preserving() {
    let mut cols = [[0.0_f32; 8]; 8];
    for c in 0..8 {
        let mut basis = [0.0_f32; 8];
        basis[c] = 1.0;
        let (y_lo, y_hi) = hadamard_8(
            F32x4::from_array([basis[0], basis[1], basis[2], basis[3]]),
            F32x4::from_array([basis[4], basis[5], basis[6], basis[7]]),
        );
        let a = y_lo.to_array();
        let b = y_hi.to_array();
        cols[c] = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
    }

    for i in 0..8 {
        for j in 0..8 {
            let dot: f32 = (0..8).map(|r| cols[i][r] * cols[j][r]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1.0e-6,
                "H^T H [{i}][{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn hadamard_16_is_strictly_orthogonal_and_energy_preserving() {
    let mut cols = [[0.0_f32; 16]; 16];
    for c in 0..16 {
        let mut basis = [0.0_f32; 16];
        basis[c] = 1.0;
        let y = hadamard_16([
            F32x4::from_array([basis[0], basis[1], basis[2], basis[3]]),
            F32x4::from_array([basis[4], basis[5], basis[6], basis[7]]),
            F32x4::from_array([basis[8], basis[9], basis[10], basis[11]]),
            F32x4::from_array([basis[12], basis[13], basis[14], basis[15]]),
        ]);
        for v in 0..4 {
            let arr = y[v].to_array();
            cols[c][v * 4..v * 4 + 4].copy_from_slice(&arr);
        }
    }
    for i in 0..16 {
        for j in 0..16 {
            let dot: f32 = (0..16).map(|r| cols[i][r] * cols[j][r]).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1.0e-6,
                "H16^T H16 [{i}][{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn hall_fdn_zero_latency_and_dry_bypass_parity() {
    let preset = test_preset(64);
    let mut dry_bank = RadiatorBank::from_preset(&preset, 48_000.0);
    assert_eq!(dry_bank.reverb_params().1, 0.0);

    let mut bypassed_bank = RadiatorBank::from_preset(&preset, 48_000.0);
    bypassed_bank.set_reverb(2.4, 0.0);

    let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 256];
    frames[0][2] = 0.4;
    frames[37][4] = -0.25;
    let (mut dl, mut dr) = (vec![0.0; 256], vec![0.0; 256]);
    let (mut bl, mut br) = (vec![0.0; 256], vec![0.0; 256]);
    dry_bank.process_block(&frames, &mut dl, &mut dr);
    bypassed_bank.process_block(&frames, &mut bl, &mut br);
    assert_eq!(dl, bl);
    assert_eq!(dr, br);

    // Active reverb: strict 0-sample pre-impulse silence + immediate frame-k output + long tail.
    let mut wet_bank = RadiatorBank::from_preset_with_space(
        &preset,
        48_000.0,
        ListenerPerspective::ConcertHall,
        LidState::Open,
    );
    wet_bank.set_reverb(1.8, 0.25);
    assert_eq!(wet_bank.reverb_params(), (1.8, 0.25));

    let k = 37;
    let mut imp = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 128];
    imp[k][2] = 0.5;
    let (mut ol, mut or) = (vec![0.0; 128], vec![0.0; 128]);
    wet_bank.process_block(&imp, &mut ol, &mut or);
    for i in 0..k {
        assert_eq!(ol[i], 0.0);
        assert_eq!(or[i], 0.0);
    }
    assert!(ol[k].abs().max(or[k].abs()) > 1.0e-5);

    // Run past 0.55 s (beyond 0.5 s body IR) and verify diffuse FDN tail persists
    // while dry bank has decayed to exact 0.0.
    dry_bank.reset();
    dry_bank.process_block(&imp, &mut dl, &mut dr);
    let silence = [[0.0_f32; PRESET_RADIATOR_ZONES]; 128];
    for _ in 0..220 {
        dry_bank.process_block(&silence, &mut dl, &mut dr);
        wet_bank.process_block(&silence, &mut ol, &mut or);
    }
    let dry_tail: f32 = dl.iter().chain(dr.iter()).map(|x| x * x).sum();
    let tail_energy: f32 = ol.iter().chain(or.iter()).map(|x| x * x).sum();
    assert!(dry_tail < 1.0e-18, "unexpected dry tail {dry_tail}");
    assert!(
        tail_energy > 1.0e-12 && tail_energy > dry_tail * 1.0e6,
        "expected active FDN tail > 0.55 s, got wet={tail_energy}, dry={dry_tail}"
    );
}

#[test]
fn hall_fdn_decay_time_scales_tail_energy_and_clamps_bounds() {
    let mut short = HallFdn::new(48_000.0, ListenerPerspective::ConcertHall);
    short.set_reverb(0.5, 0.4);
    let mut long = HallFdn::new(48_000.0, ListenerPerspective::ConcertHall);
    long.set_reverb(3.0, 0.4);

    // Clamping checks
    let mut clamp_test = HallFdn::new(48_000.0, ListenerPerspective::Player);
    clamp_test.set_reverb(9.0, 2.0);
    assert_eq!(clamp_test.reverb_params(), (3.2, 1.0));
    clamp_test.set_reverb(f32::NAN, f32::INFINITY);
    assert_eq!(clamp_test.reverb_params(), (0.0, 0.0));

    let n = 48_000; // 1.0 s
    let mut l_s = vec![0.0_f32; n];
    let mut r_s = vec![0.0_f32; n];
    let mut l_l = vec![0.0_f32; n];
    let mut r_l = vec![0.0_f32; n];
    l_s[0] = 0.6;
    r_s[0] = -0.4;
    l_l[0] = 0.6;
    r_l[0] = -0.4;

    short.process_block(&mut l_s, &mut r_s);
    long.process_block(&mut l_l, &mut r_l);

    let e_short: f64 = l_s[24_000..]
        .iter()
        .chain(r_s[24_000..].iter())
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    let e_long: f64 = l_l[24_000..]
        .iter()
        .chain(r_l[24_000..].iter())
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    assert!(
        e_long > e_short * 4.0,
        "longer decay should preserve much more tail energy: short={e_short}, long={e_long}"
    );
}

fn band_iacc(l: &[f32], r: &[f32], sr: f64, f_lo: f64, f_hi: f64) -> f64 {
    let fft_len = 16_384_usize;
    let mut pl = vec![0.0_f32; fft_len];
    let mut pr = vec![0.0_f32; fft_len];
    let copy_n = l.len().min(fft_len);
    pl[..copy_n].copy_from_slice(&l[..copy_n]);
    pr[..copy_n].copy_from_slice(&r[..copy_n]);

    let mut fft = RealFft::new(fft_len);
    let spec_len = fft.spectrum_len();
    let (mut re_l, mut im_l) = (vec![0.0_f32; spec_len], vec![0.0_f32; spec_len]);
    let (mut re_r, mut im_r) = (vec![0.0_f32; spec_len], vec![0.0_f32; spec_len]);
    fft.forward_split(&pl, &mut re_l, &mut im_l);
    fft.forward_split(&pr, &mut re_r, &mut im_r);

    let k0 = ((f_lo * fft_len as f64 / sr).ceil() as usize).max(1);
    let k1 = ((f_hi * fft_len as f64 / sr).floor() as usize).min(spec_len - 1);
    let mut cross = 0.0_f64;
    let mut el = 0.0_f64;
    let mut er = 0.0_f64;
    for k in k0..=k1 {
        let al_re = re_l[k] as f64;
        let al_im = im_l[k] as f64;
        let ar_re = re_r[k] as f64;
        let ar_im = im_r[k] as f64;
        cross += al_re * ar_re + al_im * ar_im;
        el += al_re * al_re + al_im * al_im;
        er += ar_re * ar_re + ar_im * ar_im;
    }
    cross / (el * er).sqrt().max(1.0e-30)
}

#[test]
fn hall_fdn_binaural_low_freq_iacc_coherent_and_diffuse_eq_active() {
    let sr = 48_000.0;
    let mut bin_fdn = HallFdn::new(sr, ListenerPerspective::BinauralHrtf);
    bin_fdn.set_reverb(2.2, 0.75);
    let mut hall_fdn = HallFdn::new(sr, ListenerPerspective::ConcertHall);
    hall_fdn.set_reverb(2.2, 0.75);

    let n = 16_384_usize;
    let mut bl = vec![0.0_f32; n];
    let mut br = vec![0.0_f32; n];
    let mut hl = vec![0.0_f32; n];
    let mut hr = vec![0.0_f32; n];
    // Feed an asymmetric stereo impulse at index 0, then measure wet tail after index 64.
    bl[0] = 0.6;
    br[0] = -0.3;
    hl[0] = 0.6;
    hr[0] = -0.3;
    bin_fdn.process_block(&mut bl, &mut br);
    hall_fdn.process_block(&mut hl, &mut hr);

    let iacc_low_bin = band_iacc(&bl[64..], &br[64..], sr, 60.0, 350.0);
    let iacc_high_bin = band_iacc(&bl[64..], &br[64..], sr, 1_200.0, 6_000.0);
    let iacc_low_hall = band_iacc(&hl[64..], &hr[64..], sr, 60.0, 350.0);
    eprintln!(
        "Wet tail IACC: Binaural <350Hz={iacc_low_bin:.3}, >1.2kHz={iacc_high_bin:.3} | Hall <350Hz={iacc_low_hall:.3}"
    );
    assert!(
        iacc_low_bin > 0.70,
        "expected coherent <400Hz binaural IACC > 0.70, got {iacc_low_bin:.3}"
    );
    assert!(
        iacc_low_bin > iacc_high_bin + 0.35,
        "expected low-freq IACC ({iacc_low_bin:.3}) >> high-freq IACC ({iacc_high_bin:.3})"
    );
    assert!(
        iacc_low_bin > iacc_low_hall + 0.25,
        "expected binaural IACC shaping to raise <350Hz correlation over raw hall FDN"
    );
}

#[test]
fn hall_fdn_reset_and_block_size_invariance() {
    let preset = test_preset(48);
    let mut bank_a = RadiatorBank::from_preset_with_space(
        &preset,
        48_000.0,
        ListenerPerspective::BinauralHrtf,
        LidState::Open,
    );
    bank_a.set_reverb(1.8, 0.30);
    let mut bank_b = bank_a.clone();

    let len = 1_024_usize;
    let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; len];
    for (i, f) in frames.iter_mut().enumerate() {
        f[i % PRESET_RADIATOR_ZONES] = ((i as f32) * 0.11).sin() * 0.12;
    }

    let (mut la, mut ra) = (vec![0.0; len], vec![0.0; len]);
    let (mut lb, mut rb) = (vec![0.0; len], vec![0.0; len]);
    for i in 0..len {
        bank_a.process_block(&frames[i..i + 1], &mut la[i..i + 1], &mut ra[i..i + 1]);
    }
    bank_b.process_block(&frames, &mut lb, &mut rb);
    for i in 0..len {
        assert!((la[i] - lb[i]).abs() < 1.0e-5, "L mismatch at {i}");
        assert!((ra[i] - rb[i]).abs() < 1.0e-5, "R mismatch at {i}");
    }

    bank_a.reset();
    let zeros = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 512];
    let (mut zl, mut zr) = (vec![1.0; 512], vec![1.0; 512]);
    bank_a.process_block(&zeros, &mut zl, &mut zr);
    assert!(zl.iter().all(|&x| x == 0.0));
    assert!(zr.iter().all(|&x| x == 0.0));

    let (mut la2, mut ra2) = (vec![0.0; len], vec![0.0; len]);
    bank_a.process_block(&frames, &mut la2, &mut ra2);
    for i in 0..len {
        assert!((la2[i] - lb[i]).abs() < 1.0e-6);
        assert!((ra2[i] - rb[i]).abs() < 1.0e-6);
    }
}

#[test]
fn hall_fdn_runtime_line_switching_4_8_16_preserves_binaural_iacc_and_stability() {
    let sr = 48_000.0;
    for &lines in &[4_u8, 8, 16] {
        let mut fdn = HallFdn::new_with_lines(sr, ListenerPerspective::BinauralHrtf, lines);
        assert_eq!(fdn.fdn_lines(), lines);
        fdn.set_reverb(2.0, 0.65);

        let n = 16_384_usize;
        let (mut l, mut r) = (vec![0.0_f32; n], vec![0.0_f32; n]);
        l[0] = 0.55;
        r[0] = -0.35;
        fdn.process_block(&mut l, &mut r);

        assert!(l.iter().all(|x| x.is_finite()));
        assert!(r.iter().all(|x| x.is_finite()));
        let iacc_lo = band_iacc(&l[64..], &r[64..], sr, 60.0, 350.0);
        let iacc_hi = band_iacc(&l[64..], &r[64..], sr, 1_200.0, 6_000.0);
        assert!(
            iacc_lo > 0.70,
            "lines={lines} <350Hz binaural IACC={iacc_lo:.3} must exceed 0.70"
        );
        assert!(
            iacc_lo > iacc_hi + 0.30,
            "lines={lines} low IACC ({iacc_lo:.3}) must exceed high IACC ({iacc_hi:.3})"
        );
    }

    // Verify hot-swapping 8 -> 4 -> 16 -> 8 on RadiatorBank is finite and updates line count.
    let preset = test_preset(32);
    let mut bank = RadiatorBank::from_preset(&preset, sr);
    bank.set_reverb(1.8, 0.25);
    assert_eq!(bank.fdn_lines(), 8);
    let frames = [[0.05_f32; PRESET_RADIATOR_ZONES]; 128];
    let (mut ol, mut or) = ([0.0_f32; 128], [0.0_f32; 128]);
    for &lines in &[4_u8, 16, 4, 8] {
        bank.set_fdn_lines(lines);
        assert_eq!(bank.fdn_lines(), lines);
        bank.process_block(&frames, &mut ol, &mut or);
        assert!(ol.iter().all(|x| x.is_finite()));
        assert!(or.iter().all(|x| x.is_finite()));
    }
}
