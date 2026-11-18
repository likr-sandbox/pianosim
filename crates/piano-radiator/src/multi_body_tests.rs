use alloc::vec;
use piano_spec::{BodyPreset, LidState, ListenerPerspective, PresetDecodeError};

use crate::nupols_tests::test_preset;
use crate::{
    body_asset_filename, embedded_body_bytes, space_blob_len_for_sample_rate, BodyArchetype,
    RadiatorBank, SpacePrepContext, CONCERT_GRAND_PRESET_BYTES, PRESET_RADIATOR_ZONES,
};

#[test]
fn body_archetype_enum_filenames_and_embedded_bytes() {
    assert_eq!(BodyArchetype::from_u8(0), Some(BodyArchetype::ConcertGrand));
    assert_eq!(
        BodyArchetype::from_u8(1),
        Some(BodyArchetype::UprightStudio)
    );
    assert_eq!(
        BodyArchetype::from_u8(2),
        Some(BodyArchetype::VintageSaloon)
    );
    assert_eq!(BodyArchetype::from_u8(3), None);

    assert_eq!(
        body_asset_filename(BodyArchetype::ConcertGrand),
        "concert_grand_v1.pbdy"
    );
    assert_eq!(
        body_asset_filename(BodyArchetype::UprightStudio),
        "upright_studio_v1.pbdy"
    );
    assert_eq!(
        body_asset_filename(BodyArchetype::VintageSaloon),
        "vintage_saloon_v1.pbdy"
    );

    let grand_bytes = embedded_body_bytes(BodyArchetype::ConcertGrand);
    let upright_bytes = embedded_body_bytes(BodyArchetype::UprightStudio);
    let saloon_bytes = embedded_body_bytes(BodyArchetype::VintageSaloon);
    assert_ne!(grand_bytes, upright_bytes);
    assert_ne!(grand_bytes, saloon_bytes);
    assert_ne!(upright_bytes, saloon_bytes);

    let grand = BodyPreset::decode(grand_bytes).expect("valid concert grand bytes");
    let upright = BodyPreset::decode(upright_bytes).expect("valid upright studio bytes");
    let saloon = BodyPreset::decode(saloon_bytes).expect("valid vintage saloon bytes");
    assert!(!grand.modes.is_empty());
    assert!(!upright.modes.is_empty());
    assert!(!saloon.modes.is_empty());
    assert_ne!(grand.y_inf, upright.y_inf);
    assert_ne!(grand.y_inf, saloon.y_inf);
    assert_ne!(upright.y_inf, saloon.y_inf);
    assert_ne!(grand.reduced_poles, upright.reduced_poles);
    assert_ne!(grand.reduced_poles, saloon.reduced_poles);
    assert_ne!(upright.reduced_poles, saloon.reduced_poles);
}

#[test]
fn all_three_embedded_pbdy_archetypes_synthesize_measurably_distinct_irs_and_audio() {
    let sr = 48_000.0;
    let archetypes = [
        BodyArchetype::ConcertGrand,
        BodyArchetype::UprightStudio,
        BodyArchetype::VintageSaloon,
    ];
    let mut presets = Vec::with_capacity(3);
    let mut irs = Vec::with_capacity(3);
    let mut outputs = Vec::with_capacity(3);
    let mut blobs = Vec::with_capacity(3);

    let mut impulse_block = [[0.0_f32; PRESET_RADIATOR_ZONES]; 128];
    impulse_block[0] = [0.25; PRESET_RADIATOR_ZONES];

    for arch in archetypes {
        let bytes = embedded_body_bytes(arch);
        let decoded = BodyPreset::decode(bytes).expect("valid embedded preset");
        let ir = RadiatorBank::synthesize_ir_for_test(
            &decoded,
            sr,
            ListenerPerspective::Player,
            LidState::Open,
        );
        let mut bank = RadiatorBank::from_preset_bytes(bytes, sr).expect("construct bank");
        let (mut out_l, mut out_r) = ([0.0_f32; 128], [0.0_f32; 128]);
        bank.process_block(&impulse_block, &mut out_l, &mut out_r);
        let e_out: f64 = out_l
            .iter()
            .chain(out_r.iter())
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        assert!(e_out > 1.0e-4, "{arch:?} produced silent output {e_out}");

        let mut prep = SpacePrepContext::new_from_preset_bytes(bytes, sr).expect("space prep");
        let blob = prep
            .prepare_space(ListenerPerspective::Player, LidState::Open)
            .to_vec();
        blobs.push(blob);
        presets.push(decoded);
        irs.push(ir);
        outputs.push((out_l, out_r));
    }

    let ir_rel_l2 = |a: &[[Vec<f32>; 2]; PRESET_RADIATOR_ZONES],
                     b: &[[Vec<f32>; 2]; PRESET_RADIATOR_ZONES]|
     -> f64 {
        let (mut diff_sq, mut norm_sq) = (0.0_f64, 0.0_f64);
        for z in 0..PRESET_RADIATOR_ZONES {
            for ch in 0..2 {
                for (&x, &y) in a[z][ch].iter().zip(b[z][ch].iter()) {
                    let fx = f64::from(x);
                    let fy = f64::from(y);
                    diff_sq += (fx - fy) * (fx - fy);
                    norm_sq += fx * fx + fy * fy;
                }
            }
        }
        (diff_sq / norm_sq.max(1.0e-30)).sqrt()
    };

    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        assert_ne!(presets[i].direct_zone_gain, presets[j].direct_zone_gain);
        assert_ne!(presets[i].zone_pos_m, presets[j].zone_pos_m);
        assert_ne!(blobs[i], blobs[j]);
        assert_ne!(outputs[i], outputs[j]);
        let rel = ir_rel_l2(&irs[i], &irs[j]);
        assert!(
            rel > 0.10,
            "{:?} vs {:?} synthesized IRs too similar (rel L2={rel:.4})",
            archetypes[i],
            archetypes[j]
        );
    }
}

#[test]
fn radiator_bank_reconfigure_from_preset_bytes_switches_body_and_rejects_invalid() {
    let sr = 48_000.0;
    let mut bank = RadiatorBank::new_concert_grand(sr);
    let orig_modes = bank.preset().modes.len();
    assert!(orig_modes > 1_000);

    let mut alt = test_preset(64);
    for g in &mut alt.direct_zone_gain {
        *g *= 1.7;
    }
    let alt_bytes = alt.encode();

    assert_eq!(
        bank.reconfigure_from_preset_bytes(&[0_u8; 8], ListenerPerspective::Player, LidState::Open),
        Err(PresetDecodeError::InvalidMagic)
    );
    assert_eq!(
        bank.reconfigure_from_preset_bytes(&[0_u8; 2], ListenerPerspective::Player, LidState::Open),
        Err(PresetDecodeError::TruncatedHeader)
    );
    assert_eq!(bank.preset().modes.len(), orig_modes);

    bank.reconfigure_from_preset_bytes(
        &alt_bytes,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    )
    .expect("valid alt preset");
    assert_eq!(bank.preset().modes.len(), 64);
    assert_eq!(bank.perspective(), ListenerPerspective::ConcertHall);
    assert_eq!(bank.lid_state(), LidState::Half);

    let frames = [[0.15_f32; PRESET_RADIATOR_ZONES]; 128];
    let (mut l_alt, mut r_alt) = ([0.0_f32; 128], [0.0_f32; 128]);
    bank.process_block(&frames, &mut l_alt, &mut r_alt);

    let mut ref_alt = RadiatorBank::from_preset_bytes_with_space(
        &alt_bytes,
        sr,
        ListenerPerspective::ConcertHall,
        LidState::Half,
    )
    .expect("valid alt");
    let (mut l_ref, mut r_ref) = ([0.0_f32; 128], [0.0_f32; 128]);
    ref_alt.process_block(&frames, &mut l_ref, &mut r_ref);
    assert_eq!(l_alt, l_ref);
    assert_eq!(r_alt, r_ref);
}

#[test]
fn space_prep_context_multi_body_blob_matches_radiator_bank_bit_exact() {
    let sr = 48_000.0;
    let mut alt = test_preset(96);
    for m in &mut alt.modes {
        m.freq_hz *= 1.12;
    }
    let alt_bytes = alt.encode();

    let mut prep = SpacePrepContext::new_from_preset_bytes(CONCERT_GRAND_PRESET_BYTES, sr)
        .expect("concert grand prep");
    assert!(prep.preset().modes.len() > 1_000);
    assert_eq!(
        prep.set_body_preset_bytes(&[1, 2, 3]),
        Err(PresetDecodeError::TruncatedHeader)
    );
    prep.set_body_preset_bytes(&alt_bytes)
        .expect("alt preset bytes");
    assert_eq!(prep.preset().modes.len(), 96);

    let blob = prep.prepare_space(ListenerPerspective::BinauralHrtf, LidState::Closed);
    assert_eq!(blob.len(), space_blob_len_for_sample_rate(sr));

    let mut fast_bank = RadiatorBank::new_concert_grand(sr);
    assert!(fast_bank.install_precomputed_space(
        ListenerPerspective::BinauralHrtf,
        LidState::Closed,
        blob
    ));

    let mut ref_bank = RadiatorBank::from_preset_bytes_with_space(
        &alt_bytes,
        sr,
        ListenerPerspective::BinauralHrtf,
        LidState::Closed,
    )
    .expect("valid alt");

    let frames = [[0.20_f32; PRESET_RADIATOR_ZONES]; 128];
    let (mut fl, mut fr, mut rl, mut rr) = (
        [0.0_f32; 128],
        [0.0_f32; 128],
        [0.0_f32; 128],
        [0.0_f32; 128],
    );
    fast_bank.process_block(&frames, &mut fl, &mut fr);
    ref_bank.process_block(&frames, &mut rl, &mut rr);
    assert_eq!(fl, rl);
    assert_eq!(fr, rr);
}

#[test]
fn velvet_wood_tail_is_uncorrelated_across_zones_and_channels() {
    let sr = 48_000.0;
    let mut preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES).expect("valid preset");
    // Isolate velvet tail by zeroing modal amplitudes while keeping non-empty mode count.
    for m in &mut preset.modes {
        m.zone_drive = [0.0; PRESET_RADIATOR_ZONES];
    }
    preset.direct_zone_gain = [0.0; PRESET_RADIATOR_ZONES];

    let irs = RadiatorBank::synthesize_ir_for_test(
        &preset,
        sr,
        ListenerPerspective::Player,
        LidState::Open,
    );
    let start = (0.015 * sr) as usize;
    let end = (0.45 * sr) as usize;

    let corr = |a: &[f32], b: &[f32]| -> f64 {
        let mut ab = 0.0_f64;
        let mut aa = 0.0_f64;
        let mut bb = 0.0_f64;
        for (&x, &y) in a[start..end].iter().zip(b[start..end].iter()) {
            let fx = f64::from(x);
            let fy = f64::from(y);
            ab += fx * fy;
            aa += fx * fx;
            bb += fy * fy;
        }
        ab / (aa * bb).sqrt().max(1.0e-30)
    };

    let e_l: f64 = irs[2][0][start..end]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum();
    assert!(e_l > 1.0e-7, "expected non-zero velvet tail, got {e_l}");
    assert_eq!(
        irs[2][0][..100]
            .iter()
            .fold(0.0_f32, |m, &x| m.max(x.abs())),
        0.0,
        "velvet tail must not alter earliest 2 ms direct onset"
    );

    let r_lr = corr(&irs[2][0], &irs[2][1]);
    let r_zone = corr(&irs[2][0], &irs[3][0]);
    assert!(
        r_lr.abs() < 0.20,
        "L/R velvet tail should be uncorrelated, got r={r_lr:.3}"
    );
    assert!(
        r_zone.abs() < 0.20,
        "cross-zone velvet tail should be uncorrelated, got r={r_zone:.3}"
    );

    // Check highpass attenuation below 2 kHz vs above 4 kHz via simple DFT bin energy.
    let fft_len = 16_384_usize;
    let mut padded = vec![0.0_f32; fft_len];
    padded[..irs[2][0].len().min(fft_len)]
        .copy_from_slice(&irs[2][0][..irs[2][0].len().min(fft_len)]);
    let mut fft = piano_simd::RealFft::new(fft_len);
    let spec_len = fft.spectrum_len();
    let (mut re, mut im) = (vec![0.0_f32; spec_len], vec![0.0_f32; spec_len]);
    fft.forward_split(&padded, &mut re, &mut im);
    let band_pwr = |f0: f64, f1: f64| -> f64 {
        let k0 = (f0 * fft_len as f64 / sr).ceil() as usize;
        let k1 = (f1 * fft_len as f64 / sr).floor() as usize;
        let sum: f64 = (k0..=k1)
            .map(|k| f64::from(re[k]).powi(2) + f64::from(im[k]).powi(2))
            .sum();
        sum / ((k1 - k0 + 1) as f64)
    };
    let p_low = band_pwr(200.0, 2_000.0);
    let p_high = band_pwr(4_000.0, 12_000.0);
    let ratio_db = 10.0 * (p_high / p_low.max(1.0e-30)).log10();
    assert!(
        ratio_db > 15.0,
        "velvet tail should be >3.5 kHz high-passed (HF/LF={ratio_db:.1} dB)"
    );
}
