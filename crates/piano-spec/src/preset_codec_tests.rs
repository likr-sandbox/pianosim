use super::{ListenerPerspective as LP, *};

fn sample_preset(num_modes: usize) -> BodyPreset {
    BodyPreset {
        y_inf: core::array::from_fn(|z| 1.4e-4 + 0.2e-4 * (z as f32)),
        direct_zone_gain: [0.05, 0.08, 0.12, 0.14, 0.13, 0.10, 0.08, 0.06, 0.22],
        zone_pos_m: core::array::from_fn(|z| {
            [-0.45 + 0.12 * (z as f32), 0.15 + 0.08 * (z as f32), 0.02]
        }),
        reduced_poles: core::array::from_fn(|p| PresetReducedPole {
            freq_hz: 54.0 + 110.0 * (p as f32),
            zeta: 0.02 + 0.003 * (p as f32),
            modal_mass_kg: 0.5 + 0.3 * (p as f32),
            zone_gains: core::array::from_fn(|z| if (p + z) % 2 == 0 { 0.25 } else { -0.18 }),
        }),
        modes: (0..num_modes)
            .map(|m| BodyModeRecord {
                freq_hz: 62.0 + 3.8 * (m as f32),
                zeta: 0.004 + 0.00001 * (m as f32),
                zone_drive: core::array::from_fn(|z| {
                    let sign = if (m + z) % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (0.002 + 0.0007 * ((m % 17) as f32) + 0.0003 * (z as f32))
                }),
                rad_coeffs: core::array::from_fn(|r| {
                    let base = if r == 1 { -0.04 } else { 0.03 };
                    base + 0.002 * (((m + r) % 13) as f32)
                }),
                cavity_flag: u8::from(m % 19 == 0),
            })
            .collect(),
    }
}

#[test]
fn perspective_lid_and_crc32_vectors_are_exact() {
    assert_eq!(LP::default(), LP::Player);
    assert_eq!(LP::from_u8(0), Some(LP::Player));
    assert_eq!(LP::from_u8(1), Some(LP::CloseStereo));
    assert_eq!(LP::from_u8(2), Some(LP::ConcertHall));
    assert_eq!(LP::from_u8(3), Some(LP::BinauralHrtf));
    assert_eq!(LP::from_u8(4), None);

    assert_eq!(LidState::default(), LidState::Open);
    for (i, &open) in [0.0_f32, 0.5, 1.0].iter().enumerate() {
        assert_eq!(LidState::from_u8(i as u8).unwrap().openness_f32(), open);
    }
    assert_eq!(LidState::from_u8(3), None);
    assert_eq!(crc32_ieee(b"123456789"), 0xcbf4_3926);
}

fn assert_quantized_close(src: &[f32], dst: &[f32]) {
    let max_v = src.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
    let tol = (5.0e-5 * max_v).max(1e-7);
    for (&s, &d) in src.iter().zip(dst.iter()) {
        assert!((s - d).abs() <= tol, "src={s} dst={d} tol={tol}");
    }
}

#[test]
fn roundtrip_accuracy_and_1500_mode_size_budget() {
    let mut preset = sample_preset(1_500);
    preset.modes[0].zone_drive = [0.0; PRESET_RADIATOR_ZONES];
    preset.modes[0].rad_coeffs = [0.0; PRESET_RAD_COEFFS];

    let encoded = preset.encode();
    assert!(encoded.len() < 100 * 1024 && encoded.len() < 500 * 1024);

    let dec = BodyPreset::decode(&encoded).expect("valid preset must decode");
    assert_eq!(dec.y_inf, preset.y_inf);
    assert_eq!(dec.direct_zone_gain, preset.direct_zone_gain);
    assert_eq!(dec.zone_pos_m, preset.zone_pos_m);
    assert_eq!(dec.reduced_poles, preset.reduced_poles);
    assert_eq!(dec.modes.len(), preset.modes.len());
    for (src, dst) in preset.modes.iter().zip(dec.modes.iter()) {
        assert_eq!(src.freq_hz, dst.freq_hz);
        assert_eq!(src.zeta, dst.zeta);
        assert_eq!(src.cavity_flag, dst.cavity_flag);
        assert_quantized_close(&src.zone_drive, &dst.zone_drive);
        assert_quantized_close(&src.rad_coeffs, &dst.rad_coeffs);
    }
}

#[test]
fn decode_rejects_corruption_crc_and_truncation() {
    use PresetDecodeError as E;
    let dec = BodyPreset::decode;
    let encoded = sample_preset(12).encode();
    let with_byte = |i: usize, v: u8| {
        let mut b = encoded.clone();
        b[i] = v;
        b
    };
    assert_eq!(dec(&with_byte(0, b'X')), Err(E::InvalidMagic));
    assert_eq!(dec(&with_byte(4, 99)), Err(E::UnsupportedVersion(99)));
    assert_eq!(dec(&encoded[..16]), Err(E::TruncatedHeader));
    assert_eq!(dec(&encoded[..encoded.len() - 1]), Err(E::TruncatedBody));

    let mut trailing = encoded.clone();
    trailing.push(0);
    assert_eq!(dec(&trailing), Err(E::TrailingBytes));

    for idx in [HEADER_LEN + 2, encoded.len() - 1] {
        let flipped = with_byte(idx, encoded[idx] ^ 0x5a);
        assert!(matches!(dec(&flipped), Err(E::ChecksumMismatch { .. })));
    }
}
