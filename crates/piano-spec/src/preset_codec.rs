//! Compact binary codec for baked concert-grand structural-acoustic body presets.
//!
//! Stores reduced-bridge cancellation poles, 9-zone drive projections (8 bridge
//! zones + 1 keybed impact zone), 5-component directional/lid radiation residues,
//! and an IEEE CRC-32 integrity footer in little-endian layout under `no_std + alloc`.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// 4-byte file header signature (`"PBDY"`).
pub const PRESET_MAGIC: [u8; 4] = *b"PBDY";

/// Current binary schema version for [`BodyPreset`] serialization.
pub const PRESET_VERSION: u32 = 1;

/// Number of bidirectional bridge zones coupled to Stage 3.
pub const PRESET_BRIDGE_ZONES: usize = 8;

/// Number of Stage 4 radiation input zones (8 bridge zones + Zone 8 keybed).
pub const PRESET_RADIATOR_ZONES: usize = 9;

/// Number of reduced bridge poles exported by Stage 3 for pole cancellation.
pub const PRESET_REDUCED_POLES: usize = 10;

/// Number of modal radiation residue components per body mode:
/// `[monopole, dipole_x, dipole_y, dipole_z, lid_coupling]`.
pub const PRESET_RAD_COEFFS: usize = 5;

const HEADER_PREFIX_BYTES: usize = 12;
const POLE_BYTES: usize = (3 + PRESET_BRIDGE_ZONES) * 4;
const HEADER_LEN: usize = HEADER_PREFIX_BYTES
    + (PRESET_BRIDGE_ZONES * 4)
    + (PRESET_RADIATOR_ZONES * 4 * 4)
    + (PRESET_REDUCED_POLES * POLE_BYTES);
const MODE_RECORD_BYTES: usize = 16 + (PRESET_RADIATOR_ZONES * 2) + (PRESET_RAD_COEFFS * 2) + 1;
const CRC_BYTES: usize = 4;

/// Virtual microphone / listening perspective for Stage 4 IR synthesis.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ListenerPerspective {
    /// Seated pianist perspective at keyboard position.
    #[default]
    Player = 0,
    /// Close stereo microphone pair near rim and soundboard belly.
    CloseStereo = 1,
    /// Concert hall audience/stage perspective with room/lid balance.
    ConcertHall = 2,
    /// Headphone binaural HRTF rendering at intimate listening geometry.
    BinauralHrtf = 3,
}

impl ListenerPerspective {
    /// Decodes a raw byte discriminant into a [`ListenerPerspective`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Player),
            1 => Some(Self::CloseStereo),
            2 => Some(Self::ConcertHall),
            3 => Some(Self::BinauralHrtf),
            _ => None,
        }
    }
}

/// Grand piano lid opening state controlling acoustic cavity/reflection projection.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum LidState {
    /// Lid fully closed.
    Closed = 0,
    /// Lid propped on the half-stick.
    Half = 1,
    /// Lid propped on the full concert stick.
    #[default]
    Open = 2,
}

impl LidState {
    /// Decodes a raw byte discriminant into a [`LidState`].
    #[must_use]
    pub const fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Closed),
            1 => Some(Self::Half),
            2 => Some(Self::Open),
            _ => None,
        }
    }

    /// Maps the discrete lid state onto normalized openness (`0.0 / 0.5 / 1.0`).
    #[must_use]
    pub const fn openness_f32(self) -> f32 {
        match self {
            Self::Closed => 0.0,
            Self::Half => 0.5,
            Self::Open => 1.0,
        }
    }
}

/// Reduced soundboard pole stored in a preset for pole-cancellation matching.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct PresetReducedPole {
    /// Modal natural frequency in Hz.
    pub freq_hz: f32,
    /// Modal viscous damping ratio `zeta`.
    pub zeta: f32,
    /// Effective modal mass in kilograms.
    pub modal_mass_kg: f32,
    /// Dimensionless zone coupling gains across the 8 bridge zones.
    pub zone_gains: [f32; PRESET_BRIDGE_ZONES],
}

/// Single structural-acoustic body mode record (orthotropic plate / rim / air cavity).
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct BodyModeRecord {
    /// Modal frequency in Hz.
    pub freq_hz: f32,
    /// Viscous damping ratio `zeta`.
    pub zeta: f32,
    /// Modal drive admittance weights across 9 radiator zones (`0..7` bridge, `8` keybed).
    pub zone_drive: [f32; PRESET_RADIATOR_ZONES],
    /// Directional radiation residue `[monopole, dipole_x, dipole_y, dipole_z, lid_coupling]`.
    pub rad_coeffs: [f32; PRESET_RAD_COEFFS],
    /// Classification flag (`0` = structural plate/rim/frame, `1` = air cavity coupled).
    pub cavity_flag: u8,
}

/// Complete serialized structural-acoustic piano body preset.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BodyPreset {
    /// High-frequency resistive bridge mobility `[m/(N*s)]` across 8 bridge zones.
    pub y_inf: [f32; PRESET_BRIDGE_ZONES],
    /// Direct feedthrough weights across 9 radiator zones (`0..7` bridge, `8` keybed).
    pub direct_zone_gain: [f32; PRESET_RADIATOR_ZONES],
    /// 3D physical zone centers `[x, y, z]` in meters relative to piano coordinates.
    pub zone_pos_m: [[f32; 3]; PRESET_RADIATOR_ZONES],
    /// Reduced Stage 3 poles cancelled out of the structural-acoustic sum.
    pub reduced_poles: [PresetReducedPole; PRESET_REDUCED_POLES],
    /// Uncancelled structural-acoustic and cavity body modes.
    pub modes: Vec<BodyModeRecord>,
}

/// Failure reasons returned when decoding an encoded [`BodyPreset`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PresetDecodeError {
    /// Input buffer is shorter than the fixed header plus CRC-32 footer.
    TruncatedHeader,
    /// Header signature bytes do not match [`PRESET_MAGIC`].
    InvalidMagic,
    /// Schema version is unsupported.
    UnsupportedVersion(u32),
    /// Encoded mode count overflows address calculations.
    InvalidModeCount,
    /// Input buffer ends before all declared mode records and CRC footer finish.
    TruncatedBody,
    /// Input buffer contains unexpected trailing bytes beyond the CRC footer.
    TrailingBytes,
    /// IEEE CRC-32 checksum verification failed.
    ChecksumMismatch { expected: u32, actual: u32 },
    /// A decoded floating-point scalar or vector element is NaN or infinite.
    NonFiniteData,
}

impl fmt::Display for PresetDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TruncatedHeader => f.write_str("truncated body preset header"),
            Self::InvalidMagic => f.write_str("invalid body preset magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported body preset version {v}"),
            Self::InvalidModeCount => f.write_str("invalid body preset mode count"),
            Self::TruncatedBody => f.write_str("truncated body preset payload"),
            Self::TrailingBytes => f.write_str("unexpected trailing bytes in body preset"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "CRC32 mismatch: expected {expected:#x}, got {actual:#x}")
            }
            Self::NonFiniteData => f.write_str("non-finite float in body preset"),
        }
    }
}

/// Computes standard IEEE 802.3 CRC-32 over `bytes`.
#[must_use]
pub fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[inline]
fn push_f32_le(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&(if v.is_finite() { v } else { 0.0 }).to_le_bytes());
}

#[inline]
fn push_f32_slice(out: &mut Vec<u8>, vals: &[f32]) {
    for &v in vals {
        push_f32_le(out, v);
    }
}

fn push_quantized_i16<const N: usize>(out: &mut Vec<u8>, vals: &[f32; N]) {
    let max_abs = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if max_abs <= 0.0 || !max_abs.is_finite() {
        push_f32_le(out, 0.0);
        out.resize(out.len() + N * 2, 0);
        return;
    }
    let (scale, inv) = (max_abs / 32_767.0, 32_767.0 / max_abs);
    push_f32_le(out, scale);
    for &src in vals {
        let s = (if src.is_finite() { src } else { 0.0 } * inv).clamp(-32_767.0, 32_767.0);
        let r = if s >= 0.0 { s + 0.5 } else { s - 0.5 } as i32;
        out.extend_from_slice(&(r.clamp(-32_767, 32_767) as i16).to_le_bytes());
    }
}

#[inline]
fn take_bytes<const K: usize>(cursor: &mut &[u8]) -> Result<[u8; K], PresetDecodeError> {
    if cursor.len() < K {
        return Err(PresetDecodeError::TruncatedBody);
    }
    let (head, rest) = cursor.split_at(K);
    *cursor = rest;
    let mut buf = [0_u8; K];
    buf.copy_from_slice(head);
    Ok(buf)
}

#[inline]
fn read_f32_le(cursor: &mut &[u8]) -> Result<f32, PresetDecodeError> {
    let val = f32::from_le_bytes(take_bytes::<4>(cursor)?);
    if val.is_finite() {
        Ok(val)
    } else {
        Err(PresetDecodeError::NonFiniteData)
    }
}

fn read_f32_array<const N: usize>(cursor: &mut &[u8]) -> Result<[f32; N], PresetDecodeError> {
    let mut out = [0.0_f32; N];
    for slot in &mut out {
        *slot = read_f32_le(cursor)?;
    }
    Ok(out)
}

fn read_dequantized_i16<const N: usize>(cursor: &mut &[u8]) -> Result<[f32; N], PresetDecodeError> {
    let scale = read_f32_le(cursor)?;
    let mut out = [0.0_f32; N];
    for slot in &mut out {
        let q = i16::from_le_bytes(take_bytes::<2>(cursor)?);
        *slot = scale * f32::from(q);
    }
    Ok(out)
}

impl BodyPreset {
    /// Serializes this preset into a compact little-endian byte stream with quantized
    /// per-mode `i16` drive/radiation vectors and an IEEE CRC-32 footer.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mode_count = u32::try_from(self.modes.len()).unwrap_or(u32::MAX);
        let used_modes = mode_count as usize;
        let mut out = Vec::with_capacity(HEADER_LEN + used_modes * MODE_RECORD_BYTES + CRC_BYTES);

        out.extend_from_slice(&PRESET_MAGIC);
        out.extend_from_slice(&PRESET_VERSION.to_le_bytes());
        out.extend_from_slice(&mode_count.to_le_bytes());
        push_f32_slice(&mut out, &self.y_inf);
        push_f32_slice(&mut out, &self.direct_zone_gain);
        for pos in &self.zone_pos_m {
            push_f32_slice(&mut out, pos);
        }
        for pole in &self.reduced_poles {
            push_f32_le(&mut out, pole.freq_hz);
            push_f32_le(&mut out, pole.zeta);
            push_f32_le(&mut out, pole.modal_mass_kg);
            push_f32_slice(&mut out, &pole.zone_gains);
        }
        for mode in self.modes.iter().take(used_modes) {
            push_f32_le(&mut out, mode.freq_hz);
            push_f32_le(&mut out, mode.zeta);
            push_quantized_i16(&mut out, &mode.zone_drive);
            push_quantized_i16(&mut out, &mode.rad_coeffs);
            out.push(mode.cavity_flag);
        }

        let crc = crc32_ieee(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Decodes a [`BodyPreset`] from binary bytes produced by [`Self::encode`].
    ///
    /// # Errors
    /// Returns a [`PresetDecodeError`] if the signature, version, byte length,
    /// CRC-32 checksum, or floating-point values are invalid.
    pub fn decode(bytes: &[u8]) -> Result<Self, PresetDecodeError> {
        let le_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
        };
        if bytes.len() < 4 {
            return Err(PresetDecodeError::TruncatedHeader);
        }
        if bytes[0..4] != PRESET_MAGIC {
            return Err(PresetDecodeError::InvalidMagic);
        }
        if bytes.len() < 8 {
            return Err(PresetDecodeError::TruncatedHeader);
        }
        let version = le_u32(4);
        if version != PRESET_VERSION {
            return Err(PresetDecodeError::UnsupportedVersion(version));
        }
        if bytes.len() < HEADER_LEN + CRC_BYTES {
            return Err(PresetDecodeError::TruncatedHeader);
        }

        let mode_count = le_u32(8) as usize;
        let expected_body = mode_count
            .checked_mul(MODE_RECORD_BYTES)
            .ok_or(PresetDecodeError::InvalidModeCount)?;
        let actual_body = bytes.len() - HEADER_LEN - CRC_BYTES;
        if actual_body != expected_body {
            return Err(if actual_body < expected_body {
                PresetDecodeError::TruncatedBody
            } else {
                PresetDecodeError::TrailingBytes
            });
        }

        let payload_end = bytes.len() - CRC_BYTES;
        let (expected_crc, actual_crc) = (crc32_ieee(&bytes[..payload_end]), le_u32(payload_end));
        if expected_crc != actual_crc {
            return Err(PresetDecodeError::ChecksumMismatch {
                expected: expected_crc,
                actual: actual_crc,
            });
        }

        let mut cursor = &bytes[HEADER_PREFIX_BYTES..payload_end];
        let y_inf = read_f32_array(&mut cursor)?;
        let direct_zone_gain = read_f32_array(&mut cursor)?;
        let mut zone_pos_m = [[0.0_f32; 3]; PRESET_RADIATOR_ZONES];
        for pos in &mut zone_pos_m {
            *pos = read_f32_array(&mut cursor)?;
        }
        let mut reduced_poles = [PresetReducedPole::default(); PRESET_REDUCED_POLES];
        for pole in &mut reduced_poles {
            pole.freq_hz = read_f32_le(&mut cursor)?;
            pole.zeta = read_f32_le(&mut cursor)?;
            pole.modal_mass_kg = read_f32_le(&mut cursor)?;
            pole.zone_gains = read_f32_array(&mut cursor)?;
        }
        let mut modes = vec![BodyModeRecord::default(); mode_count];
        for mode in &mut modes {
            mode.freq_hz = read_f32_le(&mut cursor)?;
            mode.zeta = read_f32_le(&mut cursor)?;
            mode.zone_drive = read_dequantized_i16(&mut cursor)?;
            mode.rad_coeffs = read_dequantized_i16(&mut cursor)?;
            mode.cavity_flag = take_bytes::<1>(&mut cursor)?[0];
        }
        Ok(Self {
            y_inf,
            direct_zone_gain,
            zone_pos_m,
            reduced_poles,
            modes,
        })
    }
}

#[cfg(test)]
mod tests {
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
}
