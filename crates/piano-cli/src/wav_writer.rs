//! Minimal zero-dependency stereo WAV writer (16-bit PCM, 24-bit TPDF PCM, 32-bit IEEE float).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const TPDF_SEED: u64 = 0x5049_414E_4F32_3442;
const PCM24_MAX: f64 = 8_388_607.0;
const PCM24_MIN: f64 = -8_388_608.0;

/// Deterministic SplitMix64 PRNG for reproducible TPDF dither generation.
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    #[inline]
    fn next_unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Triangular PDF dither sample in `(-1.0, +1.0)` (subset of `[-1.0, +1.0)`) LSBs.
    #[inline]
    fn next_tpdf_lsb(&mut self) -> f64 {
        self.next_unit_f64() - self.next_unit_f64()
    }
}

/// Signal integrity summary computed from rendered floating-point stereo audio.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AudioMetrics {
    pub frames: usize,
    pub peak_l: f32,
    pub peak_r: f32,
    pub rms_mono: f32,
    pub non_finite_count: usize,
}

/// Computes peak, mono RMS, and non-finite sample counts for stereo buffers.
#[must_use]
pub fn analyze_stereo(left: &[f32], right: &[f32]) -> AudioMetrics {
    let frames = left.len().min(right.len());
    let mut peak_l = 0.0_f32;
    let mut peak_r = 0.0_f32;
    let mut sum_sq = 0.0_f64;
    let mut non_finite_count = 0_usize;

    for i in 0..frames {
        let l = left[i];
        let r = right[i];
        if !l.is_finite() || !r.is_finite() {
            non_finite_count += 1;
            continue;
        }
        peak_l = peak_l.max(l.abs());
        peak_r = peak_r.max(r.abs());
        let mono = 0.5 * (f64::from(l) + f64::from(r));
        sum_sq += mono * mono;
    }

    let rms_mono = if frames > 0 {
        (sum_sq / (frames as f64)).sqrt() as f32
    } else {
        0.0
    };

    AudioMetrics {
        frames,
        peak_l,
        peak_r,
        rms_mono,
        non_finite_count,
    }
}

fn open_wav_writer(
    path: &Path,
    sample_rate: u32,
    frames: usize,
    audio_format: u16,
    bits_per_sample: u16,
) -> Result<BufWriter<File>, String> {
    let block_align = 2_u16 * (bits_per_sample / 8);
    let data_bytes = frames
        .checked_mul(usize::from(block_align))
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| String::from("WAV payload exceeds 4 GiB RIFF limit"))?;
    let riff_size = data_bytes
        .checked_add(36)
        .ok_or_else(|| String::from("WAV header size overflow"))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
    }

    let file =
        File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    let mut w = BufWriter::new(file);
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));

    let write_err = |e: std::io::Error| format!("failed writing {}: {e}", path.display());
    w.write_all(b"RIFF").map_err(write_err)?;
    w.write_all(&riff_size.to_le_bytes()).map_err(write_err)?;
    w.write_all(b"WAVE").map_err(write_err)?;
    w.write_all(b"fmt ").map_err(write_err)?;
    w.write_all(&16_u32.to_le_bytes()).map_err(write_err)?;
    w.write_all(&audio_format.to_le_bytes())
        .map_err(write_err)?;
    w.write_all(&2_u16.to_le_bytes()).map_err(write_err)?;
    w.write_all(&sample_rate.to_le_bytes()).map_err(write_err)?;
    w.write_all(&byte_rate.to_le_bytes()).map_err(write_err)?;
    w.write_all(&block_align.to_le_bytes()).map_err(write_err)?;
    w.write_all(&bits_per_sample.to_le_bytes())
        .map_err(write_err)?;
    w.write_all(b"data").map_err(write_err)?;
    w.write_all(&data_bytes.to_le_bytes()).map_err(write_err)?;

    Ok(w)
}

fn validate_stereo(left: &[f32], right: &[f32]) -> Result<usize, String> {
    if left.len() != right.len() {
        return Err(format!(
            "stereo length mismatch: L={}, R={}",
            left.len(),
            right.len()
        ));
    }
    Ok(left.len())
}

/// Writes interleaved 16-bit PCM stereo RIFF/WAVE audio to `path`.
pub fn write_stereo_pcm16_wav(
    path: &Path,
    sample_rate: u32,
    left: &[f32],
    right: &[f32],
) -> Result<(), String> {
    let frames = validate_stereo(left, right)?;
    let mut w = open_wav_writer(path, sample_rate, frames, WAVE_FORMAT_PCM, 16)?;
    let write_err = |e: std::io::Error| format!("failed writing {}: {e}", path.display());

    for i in 0..frames {
        let sl = float_to_i16(left[i]);
        let sr = float_to_i16(right[i]);
        w.write_all(&sl.to_le_bytes()).map_err(write_err)?;
        w.write_all(&sr.to_le_bytes()).map_err(write_err)?;
    }

    w.flush().map_err(write_err)?;
    Ok(())
}

/// Writes interleaved 24-bit PCM stereo RIFF/WAVE audio with deterministic TPDF dither to `path`.
pub fn write_stereo_pcm24_tpdf_wav(
    path: &Path,
    sample_rate: u32,
    left: &[f32],
    right: &[f32],
) -> Result<(), String> {
    let frames = validate_stereo(left, right)?;
    let mut w = open_wav_writer(path, sample_rate, frames, WAVE_FORMAT_PCM, 24)?;
    let write_err = |e: std::io::Error| format!("failed writing {}: {e}", path.display());
    let mut rng = SplitMix64(TPDF_SEED);

    for i in 0..frames {
        let dl = rng.next_tpdf_lsb();
        let dr = rng.next_tpdf_lsb();
        let sl = float_to_i24_bytes(left[i], dl);
        let sr = float_to_i24_bytes(right[i], dr);
        w.write_all(&sl).map_err(write_err)?;
        w.write_all(&sr).map_err(write_err)?;
    }

    w.flush().map_err(write_err)?;
    Ok(())
}

/// Writes interleaved 32-bit IEEE float stereo RIFF/WAVE audio to `path`.
pub fn write_stereo_f32_wav(
    path: &Path,
    sample_rate: u32,
    left: &[f32],
    right: &[f32],
) -> Result<(), String> {
    let frames = validate_stereo(left, right)?;
    let mut w = open_wav_writer(path, sample_rate, frames, WAVE_FORMAT_IEEE_FLOAT, 32)?;
    let write_err = |e: std::io::Error| format!("failed writing {}: {e}", path.display());

    for i in 0..frames {
        let sl = sanitize_f32(left[i]);
        let sr = sanitize_f32(right[i]);
        w.write_all(&sl.to_le_bytes()).map_err(write_err)?;
        w.write_all(&sr.to_le_bytes()).map_err(write_err)?;
    }

    w.flush().map_err(write_err)?;
    Ok(())
}

#[inline]
fn float_to_i16(sample: f32) -> i16 {
    let clamped = if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (clamped * 32_767.0).round() as i16
}

#[inline]
fn float_to_i24_bytes(sample: f32, dither_lsb: f64) -> [u8; 3] {
    let clamped = if sample.is_finite() {
        f64::from(sample.clamp(-1.0, 1.0))
    } else {
        0.0
    };
    let q = (clamped * PCM24_MAX + dither_lsb)
        .round()
        .clamp(PCM24_MIN, PCM24_MAX) as i32;
    let le = q.to_le_bytes();
    [le[0], le[1], le[2]]
}

#[inline]
fn sanitize_f32(sample: f32) -> f32 {
    if sample.is_finite() {
        sample
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_wav(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "pianosim_{label}_{}_{nanos}.wav",
            std::process::id()
        ))
    }

    fn unpack_i24(bytes: &[u8]) -> i32 {
        let raw = i32::from(bytes[0]) | (i32::from(bytes[1]) << 8) | (i32::from(bytes[2]) << 16);
        (raw << 8) >> 8
    }

    #[test]
    fn writes_valid_riff_wave_header_and_payload() {
        let out = temp_wav("pcm16");
        let left = [0.0_f32, 0.5, -0.5, 1.0];
        let right = [0.0_f32, -0.25, 0.25, -1.0];

        write_stereo_pcm16_wav(&out, 48_000, &left, &right).expect("write wav");
        let bytes = fs::read(&out).expect("read wav");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[32..34].try_into().unwrap()), 4);
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(bytes.len(), 44 + 4 * 4);
        let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_len, 16);

        let _ = fs::remove_file(&out);
    }

    #[test]
    fn writes_pcm24_tpdf_with_valid_header_bounded_noise_and_zero_mean() {
        let out_a = temp_wav("pcm24_a");
        let out_b = temp_wav("pcm24_b");
        let frames = 8_192_usize;
        let zeros = vec![0.0_f32; frames];

        write_stereo_pcm24_tpdf_wav(&out_a, 48_000, &zeros, &zeros).expect("write pcm24 a");
        write_stereo_pcm24_tpdf_wav(&out_b, 48_000, &zeros, &zeros).expect("write pcm24 b");
        let bytes = fs::read(&out_a).expect("read pcm24 a");
        let bytes_b = fs::read(&out_b).expect("read pcm24 b");
        assert_eq!(bytes, bytes_b, "TPDF output must be deterministic");

        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            48_000
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            48_000 * 6
        );
        assert_eq!(u16::from_le_bytes(bytes[32..34].try_into().unwrap()), 6);
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 24);
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
            (frames * 6) as u32
        );
        assert_eq!(bytes.len(), 44 + frames * 6);

        let mut sum_l = 0_i64;
        let mut sum_r = 0_i64;
        let mut nonzero = 0_usize;
        for f in 0..frames {
            let base = 44 + f * 6;
            let ql = unpack_i24(&bytes[base..base + 3]);
            let qr = unpack_i24(&bytes[base + 3..base + 6]);
            assert!(
                (-1..=1).contains(&ql),
                "TPDF noise out of [-1, 1] LSB: {ql}"
            );
            assert!(
                (-1..=1).contains(&qr),
                "TPDF noise out of [-1, 1] LSB: {qr}"
            );
            sum_l += i64::from(ql);
            sum_r += i64::from(qr);
            if ql != 0 || qr != 0 {
                nonzero += 1;
            }
        }
        let mean_l = (sum_l as f64) / (frames as f64);
        let mean_r = (sum_r as f64) / (frames as f64);
        assert!(mean_l.abs() < 0.03, "non-zero TPDF mean L: {mean_l}");
        assert!(mean_r.abs() < 0.03, "non-zero TPDF mean R: {mean_r}");
        assert!(nonzero > 100, "TPDF dither should be active");

        // Verify extreme clamping [-1.0, +1.0] packs exact 24-bit signed limits.
        let out_ext = temp_wav("pcm24_ext");
        write_stereo_pcm24_tpdf_wav(&out_ext, 48_000, &[1.5, -1.5], &[-1.0, 1.0]).expect("ext");
        let ext = fs::read(&out_ext).expect("read ext");
        let q0l = unpack_i24(&ext[44..47]);
        let q1l = unpack_i24(&ext[50..53]);
        assert!(q0l <= 8_388_607 && q0l >= 8_388_606);
        assert!((-8_388_608..=-8_388_606).contains(&q1l));

        let _ = fs::remove_file(&out_a);
        let _ = fs::remove_file(&out_b);
        let _ = fs::remove_file(&out_ext);
    }

    #[test]
    fn writes_f32_ieee_float_wav_with_exact_sample_fidelity() {
        let out = temp_wav("f32");
        let left = [0.0_f32, 0.123_456_79, -0.987_654_3, f32::NAN, 1.25];
        let right = [-0.5_f32, 0.25, f32::INFINITY, f32::NEG_INFINITY, -1.5];

        write_stereo_f32_wav(&out, 96_000, &left, &right).expect("write f32");
        let bytes = fs::read(&out).expect("read f32");

        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 3);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            96_000
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            96_000 * 8
        );
        assert_eq!(u16::from_le_bytes(bytes[32..34].try_into().unwrap()), 8);
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 32);
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
            (left.len() * 8) as u32
        );
        assert_eq!(bytes.len(), 44 + left.len() * 8);

        for i in 0..left.len() {
            let base = 44 + i * 8;
            let got_l = f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let got_r = f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
            let exp_l = if left[i].is_finite() { left[i] } else { 0.0 };
            let exp_r = if right[i].is_finite() { right[i] } else { 0.0 };
            assert_eq!(got_l.to_bits(), exp_l.to_bits(), "L mismatch at {i}");
            assert_eq!(got_r.to_bits(), exp_r.to_bits(), "R mismatch at {i}");
        }

        let _ = fs::remove_file(&out);
    }
}
