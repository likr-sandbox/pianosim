//! Minimal zero-dependency stereo 16-bit PCM WAV writer.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

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

/// Writes interleaved 16-bit PCM stereo RIFF/WAVE audio to `path`.
pub fn write_stereo_pcm16_wav(
    path: &Path,
    sample_rate: u32,
    left: &[f32],
    right: &[f32],
) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "stereo length mismatch: L={}, R={}",
            left.len(),
            right.len()
        ));
    }
    let frames = left.len();
    let data_bytes = frames
        .checked_mul(4)
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

    let byte_rate = sample_rate.saturating_mul(4);
    let block_align: u16 = 4;
    let bits_per_sample: u16 = 16;

    let write_err = |e: std::io::Error| format!("failed writing {}: {e}", path.display());
    w.write_all(b"RIFF").map_err(write_err)?;
    w.write_all(&riff_size.to_le_bytes()).map_err(write_err)?;
    w.write_all(b"WAVE").map_err(write_err)?;
    w.write_all(b"fmt ").map_err(write_err)?;
    w.write_all(&16_u32.to_le_bytes()).map_err(write_err)?;
    w.write_all(&1_u16.to_le_bytes()).map_err(write_err)?;
    w.write_all(&2_u16.to_le_bytes()).map_err(write_err)?;
    w.write_all(&sample_rate.to_le_bytes()).map_err(write_err)?;
    w.write_all(&byte_rate.to_le_bytes()).map_err(write_err)?;
    w.write_all(&block_align.to_le_bytes()).map_err(write_err)?;
    w.write_all(&bits_per_sample.to_le_bytes())
        .map_err(write_err)?;
    w.write_all(b"data").map_err(write_err)?;
    w.write_all(&data_bytes.to_le_bytes()).map_err(write_err)?;

    for i in 0..frames {
        let sl = float_to_i16(left[i]);
        let sr = float_to_i16(right[i]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn writes_valid_riff_wave_header_and_payload() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let out = std::env::temp_dir().join(format!("pianosim_wav_test_{nanos}.wav"));
        let left = [0.0_f32, 0.5, -0.5, 1.0];
        let right = [0.0_f32, -0.25, 0.25, -1.0];

        write_stereo_pcm16_wav(&out, 48_000, &left, &right).expect("write wav");
        let bytes = fs::read(&out).expect("read wav");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(bytes.len(), 44 + 4 * 4);
        let data_len = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_len, 16);

        let _ = fs::remove_file(&out);
    }
}
