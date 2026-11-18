//! `diag-buzz`: pure-Rust `3-12 kHz` non-attack buzz-burst detector across demo WAVs & MIDIs.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use piano_simd::RealFft;
use piano_spec::EventKind;

use crate::check_lines::find_repo_root;
use crate::render_cmd::SHOWCASE_DEMOS;
use crate::smf_parser::parse_smf_file;

use super::parse_flag_value;

const FFT_LEN: usize = 1_024;
const HOP: usize = 256;
const MIN_FRAME_RMS: f64 = 3.16e-4; // -70 dBFS
const RATIO_THRESHOLD_DB: f64 = -22.0;
const JUMP_LEVEL_FLOOR_DB: f64 = -65.0;
const JUMP_DELTA_DB: f64 = 8.0;
const D3_THRESHOLD: f32 = 0.005;
const MERGE_GAP_SECS: f64 = 0.025;

#[derive(Clone, Debug)]
pub struct WavAudio {
    pub sample_rate: u32,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BuzzBurst {
    pub start_s: f64,
    pub end_s: f64,
    pub max_ratio_db: f64,
    pub max_buzz_db: f64,
    pub max_d3: f32,
}

#[derive(Clone, Debug)]
pub struct BuzzReport {
    pub total_frames: usize,
    pub evaluated_frames: usize,
    pub bursts: Vec<BuzzBurst>,
    pub worst_ratio_db: f64,
    pub worst_d3: f32,
}

struct BuzzDiagOptions {
    wav_dir: PathBuf,
    demos_dir: Option<PathBuf>,
}

fn parse_options(args: &[String]) -> Result<BuzzDiagOptions, String> {
    let usage = super::diag_usage();
    let mut wav_dir: Option<PathBuf> = None;
    let mut demos_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        if let Some(v) = parse_flag_value(args, &mut i, "--demos-dir", usage)? {
            demos_dir = Some(PathBuf::from(v));
        } else if args[i].starts_with('-') {
            return Err(String::from(usage));
        } else if wav_dir.is_none() {
            wav_dir = Some(PathBuf::from(&args[i]));
            i += 1;
        } else {
            return Err(String::from(usage));
        }
    }
    Ok(BuzzDiagOptions {
        wav_dir: wav_dir.unwrap_or_else(|| PathBuf::from("demos")),
        demos_dir,
    })
}

/// Reads a 16-bit PCM, 24-bit PCM, or 32-bit IEEE float RIFF/WAVE file into stereo `f32` buffers.
pub fn read_wav_stereo(path: &Path) -> Result<WavAudio, String> {
    let bytes =
        fs::read(path).map_err(|e| format!("failed to read WAV '{}': {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("invalid RIFF/WAVE header in '{}'", path.display()));
    }

    let mut pos = 12_usize;
    let mut audio_format = 0_u16;
    let mut channels = 0_u16;
    let mut sample_rate = 0_u32;
    let mut bits_per_sample = 0_u16;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let payload_start = pos + 8;
        let payload_end = payload_start
            .checked_add(size)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| format!("truncated WAV chunk in '{}'", path.display()))?;
        if id == b"fmt " {
            if size < 16 {
                return Err(format!("truncated fmt chunk in '{}'", path.display()));
            }
            let fmt = &bytes[payload_start..payload_end];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if id == b"data" {
            data_slice = Some(&bytes[payload_start..payload_end]);
        }
        pos = payload_end + (size & 1);
    }

    let data = data_slice.ok_or_else(|| format!("missing data chunk in '{}'", path.display()))?;
    if sample_rate == 0 || (channels != 1 && channels != 2) {
        return Err(format!(
            "unsupported WAV format in '{}': ch={channels}, sr={sample_rate}",
            path.display()
        ));
    }
    let bytes_per_sample = usize::from(bits_per_sample / 8);
    let frame_bytes = bytes_per_sample * usize::from(channels);
    if frame_bytes == 0 {
        return Err(format!("invalid sample size in '{}'", path.display()));
    }
    let frames = data.len() / frame_bytes;
    let mut left = Vec::with_capacity(frames);
    let mut right = Vec::with_capacity(frames);

    for f in 0..frames {
        let base = f * frame_bytes;
        let l = decode_sample(
            &data[base..base + bytes_per_sample],
            audio_format,
            bits_per_sample,
        )?;
        let r = if channels == 2 {
            decode_sample(
                &data[base + bytes_per_sample..base + 2 * bytes_per_sample],
                audio_format,
                bits_per_sample,
            )?
        } else {
            l
        };
        left.push(l);
        right.push(r);
    }

    Ok(WavAudio {
        sample_rate,
        left,
        right,
    })
}

fn decode_sample(chunk: &[u8], audio_format: u16, bits_per_sample: u16) -> Result<f32, String> {
    match (audio_format, bits_per_sample) {
        (1, 16) => {
            let v = i16::from_le_bytes([chunk[0], chunk[1]]);
            Ok(f32::from(v) / 32_768.0)
        }
        (1, 24) => {
            let raw =
                i32::from(chunk[0]) | (i32::from(chunk[1]) << 8) | (i32::from(chunk[2]) << 16);
            let signed = (raw << 8) >> 8;
            Ok((signed as f32) / 8_388_608.0)
        }
        (3, 32) => Ok(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
        _ => Err(format!(
            "unsupported WAV encoding: format={audio_format}, bits={bits_per_sample}"
        )),
    }
}

/// Detects `3-12 kHz` non-attack buzz bursts outside NoteOn `[t_on - 5 ms, t_on + 35 ms]` windows.
#[must_use]
pub fn detect_buzz_bursts(
    left: &[f32],
    right: &[f32],
    sr: u32,
    note_on_samples: &[usize],
) -> BuzzReport {
    let n_samples = left.len().min(right.len());
    let mono: Vec<f32> = (0..n_samples).map(|i| 0.5 * (left[i] + right[i])).collect();
    let sr_f = f64::from(sr);
    let pre_mask = (0.005 * sr_f).round() as usize;
    let post_mask = (0.035 * sr_f).round() as usize;

    let mut is_attack = vec![false; n_samples];
    for &s_on in note_on_samples {
        let lo = s_on.saturating_sub(pre_mask);
        let hi = s_on.saturating_add(post_mask).min(n_samples);
        is_attack[lo..hi].fill(true);
    }

    let mut hann = vec![0.0_f32; FFT_LEN];
    let mut win_sq_sum = 0.0_f64;
    let denom = (FFT_LEN - 1) as f64;
    for (i, w) in hann.iter_mut().enumerate() {
        let v = 0.5 * (1.0 - (std::f64::consts::TAU * (i as f64) / denom).cos());
        *w = v as f32;
        win_sq_sum += v * v;
    }

    let mut fft = RealFft::new(FFT_LEN);
    let spec_len = fft.spectrum_len();
    let mut windowed = vec![0.0_f32; FFT_LEN];
    let mut re = vec![0.0_f32; spec_len];
    let mut im = vec![0.0_f32; spec_len];

    let bin_hz = sr_f / (FFT_LEN as f64);
    let low_lo = ((80.0 / bin_hz).ceil() as usize).max(1);
    let low_hi = ((2_000.0 / bin_hz).floor() as usize).min(spec_len - 1);
    let buzz_lo = ((3_000.0 / bin_hz).ceil() as usize).max(1);
    let buzz_hi = ((12_000.0 / bin_hz).floor() as usize).min(spec_len - 1);

    let mut total_frames = 0_usize;
    let mut evaluated_frames = 0_usize;
    let mut prev_buzz_db: Option<f64> = None;
    let mut worst_ratio_db = f64::NEG_INFINITY;
    let mut worst_d3 = 0.0_f32;
    let mut bursts: Vec<BuzzBurst> = Vec::new();
    let merge_gap_frames = MERGE_GAP_SECS * sr_f;

    let mut pos = 0_usize;
    while pos + FFT_LEN <= n_samples {
        total_frames += 1;
        if is_attack[pos..pos + FFT_LEN].iter().any(|&a| a) {
            prev_buzz_db = None;
            pos += HOP;
            continue;
        }

        let frame = &mono[pos..pos + FFT_LEN];
        let mut sq_sum = 0.0_f64;
        let mut max_d3 = 0.0_f32;
        for (i, &x) in frame.iter().enumerate() {
            sq_sum += f64::from(x) * f64::from(x);
            windowed[i] = x * hann[i];
            if i >= 3 {
                let d3 = (frame[i] - 3.0 * frame[i - 1] + 3.0 * frame[i - 2] - frame[i - 3]).abs();
                if d3 > max_d3 {
                    max_d3 = d3;
                }
            }
        }
        let rms = (sq_sum / (FFT_LEN as f64)).sqrt();
        if rms <= MIN_FRAME_RMS {
            prev_buzz_db = None;
            pos += HOP;
            continue;
        }
        evaluated_frames += 1;

        fft.forward_split(&windowed, &mut re, &mut im);
        let p_low: f64 = (low_lo..=low_hi)
            .map(|k| f64::from(re[k]).powi(2) + f64::from(im[k]).powi(2))
            .sum();
        let p_buzz: f64 = (buzz_lo..=buzz_hi)
            .map(|k| f64::from(re[k]).powi(2) + f64::from(im[k]).powi(2))
            .sum();

        let norm = (FFT_LEN as f64) * win_sq_sum * 0.5;
        let p_buzz_norm = p_buzz / norm.max(1.0);
        let p_buzz_db = 10.0 * p_buzz_norm.max(1.0e-30).log10();
        let ratio_db = 10.0 * (p_buzz / p_low.max(1.0e-30)).log10();
        let jump_db = prev_buzz_db.map_or(0.0, |prev| p_buzz_db - prev);
        prev_buzz_db = Some(p_buzz_db);

        worst_ratio_db = worst_ratio_db.max(ratio_db);
        worst_d3 = worst_d3.max(max_d3);

        let flagged = ratio_db > RATIO_THRESHOLD_DB
            || (p_buzz_db > JUMP_LEVEL_FLOOR_DB && jump_db >= JUMP_DELTA_DB)
            || max_d3 > D3_THRESHOLD;
        if flagged {
            let t0 = (pos as f64) / sr_f;
            let t1 = ((pos + FFT_LEN) as f64) / sr_f;
            if let Some(last) = bursts.last_mut() {
                if (t0 - last.end_s) * sr_f <= merge_gap_frames {
                    last.end_s = t1;
                    last.max_ratio_db = last.max_ratio_db.max(ratio_db);
                    last.max_buzz_db = last.max_buzz_db.max(p_buzz_db);
                    last.max_d3 = last.max_d3.max(max_d3);
                    pos += HOP;
                    continue;
                }
            }
            bursts.push(BuzzBurst {
                start_s: t0,
                end_s: t1,
                max_ratio_db: ratio_db,
                max_buzz_db: p_buzz_db,
                max_d3,
            });
        }
        pos += HOP;
    }

    BuzzReport {
        total_frames,
        evaluated_frames,
        bursts,
        worst_ratio_db,
        worst_d3,
    }
}

/// Runs `diag-buzz` across the 6 canonical demo WAVs and MIDIs.
pub fn cmd_diag_buzz(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let demos_dir = match opts.demos_dir {
        Some(dir) => dir,
        None => {
            let cwd = env::current_dir().map_err(|e| format!("failed to get cwd: {e}"))?;
            find_repo_root(&cwd).join("apps/web/public/demos")
        }
    };

    let mut total_bursts = 0_usize;
    println!(
        "diag-buzz [wav_dir={}, mid_dir={}]",
        opts.wav_dir.display(),
        demos_dir.display()
    );
    for demo in &SHOWCASE_DEMOS {
        let wav_path = opts.wav_dir.join(demo.wav_filename);
        let mid_path = demos_dir.join(demo.mid_filename);
        let wav = read_wav_stereo(&wav_path)?;
        let smf = parse_smf_file(&mid_path, wav.sample_rate)?;
        let note_ons: Vec<usize> = smf
            .events
            .iter()
            .filter_map(|&(sample, kind)| match kind {
                EventKind::NoteOn { velocity_u16, .. } if velocity_u16 > 0 => Some(sample),
                _ => None,
            })
            .collect();
        let report = detect_buzz_bursts(&wav.left, &wav.right, wav.sample_rate, &note_ons);
        total_bursts += report.bursts.len();
        println!(
            "  {}: bursts={}, eval_frames={}/{}, worst_ratio={:+.2} dB, worst_d3={:.5}",
            demo.wav_filename,
            report.bursts.len(),
            report.evaluated_frames,
            report.total_frames,
            report.worst_ratio_db,
            report.worst_d3
        );
        for b in &report.bursts {
            println!(
                "    burst [{:.3}..{:.3} s]: ratio={:+.2} dB, buzz={:+.2} dBFS, max_d3={:.5}",
                b.start_s, b.end_s, b.max_ratio_db, b.max_buzz_db, b.max_d3
            );
        }
    }
    if total_bursts > 0 {
        Err(format!(
            "diag-buzz FAILED: detected {total_bursts} non-attack 3-12 kHz buzz bursts"
        ))
    } else {
        println!("diag-buzz OK: 0 non-attack 3-12 kHz buzz bursts across all 6 canonical demos");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav_writer::{
        write_stereo_f32_wav, write_stereo_pcm16_wav, write_stereo_pcm24_tpdf_wav,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn read_wav_stereo_supports_pcm16_pcm24_and_f32() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let base =
            std::env::temp_dir().join(format!("pianosim_buzz_wav_{}_{nanos}", std::process::id()));
        let l = [0.0_f32, 0.25, -0.5, 0.75];
        let r = [0.0_f32, -0.25, 0.5, -0.75];

        let p16 = base.with_extension("16.wav");
        write_stereo_pcm16_wav(&p16, 48_000, &l, &r).unwrap();
        let w16 = read_wav_stereo(&p16).unwrap();
        assert_eq!(w16.sample_rate, 48_000);
        assert!((w16.left[2] - (-0.5)).abs() < 1e-4);

        let p24 = base.with_extension("24.wav");
        write_stereo_pcm24_tpdf_wav(&p24, 48_000, &l, &r).unwrap();
        let w24 = read_wav_stereo(&p24).unwrap();
        assert!((w24.right[3] - (-0.75)).abs() < 1e-5);

        let p32 = base.with_extension("32.wav");
        write_stereo_f32_wav(&p32, 96_000, &l, &r).unwrap();
        let w32 = read_wav_stereo(&p32).unwrap();
        assert_eq!(w32.sample_rate, 96_000);
        assert_eq!(w32.left, l);
        assert_eq!(w32.right, r);

        let _ = fs::remove_file(&p16);
        let _ = fs::remove_file(&p24);
        let _ = fs::remove_file(&p32);
    }

    #[test]
    fn detect_buzz_bursts_flags_non_attack_hf_burst_and_ignores_clean_or_attack() {
        let sr = 48_000_u32;
        let len = 48_000_usize;
        let mut clean = vec![0.0_f32; len];
        for (n, s) in clean.iter_mut().enumerate() {
            let t = (n as f64) / f64::from(sr);
            *s = (0.2 * (std::f64::consts::TAU * 220.0 * t).sin()) as f32;
        }
        // Inject 5 kHz burst inside NoteOn attack window [0..35 ms]: must be masked.
        for (n, s) in clean.iter_mut().enumerate().take(600).skip(200) {
            let t = (n as f64) / f64::from(sr);
            *s += (0.08 * (std::f64::consts::TAU * 5_500.0 * t).sin()) as f32;
        }
        let r_clean = detect_buzz_bursts(&clean, &clean, sr, &[0]);
        assert_eq!(
            r_clean.bursts.len(),
            0,
            "attack-masked clean signal should have 0 bursts, got {:?}",
            r_clean.bursts
        );

        // Inject non-attack 5.5 kHz buzz burst at t = 0.50..0.52 s: must be flagged.
        let mut dirty = clean;
        for (n, s) in dirty.iter_mut().enumerate().take(24_960).skip(24_000) {
            let t = (n as f64) / f64::from(sr);
            *s += (0.08 * (std::f64::consts::TAU * 5_500.0 * t).sin()) as f32;
        }
        let r_dirty = detect_buzz_bursts(&dirty, &dirty, sr, &[0]);
        assert!(
            !r_dirty.bursts.is_empty(),
            "non-attack 5.5 kHz burst must be detected"
        );
    }
}
