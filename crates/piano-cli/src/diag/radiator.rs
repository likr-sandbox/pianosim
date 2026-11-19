//! `diag-radiator`: per-zone unit-impulse IR inspection and spectral-tilt metrics.

use std::fs;
use std::path::PathBuf;

use piano_engine::{PerfProfile, DEFAULT_REVERB_DECAY_S};
use piano_radiator::{embedded_body_bytes, BodyArchetype, RadiatorBank};
use piano_simd::RealFft;
use piano_spec::{LidState, ListenerPerspective, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES};

use crate::cli_profile::CliProfileArgs;
use crate::scenarios::spectral_utils::{mono_mix, ols_slope, slice_rms};
use crate::wav_writer::write_stereo_f32_wav;

use super::parse_flag_value;

const SR: f64 = 48_000.0;
const SR_U32: u32 = 48_000;
const FFT_LEN: usize = 32_768;

const OCTAVE_BANDS_HZ: [(f64, f64); 7] = [
    (80.0, 125.0),
    (125.0, 250.0),
    (250.0, 500.0),
    (500.0, 1_000.0),
    (1_000.0, 2_000.0),
    (2_000.0, 4_000.0),
    (4_000.0, 8_000.0),
];

#[derive(Clone, Debug)]
pub struct ZoneSpectralReport {
    pub zone: usize,
    pub h0_direct: f32,
    pub total_rms: f64,
    pub decay_db: f64,
    pub band_db: [f64; 7],
    pub band_lf_45_90_db: f64,
    pub band_mid_300_700_db: f64,
    pub slope_db_per_oct: f64,
    pub lf_rolloff_db: f64,
    pub mid_hump_db: f64,
}

struct RadiatorDiagOptions {
    out_dir: Option<PathBuf>,
    body: BodyArchetype,
    perspective: ListenerPerspective,
    lid: LidState,
    reverb_mix: f32,
    profile: PerfProfile,
    profile_label: String,
}

fn parse_body(raw: &str) -> Result<BodyArchetype, String> {
    match raw {
        "concert-grand" => Ok(BodyArchetype::ConcertGrand),
        "upright-studio" => Ok(BodyArchetype::UprightStudio),
        "vintage-saloon" => Ok(BodyArchetype::VintageSaloon),
        _ => Err(format!(
            "invalid --body '{raw}' (expected concert-grand, upright-studio, or vintage-saloon)"
        )),
    }
}

fn parse_perspective(raw: &str) -> Result<ListenerPerspective, String> {
    match raw {
        "player" => Ok(ListenerPerspective::Player),
        "close" | "close-stereo" => Ok(ListenerPerspective::CloseStereo),
        "hall" | "concert-hall" => Ok(ListenerPerspective::ConcertHall),
        "binaural" | "binaural-hrtf" => Ok(ListenerPerspective::BinauralHrtf),
        _ => Err(format!(
            "invalid --perspective '{raw}' (expected player, close, hall, or binaural)"
        )),
    }
}

fn parse_lid(raw: &str) -> Result<LidState, String> {
    match raw {
        "closed" => Ok(LidState::Closed),
        "half" => Ok(LidState::Half),
        "open" => Ok(LidState::Open),
        _ => Err(format!(
            "invalid --lid '{raw}' (expected closed, half, or open)"
        )),
    }
}

fn parse_options(args: &[String]) -> Result<RadiatorDiagOptions, String> {
    let usage = super::diag_usage();
    let mut out_dir: Option<PathBuf> = None;
    let mut body = BodyArchetype::ConcertGrand;
    let mut perspective = ListenerPerspective::Player;
    let mut lid = LidState::Open;
    let mut reverb_mix = 0.0_f32;
    let mut cli_profile = CliProfileArgs::default();

    let mut i = 0;
    while i < args.len() {
        if cli_profile.try_consume_arg(args, &mut i)? {
            continue;
        }
        if let Some(v) = parse_flag_value(args, &mut i, "--body", usage)? {
            body = parse_body(v)?;
        } else if let Some(v) = parse_flag_value(args, &mut i, "--perspective", usage)? {
            perspective = parse_perspective(v)?;
        } else if let Some(v) = parse_flag_value(args, &mut i, "--lid", usage)? {
            lid = parse_lid(v)?;
        } else if let Some(v) = parse_flag_value(args, &mut i, "--reverb-mix", usage)? {
            let mix: f32 = v
                .parse()
                .map_err(|_| format!("invalid --reverb-mix '{v}'"))?;
            if !mix.is_finite() || !(0.0..=1.0).contains(&mix) {
                return Err(format!("--reverb-mix {v} out of range [0.0, 1.0]"));
            }
            reverb_mix = mix;
        } else if args[i].starts_with('-') {
            return Err(String::from(usage));
        } else if out_dir.is_none() {
            out_dir = Some(PathBuf::from(&args[i]));
            i += 1;
        } else {
            return Err(String::from(usage));
        }
    }

    Ok(RadiatorDiagOptions {
        out_dir,
        body,
        perspective,
        lid,
        reverb_mix,
        profile: cli_profile.resolve(),
        profile_label: cli_profile.label(),
    })
}

fn fft_mag_sq(signal: &[f32], fft_len: usize) -> Vec<f64> {
    let mut padded = vec![0.0_f32; fft_len];
    let copy_n = signal.len().min(fft_len);
    padded[..copy_n].copy_from_slice(&signal[..copy_n]);
    let mut fft = RealFft::new(fft_len);
    let spec_len = fft.spectrum_len();
    let mut re = vec![0.0_f32; spec_len];
    let mut im = vec![0.0_f32; spec_len];
    fft.forward_split(&padded, &mut re, &mut im);
    (0..spec_len)
        .map(|k| f64::from(re[k]).powi(2) + f64::from(im[k]).powi(2))
        .collect()
}

fn band_mean_db(mag_sq: &[f64], sr: f64, fft_len: usize, f_lo: f64, f_hi: f64) -> f64 {
    let k0 = ((f_lo * fft_len as f64 / sr).ceil() as usize).max(1);
    let k1 = ((f_hi * fft_len as f64 / sr).floor() as usize).min(mag_sq.len() - 1);
    if k1 < k0 {
        return -120.0;
    }
    let sum: f64 = mag_sq[k0..=k1].iter().sum();
    let mean = sum / ((k1 - k0 + 1) as f64);
    10.0 * mean.max(1.0e-30).log10()
}

/// Computes spectral slope, `<90 Hz` rolloff, and `300-700 Hz` mid-hump metrics for a mono IR.
#[must_use]
pub fn analyze_zone_ir(zone: usize, mono: &[f32], sr: f64) -> ZoneSpectralReport {
    let mag_sq = fft_mag_sq(mono, FFT_LEN);
    let mut band_db = [0.0_f64; 7];
    let mut slope_pts = Vec::with_capacity(7);
    for (idx, &(f_lo, f_hi)) in OCTAVE_BANDS_HZ.iter().enumerate() {
        let db = band_mean_db(&mag_sq, sr, FFT_LEN, f_lo, f_hi);
        band_db[idx] = db;
        let f_center = (f_lo * f_hi).sqrt();
        slope_pts.push((f_center.log2(), db));
    }

    let band_lf_45_90_db = band_mean_db(&mag_sq, sr, FFT_LEN, 45.0, 90.0);
    let band_mid_300_700_db = band_mean_db(&mag_sq, sr, FFT_LEN, 300.0, 700.0);
    let slope_db_per_oct = ols_slope(&slope_pts);
    // Negative means <90 Hz is lower than 125-250 Hz; target >= -6.0 dB (no deep LF cliff).
    let lf_rolloff_db = band_lf_45_90_db - band_db[1];
    // Positive means 300-700 Hz protrudes above neighbor octaves (125-250 and 1k-2k).
    let mid_hump_db = band_mid_300_700_db - 0.5 * (band_db[1] + band_db[4]);

    let n_50ms = ((0.050 * sr).round() as usize).min(mono.len());
    let n_450ms = ((0.450 * sr).round() as usize).min(mono.len());
    let rms_early = slice_rms(&mono[..n_50ms.max(1)]);
    let rms_tail = if n_450ms < mono.len() {
        slice_rms(&mono[n_450ms..])
    } else {
        0.0
    };
    let decay_db = 20.0 * (rms_early.max(1.0e-15) / rms_tail.max(1.0e-15)).log10();

    ZoneSpectralReport {
        zone,
        h0_direct: mono.first().copied().unwrap_or(0.0),
        total_rms: slice_rms(mono),
        decay_db,
        band_db,
        band_lf_45_90_db,
        band_mid_300_700_db,
        slope_db_per_oct,
        lf_rolloff_db,
        mid_hump_db,
    }
}

/// Runs `diag-radiator` unit-impulse sweep across zones `0..=8`.
pub fn cmd_diag_radiator(args: &[String]) -> Result<(), String> {
    let opts = parse_options(args)?;
    let bytes = embedded_body_bytes(opts.body);
    let mut bank = RadiatorBank::from_preset_bytes_with_profile(
        bytes,
        SR,
        opts.perspective,
        opts.lid,
        &opts.profile,
    )
    .map_err(|e| format!("failed to decode body preset: {e:?}"))?;
    bank.set_reverb(DEFAULT_REVERB_DECAY_S, opts.reverb_mix);

    if let Some(ref dir) = opts.out_dir {
        fs::create_dir_all(dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;
    }

    let ir_len = bank.ir_len_samples();
    let mut reports = Vec::with_capacity(PRESET_RADIATOR_ZONES);
    let mut frames = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; ir_len];
    let mut out_l = vec![0.0_f32; ir_len];
    let mut out_r = vec![0.0_f32; ir_len];

    for z in 0..PRESET_RADIATOR_ZONES {
        bank.reset();
        frames.fill([0.0_f32; PRESET_RADIATOR_ZONES]);
        frames[0][z] = 1.0;

        let mut pos = 0;
        while pos < ir_len {
            let n = (ir_len - pos).min(128);
            bank.process_block(
                &frames[pos..pos + n],
                &mut out_l[pos..pos + n],
                &mut out_r[pos..pos + n],
            );
            pos += n;
        }

        if let Some(ref dir) = opts.out_dir {
            let wav_path = dir.join(format!("zone_{z}.wav"));
            write_stereo_f32_wav(&wav_path, SR_U32, &out_l, &out_r)?;
        }

        let mono = mono_mix(&out_l, &out_r);
        reports.push(analyze_zone_ir(z, &mono, SR));
    }

    println!(
        "diag-radiator [{:?} / {:?} / {:?}, perf={}, reverb_mix={:.2}, sr={SR_U32}, ir_len={ir_len}]",
        opts.body, opts.perspective, opts.lid, opts.profile_label, opts.reverb_mix
    );
    println!(
        "Zone |   h[0]  |    RMS   | Decay | Slope(dB/oct) | <90Hz(dB) | MidHump(dB) | 45-90 | 125-250 | 300-700 | 1k-2k | 4k-8k"
    );
    println!(
        "-----+---------+----------+-------+---------------+-----------+-------------+-------+---------+---------+-------+------"
    );
    for r in &reports {
        println!(
            "  z{} | {:+.4} | {:.6} | {:5.1} |     {:+.2}     |   {:+.2}   |    {:+.2}    | {:5.1} |  {:5.1}  |  {:5.1}  | {:5.1} | {:5.1}",
            r.zone,
            r.h0_direct,
            r.total_rms,
            r.decay_db,
            r.slope_db_per_oct,
            r.lf_rolloff_db,
            r.mid_hump_db,
            r.band_lf_45_90_db,
            r.band_db[1],
            r.band_mid_300_700_db,
            r.band_db[4],
            r.band_db[6]
        );
    }

    let bridge = &reports[..PRESET_BRIDGE_ZONES];
    let n_b = bridge.len() as f64;
    let avg_slope = bridge.iter().map(|r| r.slope_db_per_oct).sum::<f64>() / n_b;
    let avg_lf = bridge.iter().map(|r| r.lf_rolloff_db).sum::<f64>() / n_b;
    let max_hump = bridge
        .iter()
        .map(|r| r.mid_hump_db)
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "Bridge zones 0..7 summary: avg_slope={avg_slope:+.2} dB/oct (target -3.0..-4.0), avg_lf_rolloff={avg_lf:+.2} dB (target >= -6.0), max_mid_hump={max_hump:+.2} dB (target <= +2.0)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_zone_ir_recovers_pink_minus_3db_per_oct_slope() {
        // Synthesize a minimum-phase-like test signal with -3 dB/oct amplitude? No, 1/sqrt(f)
        // amplitude gives 1/f power (-3.01 dB/oct).
        let n = 16_384;
        let mut fft = RealFft::new(n);
        let spec_len = fft.spectrum_len();
        let mut re = vec![0.0_f32; spec_len];
        let mut im = vec![0.0_f32; spec_len];
        for k in 1..spec_len {
            let f = (k as f64) * SR / (n as f64);
            let mag = (160.0 / f.max(20.0)).sqrt();
            let phase = (k as f64) * 1.618_033_9;
            re[k] = (mag * phase.cos()) as f32;
            im[k] = (mag * phase.sin()) as f32;
        }
        let mut time = vec![0.0_f32; n];
        fft.inverse_split(&re, &im, &mut time);
        let report = analyze_zone_ir(0, &time, SR);
        assert!(
            (-3.4..=-2.6).contains(&report.slope_db_per_oct),
            "expected ~ -3.01 dB/oct slope, got {:.2}",
            report.slope_db_per_oct
        );
        assert!(
            report.mid_hump_db.abs() < 0.8,
            "smooth 1/f spectrum should have near-zero mid hump, got {:.2}",
            report.mid_hump_db
        );
    }
}
