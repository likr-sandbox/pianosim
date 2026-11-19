//! Off-thread spatial IR + NUPOLS FFT spectrum precomputation (`SpaceBlob` v2)
//! for zero-stall `AudioWorklet` space & profile switching (`no_std` + `alloc`).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use piano_simd::{AlignedBuf, RealFft, SplitComplexBuf};
use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PerfProfile, PresetDecodeError,
    PRESET_RADIATOR_ZONES,
};

use crate::block_fft_head::{BLOCK_FFT_PARTITIONS, BLOCK_FFT_SIZE, BLOCK_FFT_SPEC_LEN};
use crate::fir_head::{snap_head_taps, FIR_COEFFS_LEN, HEAD_TAPS};
use crate::ir_synth::{
    ir_len_for_sample_rate, ir_len_for_sample_rate_secs, path_index, synthesize_irs_with_duration,
    IrSynthScratch, NUM_IR_PATHS,
};
use crate::math::clamp_f64;
use crate::nupols_collapse::{
    collapse_zone_spectra_causal, compute_mimo_p_count, compute_p_count, MAX_TIER_FFT_LEN,
    SUB_TIER_16_CFG, SUB_TIER_32_CFG, TIER_CONFIGS,
};
use crate::{empty_fallback_preset, CONCERT_GRAND_PRESET_BYTES};

/// Magic tag for v2 parameterized spatial NUPOLS blobs (`"PBSQ"`).
pub const SPACE_BLOB_MAGIC_V2: u32 = 0x5042_5351;
/// Number of `f32` words in the v2 spatial NUPOLS blob header.
pub const SPACE_BLOB_HEADER_LEN: usize = 8;
/// Header flag bit 0: use 32-sample block-FFT head (`BlockFftHead`).
pub const SPACE_BLOB_FLAG_BLOCK_FFT: u32 = 1 << 0;
/// Header flag bit 1: blob includes precomputed `block_fft_head` or `sub_tier` spectra.
pub const SPACE_BLOB_FLAG_HEAD_SPECTRA: u32 = 1 << 1;

pub(crate) fn head_spectra_float_len(fir_block_fft: bool, head_taps: u8) -> usize {
    if fir_block_fft {
        BLOCK_FFT_PARTITIONS * NUM_IR_PATHS * BLOCK_FFT_SPEC_LEN * 2
    } else {
        let snapped = snap_head_taps(head_taps);
        if snapped == 32 {
            compute_p_count(SUB_TIER_32_CFG, HEAD_TAPS)
                * NUM_IR_PATHS
                * (SUB_TIER_32_CFG.block_size + 1)
                * 2
        } else if snapped == 16 {
            compute_p_count(SUB_TIER_16_CFG, HEAD_TAPS)
                * NUM_IR_PATHS
                * (SUB_TIER_16_CFG.block_size + 1)
                * 2
        } else {
            0
        }
    }
}

pub(crate) fn space_blob_len_for_params(ir_len: usize, split_tap: u16) -> usize {
    let mut spectra_floats = 0usize;
    for &cfg in &TIER_CONFIGS {
        let p_count = compute_p_count(cfg, ir_len);
        let mimo_p_count = compute_mimo_p_count(cfg, p_count, split_tap);
        let spec_floats = (cfg.block_size + 1) * 2;
        spectra_floats += mimo_p_count * NUM_IR_PATHS * spec_floats;
        spectra_floats += (p_count - mimo_p_count) * 2 * spec_floats;
    }
    SPACE_BLOB_HEADER_LEN + FIR_COEFFS_LEN + spectra_floats
}

/// Computes the exact number of `f32` elements in a serialized spatial NUPOLS blob
/// for the given [`PerfProfile`].
#[must_use]
pub fn space_blob_len_for_profile(sample_rate: f64, profile: &PerfProfile) -> usize {
    let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
    let prof = profile.sanitized();
    let ir_len = ir_len_for_sample_rate_secs(sr, f64::from(prof.ir_duration_secs));
    space_blob_len_for_params(ir_len, prof.spatial_tail_split_tap)
        + head_spectra_float_len(prof.fir_block_fft, prof.fir_head_taps)
}

/// Computes the maximum (`Full`) `f32` element count for preallocating staging buffers.
#[must_use]
pub fn space_blob_len_for_sample_rate(sample_rate: f64) -> usize {
    space_blob_len_for_profile(sample_rate, &PerfProfile::full())
}

/// Off-thread precomputation engine producing serialized spatial FIR + NUPOLS blobs
/// without touching real-time audio delay lines.
pub struct SpacePrepContext {
    preset: BodyPreset,
    sample_rate: f64,
    profile: PerfProfile,
    scratch: IrSynthScratch,
    ffts: [RealFft; 4],
    sub_fft_32: RealFft,
    sub_fft_16: RealFft,
    time_scratch: AlignedBuf,
    collapse_buf: Vec<SplitComplexBuf>,
    spec_re: Vec<f32>,
    spec_im: Vec<f32>,
    out_blob: Vec<f32>,
}

impl SpacePrepContext {
    /// Constructs a spatial prep context from the baked concert-grand preset (`Full` profile).
    #[must_use]
    pub fn new_concert_grand(sample_rate: f64) -> Self {
        Self::new_with_profile(sample_rate, &PerfProfile::full())
    }

    /// Constructs a spatial prep context from the baked concert-grand preset with `profile`.
    #[must_use]
    pub fn new_with_profile(sample_rate: f64, profile: &PerfProfile) -> Self {
        let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES)
            .unwrap_or_else(|_| empty_fallback_preset());
        Self::new_from_preset_with_profile(preset, sample_rate, profile)
    }

    /// Constructs a spatial prep context from decoded `BodyPreset` (`Full` profile).
    #[must_use]
    pub fn new_from_preset(preset: BodyPreset, sample_rate: f64) -> Self {
        Self::new_from_preset_with_profile(preset, sample_rate, &PerfProfile::full())
    }

    /// Constructs a spatial prep context from decoded `BodyPreset` and [`PerfProfile`].
    #[must_use]
    pub fn new_from_preset_with_profile(
        preset: BodyPreset,
        sample_rate: f64,
        profile: &PerfProfile,
    ) -> Self {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let max_ir_len = ir_len_for_sample_rate(sr);
        let max_spec = MAX_TIER_FFT_LEN / 2 + 1;
        let mut collapse_buf = Vec::with_capacity(NUM_IR_PATHS);
        for _ in 0..NUM_IR_PATHS {
            collapse_buf.push(SplitComplexBuf::zeroed(max_spec));
        }
        Self {
            preset,
            sample_rate: sr,
            profile: profile.sanitized(),
            scratch: IrSynthScratch::new(max_ir_len),
            ffts: [
                RealFft::new(TIER_CONFIGS[0].block_size * 2),
                RealFft::new(TIER_CONFIGS[1].block_size * 2),
                RealFft::new(TIER_CONFIGS[2].block_size * 2),
                RealFft::new(TIER_CONFIGS[3].block_size * 2),
            ],
            sub_fft_32: RealFft::new(BLOCK_FFT_SIZE * 2),
            sub_fft_16: RealFft::new(SUB_TIER_16_CFG.block_size * 2),
            time_scratch: AlignedBuf::zeroed(MAX_TIER_FFT_LEN),
            collapse_buf,
            spec_re: vec![0.0; max_spec],
            spec_im: vec![0.0; max_spec],
            out_blob: Vec::with_capacity(
                space_blob_len_for_sample_rate(sr) + head_spectra_float_len(true, 32),
            ),
        }
    }

    /// Decodes `PIANOBOD` bytes and constructs a spatial prep context (`Full` profile).
    pub fn new_from_preset_bytes(
        bytes: &[u8],
        sample_rate: f64,
    ) -> Result<Self, PresetDecodeError> {
        Self::new_from_preset_bytes_with_profile(bytes, sample_rate, &PerfProfile::full())
    }

    /// Decodes `PIANOBOD` bytes and constructs a spatial prep context with `profile`.
    pub fn new_from_preset_bytes_with_profile(
        bytes: &[u8],
        sample_rate: f64,
        profile: &PerfProfile,
    ) -> Result<Self, PresetDecodeError> {
        let preset = BodyPreset::decode(bytes)?;
        Ok(Self::new_from_preset_with_profile(
            preset,
            sample_rate,
            profile,
        ))
    }

    /// Replaces the active body preset in-place from serialized `PIANOBOD` bytes.
    pub fn set_body_preset_bytes(&mut self, bytes: &[u8]) -> Result<(), PresetDecodeError> {
        self.preset = BodyPreset::decode(bytes)?;
        Ok(())
    }

    /// Replaces the active body preset in-place.
    pub fn set_body_preset(&mut self, preset: BodyPreset) {
        self.preset = preset;
    }

    /// Updates the performance profile used for subsequent [`Self::prepare_space`] calls.
    pub fn set_perf_profile(&mut self, profile: &PerfProfile) {
        self.profile = profile.sanitized();
    }

    /// Updates Stage 4 parameters directly for subsequent [`Self::prepare_space`] calls.
    pub fn set_stage4_profile(
        &mut self,
        ir_duration_secs: f32,
        spatial_tail_split_tap: u16,
        fir_head_taps: u8,
        fir_block_fft: bool,
    ) {
        let mut next = self.profile;
        next.ir_duration_secs = ir_duration_secs;
        next.spatial_tail_split_tap = spatial_tail_split_tap;
        next.fir_head_taps = fir_head_taps;
        next.fir_block_fft = fir_block_fft;
        self.profile = next.sanitized();
    }

    /// Active [`PerfProfile`] used by this spatial prep context.
    #[must_use]
    pub const fn perf_profile(&self) -> &PerfProfile {
        &self.profile
    }

    /// Returns the active [`BodyPreset`] used by this spatial prep context.
    #[must_use]
    pub fn preset(&self) -> &BodyPreset {
        &self.preset
    }

    /// Synthesizes IRs and pre-FFTs all NUPOLS partitions into a flat POD `f32` slice.
    pub fn prepare_space(
        &mut self,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> &[f32] {
        synthesize_irs_with_duration(
            &self.preset,
            perspective,
            lid_state,
            self.sample_rate,
            f64::from(self.profile.ir_duration_secs),
            &mut self.scratch,
        );
        let ir_len = self.scratch.ir_len;
        let split_tap = self.profile.spatial_tail_split_tap;
        let snapped_head = snap_head_taps(self.profile.fir_head_taps);
        let block_fft = self.profile.fir_block_fft;
        let head_spec_len = head_spectra_float_len(block_fft, snapped_head as u8);
        let flags = if block_fft {
            SPACE_BLOB_FLAG_BLOCK_FFT
        } else {
            0
        } | if head_spec_len > 0 {
            SPACE_BLOB_FLAG_HEAD_SPECTRA
        } else {
            0
        };

        self.out_blob.clear();
        self.out_blob.push(f32::from_bits(SPACE_BLOB_MAGIC_V2));
        self.out_blob.push(self.sample_rate as f32);
        self.out_blob.push(f32::from_bits(perspective as u32));
        self.out_blob.push(f32::from_bits(lid_state as u32));
        self.out_blob.push(f32::from_bits(ir_len as u32));
        self.out_blob
            .push(f32::from_bits(u32::from(self.profile.fir_head_taps)));
        self.out_blob.push(f32::from_bits(u32::from(split_tap)));
        self.out_blob.push(f32::from_bits(flags));

        let ir_flat = &self.scratch.ir_flat;
        let copy_len = ir_len.min(HEAD_TAPS);
        let mut rev_l = [0.0_f32; HEAD_TAPS];
        let mut rev_r = [0.0_f32; HEAD_TAPS];
        for z in 0..PRESET_RADIATOR_ZONES {
            rev_l.fill(0.0);
            rev_r.fill(0.0);
            let l_start = path_index(z, 0) * ir_len;
            let r_start = path_index(z, 1) * ir_len;
            for k in 0..copy_len {
                let dst = (HEAD_TAPS - 1) - k;
                rev_l[dst] = ir_flat[l_start + k];
                rev_r[dst] = ir_flat[r_start + k];
            }
            self.out_blob.extend_from_slice(&rev_l);
            self.out_blob.extend_from_slice(&rev_r);
        }

        let work_all = self.time_scratch.as_mut_slice();
        for (ti, &cfg) in TIER_CONFIGS.iter().enumerate() {
            let p_count = compute_p_count(cfg, ir_len);
            if p_count == 0 {
                continue;
            }
            let mimo_p_count = compute_mimo_p_count(cfg, p_count, split_tap);
            let b = cfg.block_size;
            let fft = &mut self.ffts[ti];
            let spec_len = fft.spectrum_len();
            let work = &mut work_all[..b * 2];
            for p in 0..p_count {
                let tap_start = cfg.ir_offset + p * b;
                if p < mimo_p_count {
                    for z in 0..PRESET_RADIATOR_ZONES {
                        for ch in 0..2 {
                            let ir_base = path_index(z, ch) * ir_len;
                            work.fill(0.0);
                            if tap_start < ir_len {
                                let c_len = b.min(ir_len - tap_start);
                                work[..c_len].copy_from_slice(
                                    &ir_flat[ir_base + tap_start..ir_base + tap_start + c_len],
                                );
                            }
                            let re = &mut self.spec_re[..spec_len];
                            let im = &mut self.spec_im[..spec_len];
                            fft.forward_split(work, re, im);
                            self.out_blob.extend_from_slice(re);
                            self.out_blob.extend_from_slice(im);
                        }
                    }
                } else {
                    for z in 0..PRESET_RADIATOR_ZONES {
                        for ch in 0..2 {
                            let ir_base = path_index(z, ch) * ir_len;
                            work.fill(0.0);
                            if tap_start < ir_len {
                                let c_len = b.min(ir_len - tap_start);
                                work[..c_len].copy_from_slice(
                                    &ir_flat[ir_base + tap_start..ir_base + tap_start + c_len],
                                );
                            }
                            let (re, im) = self.collapse_buf[path_index(z, ch)].planes_mut();
                            fft.forward_split(work, &mut re[..spec_len], &mut im[..spec_len]);
                        }
                    }
                    for ch in 0..2 {
                        let re = &mut self.spec_re[..spec_len];
                        let im = &mut self.spec_im[..spec_len];
                        collapse_zone_spectra_causal(&self.collapse_buf, fft, work, ch, re, im);
                        self.out_blob.extend_from_slice(re);
                        self.out_blob.extend_from_slice(im);
                    }
                }
            }
        }

        if block_fft {
            let b = BLOCK_FFT_SIZE;
            let spec_len = self.sub_fft_32.spectrum_len();
            let work = &mut work_all[..b * 2];
            for p in 0..BLOCK_FFT_PARTITIONS {
                let tap_start = p * b;
                for z in 0..PRESET_RADIATOR_ZONES {
                    for ch in 0..2 {
                        let ir_base = path_index(z, ch) * ir_len;
                        work.fill(0.0);
                        let c_len = b.min(ir_len.saturating_sub(tap_start));
                        if c_len > 0 {
                            work[..c_len].copy_from_slice(
                                &ir_flat[ir_base + tap_start..ir_base + tap_start + c_len],
                            );
                        }
                        let re = &mut self.spec_re[..spec_len];
                        let im = &mut self.spec_im[..spec_len];
                        self.sub_fft_32.forward_split(work, re, im);
                        self.out_blob.extend_from_slice(re);
                        self.out_blob.extend_from_slice(im);
                    }
                }
            }
        } else if snapped_head < HEAD_TAPS {
            let (cfg, fft) = if snapped_head <= 16 {
                (SUB_TIER_16_CFG, &mut self.sub_fft_16)
            } else {
                (SUB_TIER_32_CFG, &mut self.sub_fft_32)
            };
            let p_count = compute_p_count(cfg, ir_len.min(HEAD_TAPS));
            let b = cfg.block_size;
            let spec_len = fft.spectrum_len();
            let work = &mut work_all[..b * 2];
            for p in 0..p_count {
                let tap_start = cfg.ir_offset + p * b;
                for z in 0..PRESET_RADIATOR_ZONES {
                    for ch in 0..2 {
                        let ir_base = path_index(z, ch) * ir_len;
                        work.fill(0.0);
                        let c_len = b.min(ir_len.saturating_sub(tap_start));
                        if c_len > 0 {
                            work[..c_len].copy_from_slice(
                                &ir_flat[ir_base + tap_start..ir_base + tap_start + c_len],
                            );
                        }
                        let re = &mut self.spec_re[..spec_len];
                        let im = &mut self.spec_im[..spec_len];
                        fft.forward_split(work, re, im);
                        self.out_blob.extend_from_slice(re);
                        self.out_blob.extend_from_slice(im);
                    }
                }
            }
        }
        &self.out_blob
    }

    /// Returns the most recently prepared spatial blob slice.
    #[must_use]
    pub fn out_blob(&self) -> &[f32] {
        &self.out_blob
    }
}
