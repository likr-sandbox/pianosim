//! Stage 4 & Layer 1B: sample-rate adaptive IR synthesis and zero-latency
//! 9-zone MIMO Staggered NUPOLS acoustic radiator (`9 zones -> L/R`).

#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod bank_config;
mod binaural_hrtf;
mod block_fft_head;
mod body_assets;
mod fir_head;
mod hall_fdn;
mod hall_hadamard;
mod ir_synth;
mod limiter;
mod math;
mod mic_perspectives;
mod modal_prepare;
mod modal_synth;
mod nupols_collapse;
mod nupols_tier;
mod nupols_tier_ops;
mod partitioned_fft;
mod partitioned_schedule;
mod pole_cancel_filter;
mod space_blob;
mod space_blob_install;
mod subrate_upsample;
mod velvet_tail;

#[cfg(test)]
mod fdn_tests;
#[cfg(test)]
mod limiter_tests;
#[cfg(test)]
mod multi_body_tests;
#[cfg(test)]
mod nupols_tests;
#[cfg(test)]
mod profile_tests;
#[cfg(test)]
mod spectral_tests;
#[cfg(test)]
mod tests;

use bank_config::empty_fallback_preset;
use block_fft_head::{BlockFftHead, BLOCK_FFT_SIZE};
pub use body_assets::{body_asset_filename, embedded_body_bytes, BodyArchetype};
use core::fmt;
use fir_head::FirHead;
use hall_fdn::HallFdn;
pub use ir_synth::{
    ir_len_for_sample_rate, ir_len_for_sample_rate_secs, synthesize_irs,
    synthesize_irs_with_duration, IrSynthScratch,
};
use limiter::StereoLimiter;
use math::clamp_f64;
use partitioned_fft::NupolsTail;
pub use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PerfPreset, PerfProfile, PresetDecodeError,
    PRESET_RADIATOR_ZONES,
};
pub use space_blob::{
    space_blob_len_for_profile, space_blob_len_for_sample_rate, SpacePrepContext,
    SPACE_BLOB_FLAG_BLOCK_FFT, SPACE_BLOB_FLAG_HEAD_SPECTRA, SPACE_BLOB_HEADER_LEN,
    SPACE_BLOB_MAGIC_V2,
};

/// Deterministic physics-baked 9-zone concert-grand `PIANOBOD` asset.
pub const CONCERT_GRAND_PRESET_BYTES: &[u8] = include_bytes!("../assets/concert_grand_v1.pbdy");

/// Stage 4 acoustic radiator combining a configurable `16/32/64`-tap SIMD FIR (or
/// optional 32-sample block-FFT) head with a zero-alloc Staggered NUPOLS tail.
#[derive(Clone)]
pub struct RadiatorBank {
    preset: BodyPreset,
    sample_rate: f64,
    perspective: ListenerPerspective,
    lid_state: LidState,
    profile: PerfProfile,
    fir_block_fft: bool,
    staged_blob_has_head_spectra: bool,
    staged_space_stream: Option<space_blob_install::StagedSpaceStream>,
    scratch: IrSynthScratch,
    head: FirHead,
    block_fft_head: BlockFftHead,
    tail: NupolsTail,
    hall_fdn: HallFdn,
    limiter: StereoLimiter,
}

impl fmt::Debug for RadiatorBank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RadiatorBank")
            .field("sample_rate", &self.sample_rate)
            .field("perspective", &self.perspective)
            .field("lid_state", &self.lid_state)
            .field("ir_len", &self.tail.active_ir_len())
            .field("modes", &self.preset.modes.len())
            .finish()
    }
}

impl RadiatorBank {
    /// Construct a radiator bank from the embedded `concert_grand_v1.pbdy` preset
    /// with default `Player` perspective, `Open` lid state, and `Full` profile.
    pub fn new_concert_grand(sample_rate: f64) -> Self {
        Self::new_concert_grand_with_space(sample_rate, ListenerPerspective::Player, LidState::Open)
    }

    /// Construct a radiator bank from the embedded `concert_grand_v1.pbdy` preset
    /// with explicit listener perspective and lid state (panic-free decode).
    pub fn new_concert_grand_with_space(
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> Self {
        Self::new_with_profile(sample_rate, perspective, lid_state, &PerfProfile::full())
    }

    /// Construct a concert-grand radiator bank with explicit space and [`PerfProfile`].
    pub fn new_with_profile(
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
        profile: &PerfProfile,
    ) -> Self {
        let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES)
            .unwrap_or_else(|_| empty_fallback_preset());
        Self::from_preset_with_profile(&preset, sample_rate, perspective, lid_state, profile)
    }

    /// Construct a radiator bank from a decoded `BodyPreset` using default
    /// `Player` perspective and `Open` lid state.
    pub fn from_preset(preset: &BodyPreset, sample_rate: f64) -> Self {
        Self::from_preset_with_space(
            preset,
            sample_rate,
            ListenerPerspective::Player,
            LidState::Open,
        )
    }

    /// Construct a radiator bank with explicit listener perspective and lid state.
    pub fn from_preset_with_space(
        preset: &BodyPreset,
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> Self {
        Self::from_preset_with_profile(
            preset,
            sample_rate,
            perspective,
            lid_state,
            &PerfProfile::full(),
        )
    }

    /// Construct a radiator bank from a decoded `BodyPreset`, space, and [`PerfProfile`].
    pub fn from_preset_with_profile(
        preset: &BodyPreset,
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
        profile: &PerfProfile,
    ) -> Self {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let prof = profile.sanitized();
        let max_ir_len = ir_len_for_sample_rate(sr);
        let active_ir_len = ir_len_for_sample_rate_secs(sr, f64::from(prof.ir_duration_secs));
        let mut tail = NupolsTail::new(max_ir_len);
        tail.set_stage4_config(
            active_ir_len,
            prof.spatial_tail_split_tap,
            prof.fir_head_taps,
            prof.fir_block_fft,
        );
        let mut bank = Self {
            preset: preset.clone(),
            sample_rate: sr,
            perspective,
            lid_state,
            profile: prof,
            fir_block_fft: prof.fir_block_fft,
            staged_blob_has_head_spectra: false,
            staged_space_stream: None,
            scratch: IrSynthScratch::new(max_ir_len),
            head: FirHead::new_with_taps(prof.fir_head_taps),
            block_fft_head: BlockFftHead::new(),
            tail,
            hall_fdn: HallFdn::new_with_lines(sr, perspective, prof.hall_fdn_lines),
            limiter: StereoLimiter::new(sr),
        };
        bank.rebuild_ir_and_load();
        bank
    }

    /// Decode serialized `PIANOBOD` bytes and construct a radiator bank.
    pub fn from_preset_bytes(bytes: &[u8], sample_rate: f64) -> Result<Self, PresetDecodeError> {
        Self::from_preset_bytes_with_space(
            bytes,
            sample_rate,
            ListenerPerspective::Player,
            LidState::Open,
        )
    }

    /// Decode serialized `PIANOBOD` bytes with explicit listener perspective and lid state.
    pub fn from_preset_bytes_with_space(
        bytes: &[u8],
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> Result<Self, PresetDecodeError> {
        Self::from_preset_bytes_with_profile(
            bytes,
            sample_rate,
            perspective,
            lid_state,
            &PerfProfile::full(),
        )
    }

    /// Decode serialized `PIANOBOD` bytes with explicit space and [`PerfProfile`].
    pub fn from_preset_bytes_with_profile(
        bytes: &[u8],
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
        profile: &PerfProfile,
    ) -> Result<Self, PresetDecodeError> {
        let preset = BodyPreset::decode(bytes)?;
        Ok(Self::from_preset_with_profile(
            &preset,
            sample_rate,
            perspective,
            lid_state,
            profile,
        ))
    }

    /// Hot-swap the Hall FDN line count (`4`, `8`, or `16`) with zero heap allocations.
    #[inline]
    pub fn set_fdn_lines(&mut self, lines: u8) {
        self.hall_fdn.set_fdn_lines(lines);
        self.profile.hall_fdn_lines = self.hall_fdn.fdn_lines();
    }

    /// Active Hall FDN feedback line count (`4`, `8`, or `16`).
    #[inline]
    #[must_use]
    pub const fn fdn_lines(&self) -> u8 {
        self.hall_fdn.fdn_lines()
    }

    /// Active [`PerfProfile`].
    #[inline]
    #[must_use]
    pub const fn perf_profile(&self) -> &PerfProfile {
        &self.profile
    }

    /// Configure hall reverb decay `[0.0..3.2 s]` and wet mix `[0.0..1.0]`
    /// (zero-alloc, bypassed completely when `mix <= 0.0 || decay_s <= 0.0`).
    #[inline]
    pub fn set_reverb(&mut self, decay_s: f32, mix: f32) {
        self.hall_fdn.set_reverb(decay_s, mix);
    }

    /// Current `(reverb_decay_s, reverb_mix)` parameters.
    #[inline]
    #[must_use]
    pub const fn reverb_params(&self) -> (f32, f32) {
        self.hall_fdn.reverb_params()
    }

    /// Active [`BodyPreset`] loaded into this radiator bank.
    #[inline]
    #[must_use]
    pub fn preset(&self) -> &BodyPreset {
        &self.preset
    }

    pub(crate) fn rebuild_ir_and_load(&mut self) {
        synthesize_irs_with_duration(
            &self.preset,
            self.perspective,
            self.lid_state,
            self.sample_rate,
            f64::from(self.profile.ir_duration_secs),
            &mut self.scratch,
        );
        let ir_len = self.scratch.ir_len;
        if self.fir_block_fft {
            self.block_fft_head
                .load_from_ir_flat(&self.scratch.ir_flat, ir_len);
        } else {
            self.head.load_from_ir_flat(&self.scratch.ir_flat, ir_len);
        }
        self.tail.load_ir_flat(&self.scratch.ir_flat, ir_len);
    }

    /// Process an arbitrary-length slice of 9-zone input frames into `(out_l, out_r)`.
    pub fn process_block(
        &mut self,
        zone_frames: &[[f32; PRESET_RADIATOR_ZONES]],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let total = zone_frames.len().min(out_l.len()).min(out_r.len());
        let mut offset = 0;
        while offset < total {
            let max_step = self.tail.samples_until_slot_end();
            let chunk = (total - offset).min(max_step);
            let z_slice = &zone_frames[offset..offset + chunk];
            let l_slice = &mut out_l[offset..offset + chunk];
            let r_slice = &mut out_r[offset..offset + chunk];

            if self.fir_block_fft {
                self.block_fft_head
                    .process_frames(z_slice, l_slice, r_slice);
            } else {
                self.head.process_frames(z_slice, l_slice, r_slice);
            }
            self.tail.process_subchunk(z_slice, l_slice, r_slice);
            self.hall_fdn.process_block(l_slice, r_slice);
            self.limiter.process_block(l_slice, r_slice);

            offset += chunk;
        }

        if total < out_l.len() {
            out_l[total..].fill(0.0);
        }
        if total < out_r.len() {
            out_r[total..].fill(0.0);
        }
    }

    /// Zero all convolution delay lines, FDN/ER/IACC buffers, limiter state,
    /// and pending NUPOLS rings immediately.
    pub fn reset(&mut self) {
        self.staged_space_stream = None;
        self.head.reset();
        self.block_fft_head.reset();
        self.tail.reset();
        self.hall_fdn.reset();
        self.limiter.reset();
    }

    /// Incrementally zero runtime delay, filter, FDN, and NUPOLS FDL state across
    /// `total_slices` deterministic slices (`slice_idx in 0..total_slices`) without
    /// altering static IR spectra or Stage 4 layout configuration.
    pub fn clear_runtime_state_slice(&mut self, slice_idx: usize, total_slices: usize) {
        if total_slices == 0 || slice_idx >= total_slices {
            return;
        }
        if slice_idx == 0 {
            self.head.reset();
            self.block_fft_head.reset();
            self.limiter.reset();
            self.tail.clear_small_state();
        }
        if slice_idx == (1 % total_slices) {
            self.hall_fdn.reset();
        }
        self.tail.clear_rings_and_fdl_slice(slice_idx, total_slices);
    }

    /// Algorithmic latency in samples (`0` for direct FIR, `32` when `fir_block_fft` is active).
    #[inline]
    pub const fn algorithmic_latency_samples(&self) -> usize {
        if self.fir_block_fft {
            BLOCK_FFT_SIZE
        } else {
            0
        }
    }

    /// Length of the active direct SIMD FIR head in taps (`16`, `32`, or `64`).
    #[inline]
    pub const fn head_taps(&self) -> usize {
        self.head.active_taps()
    }

    /// Active impulse-response length in samples.
    #[inline]
    pub fn ir_len_samples(&self) -> usize {
        self.tail.active_ir_len()
    }

    /// Configured host sample rate in Hz.
    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Current listener perspective.
    #[inline]
    pub fn perspective(&self) -> ListenerPerspective {
        self.perspective
    }

    /// Current piano lid state.
    #[inline]
    pub fn lid_state(&self) -> LidState {
        self.lid_state
    }

    /// Read one synthesized zone-to-ear IR slice (`zone` in `0..9`, `ch` in `0..2`).
    pub fn ir_slice(&self, zone: usize, ch: usize) -> Option<&[f32]> {
        if zone < PRESET_RADIATOR_ZONES && ch < 2 {
            Some(self.scratch.path_slice(zone, ch))
        } else {
            None
        }
    }
}
