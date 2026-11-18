//! Stage 4 & Layer 1B: sample-rate adaptive IR synthesis and zero-latency
//! 9-zone MIMO Staggered NUPOLS acoustic radiator (`9 zones -> L/R`).

#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod binaural_hrtf;
mod body_assets;
mod fir_head;
mod hall_fdn;
mod ir_synth;
mod limiter;
mod math;
mod mic_perspectives;
mod modal_prepare;
mod modal_synth;
mod nupols_tier;
mod partitioned_fft;
mod pole_cancel_filter;
mod space_blob;
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
mod spectral_tests;
#[cfg(test)]
mod tests;

use alloc::vec::Vec;
pub use body_assets::{body_asset_filename, embedded_body_bytes, BodyArchetype};
use core::fmt;
use fir_head::{FirHead, HEAD_TAPS};
use hall_fdn::HallFdn;
pub use ir_synth::{ir_len_for_sample_rate, synthesize_irs, IrSynthScratch};
use limiter::StereoLimiter;
use math::clamp_f64;
use partitioned_fft::NupolsTail;
pub use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PresetDecodeError, PRESET_RADIATOR_ZONES,
};
use piano_spec::{PresetReducedPole, PRESET_BRIDGE_ZONES, PRESET_REDUCED_POLES};
pub use space_blob::{space_blob_len_for_sample_rate, SpacePrepContext};

/// Deterministic physics-baked 9-zone concert-grand `PIANOBOD` asset.
pub const CONCERT_GRAND_PRESET_BYTES: &[u8] = include_bytes!("../assets/concert_grand_v1.pbdy");

/// Stage 4 acoustic radiator combining a 64-tap zero-latency `F32x4` direct FIR
/// head with a zero-allocation Staggered MIMO NUPOLS convolution tail.
#[derive(Clone)]
pub struct RadiatorBank {
    preset: BodyPreset,
    sample_rate: f64,
    perspective: ListenerPerspective,
    lid_state: LidState,
    scratch: IrSynthScratch,
    head: FirHead,
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
            .field("ir_len", &self.scratch.ir_len)
            .field("modes", &self.preset.modes.len())
            .finish()
    }
}

fn empty_fallback_preset() -> BodyPreset {
    BodyPreset {
        y_inf: [1.8e-4; PRESET_BRIDGE_ZONES],
        direct_zone_gain: [0.0; PRESET_RADIATOR_ZONES],
        zone_pos_m: [[0.0; 3]; PRESET_RADIATOR_ZONES],
        reduced_poles: [PresetReducedPole {
            freq_hz: 100.0,
            zeta: 0.02,
            modal_mass_kg: 0.0,
            zone_gains: [0.0; PRESET_BRIDGE_ZONES],
        }; PRESET_REDUCED_POLES],
        modes: Vec::new(),
    }
}

impl RadiatorBank {
    /// Construct a radiator bank from the embedded `concert_grand_v1.pbdy` preset
    /// with default `Player` perspective and `Open` lid state.
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
        let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES)
            .unwrap_or_else(|_| empty_fallback_preset());
        Self::from_preset_with_space(&preset, sample_rate, perspective, lid_state)
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
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let ir_len = ir_len_for_sample_rate(sr);
        let mut bank = Self {
            preset: preset.clone(),
            sample_rate: sr,
            perspective,
            lid_state,
            scratch: IrSynthScratch::new(ir_len),
            head: FirHead::new(),
            tail: NupolsTail::new(ir_len),
            hall_fdn: HallFdn::new(sr, perspective),
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
        let preset = BodyPreset::decode(bytes)?;
        Ok(Self::from_preset_with_space(
            &preset,
            sample_rate,
            perspective,
            lid_state,
        ))
    }

    /// Re-synthesize all 18 spatialized IRs and reload FIR/NUPOLS partitions in-place.
    pub fn reconfigure_space(&mut self, perspective: ListenerPerspective, lid_state: LidState) {
        if self.perspective == perspective && self.lid_state == lid_state {
            return;
        }
        self.perspective = perspective;
        self.lid_state = lid_state;
        self.hall_fdn.set_perspective(perspective);
        self.rebuild_ir_and_load();
    }

    /// Decode serialized `PIANOBOD` bytes, switch the active body preset, and
    /// re-synthesize FIR/NUPOLS partitions in-place.
    pub fn reconfigure_from_preset_bytes(
        &mut self,
        bytes: &[u8],
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> Result<(), PresetDecodeError> {
        let preset = BodyPreset::decode(bytes)?;
        self.reconfigure_from_preset(preset, perspective, lid_state);
        Ok(())
    }

    /// Switch the active body preset and re-synthesize FIR/NUPOLS partitions in-place.
    pub fn reconfigure_from_preset(
        &mut self,
        preset: BodyPreset,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) {
        self.preset = preset;
        self.perspective = perspective;
        self.lid_state = lid_state;
        self.hall_fdn.set_perspective(perspective);
        self.rebuild_ir_and_load();
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

    fn rebuild_ir_and_load(&mut self) {
        synthesize_irs(
            &self.preset,
            self.perspective,
            self.lid_state,
            self.sample_rate,
            &mut self.scratch,
        );
        let ir_len = self.scratch.ir_len;
        self.head.load_from_ir_flat(&self.scratch.ir_flat, ir_len);
        self.tail.load_ir_flat(&self.scratch.ir_flat, ir_len);
    }

    /// Process an arbitrary-length slice of 9-zone input frames with zero allocations
    /// and exact `0`-sample algorithmic latency into `(out_l, out_r)`.
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

            self.head.process_frames(z_slice, l_slice, r_slice);
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
        self.head.reset();
        self.tail.reset();
        self.hall_fdn.reset();
        self.limiter.reset();
    }

    /// Algorithmic latency in samples (always exact `0` via split FIR + NUPOLS).
    #[inline]
    pub const fn algorithmic_latency_samples(&self) -> usize {
        0
    }

    /// Length of the zero-latency direct SIMD FIR head in taps.
    #[inline]
    pub const fn head_taps(&self) -> usize {
        HEAD_TAPS
    }

    /// Active impulse-response length in samples (`0.5 s * sample_rate`).
    #[inline]
    pub fn ir_len_samples(&self) -> usize {
        self.scratch.ir_len
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

    /// Synthesize all 18 spatialized/pole-cancelled IRs (`[zone][L/R][n]`) for
    /// offline inspection and regression tests.
    pub fn synthesize_ir_for_test(
        preset: &BodyPreset,
        sample_rate: f64,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> [[Vec<f32>; 2]; PRESET_RADIATOR_ZONES] {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let ir_len = ir_len_for_sample_rate(sr);
        let mut scratch = IrSynthScratch::new(ir_len);
        synthesize_irs(preset, perspective, lid_state, sr, &mut scratch);
        core::array::from_fn(|z| {
            [
                scratch.path_slice(z, 0).to_vec(),
                scratch.path_slice(z, 1).to_vec(),
            ]
        })
    }
}
