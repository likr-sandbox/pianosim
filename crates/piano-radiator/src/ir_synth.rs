//! Sample-rate adaptive `0.5 s` 18-path (`9 zones x L/R`) body IR synthesizer
//! combining `modal_synth` with 3D projection, direct `h[0]` radiation, and
//! exact discrete trapezoidal Stage 3 bridge pole cancellation.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use piano_spec::{BodyPreset, LidState, ListenerPerspective, PRESET_RADIATOR_ZONES};

use crate::binaural_hrtf::{apply_causal_fractional_delay, Biquad};
use crate::math::{clamp_f64, round_f64, sin_cos_f64, PI};
use crate::mic_perspectives::{compute_all_zone_paths, ChannelPathSpec};
use crate::modal_synth::{synthesize_all_modes, ModalWorkspace};
use crate::pole_cancel_filter::apply_stage3_pole_cancellation;
use crate::velvet_tail::inject_velvet_wood_tail;

pub(crate) use crate::modal_synth::{path_index, NUM_IR_PATHS};

/// Fixed physical IR duration synthesized at any host sample rate `[seconds]`.
pub(crate) const IR_DURATION_SECS: f64 = 0.5;
const MIN_IR_DURATION_SECS: f64 = 0.05;

/// Compute IR sample length for `sample_rate` and `ir_duration_secs` (`0.05..=0.50 s`, min 256).
#[inline]
pub fn ir_len_for_sample_rate_secs(sample_rate: f64, ir_duration_secs: f64) -> usize {
    let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
    let dur = if ir_duration_secs.is_finite() {
        clamp_f64(ir_duration_secs, MIN_IR_DURATION_SECS, IR_DURATION_SECS)
    } else {
        IR_DURATION_SECS
    };
    (round_f64(dur * sr) as usize).clamp(256, 192_000)
}

/// Compute IR sample length for `sample_rate` (`0.5 s`, minimum 256 samples).
#[inline]
pub fn ir_len_for_sample_rate(sample_rate: f64) -> usize {
    ir_len_for_sample_rate_secs(sample_rate, IR_DURATION_SECS)
}

/// Reusable scratch storage so `reconfigure_space` allocates zero heap memory
/// once `ir_len` is initialized.
#[derive(Clone)]
pub struct IrSynthScratch {
    pub ir_len: usize,
    pub ir_flat: Vec<f32>,
    modal_ws: ModalWorkspace,
    delay_scratch: Vec<f32>,
    refl_scratch: Vec<f32>,
}

impl IrSynthScratch {
    pub fn new(ir_len: usize) -> Self {
        Self {
            ir_len,
            ir_flat: vec![0.0_f32; NUM_IR_PATHS * ir_len],
            modal_ws: ModalWorkspace::new(ir_len),
            delay_scratch: vec![0.0_f32; ir_len],
            refl_scratch: vec![0.0_f32; ir_len],
        }
    }

    pub(crate) fn ensure_len(&mut self, ir_len: usize) {
        self.ir_len = ir_len;
        let total = NUM_IR_PATHS * ir_len;
        if self.ir_flat.len() != total {
            self.ir_flat.resize(total, 0.0);
        } else {
            self.ir_flat.fill(0.0);
        }
        if self.delay_scratch.len() != ir_len {
            self.delay_scratch.resize(ir_len, 0.0);
        }
        if self.refl_scratch.len() != ir_len {
            self.refl_scratch.resize(ir_len, 0.0);
        }
    }

    #[inline]
    pub(crate) fn path_slice(&self, zone: usize, ch: usize) -> &[f32] {
        let start = path_index(zone, ch) * self.ir_len;
        &self.ir_flat[start..start + self.ir_len]
    }
}

/// Synthesize all 18 spatialized and pole-cancelled IRs into `scratch.ir_flat`.
pub fn synthesize_irs(
    preset: &BodyPreset,
    perspective: ListenerPerspective,
    lid_state: LidState,
    sample_rate: f64,
    scratch: &mut IrSynthScratch,
) {
    synthesize_irs_with_duration(
        preset,
        perspective,
        lid_state,
        sample_rate,
        IR_DURATION_SECS,
        scratch,
    );
}

/// Synthesize all 18 spatialized and pole-cancelled IRs with explicit duration (`0.05..=0.50 s`).
pub fn synthesize_irs_with_duration(
    preset: &BodyPreset,
    perspective: ListenerPerspective,
    lid_state: LidState,
    sample_rate: f64,
    ir_duration_secs: f64,
    scratch: &mut IrSynthScratch,
) {
    let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
    let ir_len = ir_len_for_sample_rate_secs(sr, ir_duration_secs);
    scratch.ensure_len(ir_len);

    let paths = compute_all_zone_paths(preset, perspective, lid_state, sr);
    synthesize_all_modes(
        preset,
        &paths,
        sr,
        ir_len,
        &mut scratch.modal_ws,
        &mut scratch.ir_flat,
    );

    inject_velvet_wood_tail(
        preset,
        &paths,
        sr,
        ir_len,
        &mut scratch.refl_scratch,
        &mut scratch.ir_flat,
    );

    // Inject calibrated zero-latency `h[0]` direct radiation, physical reflections,
    // acoustic baffle/air/HRTF filtering, and causal fractional path/ITD delays.
    let mut z = 0;
    while z < PRESET_RADIATOR_ZONES {
        let z_direct = if preset.direct_zone_gain[z].is_finite() {
            preset.direct_zone_gain[z]
        } else {
            0.0
        };
        let l_start = path_index(z, 0) * ir_len;
        apply_channel_post(
            &mut scratch.ir_flat[l_start..l_start + ir_len],
            z_direct,
            &paths[z].left,
            sr,
            &mut scratch.refl_scratch,
            &mut scratch.delay_scratch,
        );
        let r_start = path_index(z, 1) * ir_len;
        apply_channel_post(
            &mut scratch.ir_flat[r_start..r_start + ir_len],
            z_direct,
            &paths[z].right,
            sr,
            &mut scratch.refl_scratch,
            &mut scratch.delay_scratch,
        );
        z += 1;
    }

    // Apply exact discrete trapezoidal Stage 3 pole cancellation on bridge zones 0..8
    // for Left and Right independently (zone 8 keybed passes through directly).
    apply_stage3_pole_cancellation(preset, sr, ir_len, &mut scratch.ir_flat, 0);
    apply_stage3_pole_cancellation(preset, sr, ir_len, &mut scratch.ir_flat, 1);

    // Apply smooth cosine fade-out taper on final 25 ms (or 1/16 of IR) to avoid truncation clicks.
    let taper_len = (round_f64(0.025 * sr) as usize).clamp(16, ir_len / 4);
    apply_end_taper(ir_len, taper_len, &mut scratch.ir_flat);
}

fn apply_channel_post(
    ir: &mut [f32],
    zone_direct_gain: f32,
    spec: &ChannelPathSpec,
    sample_rate: f64,
    refl_scratch: &mut [f32],
    delay_scratch: &mut [f32],
) {
    let n_len = ir.len();
    if n_len == 0 {
        return;
    }

    let direct_val = 20.0 * zone_direct_gain * spec.direct_gain;
    if direct_val != 0.0 {
        delay_scratch[0] = direct_val;
        delay_scratch[1..n_len].fill(0.0);
        let tilt_lo = Biquad::high_shelf(165.0, -3.0, 0.55, sample_rate);
        let tilt_hi = Biquad::high_shelf(920.0, -8.0, 0.65, sample_rate);
        let tilt_air = Biquad::high_shelf(3_180.0, -16.7, 0.85, sample_rate);
        let mid_dip = Biquad::peaking_eq(385.0, 1.24, -8.6, sample_rate);
        Biquad::process_chain_in_place(
            &[tilt_lo, tilt_hi, tilt_air, mid_dip],
            &mut delay_scratch[..n_len],
        );
        let mut i = 0;
        while i < n_len {
            ir[i] += delay_scratch[i];
            i += 1;
        }
    }

    if spec.lid_refl_gain != 0.0 || spec.hall_refl_gain != 0.0 {
        refl_scratch[..n_len].copy_from_slice(ir);
        add_fractional_echo(
            ir,
            &refl_scratch[..n_len],
            spec.lid_refl_delay_samples,
            spec.lid_refl_gain,
        );
        add_fractional_echo(
            ir,
            &refl_scratch[..n_len],
            spec.hall_refl_delay_samples,
            spec.hall_refl_gain,
        );
    }

    Biquad::process_chain_in_place(
        &[
            spec.lid_lowpass,
            spec.lid_shelf,
            spec.air_filter,
            spec.hrtf_head_shadow,
            spec.hrtf_concha,
            spec.hrtf_pinna_notch,
        ],
        ir,
    );

    if spec.delay_samples > 1.0e-5 {
        apply_causal_fractional_delay(ir, delay_scratch, spec.delay_samples);
    }
}

fn add_fractional_echo(dst: &mut [f32], src: &[f32], delay_samples: f64, gain: f32) {
    if gain == 0.0 || delay_samples < 1.0 {
        return;
    }
    let d_int = delay_samples as usize;
    let frac = (delay_samples - (d_int as f64)) as f32;
    let w0 = gain * (1.0 - frac);
    let w1 = gain * frac;
    let mut n = d_int;
    while n < dst.len() {
        dst[n] += w0 * src[n - d_int];
        if n > d_int {
            dst[n] += w1 * src[n - d_int - 1];
        }
        n += 1;
    }
}

fn apply_end_taper(ir_len: usize, taper_len: usize, ir_flat: &mut [f32]) {
    if taper_len == 0 || taper_len > ir_len {
        return;
    }
    let start = ir_len - taper_len;
    let denom = taper_len as f64;
    let mut k = 0;
    while k < taper_len {
        let phase = PI * ((k + 1) as f64) / denom;
        let (_, cos_p) = sin_cos_f64(phase);
        let gain = (0.5 * (1.0 + cos_p)) as f32;
        let mut p = 0;
        while p < NUM_IR_PATHS {
            ir_flat[p * ir_len + start + k] *= gain;
            p += 1;
        }
        k += 1;
    }
}
