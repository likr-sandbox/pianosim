//! Sample-rate adaptive `0.5 s` 18-path (`9 zones x L/R`) body IR synthesizer
//! combining `modal_synth` with 3D projection, direct `h[0]` radiation, and
//! exact discrete trapezoidal Stage 3 bridge pole cancellation.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use piano_spec::{
    BodyPreset, LidState, ListenerPerspective, PRESET_BRIDGE_ZONES, PRESET_RADIATOR_ZONES,
};

use crate::binaural_hrtf::{apply_causal_fractional_delay, Biquad};
use crate::math::{clamp_f64, exp_f64, invert_10x10, min_f64, round_f64, sin_cos_f64, PI};
use crate::mic_perspectives::{compute_all_zone_paths, ChannelPathSpec, ZonePathPair};
use crate::modal_synth::{synthesize_all_modes, ModalWorkspace};

pub(crate) use crate::modal_synth::{path_index, NUM_IR_PATHS};

/// Fixed physical IR duration synthesized at any host sample rate `[seconds]`.
pub(crate) const IR_DURATION_SECS: f64 = 0.5;
const NUM_REDUCED_POLES: usize = 10;
const VELVET_START_SECS: f64 = 0.005;
const VELVET_END_SECS: f64 = 0.485;
const VELVET_PULSE_RATE_HZ: f64 = 2_400.0;
const VELVET_DECAY_TAU_SECS: f64 = 0.085;
const VELVET_ATTACK_RAMP_SECS: f64 = 0.004;
const VELVET_HPF_HZ: f64 = 3_800.0;
const VELVET_BASE_GAIN: f64 = 0.0022;

/// Compute IR sample length for `sample_rate` (`0.5 s`, minimum 256 samples).
#[inline]
pub fn ir_len_for_sample_rate(sample_rate: f64) -> usize {
    let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
    (round_f64(IR_DURATION_SECS * sr) as usize).clamp(256, 192_000)
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
    let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
    let ir_len = ir_len_for_sample_rate(sr);
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
            &mut scratch.refl_scratch,
            &mut scratch.delay_scratch,
        );
        let r_start = path_index(z, 1) * ir_len;
        apply_channel_post(
            &mut scratch.ir_flat[r_start..r_start + ir_len],
            z_direct,
            &paths[z].right,
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

#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Injects a `(zone, ch)`-uncorrelated `> 3.5 kHz` sparse velvet-noise wood
/// diffusion tail into `ir_flat` with sample-rate-normalized energy and density.
fn inject_velvet_wood_tail(
    preset: &BodyPreset,
    paths: &[ZonePathPair; PRESET_RADIATOR_ZONES],
    sr: f64,
    ir_len: usize,
    refl_scratch: &mut [f32],
    ir_flat: &mut [f32],
) {
    if preset.modes.is_empty() || VELVET_HPF_HZ >= 0.44 * sr {
        return;
    }
    let onset_idx = (round_f64(VELVET_START_SECS * sr) as usize).clamp(8, ir_len);
    if onset_idx + 16 >= ir_len {
        return;
    }

    let dt_cell = 1.0 / VELVET_PULSE_RATE_HZ;
    let num_cells =
        round_f64((VELVET_END_SECS - VELVET_START_SECS) * VELVET_PULSE_RATE_HZ) as usize;
    let hpf = Biquad::highpass_2nd(VELVET_HPF_HZ, core::f64::consts::FRAC_1_SQRT_2, sr);
    let hpf_chain = [hpf, hpf];

    let mut z = 0;
    while z < PRESET_RADIATOR_ZONES {
        let zone_trim = if z < 8 { 1.0 } else { 0.35 };
        let mut ch = 0;
        while ch < 2 {
            let proj0 = if ch == 0 {
                paths[z].left.mode_proj[0]
            } else {
                paths[z].right.mode_proj[0]
            };
            let amp_base = VELVET_BASE_GAIN * zone_trim * (proj0 as f64);
            if amp_base > 0.0 {
                let tail_buf = &mut refl_scratch[onset_idx..ir_len];
                tail_buf.fill(0.0);

                let mut rng = 0x7665_6c76_6574_5f31_u64
                    ^ ((z as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    ^ ((ch as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
                let _ = splitmix64_next(&mut rng);
                let _ = splitmix64_next(&mut rng);

                let mut m = 0;
                while m < num_cells {
                    let r_pos = splitmix64_next(&mut rng);
                    let r_sgn = splitmix64_next(&mut rng);
                    let u = ((r_pos >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
                    let sign = if (r_sgn & 1) == 0 { 1.0 } else { -1.0 };
                    let t_rel = ((m as f64) + u) * dt_cell;
                    let idx = round_f64((VELVET_START_SECS + t_rel) * sr) as usize;
                    if idx >= onset_idx && idx < ir_len {
                        let attack = min_f64(1.0, t_rel / VELVET_ATTACK_RAMP_SECS);
                        let env = attack * exp_f64(-t_rel / VELVET_DECAY_TAU_SECS);
                        refl_scratch[idx] += (sign * amp_base * env) as f32;
                    }
                    m += 1;
                }

                Biquad::process_chain_in_place(&hpf_chain, &mut refl_scratch[onset_idx..ir_len]);

                let dst_base = path_index(z, ch) * ir_len;
                let dst = &mut ir_flat[dst_base + onset_idx..dst_base + ir_len];
                let src = &refl_scratch[onset_idx..ir_len];
                let mut i = 0;
                while i < dst.len() {
                    dst[i] += src[i];
                    i += 1;
                }
            }
            ch += 1;
        }
        z += 1;
    }
}

fn apply_channel_post(
    ir: &mut [f32],
    zone_direct_gain: f32,
    spec: &ChannelPathSpec,
    refl_scratch: &mut [f32],
    delay_scratch: &mut [f32],
) {
    let n_len = ir.len();
    if n_len == 0 {
        return;
    }

    ir[0] += zone_direct_gain * spec.direct_gain;

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

/// Exact discrete trapezoidal pole cancellation `H_total(z) = Z_red_norm(z)^T * H_rad(z)`
/// for bridge zones `0..8` at host `sample_rate`.
fn apply_stage3_pole_cancellation(
    preset: &BodyPreset,
    sample_rate: f64,
    ir_len: usize,
    ir_flat: &mut [f32],
    ch: usize,
) {
    let mut y_inf_inv = [0.0_f64; PRESET_BRIDGE_ZONES];
    let mut z = 0;
    while z < PRESET_BRIDGE_ZONES {
        let y = preset.y_inf[z] as f64;
        if !y.is_finite() || y <= 1.0e-12 {
            return;
        }
        y_inf_inv[z] = 1.0 / y;
        z += 1;
    }

    let h = 0.5 / sample_rate;
    let mut k_stiff = [0.0_f64; NUM_REDUCED_POLES];
    let mut d_plus = [1.0_f64; NUM_REDUCED_POLES];
    let mut d_minus = [1.0_f64; NUM_REDUCED_POLES];
    let mut phi = [[0.0_f64; PRESET_BRIDGE_ZONES]; NUM_REDUCED_POLES];
    let mut any_active = false;

    let mut m = 0;
    while m < NUM_REDUCED_POLES {
        let pole = &preset.reduced_poles[m];
        let freq = pole.freq_hz as f64;
        let zeta = pole.zeta as f64;
        let mass = pole.modal_mass_kg as f64;
        if freq.is_finite()
            && zeta.is_finite()
            && mass.is_finite()
            && freq > 0.5
            && mass > 1.0e-12
            && zeta >= 0.0
        {
            any_active = true;
            let omega = 2.0 * PI * freq;
            let cm = 2.0 * zeta * omega * mass;
            let km = omega * omega * mass;
            k_stiff[m] = km;
            d_plus[m] = mass + h * cm + h * h * km;
            d_minus[m] = mass - h * cm - h * h * km;
            let mut bz = 0;
            while bz < PRESET_BRIDGE_ZONES {
                let g = pole.zone_gains[bz] as f64;
                phi[m][bz] = if g.is_finite() { g } else { 0.0 };
                bz += 1;
            }
        }
        m += 1;
    }
    if !any_active {
        return;
    }

    let mut a_inv = [[0.0_f64; NUM_REDUCED_POLES]; NUM_REDUCED_POLES];
    let mut i = 0;
    while i < NUM_REDUCED_POLES {
        a_inv[i][i] = d_plus[i];
        let mut j = 0;
        while j < NUM_REDUCED_POLES {
            let mut sum = 0.0_f64;
            let mut bz = 0;
            while bz < PRESET_BRIDGE_ZONES {
                sum += phi[i][bz] * y_inf_inv[bz] * phi[j][bz];
                bz += 1;
            }
            a_inv[i][j] += h * sum;
            j += 1;
        }
        i += 1;
    }

    let Some(a_inv_inv) = invert_10x10(a_inv) else {
        return;
    };

    let mut eta = [0.0_f64; NUM_REDUCED_POLES];
    let mut prev_eta_dot = [0.0_f64; NUM_REDUCED_POLES];
    let mut prev_f = [0.0_f64; PRESET_BRIDGE_ZONES];

    let mut n = 0;
    while n < ir_len {
        let mut x = [0.0_f64; PRESET_BRIDGE_ZONES];
        let mut bz = 0;
        while bz < PRESET_BRIDGE_ZONES {
            x[bz] = ir_flat[path_index(bz, ch) * ir_len + n] as f64;
            bz += 1;
        }

        let mut rhs = [0.0_f64; NUM_REDUCED_POLES];
        let mut mode_i = 0;
        while mode_i < NUM_REDUCED_POLES {
            let mut phi_sum = 0.0_f64;
            let mut bz2 = 0;
            while bz2 < PRESET_BRIDGE_ZONES {
                phi_sum += phi[mode_i][bz2] * (x[bz2] + prev_f[bz2]);
                bz2 += 1;
            }
            rhs[mode_i] = h * phi_sum + d_minus[mode_i] * prev_eta_dot[mode_i]
                - 2.0 * h * k_stiff[mode_i] * eta[mode_i];
            mode_i += 1;
        }

        let mut eta_dot = [0.0_f64; NUM_REDUCED_POLES];
        let mut r = 0;
        while r < NUM_REDUCED_POLES {
            let mut acc = 0.0_f64;
            let mut c = 0;
            while c < NUM_REDUCED_POLES {
                acc += a_inv_inv[r][c] * rhs[c];
                c += 1;
            }
            eta_dot[r] = acc;
            r += 1;
        }

        let mut bz3 = 0;
        while bz3 < PRESET_BRIDGE_ZONES {
            let mut phi_eta_dot = 0.0_f64;
            let mut mi = 0;
            while mi < NUM_REDUCED_POLES {
                phi_eta_dot += phi[mi][bz3] * eta_dot[mi];
                mi += 1;
            }
            let f_val = x[bz3] - y_inf_inv[bz3] * phi_eta_dot;
            ir_flat[path_index(bz3, ch) * ir_len + n] = f_val as f32;
            prev_f[bz3] = f_val;
            bz3 += 1;
        }

        let mut mi2 = 0;
        while mi2 < NUM_REDUCED_POLES {
            eta[mi2] += h * (eta_dot[mi2] + prev_eta_dot[mi2]);
            prev_eta_dot[mi2] = eta_dot[mi2];
            mi2 += 1;
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
