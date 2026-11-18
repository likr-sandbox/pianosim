//! High-frequency sparse velvet-noise wood diffusion tail synthesis.

use piano_spec::{BodyPreset, PRESET_RADIATOR_ZONES};

use crate::binaural_hrtf::Biquad;
use crate::math::{exp_f64, min_f64, round_f64};
use crate::mic_perspectives::ZonePathPair;
use crate::modal_synth::path_index;

const VELVET_START_SECS: f64 = 0.005;
const VELVET_END_SECS: f64 = 0.485;
const VELVET_PULSE_RATE_HZ: f64 = 2_400.0;
const VELVET_DECAY_TAU_SECS: f64 = 0.085;
const VELVET_ATTACK_RAMP_SECS: f64 = 0.004;
const VELVET_HPF_HZ: f64 = 3_800.0;
pub(crate) const VELVET_BASE_GAIN: f64 = 0.0008;

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
pub(crate) fn inject_velvet_wood_tail(
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
