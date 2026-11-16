//! 64-mode analytical coupled-form ZOH duplex scale bank (`16` `F32x4` SIMD vectors).
//!
//! Passively and bidirectionally coupled ONLY to treble/tenor bridge **Zones 2..7**
//! via exact WDF port scattering so bass Zones 0..1 remain uncoupled and `E_duplex >= 0` holds.

use piano_simd::F32x4;

use crate::math::{coupled_pole_f32, exp_f64};
use crate::zone_proj::{zone_stereo_pan, NUM_BRIDGE_ZONES, TREBLE_DUPLEX_ZONE_START};

/// Total analytical duplex scale resonant modes (`16 * 4 = 64`).
pub const NUM_DUPLEX_MODES: usize = 64;

/// Number of `F32x4` SIMD vectors in `DuplexBank`.
pub const NUM_DUPLEX_VECS: usize = 16;

/// Zone assignment (`2..=7`) for each of the 16 `F32x4` duplex mode vectors.
pub const DUPLEX_VEC_ZONES: [usize; NUM_DUPLEX_VECS] =
    [2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7];

const ZONE_FREQ_BANDS_HZ: [(f64, f64); 6] = [
    (1_400.0, 2_800.0),  // Zone 2 (8 modes)
    (1_900.0, 3_900.0),  // Zone 3 (8 modes)
    (2_500.0, 5_800.0),  // Zone 4 (12 modes)
    (3_600.0, 7_800.0),  // Zone 5 (12 modes)
    (5_100.0, 10_000.0), // Zone 6 (12 modes)
    (6_800.0, 12_000.0), // Zone 7 (12 modes)
];

const DUPLEX_TARGET_R_PER_ZONE: f64 = 0.018;
const DUPLEX_AUDIO_GAIN: f32 = 0.30;

/// 64-mode SIMD coupled-form ZOH duplex scale resonator bank (`Zones 2..7`).
#[derive(Clone, Debug)]
pub struct DuplexBank {
    u: [F32x4; NUM_DUPLEX_VECS],
    v: [F32x4; NUM_DUPLEX_VECS],
    d: [F32x4; NUM_DUPLEX_VECS],
    s: [F32x4; NUM_DUPLEX_VECS],
    beta: [F32x4; NUM_DUPLEX_VECS],
    c_audio: [F32x4; NUM_DUPLEX_VECS],
    freqs_hz: [f32; NUM_DUPLEX_MODES],
    r_dup: [f64; NUM_BRIDGE_ZONES],
    alpha_dup: f64,
    ts: f64,
    last_stereo: (f32, f32),
}

impl DuplexBank {
    /// Constructs default passive `DuplexBank` (`duplex_gain = 1.0`, damped).
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_params(sample_rate, 1.0, false)
    }

    /// Constructs `DuplexBank` with explicit coupling gain and lossless flag (`zeta = 0`).
    #[must_use]
    pub fn new_with_params(sample_rate: f32, duplex_gain: f32, lossless: bool) -> Self {
        let safe_sr = f64::from(if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        });
        let ts = 1.0 / safe_sr;
        let alpha_dup = 1.0 / ts;
        let gain = f64::from(if duplex_gain.is_finite() {
            duplex_gain.max(0.0)
        } else {
            0.0
        });

        let mut counts_per_zone = [0usize; NUM_BRIDGE_ZONES];
        for &z in &DUPLEX_VEC_ZONES {
            counts_per_zone[z] += 4;
        }

        let mut d = [F32x4::splat(1.0); NUM_DUPLEX_VECS];
        let mut s = [F32x4::zero(); NUM_DUPLEX_VECS];
        let mut beta = [F32x4::zero(); NUM_DUPLEX_VECS];
        let mut c_audio = [F32x4::zero(); NUM_DUPLEX_VECS];
        let mut freqs_hz = [0.0_f32; NUM_DUPLEX_MODES];
        let mut r_dup = [0.0_f64; NUM_BRIDGE_ZONES];
        let mut seen_in_zone = [0usize; NUM_BRIDGE_ZONES];

        for vec_idx in 0..NUM_DUPLEX_VECS {
            let z = DUPLEX_VEC_ZONES[vec_idx];
            let (f_min, f_max) = ZONE_FREQ_BANDS_HZ[z - TREBLE_DUPLEX_ZONE_START];
            let total_in_z = counts_per_zone[z];
            let beta_mag = if gain > 0.0 {
                let per_mode = 2.0 * DUPLEX_TARGET_R_PER_ZONE * ts / (total_in_z as f64);
                gain * crate::math::sqrt_f64(per_mode)
            } else {
                0.0
            };

            let (mut d_arr, mut s_arr, mut b_arr, mut c_arr) =
                ([1.0_f32; 4], [0.0_f32; 4], [0.0_f32; 4], [0.0_f32; 4]);
            let ratio_ln = if f_max > f_min {
                let mut x = f_max / f_min;
                // Compute ln(f_max/f_min) via 64-step Halley/midpoint or simple series:
                let mut ln_approx = 0.0_f64;
                while x > 2.0 {
                    x *= 0.5;
                    ln_approx += core::f64::consts::LN_2;
                }
                let u = (x - 1.0) / (x + 1.0);
                let u2 = u * u;
                ln_approx + 2.0 * u * (1.0 + u2 / 3.0 + u2 * u2 / 5.0 + u2 * u2 * u2 / 7.0)
            } else {
                0.0
            };

            for lane in 0..4 {
                let m_in_z = seen_in_zone[z];
                seen_in_zone[z] += 1;
                let frac = (m_in_z as f64) / ((total_in_z - 1).max(1) as f64);
                let jitter = 1.0 + 0.015 * (((m_in_z * 7 + z * 3) % 5) as f64 - 2.0) * 0.5;
                let f_hz = (f_min * exp_f64(frac * ratio_ln) * jitter).min(0.47 * safe_sr);
                freqs_hz[vec_idx * 4 + lane] = f_hz as f32;

                let t60 = if lossless {
                    f64::INFINITY
                } else {
                    0.55 / (1.0 + 0.12 * (f_hz * 1.0e-3))
                };
                if let Some((df, sf)) = coupled_pole_f32(f_hz, safe_sr, t60) {
                    d_arr[lane] = df;
                    s_arr[lane] = sf;
                    let sign = if (m_in_z & 1) == 0 { 1.0 } else { -1.0 };
                    let b_val = (sign * beta_mag) as f32;
                    b_arr[lane] = b_val;
                    c_arr[lane] = (sign * f64::from(DUPLEX_AUDIO_GAIN)) as f32;
                    r_dup[z] += 0.5 * alpha_dup * f64::from(b_val) * f64::from(b_val);
                }
            }
            d[vec_idx] = F32x4::from_array(d_arr);
            s[vec_idx] = F32x4::from_array(s_arr);
            beta[vec_idx] = F32x4::from_array(b_arr);
            c_audio[vec_idx] = F32x4::from_array(c_arr);
        }

        Self {
            u: [F32x4::zero(); NUM_DUPLEX_VECS],
            v: [F32x4::zero(); NUM_DUPLEX_VECS],
            d,
            s,
            beta,
            c_audio,
            freqs_hz,
            r_dup,
            alpha_dup,
            ts,
            last_stereo: (0.0, 0.0),
        }
    }

    /// Resets all 64 duplex resonator states and cached stereo output to zero.
    pub fn reset(&mut self) {
        self.u = [F32x4::zero(); NUM_DUPLEX_VECS];
        self.v = [F32x4::zero(); NUM_DUPLEX_VECS];
        self.last_stereo = (0.0, 0.0);
    }

    /// Returns passive WDF port resistances on the 8 bridge zones (`0.0` on bass Zones 0..2).
    #[must_use]
    #[inline]
    pub const fn port_resistances(&self) -> &[f64; NUM_BRIDGE_ZONES] {
        &self.r_dup
    }

    /// Computes instantaneous free forces `[N]` exerted by duplex modes onto Zones 2..7.
    #[must_use]
    #[inline]
    pub fn free_forces(&self) -> [f64; NUM_BRIDGE_ZONES] {
        let mut f_free = [0.0_f64; NUM_BRIDGE_ZONES];
        for (vec_idx, &z) in DUPLEX_VEC_ZONES.iter().enumerate() {
            let sum = (self.v[vec_idx] * self.beta[vec_idx]).reduce_sum();
            f_free[z] += self.alpha_dup * f64::from(sum);
        }
        f_free
    }

    /// Steps all 64 SIMD duplex modes coupled to solved bridge zone velocities `v_zone[2..8]`,
    /// writes finite per-zone duplex contributions into `dup_zone_out` (with bass Zones `0..2`
    /// strictly `0.0`), and returns the new `(left, right)` stereo output.
    #[inline]
    pub fn step_with_zone_velocities_and_zones(
        &mut self,
        v_zone: &[f32; NUM_BRIDGE_ZONES],
        dup_zone_out: &mut [f32; NUM_BRIDGE_ZONES],
    ) -> (f32, f32) {
        let mut zone_acc = [0.0_f32; NUM_BRIDGE_ZONES];
        for (vec_idx, &z) in DUPLEX_VEC_ZONES.iter().enumerate() {
            let vz = v_zone[z];
            let v_br = F32x4::splat(if vz.is_finite() { vz } else { 0.0 });
            let v_coupled = self.v[vec_idx] - self.beta[vec_idx] * v_br;

            let u0 = self.u[vec_idx];
            let d0 = self.d[vec_idx];
            let s0 = self.s[vec_idx];
            let du = d0.mul_add(u0, s0 * v_coupled);
            let dv = s0 * u0 - d0 * v_coupled;
            let un = u0 - du;
            let vn = v_coupled + dv;
            self.u[vec_idx] = un;
            self.v[vec_idx] = vn;

            zone_acc[z] += (vn * self.c_audio[vec_idx]).reduce_sum();
        }

        dup_zone_out[..TREBLE_DUPLEX_ZONE_START].fill(0.0);
        let (mut out_l, mut out_r) = (0.0_f32, 0.0_f32);
        for (z, (&acc, dst)) in zone_acc
            .iter()
            .zip(dup_zone_out.iter_mut())
            .enumerate()
            .skip(TREBLE_DUPLEX_ZONE_START)
        {
            let safe_acc = if acc.is_finite() { acc } else { 0.0 };
            *dst = safe_acc;
            let (pan_l, pan_r) = zone_stereo_pan(z);
            out_l += safe_acc * pan_l;
            out_r += safe_acc * pan_r;
        }
        self.last_stereo = (out_l, out_r);
        (out_l, out_r)
    }

    /// Steps all 64 SIMD duplex modes coupled to solved bridge zone velocities `v_zone[2..8]`
    /// and returns the new `(left, right)` stereo output.
    #[inline]
    pub fn step_with_zone_velocities(&mut self, v_zone: &[f32; NUM_BRIDGE_ZONES]) -> (f32, f32) {
        let mut unused_zones = [0.0_f32; NUM_BRIDGE_ZONES];
        self.step_with_zone_velocities_and_zones(v_zone, &mut unused_zones)
    }

    /// Returns the most recent `(left, right)` stereo duplex output.
    #[must_use]
    #[inline]
    pub const fn stereo_out(&self) -> (f32, f32) {
        self.last_stereo
    }

    /// Returns total instantaneous duplex modal energy `E_duplex >= 0` in Joules.
    #[must_use]
    pub fn energy_joules(&self) -> f64 {
        let mut sum_sq = 0.0_f64;
        for vec_idx in 0..NUM_DUPLEX_VECS {
            let ua = self.u[vec_idx].to_array();
            let va = self.v[vec_idx].to_array();
            for lane in 0..4 {
                let u64v = f64::from(ua[lane]);
                let v64v = f64::from(va[lane]);
                sum_sq += u64v * u64v + v64v * v64v;
            }
        }
        0.5 * self.alpha_dup * self.ts * sum_sq
    }

    /// Returns analytical resonant frequency in Hz for mode `mode_idx` (`0..64`).
    #[must_use]
    pub fn mode_freq_hz(&self, mode_idx: usize) -> f32 {
        self.freqs_hz.get(mode_idx).copied().unwrap_or(0.0)
    }
}
