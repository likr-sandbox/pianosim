//! Per-key damper-rail clearance tile stepping, decimated control-rate interpolation,
//! peel-off adhesion pulse generation, and continuous sustain clearance curve.

use piano_spec::DAMPED_KEYS;

use crate::math::sin_cos_f64;
use crate::pedals::{
    PedalBank, CLEARANCE_OPEN_M, CLEARANCE_PRELOAD_M, HALF_PEDAL_HIGH, HALF_PEDAL_HI_CLEARANCE_M,
    HALF_PEDAL_LOW, PEEL_ARM_CLEARANCE_M, PEEL_MAX_PEAK_N, PEEL_MIN_PEAK_N, SETTLE_EPS_M,
};

#[inline]
pub(crate) fn snap_control_decim(decim: u8) -> u8 {
    if decim <= 1 {
        1
    } else if decim <= 3 {
        2
    } else if decim <= 6 {
        4
    } else if decim <= 12 {
        8
    } else {
        16
    }
}

#[inline]
pub(crate) fn alpha_for_decim(alpha_1x: f32, decim: u8) -> f32 {
    let one_m = 1.0 - alpha_1x;
    match decim {
        1 => alpha_1x,
        2 => 1.0 - one_m * one_m,
        4 => {
            let p2 = one_m * one_m;
            1.0 - p2 * p2
        }
        8 => {
            let p2 = one_m * one_m;
            let p4 = p2 * p2;
            1.0 - p4 * p4
        }
        _ => {
            let p2 = one_m * one_m;
            let p4 = p2 * p2;
            let p8 = p4 * p4;
            1.0 - p8 * p8
        }
    }
}

#[inline]
fn alpha_for_step(alpha_1x: f32, step: usize) -> f32 {
    let one_m = 1.0 - alpha_1x;
    match step {
        0 => 0.0,
        1 => alpha_1x,
        2 => 1.0 - one_m * one_m,
        3 => 1.0 - one_m * one_m * one_m,
        4 => {
            let sq = one_m * one_m;
            1.0 - sq * sq
        }
        _ => 1.0 - crate::damper_pad_symp::pow_usize(one_m, step),
    }
}

#[inline]
fn eval_peel_hann2(pos: u16, peel_len: u16, peak_n: f32) -> f32 {
    let u = (f64::from(pos) + 0.5) / f64::from(peel_len);
    let (_, cos_tau) = sin_cos_f64(core::f64::consts::TAU * u);
    let c = 0.5 * (1.0 - cos_tau);
    peak_n * ((c * c) as f32)
}

impl PedalBank {
    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and per-sample peel-off adhesion
    /// forces (`N`) into `out_peel_n[..n]`. Returns `(min_clear_m, any_peel_nonzero)`.
    #[inline]
    pub fn step_key_clearance_tile_with_peel(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
        out_peel_n: &mut [f32; 8],
    ) -> (f32, bool) {
        let n = t_len.min(8);
        if key_idx >= DAMPED_KEYS {
            out_clears[..n].fill(CLEARANCE_OPEN_M);
            out_peel_n[..n].fill(0.0);
            return (CLEARANCE_OPEN_M, false);
        }
        if n == 0 {
            return (self.clearance_m[key_idx], false);
        }
        let peel_active = self.peel_pos[key_idx] < self.peel_len;
        if !self.settling[key_idx] && !peel_active {
            let cur = self.clearance_m[key_idx];
            out_clears[..n].fill(cur);
            out_peel_n[..n].fill(0.0);
            if self.last_peel_n[key_idx] != 0.0 {
                self.last_peel_n[key_idx] = 0.0;
            }
            return (cur, false);
        }

        let target = if self.key_down[key_idx] || self.sostenuto_latched[key_idx] {
            CLEARANCE_OPEN_M
        } else {
            sustain_clearance_m(self.sustain, self.release_preload_m[key_idx])
        };
        let mut cur = self.clearance_m[key_idx];
        let mut min_clear = f32::INFINITY;
        let mut settled = !self.settling[key_idx];
        let mut any_peel = false;

        let last_peel = if self.control_decim <= 1 {
            let alpha = if target >= cur {
                self.alpha_lift
            } else {
                self.alpha_down[key_idx]
            };
            let mut last_p = 0.0_f32;
            for i in 0..n {
                let prev = cur;
                if settled || (target - cur).abs() < SETTLE_EPS_M {
                    cur = target;
                    settled = true;
                } else {
                    cur += alpha * (target - cur);
                    if (target - cur).abs() < SETTLE_EPS_M {
                        cur = target;
                        settled = true;
                    }
                }
                out_clears[i] = cur;
                min_clear = min_clear.min(cur);
                self.update_peel_trigger(key_idx, prev, cur);

                let f_peel = if self.peel_pos[key_idx] < self.peel_len {
                    let f = eval_peel_hann2(
                        self.peel_pos[key_idx],
                        self.peel_len,
                        self.peel_peak_n[key_idx],
                    );
                    self.peel_pos[key_idx] += 1;
                    any_peel |= f != 0.0;
                    f
                } else {
                    0.0
                };
                out_peel_n[i] = f_peel;
                last_p = f_peel;
            }
            last_p
        } else {
            let (alpha_1x, alpha_d) = if target >= cur {
                (self.alpha_lift, self.alpha_lift_d)
            } else {
                (self.alpha_down[key_idx], self.alpha_down_d[key_idx])
            };
            let decim = usize::from(self.control_decim);
            let mut prev_peel = self.last_peel_n[key_idx];
            let mut b = 0_usize;
            while b < n {
                let step = decim.min(n - b);
                let cur_prev = cur;
                if settled || (target - cur).abs() < SETTLE_EPS_M {
                    cur = target;
                    settled = true;
                } else {
                    let a_step = if step == decim {
                        alpha_d
                    } else {
                        alpha_for_step(alpha_1x, step)
                    };
                    cur += a_step * (target - cur);
                    if (target - cur).abs() < SETTLE_EPS_M {
                        cur = target;
                        settled = true;
                    }
                }
                let inv_step = 1.0 / (step as f32);
                let dc = cur - cur_prev;
                for j in 0..step {
                    let c_j = if j + 1 == step {
                        cur
                    } else {
                        cur_prev + (((j + 1) as f32) * inv_step) * dc
                    };
                    out_clears[b + j] = c_j;
                    min_clear = min_clear.min(c_j);
                }
                self.update_peel_trigger(key_idx, cur_prev, cur);

                let f_peel_end = if self.peel_pos[key_idx] < self.peel_len {
                    let rem = self.peel_len - self.peel_pos[key_idx];
                    let adv = (step as u16).min(rem);
                    let next_pos = self.peel_pos[key_idx] + adv;
                    self.peel_pos[key_idx] = next_pos;
                    if next_pos < self.peel_len {
                        eval_peel_hann2(next_pos - 1, self.peel_len, self.peel_peak_n[key_idx])
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                if prev_peel == 0.0 && f_peel_end == 0.0 {
                    out_peel_n[b..b + step].fill(0.0);
                } else {
                    let dp = f_peel_end - prev_peel;
                    for j in 0..step {
                        let f_j = if j + 1 == step {
                            f_peel_end
                        } else {
                            prev_peel + (((j + 1) as f32) * inv_step) * dp
                        };
                        out_peel_n[b + j] = f_j;
                        any_peel |= f_j != 0.0;
                    }
                }
                prev_peel = f_peel_end;
                b += step;
            }
            prev_peel
        };

        self.clearance_m[key_idx] = cur;
        self.last_peel_n[key_idx] = last_peel;
        if settled {
            self.settling[key_idx] = false;
        }
        (min_clear, any_peel)
    }

    #[inline]
    fn update_peel_trigger(&mut self, key_idx: usize, prev: f32, cur: f32) {
        if self.peel_played[key_idx] && cur <= PEEL_ARM_CLEARANCE_M {
            self.peel_armed[key_idx] = true;
            self.peel_min_clear_m[key_idx] = self.peel_min_clear_m[key_idx].min(cur);
        } else if self.peel_armed[key_idx] && prev < 0.0 && cur >= 0.0 {
            let x = (key_idx as f32) / ((DAMPED_KEYS - 1) as f32);
            let depth = (-self.peel_min_clear_m[key_idx] / (-CLEARANCE_PRELOAD_M)).clamp(0.25, 1.0);
            let peak = PEEL_MIN_PEAK_N
                + (PEEL_MAX_PEAK_N - PEEL_MIN_PEAK_N)
                    * (0.30 + 0.70 * (1.0 - x))
                    * (0.55 + 0.45 * depth);
            self.peel_peak_n[key_idx] = peak.clamp(PEEL_MIN_PEAK_N, PEEL_MAX_PEAK_N);
            self.peel_pos[key_idx] = 0;
            self.peel_armed[key_idx] = false;
            self.peel_min_clear_m[key_idx] = 0.0;
        }
    }

    /// Advances the 1-pole damper rail smoother across an 8-sample tile (`n = t_len.min(8)`),
    /// storing per-sample clearances into `out_clears[..n]` and returning the tile minimum.
    #[inline]
    pub fn step_key_clearance_tile(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; 8],
    ) -> f32 {
        let n = t_len.min(8);
        if key_idx >= DAMPED_KEYS {
            out_clears[..n].fill(CLEARANCE_OPEN_M);
            return CLEARANCE_OPEN_M;
        }
        if n == 0 {
            return self.clearance_m[key_idx];
        }
        if !self.settling[key_idx] && self.peel_pos[key_idx] >= self.peel_len {
            let cur = self.clearance_m[key_idx];
            out_clears[..n].fill(cur);
            if self.last_peel_n[key_idx] != 0.0 {
                self.last_peel_n[key_idx] = 0.0;
            }
            return cur;
        }
        let mut tmp_peel = [0.0_f32; 8];
        self.step_key_clearance_tile_with_peel(key_idx, t_len, out_clears, &mut tmp_peel)
            .0
    }

    /// Advances the 1-pole damper rail smoother for `key_idx` by one host sample.
    #[inline]
    pub fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        let mut tmp = [0.0_f32; 8];
        self.step_key_clearance_tile(key_idx, 1, &mut tmp)
    }
}

/// Smooth strictly monotonic sustain-pedal lift curve mapping `s in [0.0, 1.0]`
/// to damper clearance in meters:
/// - `0.0 -> preload_m` (`-0.30 mm` nominal)
/// - `0.5 -> +32 um` (`+0.032 mm` half-pedal grazing zone)
/// - `1.0 -> +5.0 mm` (open gap)
#[must_use]
#[inline]
pub fn sustain_clearance_m(sustain: f32, preload_m: f32) -> f32 {
    let s = sustain.clamp(0.0, 1.0);
    if s <= HALF_PEDAL_LOW {
        let t = s / HALF_PEDAL_LOW;
        let smooth = 0.25 * t + 0.75 * (t * t * (3.0 - 2.0 * t));
        preload_m * (1.0 - smooth)
    } else if s <= HALF_PEDAL_HIGH {
        let t = (s - HALF_PEDAL_LOW) / (HALF_PEDAL_HIGH - HALF_PEDAL_LOW);
        t * HALF_PEDAL_HI_CLEARANCE_M
    } else {
        let span = 1.0 - HALF_PEDAL_HIGH;
        let ds = s - HALF_PEDAL_HIGH;
        let t = ds / span;
        let slope_lo = HALF_PEDAL_HI_CLEARANCE_M / (HALF_PEDAL_HIGH - HALF_PEDAL_LOW);
        let linear = HALF_PEDAL_HI_CLEARANCE_M + slope_lo * ds;
        linear + (CLEARANCE_OPEN_M - HALF_PEDAL_HI_CLEARANCE_M - slope_lo * span) * (t * t)
    }
}
