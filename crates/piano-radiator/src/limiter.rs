//! Zero-latency stereo-linked `C^2` soft-ceiling safety limiter (`|out| <= 1.0`,
//! non-finite -> `0.0`).

use crate::math::{clamp_f64, exp_f64};

/// Linear pass-through threshold; signals whose peak and envelope stay at or
/// below this level pass through with exact `1.0` unity gain.
pub(crate) const LINEAR_THRESHOLD: f32 = 0.91;
const HEADROOM: f32 = 1.0 - LINEAR_THRESHOLD;
const DETECTOR_CEILING: f32 = 4.0;

const ATTACK_SECS: f64 = 0.0002; // 0.2 ms
const SLOW_ATTACK_SECS: f64 = 0.012; // 12 ms integration for slow release stage
const RELEASE_FAST_SECS: f64 = 0.035; // 35 ms
const RELEASE_SLOW_SECS: f64 = 0.180; // 180 ms

/// Evaluate the `C^2`-continuous soft ceiling curve `f(u)` for `u >= 0`:
/// - `f(u) = u` for `u <= LINEAR_THRESHOLD`
/// - `f(T) = T`, `f'(T+) = 1`, `f''(T+) = 0`, `lim_{u->inf} f(u) = 1.0`.
#[inline]
pub(crate) fn soft_ceiling_curve(u: f32) -> f32 {
    if u <= LINEAR_THRESHOLD {
        return u;
    }
    let t = (u - LINEAR_THRESHOLD) / HEADROOM;
    let t2 = t * t;
    let phi = (t + t2) / (1.0 + t + t2);
    let shaped = LINEAR_THRESHOLD + HEADROOM * phi;
    if shaped > 1.0 {
        1.0
    } else {
        shaped
    }
}

/// Stateful zero-latency stereo-linked peak limiter with 0.2 ms attack,
/// 2-stage (35 ms / 180 ms) release, and `C^2` soft ceiling.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StereoLimiter {
    att_coeff: f32,
    att_slow_coeff: f32,
    rel_fast_coeff: f32,
    rel_slow_coeff: f32,
    env_fast: f32,
    env_slow: f32,
}

impl StereoLimiter {
    pub(crate) fn new(sample_rate: f64) -> Self {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        Self {
            att_coeff: (1.0 - exp_f64(-1.0 / (ATTACK_SECS * sr))) as f32,
            att_slow_coeff: (1.0 - exp_f64(-1.0 / (SLOW_ATTACK_SECS * sr))) as f32,
            rel_fast_coeff: (1.0 - exp_f64(-1.0 / (RELEASE_FAST_SECS * sr))) as f32,
            rel_slow_coeff: (1.0 - exp_f64(-1.0 / (RELEASE_SLOW_SECS * sr))) as f32,
            env_fast: 0.0,
            env_slow: 0.0,
        }
    }

    #[inline]
    pub(crate) fn reset(&mut self) {
        self.env_fast = 0.0;
        self.env_slow = 0.0;
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn envelope(&self) -> f32 {
        self.env_fast.max(self.env_slow)
    }

    /// Process `(out_l, out_r)` in-place with zero algorithmic latency and a shared
    /// stereo-linked gain factor.
    #[inline]
    pub(crate) fn process_block(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        let n = out_l.len().min(out_r.len());
        let att_f = self.att_coeff;
        let att_s = self.att_slow_coeff;
        let rel_f = self.rel_fast_coeff;
        let rel_s = self.rel_slow_coeff;
        let mut ef = self.env_fast;
        let mut es = self.env_slow;

        let mut i = 0;
        while i < n {
            let l_in = out_l[i];
            let r_in = out_r[i];
            let l = if l_in.is_finite() { l_in } else { 0.0 };
            let r = if r_in.is_finite() { r_in } else { 0.0 };

            let al = if l < 0.0 { -l } else { l };
            let ar = if r < 0.0 { -r } else { r };
            let peak = if al > ar { al } else { ar };
            let det = if peak > DETECTOR_CEILING {
                DETECTOR_CEILING
            } else {
                peak
            };

            let c_f = if det > ef { att_f } else { rel_f };
            ef += c_f * (det - ef);
            let c_s = if det > es { att_s } else { rel_s };
            es += c_s * (det - es);

            let env = if ef > es { ef } else { es };
            let ctrl = if peak > env { peak } else { env };

            if ctrl <= LINEAR_THRESHOLD {
                out_l[i] = l;
                out_r[i] = r;
            } else {
                let gain = soft_ceiling_curve(ctrl) / ctrl;
                out_l[i] = (l * gain).clamp(-1.0, 1.0);
                out_r[i] = (r * gain).clamp(-1.0, 1.0);
            }
            i += 1;
        }

        if ef < 1.0e-20 {
            ef = 0.0;
        }
        if es < 1.0e-20 {
            es = 0.0;
        }
        self.env_fast = ef;
        self.env_slow = es;
    }
}
