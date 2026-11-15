//! Zero-latency transparent safety limiter (`|out| <= 1.0`, non-finite -> `0.0`).

const LINEAR_THRESHOLD: f32 = 0.92;
const HEADROOM: f32 = 1.0 - LINEAR_THRESHOLD;

/// Apply a zero-latency `C^1` soft limiter to a single sample, sanitizing non-finite inputs.
#[inline]
pub(crate) fn limit_sample(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let ax = if x < 0.0 { -x } else { x };
    if ax <= LINEAR_THRESHOLD {
        return x;
    }
    let excess = ax - LINEAR_THRESHOLD;
    let mag = LINEAR_THRESHOLD + HEADROOM * (excess / (HEADROOM + excess));
    let clamped = if mag > 1.0 { 1.0 } else { mag };
    if x < 0.0 {
        -clamped
    } else {
        clamped
    }
}

/// Sanitize and transparently soft-limit stereo output buffers in place.
#[inline]
pub(crate) fn limit_stereo_buffers(out_l: &mut [f32], out_r: &mut [f32]) {
    let n = out_l.len().min(out_r.len());
    let mut i = 0;
    while i < n {
        out_l[i] = limit_sample(out_l[i]);
        out_r[i] = limit_sample(out_r[i]);
        i += 1;
    }
}
