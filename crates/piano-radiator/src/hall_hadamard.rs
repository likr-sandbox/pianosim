//! Orthogonal 4x4 / 8x8 / 16x16 Walsh-Hadamard transforms, prime delay
//! helpers, and listener perspective tuning for [`crate::hall_fdn::HallFdn`].

use piano_simd::F32x4;
use piano_spec::ListenerPerspective;

pub(crate) const MAX_FDN_LINES: usize = 16;
pub(crate) const NUM_ER_TAPS: usize = 6;

pub(crate) const INV_SQRT_4: f32 = 0.5;
pub(crate) const INV_SQRT_8: f32 = 0.353_553_38;
pub(crate) const INV_SQRT_16: f32 = 0.25;

pub(crate) const FDN_BASE_DELAYS_S: [f64; MAX_FDN_LINES] = [
    0.0233, 0.0289, 0.0351, 0.0427, 0.0513, 0.0619, 0.0731, 0.0853, 0.0923, 0.0991, 0.1063, 0.1139,
    0.1213, 0.1289, 0.1361, 0.1439,
];

pub(crate) const ER_BASE_DELAYS_S: [f64; NUM_ER_TAPS] =
    [0.0073, 0.0117, 0.0169, 0.0231, 0.0317, 0.0423];

#[inline]
pub(crate) const fn snap_fdn_lines(lines: u8) -> usize {
    if lines <= 6 {
        4
    } else if lines <= 12 {
        8
    } else {
        16
    }
}

#[inline]
pub(crate) fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut d = 5usize;
    while d * d <= n {
        if n.is_multiple_of(d) || n.is_multiple_of(d + 2) {
            return false;
        }
        d += 6;
    }
    true
}

#[inline]
pub(crate) fn next_prime_above(mut n: usize, min_exclusive: usize) -> usize {
    if n <= min_exclusive {
        n = min_exclusive + 1;
    }
    while !is_prime(n) {
        n += 1;
    }
    n
}

#[inline]
pub(crate) fn next_pow2(mut n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n + 1
}

#[inline]
fn walsh_4_unscaled(x: F32x4) -> F32x4 {
    let sign2 = F32x4::from_array([1.0, 1.0, -1.0, -1.0]);
    let w = x * sign2 + x.slide2(x);
    let lo = w.unpack_lo(w);
    let hi = w.unpack_hi(w);
    lo.unpack_lo(hi) + lo.unpack_hi(hi) * sign2
}

#[inline]
fn walsh_8_unscaled(lo: F32x4, hi: F32x4) -> (F32x4, F32x4) {
    (walsh_4_unscaled(lo + hi), walsh_4_unscaled(lo - hi))
}

/// 4x4 orthogonal Walsh-Hadamard transform on one `F32x4` vector (`H^T H = I`).
#[inline]
pub(crate) fn hadamard_4(x: F32x4) -> F32x4 {
    walsh_4_unscaled(x) * F32x4::splat(INV_SQRT_4)
}

/// 8x8 orthogonal Walsh-Hadamard transform on two `F32x4` vectors (`H^T H = I`).
#[inline]
pub(crate) fn hadamard_8(lo: F32x4, hi: F32x4) -> (F32x4, F32x4) {
    let scale = F32x4::splat(INV_SQRT_8);
    let (y_lo, y_hi) = walsh_8_unscaled(lo, hi);
    (y_lo * scale, y_hi * scale)
}

/// 16x16 orthogonal Walsh-Hadamard transform on four `F32x4` vectors (`H^T H = I`).
#[inline]
pub(crate) fn hadamard_16(v: [F32x4; 4]) -> [F32x4; 4] {
    let scale = F32x4::splat(INV_SQRT_16);
    let (y0, y1) = walsh_8_unscaled(v[0] + v[2], v[1] + v[3]);
    let (y2, y3) = walsh_8_unscaled(v[0] - v[2], v[1] - v[3]);
    [y0 * scale, y1 * scale, y2 * scale, y3 * scale]
}

#[derive(Clone, Copy)]
pub(crate) struct PerspectiveTuning {
    pub(crate) er_time_scale: f64,
    pub(crate) er_gain_scale: f32,
    pub(crate) fdn_gain_scale: f32,
    pub(crate) er_tap_gains_l: [f32; NUM_ER_TAPS],
    pub(crate) er_tap_gains_r: [f32; NUM_ER_TAPS],
}

pub(crate) fn perspective_tuning(perspective: ListenerPerspective) -> PerspectiveTuning {
    match perspective {
        ListenerPerspective::Player => PerspectiveTuning {
            er_time_scale: 0.72,
            er_gain_scale: 0.24,
            fdn_gain_scale: 0.42,
            er_tap_gains_l: [0.38, 0.22, 0.30, 0.16, 0.20, 0.12],
            er_tap_gains_r: [0.22, 0.38, 0.18, 0.28, 0.14, 0.19],
        },
        ListenerPerspective::CloseStereo => PerspectiveTuning {
            er_time_scale: 0.58,
            er_gain_scale: 0.16,
            fdn_gain_scale: 0.30,
            er_tap_gains_l: [0.34, 0.24, 0.26, 0.18, 0.16, 0.11],
            er_tap_gains_r: [0.24, 0.34, 0.18, 0.26, 0.12, 0.15],
        },
        ListenerPerspective::ConcertHall => PerspectiveTuning {
            er_time_scale: 1.15,
            er_gain_scale: 0.34,
            fdn_gain_scale: 0.58,
            er_tap_gains_l: [0.40, 0.28, 0.34, 0.24, 0.26, 0.18],
            er_tap_gains_r: [0.28, 0.40, 0.24, 0.34, 0.18, 0.26],
        },
        ListenerPerspective::BinauralHrtf => PerspectiveTuning {
            er_time_scale: 0.88,
            er_gain_scale: 0.28,
            fdn_gain_scale: 0.48,
            er_tap_gains_l: [0.42, 0.18, 0.31, 0.15, 0.22, 0.12],
            er_tap_gains_r: [0.18, 0.42, 0.15, 0.31, 0.12, 0.22],
        },
    }
}
