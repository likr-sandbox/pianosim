//! Stage 2 <-> Stage 3 Wave Digital Filter (WDF) displacement-bridge port
//! precomputation, L1d-packed prefix tables, and contiguous tail SoA (`no_std`).

use alloc::boxed::Box;
use core::ops::Range;

use piano_simd::F32x4;
use piano_spec::{partial_frequency_hz, NUM_KEYS, TOTAL_STRINGS};

use crate::modal_soa::{sqrt_f64, ModalSoa, C_HORIZ, C_VERT};

/// Alternating spatial mode sign `[+C_VERT, -C_VERT, +C_VERT, -C_VERT]` for vertical prefix vectors.
pub(crate) const C_SIGN_V_ARR: [f32; 4] = [C_VERT, -C_VERT, C_VERT, -C_VERT];
/// Alternating spatial mode sign `[+C_HORIZ, -C_HORIZ, +C_HORIZ, -C_HORIZ]` for horizontal prefix vectors.
pub(crate) const C_SIGN_H_ARR: [f32; 4] = [C_HORIZ, -C_HORIZ, C_HORIZ, -C_HORIZ];

/// Stage 3 bidirectional reduced bridge and duplex scale driver (`0` runtime allocs).
pub trait BridgePortDriver {
    /// Per-string WDF mechanical port resistances `R_0,s > 0` (`N*s/m`).
    fn port_resistances(&self) -> &[f32; TOTAL_STRINGS];

    /// Partition-of-unity 8-zone bridge projection `(z0, w0, z1, w1)` per string.
    fn zone_weights(&self) -> &[(u8, f32, u8, f32); TOTAL_STRINGS];

    /// Advances bridge admittance, 8-zone velocity projection, and duplex bank by 1 sample.
    fn step_bridge_sample(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
    ) -> (f32, f32);

    /// Advances bridge admittance, 8-zone velocity projection, and per-zone duplex bank (`Zones 2..7`).
    #[inline]
    fn step_bridge_sample_with_duplex_zones(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; 8],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; 8],
        dup_zone_out: &mut [f32; 8],
    ) -> (f32, f32) {
        dup_zone_out.fill(0.0);
        self.step_bridge_sample(a_waves, zone_str_audio, b_waves_out, zone_out)
    }
}

/// 32-byte L1d-packed coupled-form modal state `(u, v)` for one 4-mode prefix vector.
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct PrefixState {
    pub u: F32x4,
    pub v: F32x4,
}

/// 48-byte L1d-packed read-only ZOH poles + WDF port vector `(d, s, beta)` (`c` is analytic!).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug)]
pub(crate) struct PrefixCoef {
    pub d: F32x4,
    pub s: F32x4,
    pub beta: F32x4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct FastPrefixMeta {
    pub a_wave_gain: f32,
    pub pan_l: f32,
    pub pan_r: f32,
}

#[rustfmt::skip]
impl FastPrefixMeta {
    pub const ZERO: Self = Self { a_wave_gain: 0.0, pan_l: 0.0, pan_r: 0.0 };
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PrefixRun {
    pub start_s: u8,
    pub count: u8,
    pub n_pol: u8,
    pub p_off: u16,
}

#[rustfmt::skip]
impl PrefixRun {
    pub const ZERO: Self = Self { start_s: 0, count: 0, n_pol: 0, p_off: 0 };
}

pub(crate) const MAX_PREFIX_RUNS: usize = 48;

/// Per-key contiguous ranges in `BridgePorts::tl_soa` across all unison strings of key `k`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct KeyTailRanges {
    pub th_end: u16,
    pub l_end: u16,
    pub pan_l: f32,
    pub pan_r: f32,
    pub tail_n_trans: f32,
    pub tail_n_long: f32,
}

#[rustfmt::skip]
impl KeyTailRanges {
    pub const ZERO: Self = Self {
        th_end: 0, l_end: 0, pan_l: 0.0, pan_r: 0.0, tail_n_trans: 0.0, tail_n_long: 0.0,
    };
}

/// Precomputed per-string prefix/tail slice bounds and WDF bridge port weights.
#[derive(Clone, Debug)]
pub(crate) struct StringBridgeMeta {
    pub v_br: Range<usize>,
    pub h_br: Range<usize>,
    pub v_tail: Range<usize>,
    pub h_tail: Range<usize>,
    pub tl_v: Range<usize>,
    pub tl_h: Range<usize>,
    pub tl_l: Range<usize>,
    pub beta_v_off: usize,
    pub beta_h_off: usize,
    pub n_vh: usize,
    pub a_wave_gain: f32,
}

#[rustfmt::skip]
impl StringBridgeMeta {
    pub(crate) const fn empty() -> Self {
        Self {
            v_br: 0..0, h_br: 0..0, v_tail: 0..0, h_tail: 0..0, tl_v: 0..0, tl_h: 0..0,
            tl_l: 0..0, beta_v_off: 0, beta_h_off: 0, n_vh: 0, a_wave_gain: 0.0,
        }
    }
}

/// Precomputed L1d-resident WDF prefix state/coefficients and key-contiguous tail `ModalSoa`.
#[derive(Clone, Debug)]
pub(crate) struct BridgePorts {
    pub state: Box<[PrefixState]>,
    pub coef: Box<[PrefixCoef]>,
    pub damp_bd: Box<[F32x4]>,
    pub damp_ad: Box<[F32x4]>,
    pub tl_soa: ModalSoa,
    pub fast_meta: [FastPrefixMeta; TOTAL_STRINGS],
    pub a_wave_gains: [f32; TOTAL_STRINGS],
    pub pan_l: [f32; TOTAL_STRINGS],
    pub pan_r: [f32; TOTAL_STRINGS],
    pub prefix_runs: [PrefixRun; MAX_PREFIX_RUNS],
    pub n_prefix_runs: usize,
    pub key_tails: [KeyTailRanges; NUM_KEYS],
    pub r0: [f32; TOTAL_STRINGS],
    pub r_damp_prefix_v: [f32; TOTAL_STRINGS],
    pub r_damp_prefix_h: [f32; TOTAL_STRINGS],
    pub tail_n_trans: [f32; TOTAL_STRINGS],
    pub tail_n_long: [f32; TOTAL_STRINGS],
    pub meta: [StringBridgeMeta; TOTAL_STRINGS],
}

pub(crate) fn count_prefix_modes(
    key_idx: usize,
    modes: usize,
    f0: f64,
    inharmonicity_b: f64,
    nyquist_hz: f64,
) -> usize {
    let (cutoff_hz, max_j) = if key_idx < 16 {
        (1_250.0_f64, 24usize)
    } else if key_idx < 28 {
        (2_500.0_f64, 12usize)
    } else if key_idx < 40 {
        (3_800.0_f64, 8usize)
    } else {
        (6_000.0_f64, 4usize)
    };
    let lim = cutoff_hz.min(nyquist_hz);
    let mut n = 0usize;
    for j in 1..=modes.min(max_j) {
        let f_j = partial_frequency_hz(f0, inharmonicity_b, j);
        if f_j > 0.0 && f_j < lim {
            n = j;
        } else {
            break;
        }
    }
    n
}

pub(crate) fn fill_pol_beta(
    out: &mut [f32],
    modes_in_prefix: usize,
    gain_64: f64,
    inharmonicity_b: f64,
) -> f64 {
    let mut sum_sq = 0.0_f64;
    for j in 1..=modes_in_prefix {
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        let jf = j as f64;
        let b_f32 = (sign * gain_64 * sqrt_f64(1.0 + inharmonicity_b * jf * jf)) as f32;
        out[j - 1] = b_f32;
        let bf = f64::from(b_f32);
        sum_sq += bf * bf;
    }
    sum_sq
}
