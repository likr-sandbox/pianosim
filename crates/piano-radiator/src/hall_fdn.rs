//! Zero-allocation SIMD 3D early reflections + 8x8 orthogonal Hadamard FDN
//! reverberator with `< 400 Hz` binaural IACC shaping and diffuse-field EQ.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use piano_simd::F32x4;
use piano_spec::ListenerPerspective;

use crate::binaural_hrtf::{Biquad, BiquadState};
use crate::math::{clamp_f64, exp_f64, powf_f64, round_f64, PI};

const NUM_FDN_LINES: usize = 8;
const NUM_ER_TAPS: usize = 6;

pub(crate) const DEFAULT_REVERB_DECAY_S: f32 = 1.8;
pub(crate) const MAX_REVERB_DECAY_S: f32 = 3.2;

const FDN_BASE_DELAYS_S: [f64; NUM_FDN_LINES] = [
    0.0233, 0.0289, 0.0351, 0.0427, 0.0513, 0.0619, 0.0731, 0.0853,
];

const ER_BASE_DELAYS_S: [f64; NUM_ER_TAPS] = [0.0073, 0.0117, 0.0169, 0.0231, 0.0317, 0.0423];

const INV_SQRT_8: f32 = 0.353_553_38;
const FDN_AIR_LPF_HZ: f64 = 5_500.0;
const IACC_CUTOFF_HZ: f64 = 400.0;

#[inline]
fn is_prime(n: usize) -> bool {
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
fn next_prime_above(mut n: usize, min_exclusive: usize) -> usize {
    if n <= min_exclusive {
        n = min_exclusive + 1;
    }
    while !is_prime(n) {
        n += 1;
    }
    n
}

#[inline]
fn next_pow2(mut n: usize) -> usize {
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

/// 8x8 orthogonal Walsh-Hadamard transform on two `F32x4` vectors (`H^T H = I`).
#[inline]
pub(crate) fn hadamard_8(lo: F32x4, hi: F32x4) -> (F32x4, F32x4) {
    let sign2 = F32x4::from_array([1.0, 1.0, -1.0, -1.0]);
    let scale = F32x4::splat(INV_SQRT_8);

    let s_lo = lo + hi;
    let s_hi = lo - hi;

    let w_lo = s_lo * sign2 + s_lo.slide2(s_lo);
    let w_hi = s_hi * sign2 + s_hi.slide2(s_hi);

    let lo_l = w_lo.unpack_lo(w_lo);
    let lo_h = w_lo.unpack_hi(w_lo);
    let y_lo = (lo_l.unpack_lo(lo_h) + lo_l.unpack_hi(lo_h) * sign2) * scale;

    let hi_l = w_hi.unpack_lo(w_hi);
    let hi_h = w_hi.unpack_hi(w_hi);
    let y_hi = (hi_l.unpack_lo(hi_h) + hi_l.unpack_hi(hi_h) * sign2) * scale;
    (y_lo, y_hi)
}

#[derive(Clone, Copy)]
struct PerspectiveTuning {
    er_time_scale: f64,
    er_gain_scale: f32,
    fdn_gain_scale: f32,
    er_tap_gains_l: [f32; NUM_ER_TAPS],
    er_tap_gains_r: [f32; NUM_ER_TAPS],
}

fn perspective_tuning(perspective: ListenerPerspective) -> PerspectiveTuning {
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

/// Real-time zero-allocation 3D early-reflection + 8x8 Hadamard FDN reverberator.
#[derive(Clone)]
pub(crate) struct HallFdn {
    sample_rate: f64,
    perspective: ListenerPerspective,
    reverb_decay_s: f32,
    reverb_mix: f32,

    er_buf_l: Vec<f32>,
    er_buf_r: Vec<f32>,
    er_mask: usize,
    er_pos: usize,
    er_delays: [usize; NUM_ER_TAPS],
    er_gains_l: [F32x4; 2],
    er_gains_r: [F32x4; 2],
    er_level: f32,
    fdn_level: f32,

    fdn_buf: Vec<f32>,
    fdn_offsets: [usize; NUM_FDN_LINES],
    fdn_lens: [usize; NUM_FDN_LINES],
    fdn_pos: [usize; NUM_FDN_LINES],

    fdn_gain_lo: F32x4,
    fdn_gain_hi: F32x4,
    lpf_coeff: F32x4,
    one_minus_lpf: F32x4,
    lpf_lo: F32x4,
    lpf_hi: F32x4,

    is_binaural: bool,
    iacc_mid_shelf: Biquad,
    iacc_side_shelf: Biquad,
    iacc_lp_mid: BiquadState,
    iacc_lp_side: BiquadState,
    diff_eq_peak: Biquad,
    diff_eq_shelf: Biquad,
    diff_eq_peak_l: BiquadState,
    diff_eq_peak_r: BiquadState,
    diff_eq_shelf_l: BiquadState,
    diff_eq_shelf_r: BiquadState,
}

impl HallFdn {
    pub(crate) fn new(sample_rate: f64, perspective: ListenerPerspective) -> Self {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let max_er_samples = (round_f64(0.060 * sr) as usize).max(64);
        let er_len = next_pow2(max_er_samples + 8);
        let er_mask = er_len - 1;

        let mut fdn_lens = [0usize; NUM_FDN_LINES];
        let mut fdn_offsets = [0usize; NUM_FDN_LINES];
        let mut total_fdn = 0usize;
        let mut prev_len = 16usize;
        for i in 0..NUM_FDN_LINES {
            let target = round_f64(FDN_BASE_DELAYS_S[i] * sr) as usize;
            let prime_len = next_prime_above(target, prev_len);
            fdn_offsets[i] = total_fdn;
            fdn_lens[i] = prime_len;
            total_fdn += prime_len;
            prev_len = prime_len;
        }

        let a_lpf = exp_f64(-2.0 * PI * FDN_AIR_LPF_HZ / sr) as f32;
        let iacc_mid_shelf = Biquad::low_shelf(IACC_CUTOFF_HZ, 2.5, 0.70, sr);
        let iacc_side_shelf = Biquad::low_shelf(IACC_CUTOFF_HZ, -18.0, 0.70, sr);

        let mut fdn = Self {
            sample_rate: sr,
            perspective,
            reverb_decay_s: DEFAULT_REVERB_DECAY_S,
            reverb_mix: 0.0,
            er_buf_l: vec![0.0_f32; er_len],
            er_buf_r: vec![0.0_f32; er_len],
            er_mask,
            er_pos: 0,
            er_delays: [1; NUM_ER_TAPS],
            er_gains_l: [F32x4::zero(); 2],
            er_gains_r: [F32x4::zero(); 2],
            er_level: 0.0,
            fdn_level: 0.0,
            fdn_buf: vec![0.0_f32; total_fdn],
            fdn_offsets,
            fdn_lens,
            fdn_pos: [0; NUM_FDN_LINES],
            fdn_gain_lo: F32x4::zero(),
            fdn_gain_hi: F32x4::zero(),
            lpf_coeff: F32x4::splat(a_lpf),
            one_minus_lpf: F32x4::splat(1.0 - a_lpf),
            lpf_lo: F32x4::zero(),
            lpf_hi: F32x4::zero(),
            is_binaural: matches!(perspective, ListenerPerspective::BinauralHrtf),
            iacc_mid_shelf,
            iacc_side_shelf,
            iacc_lp_mid: BiquadState::default(),
            iacc_lp_side: BiquadState::default(),
            diff_eq_peak: Biquad::peaking_eq(3_000.0, 1.1, 2.0, sr),
            diff_eq_shelf: Biquad::high_shelf(8_000.0, -2.5, 0.75, sr),
            diff_eq_peak_l: BiquadState::default(),
            diff_eq_peak_r: BiquadState::default(),
            diff_eq_shelf_l: BiquadState::default(),
            diff_eq_shelf_r: BiquadState::default(),
        };
        fdn.update_perspective_tuning();
        fdn.update_decay_gains();
        fdn
    }

    pub(crate) fn set_perspective(&mut self, perspective: ListenerPerspective) {
        if self.perspective == perspective {
            return;
        }
        self.perspective = perspective;
        self.is_binaural = matches!(perspective, ListenerPerspective::BinauralHrtf);
        self.update_perspective_tuning();
    }

    pub(crate) fn set_reverb(&mut self, decay_s: f32, mix: f32) {
        let was_active = self.is_active();
        let d = if decay_s.is_finite() {
            decay_s.clamp(0.0, MAX_REVERB_DECAY_S)
        } else {
            0.0
        };
        let m = if mix.is_finite() {
            mix.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.reverb_decay_s = d;
        self.reverb_mix = m;
        self.update_decay_gains();
        if was_active && !self.is_active() {
            self.reset();
        }
    }

    #[inline]
    pub(crate) const fn reverb_params(&self) -> (f32, f32) {
        (self.reverb_decay_s, self.reverb_mix)
    }

    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        self.reverb_mix > 0.0 && self.reverb_decay_s > 0.0
    }

    pub(crate) fn reset(&mut self) {
        self.er_buf_l.fill(0.0);
        self.er_buf_r.fill(0.0);
        self.er_pos = 0;
        self.fdn_buf.fill(0.0);
        self.fdn_pos.fill(0);
        self.lpf_lo = F32x4::zero();
        self.lpf_hi = F32x4::zero();
        self.iacc_lp_mid.reset();
        self.iacc_lp_side.reset();
        self.diff_eq_peak_l.reset();
        self.diff_eq_peak_r.reset();
        self.diff_eq_shelf_l.reset();
        self.diff_eq_shelf_r.reset();
    }

    fn update_perspective_tuning(&mut self) {
        let cfg = perspective_tuning(self.perspective);
        let max_delay = self.er_mask;
        for (i, &base_s) in ER_BASE_DELAYS_S.iter().enumerate() {
            let d = (round_f64(base_s * cfg.er_time_scale * self.sample_rate) as usize)
                .clamp(1, max_delay);
            self.er_delays[i] = d;
        }
        let gl = &cfg.er_tap_gains_l;
        let gr = &cfg.er_tap_gains_r;
        self.er_gains_l = [
            F32x4::from_array([gl[0], gl[1], gl[2], gl[3]]),
            F32x4::from_array([gl[4], gl[5], 0.0, 0.0]),
        ];
        self.er_gains_r = [
            F32x4::from_array([gr[0], gr[1], gr[2], gr[3]]),
            F32x4::from_array([gr[4], gr[5], 0.0, 0.0]),
        ];
        self.er_level = cfg.er_gain_scale;
        self.fdn_level = cfg.fdn_gain_scale;
    }

    fn update_decay_gains(&mut self) {
        if self.reverb_decay_s <= 0.0 {
            self.fdn_gain_lo = F32x4::zero();
            self.fdn_gain_hi = F32x4::zero();
            return;
        }
        let decay_s = self.reverb_decay_s as f64;
        let mut g = [0.0_f32; NUM_FDN_LINES];
        for (i, slot) in g.iter_mut().enumerate() {
            let len_s = (self.fdn_lens[i] as f64) / self.sample_rate;
            let gain = powf_f64(10.0, -3.0 * len_s / decay_s).min(0.9995);
            *slot = gain as f32;
        }
        self.fdn_gain_lo = F32x4::from_array([g[0], g[1], g[2], g[3]]);
        self.fdn_gain_hi = F32x4::from_array([g[4], g[5], g[6], g[7]]);
    }

    /// Apply 3D early reflections + 8x8 FDN diffuse reverb in-place with 0-sample latency.
    pub(crate) fn process_block(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        if !self.is_active() {
            return;
        }
        let n = out_l.len().min(out_r.len());
        let dry_gain = 1.0 - 0.20 * self.reverb_mix;
        let wet_gain = self.reverb_mix * 0.72;
        let er_level = self.er_level;
        let fdn_level = self.fdn_level;

        let read_l_lo = F32x4::from_array([0.5, 0.0, -0.5, 0.0]);
        let read_l_hi = F32x4::from_array([0.5, 0.0, -0.5, 0.0]);
        let read_r_lo = F32x4::from_array([0.0, 0.5, 0.0, -0.5]);
        let read_r_hi = F32x4::from_array([0.0, 0.5, 0.0, -0.5]);
        let inj_scale = F32x4::splat(INV_SQRT_8);
        let anti_denorm = F32x4::splat(1.0e-25);

        let er_mask = self.er_mask;
        let mut er_pos = self.er_pos;
        let d0 = self.er_delays[0];
        let d1 = self.er_delays[1];
        let d2 = self.er_delays[2];
        let d3 = self.er_delays[3];
        let d4 = self.er_delays[4];
        let d5 = self.er_delays[5];
        let er_gl0 = self.er_gains_l[0];
        let er_gl1 = self.er_gains_l[1];
        let er_gr0 = self.er_gains_r[0];
        let er_gr1 = self.er_gains_r[1];

        let mut i = 0;
        while i < n {
            let dry_l = out_l[i];
            let dry_r = out_r[i];

            self.er_buf_l[er_pos] = dry_l;
            self.er_buf_r[er_pos] = dry_r;

            let p0 = er_pos.wrapping_sub(d0) & er_mask;
            let p1 = er_pos.wrapping_sub(d1) & er_mask;
            let p2 = er_pos.wrapping_sub(d2) & er_mask;
            let p3 = er_pos.wrapping_sub(d3) & er_mask;
            let p4 = er_pos.wrapping_sub(d4) & er_mask;
            let p5 = er_pos.wrapping_sub(d5) & er_mask;
            er_pos = (er_pos + 1) & er_mask;

            let er_vl0 = F32x4::from_array([
                self.er_buf_l[p0],
                self.er_buf_r[p1],
                self.er_buf_l[p2],
                self.er_buf_r[p3],
            ]);
            let er_vl1 = F32x4::from_array([self.er_buf_l[p4], self.er_buf_r[p5], 0.0, 0.0]);
            let er_l = (er_vl0 * er_gl0 + er_vl1 * er_gl1).reduce_sum() * er_level;

            let er_vr0 = F32x4::from_array([
                self.er_buf_r[p0],
                self.er_buf_l[p1],
                self.er_buf_r[p2],
                self.er_buf_l[p3],
            ]);
            let er_vr1 = F32x4::from_array([self.er_buf_r[p4], self.er_buf_l[p5], 0.0, 0.0]);
            let er_r = (er_vr0 * er_gr0 + er_vr1 * er_gr1).reduce_sum() * er_level;

            let mut x_arr = [0.0_f32; NUM_FDN_LINES];
            let mut k = 0;
            while k < NUM_FDN_LINES {
                x_arr[k] = self.fdn_buf[self.fdn_offsets[k] + self.fdn_pos[k]];
                k += 1;
            }
            let x_lo = F32x4::from_array([x_arr[0], x_arr[1], x_arr[2], x_arr[3]]);
            let x_hi = F32x4::from_array([x_arr[4], x_arr[5], x_arr[6], x_arr[7]]);

            self.lpf_lo = x_lo * self.one_minus_lpf + self.lpf_lo * self.lpf_coeff;
            self.lpf_hi = x_hi * self.one_minus_lpf + self.lpf_hi * self.lpf_coeff;
            let d_lo = self.lpf_lo * self.fdn_gain_lo;
            let d_hi = self.lpf_hi * self.fdn_gain_hi;

            let fdn_l = (d_lo * read_l_lo + d_hi * read_l_hi).reduce_sum() * fdn_level;
            let fdn_r = (d_lo * read_r_lo + d_hi * read_r_hi).reduce_sum() * fdn_level;

            let (y_lo, y_hi) = hadamard_8(d_lo, d_hi);
            let inj_l = er_l + 0.45 * dry_l;
            let inj_r = er_r + 0.45 * dry_r;
            let mid_in = 0.5 * (inj_l + inj_r);
            let side_in = 0.5 * (inj_l - inj_r);
            let in_lo = F32x4::from_array([
                mid_in + side_in,
                mid_in - side_in,
                -mid_in + side_in,
                -mid_in - side_in,
            ]) * inj_scale;
            let in_hi = F32x4::from_array([
                mid_in - side_in,
                -mid_in + side_in,
                mid_in + side_in,
                -mid_in - side_in,
            ]) * inj_scale;

            let w_lo = ((y_lo + in_lo) + anti_denorm) - anti_denorm;
            let w_hi = ((y_hi + in_hi) + anti_denorm) - anti_denorm;
            let w0 = w_lo.to_array();
            let w1 = w_hi.to_array();
            let w_all = [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]];
            let mut m = 0;
            while m < NUM_FDN_LINES {
                let pos = self.fdn_pos[m];
                self.fdn_buf[self.fdn_offsets[m] + pos] = w_all[m];
                let next_pos = pos + 1;
                self.fdn_pos[m] = if next_pos >= self.fdn_lens[m] {
                    0
                } else {
                    next_pos
                };
                m += 1;
            }

            let mut wet_l = er_l + fdn_l;
            let mut wet_r = er_r + fdn_r;
            if self.is_binaural {
                let mid = 0.5 * (wet_l + wet_r);
                let side = 0.5 * (wet_l - wet_r);
                let mid_eq = self.iacc_lp_mid.process(&self.iacc_mid_shelf, mid);
                let side_eq = self.iacc_lp_side.process(&self.iacc_side_shelf, side);
                let adj_l = mid_eq + side_eq;
                let adj_r = mid_eq - side_eq;
                let eq_l = self.diff_eq_peak_l.process(&self.diff_eq_peak, adj_l);
                let eq_r = self.diff_eq_peak_r.process(&self.diff_eq_peak, adj_r);
                wet_l = self.diff_eq_shelf_l.process(&self.diff_eq_shelf, eq_l);
                wet_r = self.diff_eq_shelf_r.process(&self.diff_eq_shelf, eq_r);
            }

            out_l[i] = dry_gain * dry_l + wet_gain * wet_l;
            out_r[i] = dry_gain * dry_r + wet_gain * wet_r;
            i += 1;
        }

        self.er_pos = er_pos;
        self.lpf_lo = (self.lpf_lo + anti_denorm) - anti_denorm;
        self.lpf_hi = (self.lpf_hi + anti_denorm) - anti_denorm;
        self.iacc_lp_mid.flush_denormals();
        self.iacc_lp_side.flush_denormals();
        self.diff_eq_peak_l.flush_denormals();
        self.diff_eq_peak_r.flush_denormals();
        self.diff_eq_shelf_l.flush_denormals();
        self.diff_eq_shelf_r.flush_denormals();
    }
}
