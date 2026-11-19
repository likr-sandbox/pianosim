//! Zero-allocation SIMD 3D early reflections + configurable 4x4/8x8/16x16 orthogonal
//! Hadamard FDN reverberator with `< 400 Hz` binaural IACC shaping and diffuse-field EQ.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::f32::consts::FRAC_1_SQRT_2;
use piano_simd::F32x4;
use piano_spec::ListenerPerspective;

use crate::binaural_hrtf::{Biquad, BiquadState};
use crate::hall_hadamard::{
    hadamard_16, hadamard_4, hadamard_8, next_pow2, next_prime_above, perspective_tuning,
    snap_fdn_lines, ER_BASE_DELAYS_S, FDN_BASE_DELAYS_S, INV_SQRT_16, INV_SQRT_4, INV_SQRT_8,
    MAX_FDN_LINES, NUM_ER_TAPS,
};
use crate::math::{clamp_f64, exp_f64, powf_f64, round_f64, PI};

pub(crate) const DEFAULT_REVERB_DECAY_S: f32 = 1.8;
pub(crate) const MAX_REVERB_DECAY_S: f32 = 3.2;

const FDN_AIR_LPF_HZ: f64 = 5_500.0;
const IACC_CUTOFF_HZ: f64 = 400.0;
const FDN_VEC_COUNT: usize = MAX_FDN_LINES / 4;

/// Real-time zero-allocation 3D early-reflection + 4x4/8x8/16x16 Hadamard FDN reverberator.
#[derive(Clone)]
pub(crate) struct HallFdn {
    sample_rate: f64,
    perspective: ListenerPerspective,
    active_lines: usize,
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
    fdn_offsets: [usize; MAX_FDN_LINES],
    fdn_lens: [usize; MAX_FDN_LINES],
    fdn_pos: [usize; MAX_FDN_LINES],

    fdn_gains: [F32x4; FDN_VEC_COUNT],
    lpf_coeff: F32x4,
    one_minus_lpf: F32x4,
    lpf_state: [F32x4; FDN_VEC_COUNT],

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
    #[cfg(test)]
    pub(crate) fn new(sample_rate: f64, perspective: ListenerPerspective) -> Self {
        Self::new_with_lines(sample_rate, perspective, 8)
    }

    pub(crate) fn new_with_lines(
        sample_rate: f64,
        perspective: ListenerPerspective,
        lines: u8,
    ) -> Self {
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let active_lines = snap_fdn_lines(lines);
        let max_er_samples = (round_f64(0.060 * sr) as usize).max(64);
        let er_len = next_pow2(max_er_samples + 8);
        let er_mask = er_len - 1;

        let mut fdn_lens = [0usize; MAX_FDN_LINES];
        let mut fdn_offsets = [0usize; MAX_FDN_LINES];
        let mut total_fdn = 0usize;
        let mut prev_len = 16usize;
        for i in 0..MAX_FDN_LINES {
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
            active_lines,
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
            fdn_pos: [0; MAX_FDN_LINES],
            fdn_gains: [F32x4::zero(); FDN_VEC_COUNT],
            lpf_coeff: F32x4::splat(a_lpf),
            one_minus_lpf: F32x4::splat(1.0 - a_lpf),
            lpf_state: [F32x4::zero(); FDN_VEC_COUNT],
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

    #[inline]
    pub(crate) const fn fdn_lines(&self) -> u8 {
        self.active_lines as u8
    }

    pub(crate) fn set_fdn_lines(&mut self, lines: u8) {
        let new_lines = snap_fdn_lines(lines);
        if new_lines == self.active_lines {
            return;
        }
        if new_lines > self.active_lines {
            for i in self.active_lines..new_lines {
                let off = self.fdn_offsets[i];
                let len = self.fdn_lens[i];
                self.fdn_buf[off..off + len].fill(0.0);
                self.fdn_pos[i] = 0;
            }
            for v in (self.active_lines / 4)..(new_lines / 4) {
                self.lpf_state[v] = F32x4::zero();
            }
        }
        self.active_lines = new_lines;
        self.update_decay_gains();
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
        self.lpf_state = [F32x4::zero(); FDN_VEC_COUNT];
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
            self.fdn_gains = [F32x4::zero(); FDN_VEC_COUNT];
            return;
        }
        let decay_s = self.reverb_decay_s as f64;
        let mut g = [0.0_f32; MAX_FDN_LINES];
        for (slot, &len) in g
            .iter_mut()
            .zip(self.fdn_lens.iter())
            .take(self.active_lines)
        {
            let len_s = (len as f64) / self.sample_rate;
            *slot = powf_f64(10.0, -3.0 * len_s / decay_s).min(0.9995) as f32;
        }
        for v in 0..FDN_VEC_COUNT {
            let base = v * 4;
            self.fdn_gains[v] = F32x4::from_array([g[base], g[base + 1], g[base + 2], g[base + 3]]);
        }
    }

    #[inline]
    fn step_fdn_lines(&mut self, inj_l: f32, inj_r: f32, fdn_level: f32) -> (f32, f32) {
        let anti_denorm = F32x4::splat(1.0e-25);
        let mid_in = 0.5 * (inj_l + inj_r);
        let side_in = 0.5 * (inj_l - inj_r);
        let n_lines = self.active_lines;
        let n_vecs = n_lines / 4;

        let mut x_arr = [0.0_f32; MAX_FDN_LINES];
        let mut k = 0;
        while k < n_lines {
            x_arr[k] = self.fdn_buf[self.fdn_offsets[k] + self.fdn_pos[k]];
            k += 1;
        }

        let mut d = [F32x4::zero(); FDN_VEC_COUNT];
        let mut v = 0;
        while v < n_vecs {
            let b = v * 4;
            let xv = F32x4::from_array([x_arr[b], x_arr[b + 1], x_arr[b + 2], x_arr[b + 3]]);
            self.lpf_state[v] = xv * self.one_minus_lpf + self.lpf_state[v] * self.lpf_coeff;
            d[v] = self.lpf_state[v] * self.fdn_gains[v];
            v += 1;
        }

        let (fdn_l, fdn_r, w) = match n_lines {
            4 => {
                let rl = F32x4::from_array([FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2, 0.0]);
                let rr = F32x4::from_array([0.0, FRAC_1_SQRT_2, 0.0, -FRAC_1_SQRT_2]);
                let fl = (d[0] * rl).reduce_sum() * fdn_level;
                let fr = (d[0] * rr).reduce_sum() * fdn_level;
                let y0 = hadamard_4(d[0]);
                let in0 = F32x4::from_array([
                    mid_in + side_in,
                    mid_in - side_in,
                    -mid_in + side_in,
                    -mid_in - side_in,
                ]) * F32x4::splat(INV_SQRT_4);
                let w0 = ((y0 + in0) + anti_denorm) - anti_denorm;
                (fl, fr, [w0, F32x4::zero(), F32x4::zero(), F32x4::zero()])
            }
            8 => {
                let rl = F32x4::from_array([0.5, 0.0, -0.5, 0.0]);
                let rr = F32x4::from_array([0.0, 0.5, 0.0, -0.5]);
                let fl = (d[0] * rl + d[1] * rl).reduce_sum() * fdn_level;
                let fr = (d[0] * rr + d[1] * rr).reduce_sum() * fdn_level;
                let (y0, y1) = hadamard_8(d[0], d[1]);
                let scale = F32x4::splat(INV_SQRT_8);
                let in0 = F32x4::from_array([
                    mid_in + side_in,
                    mid_in - side_in,
                    -mid_in + side_in,
                    -mid_in - side_in,
                ]) * scale;
                let in1 = F32x4::from_array([
                    mid_in - side_in,
                    -mid_in + side_in,
                    mid_in + side_in,
                    -mid_in - side_in,
                ]) * scale;
                let w0 = ((y0 + in0) + anti_denorm) - anti_denorm;
                let w1 = ((y1 + in1) + anti_denorm) - anti_denorm;
                (fl, fr, [w0, w1, F32x4::zero(), F32x4::zero()])
            }
            _ => {
                let rl = F32x4::from_array([INV_SQRT_8, 0.0, -INV_SQRT_8, 0.0]);
                let rr = F32x4::from_array([0.0, INV_SQRT_8, 0.0, -INV_SQRT_8]);
                let fl = ((d[0] + d[1] + d[2] + d[3]) * rl).reduce_sum() * fdn_level;
                let fr = ((d[0] + d[1] + d[2] + d[3]) * rr).reduce_sum() * fdn_level;
                let y = hadamard_16(d);
                let scale = F32x4::splat(INV_SQRT_16);
                let in0 = F32x4::from_array([
                    mid_in + side_in,
                    mid_in - side_in,
                    -mid_in + side_in,
                    -mid_in - side_in,
                ]) * scale;
                let in1 = F32x4::from_array([
                    mid_in - side_in,
                    -mid_in + side_in,
                    mid_in + side_in,
                    -mid_in - side_in,
                ]) * scale;
                let in2 = F32x4::from_array([
                    -mid_in + side_in,
                    mid_in + side_in,
                    -mid_in - side_in,
                    mid_in - side_in,
                ]) * scale;
                let in3 = F32x4::from_array([
                    -mid_in - side_in,
                    -mid_in + side_in,
                    mid_in - side_in,
                    mid_in + side_in,
                ]) * scale;
                let w0 = ((y[0] + in0) + anti_denorm) - anti_denorm;
                let w1 = ((y[1] + in1) + anti_denorm) - anti_denorm;
                let w2 = ((y[2] + in2) + anti_denorm) - anti_denorm;
                let w3 = ((y[3] + in3) + anti_denorm) - anti_denorm;
                (fl, fr, [w0, w1, w2, w3])
            }
        };

        let mut v_idx = 0;
        while v_idx < n_vecs {
            let arr = w[v_idx].to_array();
            let base = v_idx * 4;
            let mut lane = 0;
            while lane < 4 {
                let m = base + lane;
                let pos = self.fdn_pos[m];
                self.fdn_buf[self.fdn_offsets[m] + pos] = arr[lane];
                let next_pos = pos + 1;
                self.fdn_pos[m] = if next_pos >= self.fdn_lens[m] {
                    0
                } else {
                    next_pos
                };
                lane += 1;
            }
            v_idx += 1;
        }
        (fdn_l, fdn_r)
    }

    /// Apply 3D early reflections + Hadamard FDN diffuse reverb in-place with 0-sample latency.
    pub(crate) fn process_block(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        if !self.is_active() {
            return;
        }
        let n = out_l.len().min(out_r.len());
        let dry_gain = 1.0 - 0.20 * self.reverb_mix;
        let wet_gain = self.reverb_mix * 0.72;
        let er_level = self.er_level;
        let fdn_level = self.fdn_level;
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

            let (fdn_l, fdn_r) =
                self.step_fdn_lines(er_l + 0.45 * dry_l, er_r + 0.45 * dry_r, fdn_level);

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
        let n_vecs = self.active_lines / 4;
        for v in 0..n_vecs {
            self.lpf_state[v] = (self.lpf_state[v] + anti_denorm) - anti_denorm;
        }
        self.iacc_lp_mid.flush_denormals();
        self.iacc_lp_side.flush_denormals();
        self.diff_eq_peak_l.flush_denormals();
        self.diff_eq_peak_r.flush_denormals();
        self.diff_eq_shelf_l.flush_denormals();
        self.diff_eq_shelf_r.flush_denormals();
    }
}
