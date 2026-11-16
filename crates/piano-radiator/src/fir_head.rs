//! Zero-latency 64-tap `F32x4` SIMD direct FIR convolution for all 9 zones -> `(L, R)`.

use piano_simd::F32x4;
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::ir_synth::path_index;

/// Exact cutoff tap index splitting zero-latency direct FIR from NUPOLS FFT tail.
pub(crate) const HEAD_TAPS: usize = 64;
const HEAD_VECS: usize = HEAD_TAPS / 4;
pub(crate) const FIR_COEFFS_LEN: usize = PRESET_RADIATOR_ZONES * HEAD_TAPS * 2;
const RING_LEN: usize = 128;
const RING_MASK: usize = RING_LEN - 1;
const RING_HIST_OFFSET: usize = RING_LEN - (HEAD_TAPS - 1); // 65
const RING_STORAGE: usize = RING_LEN * 2 + 8; // 264

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct AlignedRing([f32; RING_STORAGE]);

#[inline(always)]
fn write_ring_sample(ring: &mut [f32; RING_STORAGE], pos: usize, x: f32) {
    ring[pos] = x;
    ring[pos + RING_LEN] = x;
    if pos < 4 {
        ring[pos + RING_LEN * 2] = x;
    }
}

/// Zero-allocation 64-tap MIMO direct FIR head (`9 zones -> L/R`).
#[derive(Clone)]
pub(crate) struct FirHead {
    /// Time-reversed 64-tap L IR packed into `16` `F32x4` vectors:
    /// `coeffs_l_rev[z][v] = [h[63-4v], h[62-4v], h[61-4v], h[60-4v]]`.
    coeffs_l_rev: [[F32x4; HEAD_VECS]; PRESET_RADIATOR_ZONES],
    /// Time-reversed 64-tap R IR packed into `16` `F32x4` vectors.
    coeffs_r_rev: [[F32x4; HEAD_VECS]; PRESET_RADIATOR_ZONES],
    history: [AlignedRing; PRESET_RADIATOR_ZONES],
    head: usize,
    zone_active: [usize; PRESET_RADIATOR_ZONES],
}

impl FirHead {
    pub(crate) fn new() -> Self {
        Self {
            coeffs_l_rev: [[F32x4::zero(); HEAD_VECS]; PRESET_RADIATOR_ZONES],
            coeffs_r_rev: [[F32x4::zero(); HEAD_VECS]; PRESET_RADIATOR_ZONES],
            history: [AlignedRing([0.0; RING_STORAGE]); PRESET_RADIATOR_ZONES],
            head: 0,
            zone_active: [0; PRESET_RADIATOR_ZONES],
        }
    }

    /// Copy the first 64 taps (`[0..64)`) of all 18 zone-to-ear IRs into aligned,
    /// time-reversed SIMD coefficient storage.
    pub(crate) fn load_from_ir_flat(&mut self, ir_flat: &[f32], ir_len: usize) {
        let copy_len = ir_len.min(HEAD_TAPS);
        let mut rev_l = [0.0_f32; HEAD_TAPS];
        let mut rev_r = [0.0_f32; HEAD_TAPS];
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            rev_l.fill(0.0);
            rev_r.fill(0.0);
            let l_start = path_index(z, 0) * ir_len;
            let r_start = path_index(z, 1) * ir_len;
            let mut k = 0;
            while k < copy_len {
                let dst_idx = (HEAD_TAPS - 1) - k;
                rev_l[dst_idx] = ir_flat[l_start + k];
                rev_r[dst_idx] = ir_flat[r_start + k];
                k += 1;
            }
            let mut v = 0;
            while v < HEAD_VECS {
                let base = v * 4;
                self.coeffs_l_rev[z][v] = F32x4::from_slice(&rev_l[base..base + 4]);
                self.coeffs_r_rev[z][v] = F32x4::from_slice(&rev_r[base..base + 4]);
                v += 1;
            }
            z += 1;
        }
    }

    pub(crate) fn load_coeffs_slice(&mut self, coeffs: &[f32]) {
        let mut cursor = 0usize;
        let mut z = 0usize;
        while z < PRESET_RADIATOR_ZONES {
            let mut v = 0usize;
            while v < HEAD_VECS {
                self.coeffs_l_rev[z][v] = F32x4::from_slice(&coeffs[cursor..cursor + 4]);
                cursor += 4;
                v += 1;
            }
            v = 0;
            while v < HEAD_VECS {
                self.coeffs_r_rev[z][v] = F32x4::from_slice(&coeffs[cursor..cursor + 4]);
                cursor += 4;
                v += 1;
            }
            z += 1;
        }
    }

    pub(crate) fn load_coeffs_range(&mut self, fir_off: usize, coeffs: &[f32]) {
        if fir_off == 0 && coeffs.len() >= FIR_COEFFS_LEN {
            self.load_coeffs_slice(&coeffs[..FIR_COEFFS_LEN]);
            return;
        }
        let end = (fir_off + coeffs.len()).min(FIR_COEFFS_LEN);
        for g_idx in fir_off..end {
            let val = coeffs[g_idx - fir_off];
            let z = g_idx / (HEAD_TAPS * 2);
            let rem = g_idx % (HEAD_TAPS * 2);
            let is_r = rem >= HEAD_TAPS;
            let tap = if is_r { rem - HEAD_TAPS } else { rem };
            let vec_slot = if is_r {
                &mut self.coeffs_r_rev[z][tap >> 2]
            } else {
                &mut self.coeffs_l_rev[z][tap >> 2]
            };
            let mut arr = vec_slot.to_array();
            arr[tap & 3] = val;
            *vec_slot = F32x4::from_array(arr);
        }
    }

    /// Zero all input history lines without altering filter coefficients.
    pub(crate) fn reset(&mut self) {
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            self.history[z].0.fill(0.0);
            z += 1;
        }
        self.zone_active.fill(0);
        self.head = 0;
    }

    /// Convolve `frames.len()` 9-zone input samples with `0`-sample algorithmic delay,
    /// processing 4 output samples at a time in `F32x4` across L/R with a contiguous
    /// SIMD scalar tail for `n % 4 != 0`.
    #[inline]
    pub(crate) fn process_frames(
        &mut self,
        frames: &[[f32; PRESET_RADIATOR_ZONES]],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = frames.len().min(out_l.len()).min(out_r.len());
        let mut w = self.head;
        let mut i = 0;

        while i + 4 <= n {
            let p0 = w;
            let p1 = (w + 1) & RING_MASK;
            let p2 = (w + 2) & RING_MASK;
            let p3 = (w + 3) & RING_MASK;
            let f0 = &frames[i];
            let f1 = &frames[i + 1];
            let f2 = &frames[i + 2];
            let f3 = &frames[i + 3];

            let mut acc_l0 = F32x4::zero();
            let mut acc_l1 = F32x4::zero();
            let mut acc_l2 = F32x4::zero();
            let mut acc_l3 = F32x4::zero();
            let mut acc_r0 = F32x4::zero();
            let mut acc_r1 = F32x4::zero();
            let mut acc_r2 = F32x4::zero();
            let mut acc_r3 = F32x4::zero();

            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                let s0 = if f0[z].is_finite() { f0[z] } else { 0.0 };
                let s1 = if f1[z].is_finite() { f1[z] } else { 0.0 };
                let s2 = if f2[z].is_finite() { f2[z] } else { 0.0 };
                let s3 = if f3[z].is_finite() { f3[z] } else { 0.0 };

                let ring = &mut self.history[z].0;
                write_ring_sample(ring, p0, s0);
                write_ring_sample(ring, p1, s1);
                write_ring_sample(ring, p2, s2);
                write_ring_sample(ring, p3, s3);

                if s0 != 0.0 || s1 != 0.0 || s2 != 0.0 || s3 != 0.0 {
                    self.zone_active[z] = HEAD_TAPS + 4;
                }

                if self.zone_active[z] == 0 {
                    z += 1;
                    continue;
                }
                self.zone_active[z] = self.zone_active[z].saturating_sub(4);

                let win = &ring[w + RING_HIST_OFFSET..w + RING_HIST_OFFSET + HEAD_TAPS + 4];
                let cl = &self.coeffs_l_rev[z];
                let cr = &self.coeffs_r_rev[z];
                let mut x_cur = F32x4::from_slice(&win[0..4]);

                let mut v = 0;
                while v < HEAD_VECS {
                    let x_next = F32x4::from_slice(&win[v * 4 + 4..v * 4 + 8]);
                    let s0 = x_cur;
                    let s1 = x_cur.slide1(x_next);
                    let s2 = x_cur.slide2(x_next);
                    let s3 = x_cur.slide3(x_next);
                    x_cur = x_next;

                    let cl_v = cl[v];
                    let cr_v = cr[v];

                    acc_l0 = s0.mul_add(cl_v.splat_lane::<0>(), acc_l0);
                    acc_l1 = s1.mul_add(cl_v.splat_lane::<1>(), acc_l1);
                    acc_l2 = s2.mul_add(cl_v.splat_lane::<2>(), acc_l2);
                    acc_l3 = s3.mul_add(cl_v.splat_lane::<3>(), acc_l3);

                    acc_r0 = s0.mul_add(cr_v.splat_lane::<0>(), acc_r0);
                    acc_r1 = s1.mul_add(cr_v.splat_lane::<1>(), acc_r1);
                    acc_r2 = s2.mul_add(cr_v.splat_lane::<2>(), acc_r2);
                    acc_r3 = s3.mul_add(cr_v.splat_lane::<3>(), acc_r3);

                    v += 1;
                }
                z += 1;
            }

            ((acc_l0 + acc_l1) + (acc_l2 + acc_l3)).write_to_slice(&mut out_l[i..i + 4]);
            ((acc_r0 + acc_r1) + (acc_r2 + acc_r3)).write_to_slice(&mut out_r[i..i + 4]);
            w = (w + 4) & RING_MASK;
            i += 4;
        }

        while i < n {
            let frame = &frames[i];
            let mut sum_l = F32x4::zero();
            let mut sum_r = F32x4::zero();

            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                let raw = frame[z];
                let x = if raw.is_finite() { raw } else { 0.0 };
                let ring = &mut self.history[z].0;
                write_ring_sample(ring, w, x);

                if x != 0.0 {
                    self.zone_active[z] = HEAD_TAPS + 4;
                }

                if self.zone_active[z] == 0 {
                    z += 1;
                    continue;
                }
                self.zone_active[z] = self.zone_active[z].saturating_sub(1);

                let win = &ring[w + RING_HIST_OFFSET..w + RING_HIST_OFFSET + HEAD_TAPS];
                let cl = &self.coeffs_l_rev[z];
                let cr = &self.coeffs_r_rev[z];
                let mut v = 0;
                while v < HEAD_VECS {
                    let xv = F32x4::from_slice(&win[v * 4..v * 4 + 4]);
                    sum_l = xv.mul_add(cl[v], sum_l);
                    sum_r = xv.mul_add(cr[v], sum_r);
                    v += 1;
                }
                z += 1;
            }

            out_l[i] = sum_l.reduce_sum();
            out_r[i] = sum_r.reduce_sum();
            w = (w + 1) & RING_MASK;
            i += 1;
        }

        self.head = w;
    }
}
