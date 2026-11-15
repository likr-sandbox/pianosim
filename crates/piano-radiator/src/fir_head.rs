//! Zero-latency 64-tap `F32x4` SIMD direct FIR convolution for all 9 zones -> `(L, R)`.

use piano_simd::F32x4;
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::ir_synth::path_index;

/// Exact cutoff tap index splitting zero-latency direct FIR from NUPOLS FFT tail.
pub(crate) const HEAD_TAPS: usize = 64;
const HEAD_VECS: usize = HEAD_TAPS / 4;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct AlignedRing([f32; HEAD_TAPS * 2]);

/// Zero-allocation 64-tap MIMO direct FIR head (`9 zones -> L/R`).
pub(crate) struct FirHead {
    coeffs_l: [[F32x4; HEAD_VECS]; PRESET_RADIATOR_ZONES],
    coeffs_r: [[F32x4; HEAD_VECS]; PRESET_RADIATOR_ZONES],
    history: [AlignedRing; PRESET_RADIATOR_ZONES],
    head: usize,
}

impl FirHead {
    pub(crate) fn new() -> Self {
        Self {
            coeffs_l: [[F32x4::zero(); HEAD_VECS]; PRESET_RADIATOR_ZONES],
            coeffs_r: [[F32x4::zero(); HEAD_VECS]; PRESET_RADIATOR_ZONES],
            history: [AlignedRing([0.0; HEAD_TAPS * 2]); PRESET_RADIATOR_ZONES],
            head: 0,
        }
    }

    /// Copy the first 64 taps (`[0..64)`) of all 18 zone-to-ear IRs into aligned FIR storage.
    pub(crate) fn load_from_ir_flat(&mut self, ir_flat: &[f32], ir_len: usize) {
        let copy_len = ir_len.min(HEAD_TAPS);
        let mut tmp_l = [0.0_f32; HEAD_TAPS];
        let mut tmp_r = [0.0_f32; HEAD_TAPS];
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            tmp_l.fill(0.0);
            tmp_r.fill(0.0);
            let l_start = path_index(z, 0) * ir_len;
            let r_start = path_index(z, 1) * ir_len;
            tmp_l[..copy_len].copy_from_slice(&ir_flat[l_start..l_start + copy_len]);
            tmp_r[..copy_len].copy_from_slice(&ir_flat[r_start..r_start + copy_len]);
            let mut v = 0;
            while v < HEAD_VECS {
                let k = v * 4;
                self.coeffs_l[z][v] = F32x4::from_slice(&tmp_l[k..k + 4]);
                self.coeffs_r[z][v] = F32x4::from_slice(&tmp_r[k..k + 4]);
                v += 1;
            }
            z += 1;
        }
    }

    /// Zero all input history lines without altering filter coefficients.
    pub(crate) fn reset(&mut self) {
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            self.history[z].0.fill(0.0);
            z += 1;
        }
        self.head = 0;
    }

    /// Convolve `frames.len()` 9-zone input samples with `0`-sample algorithmic delay,
    /// writing the direct-FIR contribution into `out_l` and `out_r`.
    #[inline]
    pub(crate) fn process_frames(
        &mut self,
        frames: &[[f32; PRESET_RADIATOR_ZONES]],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = frames.len().min(out_l.len()).min(out_r.len());
        let mut head = self.head;

        let mut i = 0;
        while i < n {
            head = head.wrapping_sub(1) & (HEAD_TAPS - 1);
            let frame = &frames[i];

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
                let raw = frame[z];
                let x = if raw.is_finite() { raw } else { 0.0 };
                let ring = &mut self.history[z].0;
                ring[head] = x;
                ring[head + HEAD_TAPS] = x;

                let h = &ring[head..head + HEAD_TAPS];
                let cl = &self.coeffs_l[z];
                let cr = &self.coeffs_r[z];

                let mut v = 0;
                while v < HEAD_VECS {
                    let k = v * 4;
                    let x0 = F32x4::from_array([h[k], h[k + 1], h[k + 2], h[k + 3]]);
                    let x1 = F32x4::from_array([h[k + 4], h[k + 5], h[k + 6], h[k + 7]]);
                    let x2 = F32x4::from_array([h[k + 8], h[k + 9], h[k + 10], h[k + 11]]);
                    let x3 = F32x4::from_array([h[k + 12], h[k + 13], h[k + 14], h[k + 15]]);

                    acc_l0 = x0 * cl[v] + acc_l0;
                    acc_l1 = x1 * cl[v + 1] + acc_l1;
                    acc_l2 = x2 * cl[v + 2] + acc_l2;
                    acc_l3 = x3 * cl[v + 3] + acc_l3;

                    acc_r0 = x0 * cr[v] + acc_r0;
                    acc_r1 = x1 * cr[v + 1] + acc_r1;
                    acc_r2 = x2 * cr[v + 2] + acc_r2;
                    acc_r3 = x3 * cr[v + 3] + acc_r3;

                    v += 4;
                }
                z += 1;
            }

            let sum_l = (acc_l0 + acc_l1) + (acc_l2 + acc_l3);
            let sum_r = (acc_r0 + acc_r1) + (acc_r2 + acc_r3);
            out_l[i] = sum_l.reduce_sum();
            out_r[i] = sum_r.reduce_sum();
            i += 1;
        }

        self.head = head;
    }
}
