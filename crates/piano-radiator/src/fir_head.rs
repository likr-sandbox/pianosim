//! Zero-latency configurable `16/32/64`-tap `F32x4` SIMD direct FIR convolution
//! for all 9 zones -> `(L, R)`.

use piano_simd::F32x4;
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::ir_synth::path_index;

/// Maximum cutoff tap index splitting zero-latency direct FIR from NUPOLS FFT tail.
pub(crate) const HEAD_TAPS: usize = 64;
const HEAD_VECS: usize = HEAD_TAPS / 4;
pub(crate) const FIR_COEFFS_LEN: usize = PRESET_RADIATOR_ZONES * HEAD_TAPS * 2;
const RING_LEN: usize = 128;
const RING_MASK: usize = RING_LEN - 1;
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

#[inline]
pub(crate) const fn snap_head_taps(taps: u8) -> usize {
    if taps <= 24 {
        16
    } else if taps <= 48 {
        32
    } else {
        64
    }
}

/// Zero-allocation `16/32/64`-tap MIMO direct FIR head (`9 zones -> L/R`).
#[derive(Clone)]
pub(crate) struct FirHead {
    active_taps: usize,
    active_vecs: usize,
    hist_offset: usize,
    /// Time-reversed IR up to 64 taps packed into `[0..active_vecs]` `F32x4` vectors:
    /// `coeffs_l_rev[z][v] = [h[T-1-4v], h[T-2-4v], h[T-3-4v], h[T-4-4v]]`.
    coeffs_l_rev: [[F32x4; HEAD_VECS]; PRESET_RADIATOR_ZONES],
    /// Time-reversed R IR packed into `[0..active_vecs]` `F32x4` vectors.
    coeffs_r_rev: [[F32x4; HEAD_VECS]; PRESET_RADIATOR_ZONES],
    history: [AlignedRing; PRESET_RADIATOR_ZONES],
    head: usize,
    zone_active: [usize; PRESET_RADIATOR_ZONES],
}

impl FirHead {
    pub(crate) fn new_with_taps(taps: u8) -> Self {
        let active_taps = snap_head_taps(taps);
        Self {
            active_taps,
            active_vecs: active_taps / 4,
            hist_offset: RING_LEN - (active_taps - 1),
            coeffs_l_rev: [[F32x4::zero(); HEAD_VECS]; PRESET_RADIATOR_ZONES],
            coeffs_r_rev: [[F32x4::zero(); HEAD_VECS]; PRESET_RADIATOR_ZONES],
            history: [AlignedRing([0.0; RING_STORAGE]); PRESET_RADIATOR_ZONES],
            head: 0,
            zone_active: [0; PRESET_RADIATOR_ZONES],
        }
    }

    #[inline]
    pub(crate) const fn active_taps(&self) -> usize {
        self.active_taps
    }

    pub(crate) fn set_active_taps(&mut self, taps: u8) {
        let new_taps = snap_head_taps(taps);
        if new_taps == self.active_taps {
            return;
        }
        self.active_taps = new_taps;
        self.active_vecs = new_taps / 4;
        self.hist_offset = RING_LEN - (new_taps - 1);
        self.reset();
    }

    /// Copy the first `active_taps` (`[0..active_taps)`) of all 18 zone-to-ear IRs
    /// into aligned, time-reversed SIMD coefficient storage.
    pub(crate) fn load_from_ir_flat(&mut self, ir_flat: &[f32], ir_len: usize) {
        let taps = self.active_taps;
        let vecs = self.active_vecs;
        let copy_len = ir_len.min(taps);
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
                let dst_idx = (taps - 1) - k;
                rev_l[dst_idx] = ir_flat[l_start + k];
                rev_r[dst_idx] = ir_flat[r_start + k];
                k += 1;
            }
            let mut v = 0;
            while v < vecs {
                let base = v * 4;
                self.coeffs_l_rev[z][v] = F32x4::from_slice(&rev_l[base..base + 4]);
                self.coeffs_r_rev[z][v] = F32x4::from_slice(&rev_r[base..base + 4]);
                v += 1;
            }
            while v < HEAD_VECS {
                self.coeffs_l_rev[z][v] = F32x4::zero();
                self.coeffs_r_rev[z][v] = F32x4::zero();
                v += 1;
            }
            z += 1;
        }
    }

    /// Load `[0..active_taps)` from a full 64-tap `SpaceBlob` head slice (`FIR_COEFFS_LEN`).
    pub(crate) fn load_coeffs_from_full_slice(&mut self, coeffs: &[f32]) {
        let vecs = self.active_vecs;
        let skip = HEAD_TAPS - self.active_taps;
        let mut z = 0usize;
        while z < PRESET_RADIATOR_ZONES {
            let z_base = z * HEAD_TAPS * 2;
            let l_base = z_base + skip;
            let r_base = z_base + HEAD_TAPS + skip;
            let mut v = 0usize;
            while v < vecs {
                let off = v * 4;
                self.coeffs_l_rev[z][v] =
                    F32x4::from_slice(&coeffs[l_base + off..l_base + off + 4]);
                self.coeffs_r_rev[z][v] =
                    F32x4::from_slice(&coeffs[r_base + off..r_base + off + 4]);
                v += 1;
            }
            while v < HEAD_VECS {
                self.coeffs_l_rev[z][v] = F32x4::zero();
                self.coeffs_r_rev[z][v] = F32x4::zero();
                v += 1;
            }
            z += 1;
        }
    }

    /// Stream raw full-64-tap reversed coefficients into `[0..HEAD_VECS]` staging slots.
    pub(crate) fn load_coeffs_range(&mut self, fir_off: usize, coeffs: &[f32]) {
        if fir_off == 0 && coeffs.len() >= FIR_COEFFS_LEN {
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

    /// Extract forward-order 64-tap IR `[h[0]..h[63]]` from full-64-tap reversed staging.
    pub(crate) fn extract_staged_ir_64(&self, z: usize, ch: usize, dst: &mut [f32; HEAD_TAPS]) {
        let vecs = if ch == 0 {
            &self.coeffs_l_rev[z]
        } else {
            &self.coeffs_r_rev[z]
        };
        let mut v = 0;
        while v < HEAD_VECS {
            let arr = vecs[v].to_array();
            let base = v * 4;
            dst[HEAD_TAPS - 1 - base] = arr[0];
            dst[HEAD_TAPS - 2 - base] = arr[1];
            dst[HEAD_TAPS - 3 - base] = arr[2];
            dst[HEAD_TAPS - 4 - base] = arr[3];
            v += 1;
        }
    }

    /// Repack staged full-64-tap reversed coefficients down to `active_vecs` in-place.
    pub(crate) fn repack_staged_to_active(&mut self) {
        let vecs = self.active_vecs;
        if vecs == HEAD_VECS {
            return;
        }
        let skip_vecs = HEAD_VECS - vecs;
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            let mut v = 0;
            while v < vecs {
                self.coeffs_l_rev[z][v] = self.coeffs_l_rev[z][skip_vecs + v];
                self.coeffs_r_rev[z][v] = self.coeffs_r_rev[z][skip_vecs + v];
                v += 1;
            }
            while v < HEAD_VECS {
                self.coeffs_l_rev[z][v] = F32x4::zero();
                self.coeffs_r_rev[z][v] = F32x4::zero();
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
        let taps = self.active_taps;
        let vecs = self.active_vecs;
        let hist_off = self.hist_offset;
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
                    self.zone_active[z] = taps + 4;
                }

                if self.zone_active[z] == 0 {
                    z += 1;
                    continue;
                }
                self.zone_active[z] = self.zone_active[z].saturating_sub(4);

                let win = &ring[w + hist_off..w + hist_off + taps + 4];
                let cl = &self.coeffs_l_rev[z];
                let cr = &self.coeffs_r_rev[z];
                let mut x_cur = F32x4::from_slice(&win[0..4]);

                let mut v = 0;
                while v < vecs {
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
                    self.zone_active[z] = taps + 4;
                }

                if self.zone_active[z] == 0 {
                    z += 1;
                    continue;
                }
                self.zone_active[z] = self.zone_active[z].saturating_sub(1);

                let win = &ring[w + hist_off..w + hist_off + taps];
                let cl = &self.coeffs_l_rev[z];
                let cr = &self.coeffs_r_rev[z];
                let mut v = 0;
                while v < vecs {
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
