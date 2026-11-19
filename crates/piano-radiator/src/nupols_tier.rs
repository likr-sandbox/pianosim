//! Partitioned FDL storage and FFT kernels for a single staggered NUPOLS tier,
//! supporting zero-alloc IR length resizing and late-partition stereo collapse.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{AlignedBuf, RealFft, SplitComplexBuf};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::fir_head::FirHead;
use crate::ir_synth::{path_index, NUM_IR_PATHS};
use crate::nupols_collapse::{
    add_simd_slice, collapse_zone_spectra_into, compute_mimo_p_count, compute_p_count, TierConfig,
    IN_RING_LEN, IN_RING_MASK, OUT_RING_LEN, OUT_RING_MASK,
};

#[derive(Clone)]
pub(crate) struct NupolsTier {
    pub(crate) cfg: TierConfig,
    pub(crate) block_size: usize,
    pub(crate) ir_offset: usize,
    pub(crate) max_p_count: usize,
    pub(crate) p_count: usize,
    pub(crate) mimo_p_count: usize,
    fft: RealFft,
    ir_spectra: Vec<SplitComplexBuf>,
    fdl: Vec<SplitComplexBuf>,
    fdl_head: usize,
    pub(crate) acc_l: SplitComplexBuf,
    pub(crate) acc_r: SplitComplexBuf,
    collapsed_x_scratch: SplitComplexBuf,
    pub(crate) acc_dirty: bool,
    pub(crate) has_pending: bool,
    pub(crate) pending_s_in_end: usize,
    pub(crate) fdl_nz: Vec<bool>,
}

impl NupolsTier {
    pub(crate) fn new(cfg: TierConfig, max_ir_len: usize) -> Self {
        let max_p_count = compute_p_count(cfg, max_ir_len);
        let fft = RealFft::new(cfg.block_size * 2);
        let spec_len = fft.spectrum_len();

        let mut ir_spectra = Vec::with_capacity(max_p_count * NUM_IR_PATHS);
        for _ in 0..(max_p_count * NUM_IR_PATHS) {
            ir_spectra.push(SplitComplexBuf::zeroed(spec_len));
        }

        let mut fdl = Vec::with_capacity(max_p_count * PRESET_RADIATOR_ZONES);
        for _ in 0..(max_p_count * PRESET_RADIATOR_ZONES) {
            fdl.push(SplitComplexBuf::zeroed(spec_len));
        }

        Self {
            cfg,
            block_size: cfg.block_size,
            ir_offset: cfg.ir_offset,
            max_p_count,
            p_count: max_p_count,
            mimo_p_count: max_p_count,
            fft,
            ir_spectra,
            fdl,
            fdl_head: 0,
            acc_l: SplitComplexBuf::zeroed(spec_len),
            acc_r: SplitComplexBuf::zeroed(spec_len),
            collapsed_x_scratch: SplitComplexBuf::zeroed(spec_len),
            acc_dirty: false,
            has_pending: false,
            pending_s_in_end: 0,
            fdl_nz: alloc::vec![false; max_p_count * PRESET_RADIATOR_ZONES],
        }
    }

    /// Reconfigure active partition count and MIMO split tap with zero allocations.
    pub(crate) fn set_active_layout(&mut self, ir_len: usize, split_tap: u16) {
        let new_p = compute_p_count(self.cfg, ir_len).min(self.max_p_count);
        let new_mimo = compute_mimo_p_count(self.cfg, new_p, split_tap);
        if new_p != self.p_count || new_mimo != self.mimo_p_count {
            self.p_count = new_p;
            self.mimo_p_count = new_mimo;
            self.reset();
        }
    }

    pub(crate) fn load_ir(&mut self, ir_flat: &[f32], ir_len: usize, time_scratch: &mut [f32]) {
        if self.p_count == 0 {
            return;
        }
        let b = self.block_size;
        let fft_len = b * 2;
        let spec_len = self.fft.spectrum_len();
        let work = &mut time_scratch[..fft_len];

        let mut p = 0;
        while p < self.p_count {
            let tap_start = self.ir_offset + p * b;
            let p_base = p * NUM_IR_PATHS;
            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                let mut ch = 0;
                while ch < 2 {
                    let ir_base = path_index(z, ch) * ir_len;
                    work.fill(0.0);
                    if tap_start < ir_len {
                        let copy_len = b.min(ir_len - tap_start);
                        work[..copy_len].copy_from_slice(
                            &ir_flat[ir_base + tap_start..ir_base + tap_start + copy_len],
                        );
                    }
                    let spec = &mut self.ir_spectra[p_base + path_index(z, ch)];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(work, re, im);
                    ch += 1;
                }
                z += 1;
            }
            if p >= self.mimo_p_count {
                let slice = &mut self.ir_spectra[p_base..p_base + NUM_IR_PATHS];
                let (cr, ci) = self.collapsed_x_scratch.planes_mut();
                collapse_zone_spectra_into(slice, spec_len, 0, cr, ci);
                let (tr, ti) = self.acc_l.planes_mut();
                collapse_zone_spectra_into(slice, spec_len, 1, tr, ti);
                slice[0]
                    .re_mut()
                    .copy_from_slice(self.collapsed_x_scratch.re());
                slice[0]
                    .im_mut()
                    .copy_from_slice(self.collapsed_x_scratch.im());
                slice[1].re_mut().copy_from_slice(self.acc_l.re());
                slice[1].im_mut().copy_from_slice(self.acc_l.im());
                self.acc_l.clear();
                self.collapsed_x_scratch.clear();
            }
            p += 1;
        }
    }

    pub(crate) fn load_sub_from_reversed_head_coeffs(
        &mut self,
        coeffs: &[f32],
        time_scratch: &mut [f32],
    ) {
        let b = self.block_size;
        let work = &mut time_scratch[..b * 2];
        for p in 0..self.p_count {
            let tap_start = self.ir_offset + p * b;
            let p_base = p * NUM_IR_PATHS;
            for z in 0..PRESET_RADIATOR_ZONES {
                for ch in 0..2 {
                    let base = z * 128 + ch * 64;
                    work.fill(0.0);
                    for k in 0..b {
                        work[k] = coeffs[base + 63 - (tap_start + k)];
                    }
                    let spec = &mut self.ir_spectra[p_base + path_index(z, ch)];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(work, re, im);
                }
            }
        }
    }

    pub(crate) fn load_sub_from_staged_fir_head(
        &mut self,
        fir_head: &FirHead,
        time_scratch: &mut [f32],
    ) {
        let b = self.block_size;
        let work = &mut time_scratch[..b * 2];
        let mut ir64 = [0.0_f32; 64];
        for z in 0..PRESET_RADIATOR_ZONES {
            for ch in 0..2 {
                fir_head.extract_staged_ir_64(z, ch, &mut ir64);
                for p in 0..self.p_count {
                    let tap_start = self.ir_offset + p * b;
                    work.fill(0.0);
                    work[..b].copy_from_slice(&ir64[tap_start..tap_start + b]);
                    let spec = &mut self.ir_spectra[p * NUM_IR_PATHS + path_index(z, ch)];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(work, re, im);
                }
            }
        }
    }

    pub(crate) fn ir_spectra_float_len(&self) -> usize {
        let spec_floats = self.fft.spectrum_len() * 2;
        let mimo_floats = self.mimo_p_count * NUM_IR_PATHS * spec_floats;
        let collapsed_floats = (self.p_count - self.mimo_p_count) * 2 * spec_floats;
        mimo_floats + collapsed_floats
    }

    pub(crate) fn import_ir_spectra(&mut self, src: &[f32]) -> usize {
        if self.p_count == 0 {
            return 0;
        }
        let spec_len = self.fft.spectrum_len();
        let mut off = 0usize;
        for p in 0..self.p_count {
            let p_base = p * NUM_IR_PATHS;
            let count = if p < self.mimo_p_count {
                NUM_IR_PATHS
            } else {
                2
            };
            for spec in &mut self.ir_spectra[p_base..p_base + count] {
                let (re, im) = spec.planes_mut();
                re.copy_from_slice(&src[off..off + spec_len]);
                off += spec_len;
                im.copy_from_slice(&src[off..off + spec_len]);
                off += spec_len;
            }
        }
        off
    }

    pub(crate) fn import_ir_spectra_range(
        &mut self,
        tier_base: usize,
        src_off: usize,
        src: &[f32],
    ) {
        if self.p_count == 0 {
            return;
        }
        let tier_end = tier_base + self.ir_spectra_float_len();
        let src_end = src_off + src.len();
        if src_end <= tier_base || src_off >= tier_end {
            return;
        }
        let spec_len = self.fft.spectrum_len();
        let mut cur = tier_base;
        for p in 0..self.p_count {
            let p_base = p * NUM_IR_PATHS;
            let count = if p < self.mimo_p_count {
                NUM_IR_PATHS
            } else {
                2
            };
            for spec in &mut self.ir_spectra[p_base..p_base + count] {
                let (re, im) = spec.planes_mut();
                for plane in [re, im] {
                    let p_end = cur + spec_len;
                    let lo = cur.max(src_off);
                    let hi = p_end.min(src_end);
                    if lo < hi {
                        plane[lo - cur..hi - cur].copy_from_slice(&src[lo - src_off..hi - src_off]);
                    }
                    cur = p_end;
                }
            }
        }
    }

    pub(crate) fn reset(&mut self) {
        for buf in &mut self.fdl {
            buf.clear();
        }
        self.acc_l.clear();
        self.acc_r.clear();
        self.collapsed_x_scratch.clear();
        self.acc_dirty = false;
        self.fdl_head = 0;
        self.has_pending = false;
        self.pending_s_in_end = 0;
        self.fdl_nz.fill(false);
    }

    #[inline]
    pub(crate) fn clear_acc(&mut self) {
        self.acc_l.clear();
        self.acc_r.clear();
        self.acc_dirty = false;
    }

    #[inline]
    pub(crate) fn step_fdl_head(&mut self) {
        if self.p_count == 0 {
            self.fdl_head = 0;
            return;
        }
        self.fdl_head = if self.fdl_head == 0 {
            self.p_count - 1
        } else {
            self.fdl_head - 1
        };
    }

    pub(crate) fn forward_zones_range(
        &mut self,
        in_ring: &[AlignedBuf; PRESET_RADIATOR_ZONES],
        s_in_end: usize,
        z_start: usize,
        z_end: usize,
        time_scratch: &mut [f32],
    ) {
        if self.p_count == 0 {
            return;
        }
        let fft_len = self.block_size * 2;
        let start = s_in_end.wrapping_sub(fft_len) & IN_RING_MASK;
        let first = (IN_RING_LEN - start).min(fft_len);
        let second = fft_len - first;
        let work = &mut time_scratch[..fft_len];

        if self.mimo_p_count == 0 {
            if z_start > 0 {
                return;
            }
            let src0 = in_ring[0].as_slice();
            work[..first].copy_from_slice(&src0[start..start + first]);
            if second > 0 {
                work[first..fft_len].copy_from_slice(&src0[..second]);
            }
            let mut z = 1;
            while z < PRESET_RADIATOR_ZONES {
                let src = in_ring[z].as_slice();
                add_simd_slice(&mut work[..first], &src[start..start + first]);
                if second > 0 {
                    add_simd_slice(&mut work[first..fft_len], &src[..second]);
                }
                z += 1;
            }
            let nz = work[..fft_len].iter().any(|&v| v != 0.0);
            let slot = self.fdl_head * PRESET_RADIATOR_ZONES;
            self.fdl_nz[slot] = nz;
            if nz {
                let (re, im) = self.fdl[slot].planes_mut();
                self.fft.forward_split(work, re, im);
            }
            return;
        }

        let mut z = z_start;
        while z < z_end {
            let src = in_ring[z].as_slice();
            work[..first].copy_from_slice(&src[start..start + first]);
            if second > 0 {
                work[first..fft_len].copy_from_slice(&src[..second]);
            }
            let nz = work[..fft_len].iter().any(|&v| v != 0.0);
            let slot = self.fdl_head * PRESET_RADIATOR_ZONES + z;
            self.fdl_nz[slot] = nz;
            if nz {
                let (re, im) = self.fdl[slot].planes_mut();
                self.fft.forward_split(work, re, im);
            }
            z += 1;
        }
    }

    pub(crate) fn accumulate_partitions(&mut self, p_start: usize, p_end: usize) {
        let p_count = self.p_count;
        if p_count == 0 {
            return;
        }
        let mut p = p_start;
        while p < p_end {
            let fdl_slot = (self.fdl_head + p) % p_count;
            let fdl_base = fdl_slot * PRESET_RADIATOR_ZONES;
            let ir_base = p * NUM_IR_PATHS;

            if self.mimo_p_count == 0 {
                if self.fdl_nz[fdl_base] {
                    self.acc_dirty = true;
                    SplitComplexBuf::fma_accumulate_lr(
                        &mut self.acc_l,
                        &mut self.acc_r,
                        &self.fdl[fdl_base],
                        &self.ir_spectra[ir_base],
                        &self.ir_spectra[ir_base + 1],
                    );
                }
            } else if p < self.mimo_p_count {
                let mut z = 0;
                while z < PRESET_RADIATOR_ZONES {
                    if self.fdl_nz[fdl_base + z] {
                        self.acc_dirty = true;
                        SplitComplexBuf::fma_accumulate_lr(
                            &mut self.acc_l,
                            &mut self.acc_r,
                            &self.fdl[fdl_base + z],
                            &self.ir_spectra[ir_base + path_index(z, 0)],
                            &self.ir_spectra[ir_base + path_index(z, 1)],
                        );
                    }
                    z += 1;
                }
            } else {
                let mut any_nz = false;
                self.collapsed_x_scratch.clear();
                let mut z = 0;
                while z < PRESET_RADIATOR_ZONES {
                    if self.fdl_nz[fdl_base + z] {
                        any_nz = true;
                        let (sr, si) = self.fdl[fdl_base + z].padded_planes();
                        let (dr, di) = self.collapsed_x_scratch.padded_planes_mut();
                        add_simd_slice(dr, sr);
                        add_simd_slice(di, si);
                    }
                    z += 1;
                }
                if any_nz {
                    self.acc_dirty = true;
                    SplitComplexBuf::fma_accumulate_lr(
                        &mut self.acc_l,
                        &mut self.acc_r,
                        &self.collapsed_x_scratch,
                        &self.ir_spectra[ir_base],
                        &self.ir_spectra[ir_base + 1],
                    );
                }
            }
            p += 1;
        }
    }

    pub(crate) fn inverse_channel_add(
        &mut self,
        ch: usize,
        out_ring: &mut AlignedBuf,
        s_in_end: usize,
        output_delay: usize,
        time_scratch: &mut [f32],
    ) {
        if !self.acc_dirty {
            return;
        }
        let b = self.block_size;
        let fft_len = b * 2;
        let write_start = s_in_end
            .wrapping_sub(b)
            .wrapping_add(self.ir_offset)
            .wrapping_add(output_delay)
            & OUT_RING_MASK;
        let work = &mut time_scratch[..fft_len];
        let (re, im) = if ch == 0 {
            self.acc_l.planes()
        } else {
            self.acc_r.planes()
        };
        self.fft.inverse_split(re, im, work);
        let first = (OUT_RING_LEN - write_start).min(b);
        let ring = out_ring.as_mut_slice();
        add_simd_slice(
            &mut ring[write_start..write_start + first],
            &work[b..b + first],
        );
        if first < b {
            add_simd_slice(&mut ring[..b - first], &work[b + first..fft_len]);
        }
    }

    pub(crate) fn accumulate_and_inverse(
        &mut self,
        out_ring_l: &mut AlignedBuf,
        out_ring_r: &mut AlignedBuf,
        s_in_end: usize,
        output_delay: usize,
        time_scratch: &mut [f32],
    ) {
        self.clear_acc();
        self.accumulate_partitions(0, self.p_count);
        self.inverse_channel_add(0, out_ring_l, s_in_end, output_delay, time_scratch);
        self.inverse_channel_add(1, out_ring_r, s_in_end, output_delay, time_scratch);
    }
}
