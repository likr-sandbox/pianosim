//! Partitioned FDL storage and FFT kernels for a single staggered NUPOLS tier,
//! supporting zero-alloc IR length resizing and late-partition stereo collapse.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{AlignedBuf, RealFft, SplitComplexBuf};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::ir_synth::{path_index, NUM_IR_PATHS};
use crate::nupols_collapse::{
    add_simd_slice, compute_p_count, TierConfig, IN_RING_LEN, IN_RING_MASK, OUT_RING_LEN,
    OUT_RING_MASK,
};

#[derive(Clone)]
pub(crate) struct NupolsTier {
    pub(crate) cfg: TierConfig,
    pub(crate) block_size: usize,
    pub(crate) ir_offset: usize,
    pub(crate) max_p_count: usize,
    pub(crate) p_count: usize,
    pub(crate) mimo_p_count: usize,
    pub(super) fft: RealFft,
    pub(super) ir_spectra: Vec<SplitComplexBuf>,
    pub(super) fdl: Vec<SplitComplexBuf>,
    pub(super) fdl_head: usize,
    pub(crate) acc_l: SplitComplexBuf,
    pub(crate) acc_r: SplitComplexBuf,
    pub(super) collapsed_x_scratch: SplitComplexBuf,
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
