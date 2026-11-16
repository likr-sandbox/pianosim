//! Partitioned FDL storage and FFT kernels for a single staggered NUPOLS tier.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{AlignedBuf, F32x4, RealFft, SplitComplexBuf};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::ir_synth::{path_index, NUM_IR_PATHS};

pub(crate) const IN_RING_LEN: usize = 16_384;
pub(crate) const IN_RING_MASK: usize = IN_RING_LEN - 1;
pub(crate) const OUT_RING_LEN: usize = 16_384;
pub(crate) const OUT_RING_MASK: usize = OUT_RING_LEN - 1;
pub(crate) const MAX_TIER_FFT_LEN: usize = 8_192;

#[derive(Clone, Copy)]
pub(crate) struct TierConfig {
    pub(crate) block_size: usize,
    pub(crate) ir_offset: usize,
    pub(crate) end_limit: usize,
}

pub(crate) const TIER_CONFIGS: [TierConfig; 4] = [
    TierConfig {
        block_size: 64,
        ir_offset: 64,
        end_limit: 512,
    },
    TierConfig {
        block_size: 256,
        ir_offset: 512,
        end_limit: 2_048,
    },
    TierConfig {
        block_size: 1_024,
        ir_offset: 2_048,
        end_limit: 8_192,
    },
    TierConfig {
        block_size: 4_096,
        ir_offset: 8_192,
        end_limit: usize::MAX,
    },
];

#[derive(Clone)]
pub(crate) struct NupolsTier {
    pub(crate) block_size: usize,
    pub(crate) ir_offset: usize,
    pub(crate) p_count: usize,
    fft: RealFft,
    ir_spectra: Vec<SplitComplexBuf>,
    fdl: Vec<SplitComplexBuf>,
    fdl_head: usize,
    pub(crate) acc_l: SplitComplexBuf,
    pub(crate) acc_r: SplitComplexBuf,
    pub(crate) acc_dirty: bool,
    pub(crate) has_pending: bool,
    pub(crate) pending_s_in_end: usize,
    pub(crate) fdl_nz: Vec<bool>,
}

impl NupolsTier {
    pub(crate) fn new(cfg: TierConfig, ir_len: usize) -> Self {
        let end = ir_len.min(cfg.end_limit);
        let p_count = if end > cfg.ir_offset {
            (end - cfg.ir_offset).div_ceil(cfg.block_size)
        } else {
            0
        };
        let fft = RealFft::new(cfg.block_size * 2);
        let spec_len = fft.spectrum_len();

        let mut ir_spectra = Vec::with_capacity(p_count * NUM_IR_PATHS);
        for _ in 0..(p_count * NUM_IR_PATHS) {
            ir_spectra.push(SplitComplexBuf::zeroed(spec_len));
        }

        let mut fdl = Vec::with_capacity(p_count * PRESET_RADIATOR_ZONES);
        for _ in 0..(p_count * PRESET_RADIATOR_ZONES) {
            fdl.push(SplitComplexBuf::zeroed(spec_len));
        }

        Self {
            block_size: cfg.block_size,
            ir_offset: cfg.ir_offset,
            p_count,
            fft,
            ir_spectra,
            fdl,
            fdl_head: 0,
            acc_l: SplitComplexBuf::zeroed(spec_len),
            acc_r: SplitComplexBuf::zeroed(spec_len),
            acc_dirty: false,
            has_pending: false,
            pending_s_in_end: 0,
            fdl_nz: alloc::vec![false; p_count * PRESET_RADIATOR_ZONES],
        }
    }

    pub(crate) fn load_ir(&mut self, ir_flat: &[f32], ir_len: usize, time_scratch: &mut [f32]) {
        if self.p_count == 0 {
            return;
        }
        let b = self.block_size;
        let fft_len = b * 2;
        let work = &mut time_scratch[..fft_len];

        let mut p = 0;
        while p < self.p_count {
            let tap_start = self.ir_offset + p * b;
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
                    let spec_idx = p * NUM_IR_PATHS + path_index(z, ch);
                    let spec = &mut self.ir_spectra[spec_idx];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(work, re, im);
                    ch += 1;
                }
                z += 1;
            }
            p += 1;
        }
    }

    pub(crate) fn ir_spectra_float_len(&self) -> usize {
        self.p_count * NUM_IR_PATHS * self.fft.spectrum_len() * 2
    }

    pub(crate) fn import_ir_spectra(&mut self, src: &[f32]) -> usize {
        if self.p_count == 0 {
            return 0;
        }
        let spec_len = self.fft.spectrum_len();
        let mut off = 0usize;
        for spec in &mut self.ir_spectra {
            let (re, im) = spec.planes_mut();
            re.copy_from_slice(&src[off..off + spec_len]);
            off += spec_len;
            im.copy_from_slice(&src[off..off + spec_len]);
            off += spec_len;
        }
        off
    }

    pub(crate) fn reset(&mut self) {
        for buf in &mut self.fdl {
            buf.clear();
        }
        self.acc_l.clear();
        self.acc_r.clear();
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
        let fft_len = self.block_size * 2;
        let start = s_in_end.wrapping_sub(fft_len) & IN_RING_MASK;
        let first = (IN_RING_LEN - start).min(fft_len);
        let second = fft_len - first;
        let work = &mut time_scratch[..fft_len];

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
                let dst = &mut self.fdl[slot];
                let (re, im) = dst.planes_mut();
                self.fft.forward_split(work, re, im);
            }
            z += 1;
        }
    }

    pub(crate) fn accumulate_partitions(&mut self, p_start: usize, p_end: usize) {
        let p_count = self.p_count;
        let mut p = p_start;
        while p < p_end {
            let fdl_slot = (self.fdl_head + p) % p_count;
            let fdl_base = fdl_slot * PRESET_RADIATOR_ZONES;
            let ir_base = p * NUM_IR_PATHS;

            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                if self.fdl_nz[fdl_base + z] {
                    self.acc_dirty = true;
                    let x_spec = &self.fdl[fdl_base + z];
                    self.acc_l
                        .fma_accumulate(x_spec, &self.ir_spectra[ir_base + path_index(z, 0)]);
                    self.acc_r
                        .fma_accumulate(x_spec, &self.ir_spectra[ir_base + path_index(z, 1)]);
                }
                z += 1;
            }
            p += 1;
        }
    }

    pub(crate) fn inverse_channel_add(
        &mut self,
        ch: usize,
        out_ring: &mut AlignedBuf,
        s_in_end: usize,
        time_scratch: &mut [f32],
    ) {
        if !self.acc_dirty {
            return;
        }
        let b = self.block_size;
        let fft_len = b * 2;
        let write_start = s_in_end.wrapping_sub(b).wrapping_add(self.ir_offset) & OUT_RING_MASK;
        let work = &mut time_scratch[..fft_len];
        let (re, im) = if ch == 0 {
            self.acc_l.planes()
        } else {
            self.acc_r.planes()
        };
        self.fft.inverse_split(re, im, work);
        add_simd_slice(
            &mut out_ring.as_mut_slice()[write_start..write_start + b],
            &work[b..fft_len],
        );
    }

    pub(crate) fn accumulate_and_inverse(
        &mut self,
        out_ring_l: &mut AlignedBuf,
        out_ring_r: &mut AlignedBuf,
        s_in_end: usize,
        time_scratch: &mut [f32],
    ) {
        self.clear_acc();
        self.accumulate_partitions(0, self.p_count);
        self.inverse_channel_add(0, out_ring_l, s_in_end, time_scratch);
        self.inverse_channel_add(1, out_ring_r, s_in_end, time_scratch);
    }
}

#[inline]
fn add_simd_slice(dst: &mut [f32], src: &[f32]) {
    let mut i = 0;
    while i + 4 <= dst.len() {
        let d = F32x4::from_slice(&dst[i..i + 4]);
        let s = F32x4::from_slice(&src[i..i + 4]);
        (d + s).write_to_slice(&mut dst[i..i + 4]);
        i += 4;
    }
    while i < dst.len() {
        dst[i] += src[i];
        i += 1;
    }
}
