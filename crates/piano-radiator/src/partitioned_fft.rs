//! Zero-allocation MIMO Staggered NUPOLS engine (`9 zones -> L/R`) for taps
//! `64..ir_len` using `piano_simd::{RealFft, SplitComplexBuf}`.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{AlignedBuf, F32x4, RealFft, SplitComplexBuf};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::fir_head::HEAD_TAPS;
use crate::ir_synth::{path_index, NUM_IR_PATHS};

const IN_RING_LEN: usize = 16_384;
const IN_RING_MASK: usize = IN_RING_LEN - 1;
const OUT_RING_LEN: usize = 16_384;
const OUT_RING_MASK: usize = OUT_RING_LEN - 1;
const MAX_TIER_FFT_LEN: usize = 8_192;

#[derive(Clone, Copy)]
struct TierConfig {
    block_size: usize,
    ir_offset: usize,
    end_limit: usize,
    phi_slots: usize,
    split_phase: bool,
}

const TIER_CONFIGS: [TierConfig; 7] = [
    TierConfig {
        block_size: 64,
        ir_offset: 64,
        end_limit: 256,
        phi_slots: 0,
        split_phase: false,
    },
    TierConfig {
        block_size: 128,
        ir_offset: 256,
        end_limit: 512,
        phi_slots: 1,
        split_phase: false,
    },
    TierConfig {
        block_size: 256,
        ir_offset: 512,
        end_limit: 1_024,
        phi_slots: 2,
        split_phase: false,
    },
    TierConfig {
        block_size: 512,
        ir_offset: 1_024,
        end_limit: 2_048,
        phi_slots: 4,
        split_phase: false,
    },
    TierConfig {
        block_size: 1_024,
        ir_offset: 2_048,
        end_limit: 4_096,
        phi_slots: 8,
        split_phase: false,
    },
    TierConfig {
        block_size: 2_048,
        ir_offset: 4_096,
        end_limit: 8_192,
        phi_slots: 16,
        split_phase: false,
    },
    TierConfig {
        block_size: 4_096,
        ir_offset: 8_192,
        end_limit: usize::MAX,
        phi_slots: 32,
        split_phase: true,
    },
];

struct NupolsTier {
    block_size: usize,
    ir_offset: usize,
    phi_slots: usize,
    split_phase: bool,
    p_count: usize,
    fft: RealFft,
    ir_spectra: Vec<SplitComplexBuf>,
    fdl: Vec<SplitComplexBuf>,
    fdl_head: usize,
    acc_l: SplitComplexBuf,
    acc_r: SplitComplexBuf,
    has_pending: bool,
    pending_s_in_end: usize,
}

impl NupolsTier {
    fn new(cfg: TierConfig, ir_len: usize) -> Self {
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
            phi_slots: cfg.phi_slots,
            split_phase: cfg.split_phase,
            p_count,
            fft,
            ir_spectra,
            fdl,
            fdl_head: 0,
            acc_l: SplitComplexBuf::zeroed(spec_len),
            acc_r: SplitComplexBuf::zeroed(spec_len),
            has_pending: false,
            pending_s_in_end: 0,
        }
    }

    fn load_ir(&mut self, ir_flat: &[f32], ir_len: usize, time_scratch: &mut [f32]) {
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

    fn reset(&mut self) {
        for buf in &mut self.fdl {
            buf.clear();
        }
        self.acc_l.clear();
        self.acc_r.clear();
        self.fdl_head = 0;
        self.has_pending = false;
        self.pending_s_in_end = 0;
    }

    fn forward_zones_range(
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
            let dst = &mut self.fdl[self.fdl_head * PRESET_RADIATOR_ZONES + z];
            let (re, im) = dst.planes_mut();
            self.fft.forward_split(work, re, im);
            z += 1;
        }
    }

    fn forward_and_push_fdl(
        &mut self,
        in_ring: &[AlignedBuf; PRESET_RADIATOR_ZONES],
        s_in_end: usize,
        time_scratch: &mut [f32],
    ) {
        self.fdl_head = if self.fdl_head == 0 {
            self.p_count - 1
        } else {
            self.fdl_head - 1
        };
        self.forward_zones_range(in_ring, s_in_end, 0, PRESET_RADIATOR_ZONES, time_scratch);
    }

    fn accumulate_partitions(&mut self, p_start: usize, p_end: usize) {
        let p_count = self.p_count;
        let mut p = p_start;
        while p < p_end {
            let fdl_slot = (self.fdl_head + p) % p_count;
            let fdl_base = fdl_slot * PRESET_RADIATOR_ZONES;
            let ir_base = p * NUM_IR_PATHS;

            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                let x_spec = &self.fdl[fdl_base + z];
                self.acc_l
                    .fma_accumulate(x_spec, &self.ir_spectra[ir_base + path_index(z, 0)]);
                self.acc_r
                    .fma_accumulate(x_spec, &self.ir_spectra[ir_base + path_index(z, 1)]);
                z += 1;
            }
            p += 1;
        }
    }

    fn inverse_channel_add(
        &mut self,
        ch: usize,
        out_ring: &mut AlignedBuf,
        s_in_end: usize,
        time_scratch: &mut [f32],
    ) {
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

    fn accumulate_and_inverse(
        &mut self,
        out_ring_l: &mut AlignedBuf,
        out_ring_r: &mut AlignedBuf,
        s_in_end: usize,
        time_scratch: &mut [f32],
    ) {
        self.acc_l.clear();
        self.acc_r.clear();
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

/// Zero-allocation Staggered MIMO NUPOLS convolution tail (`taps 64..ir_len`).
pub(crate) struct NupolsTail {
    ir_len: usize,
    sample_pos: usize,
    in_ring: [AlignedBuf; PRESET_RADIATOR_ZONES],
    out_ring_l: AlignedBuf,
    out_ring_r: AlignedBuf,
    time_scratch: AlignedBuf,
    tiers: Vec<NupolsTier>,
}

impl NupolsTail {
    pub(crate) fn new(ir_len: usize) -> Self {
        let in_ring = [
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
            AlignedBuf::zeroed(IN_RING_LEN),
        ];
        let mut tiers = Vec::with_capacity(TIER_CONFIGS.len());
        for &cfg in &TIER_CONFIGS {
            tiers.push(NupolsTier::new(cfg, ir_len));
        }
        Self {
            ir_len,
            sample_pos: 0,
            in_ring,
            out_ring_l: AlignedBuf::zeroed(OUT_RING_LEN),
            out_ring_r: AlignedBuf::zeroed(OUT_RING_LEN),
            time_scratch: AlignedBuf::zeroed(MAX_TIER_FFT_LEN),
            tiers,
        }
    }

    pub(crate) fn load_ir_flat(&mut self, ir_flat: &[f32], ir_len: usize) {
        if ir_len != self.ir_len {
            self.ir_len = ir_len;
            self.tiers.clear();
            for &cfg in &TIER_CONFIGS {
                self.tiers.push(NupolsTier::new(cfg, ir_len));
            }
        }
        let work = self.time_scratch.as_mut_slice();
        for tier in &mut self.tiers {
            tier.load_ir(ir_flat, ir_len, work);
        }
    }

    pub(crate) fn reset(&mut self) {
        for r in &mut self.in_ring {
            r.clear();
        }
        self.out_ring_l.clear();
        self.out_ring_r.clear();
        for tier in &mut self.tiers {
            tier.reset();
        }
        self.sample_pos = 0;
    }

    /// Add scheduled tail outputs into `(out_l, out_r)`, record `frames` into the
    /// input ring, and fire staggered FFT tiers on `64`-sample slot boundaries.
    ///
    /// Expects `frames.len() <= 64 - (sample_pos & 63)` so it never crosses a `64`-sample slot.
    pub(crate) fn process_subchunk(
        &mut self,
        frames: &[[f32; PRESET_RADIATOR_ZONES]],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let chunk = frames.len().min(out_l.len()).min(out_r.len());
        if chunk == 0 {
            return;
        }

        // 1. Add already-scheduled future convolution outputs and zero consumed slots.
        let out_start = self.sample_pos & OUT_RING_MASK;
        let ring_l = self.out_ring_l.as_mut_slice();
        let ring_r = self.out_ring_r.as_mut_slice();
        let mut i = 0;
        while i + 4 <= chunk {
            let ol = F32x4::from_slice(&out_l[i..i + 4])
                + F32x4::from_slice(&ring_l[out_start + i..out_start + i + 4]);
            let or = F32x4::from_slice(&out_r[i..i + 4])
                + F32x4::from_slice(&ring_r[out_start + i..out_start + i + 4]);
            ol.write_to_slice(&mut out_l[i..i + 4]);
            or.write_to_slice(&mut out_r[i..i + 4]);
            F32x4::zero().write_to_slice(&mut ring_l[out_start + i..out_start + i + 4]);
            F32x4::zero().write_to_slice(&mut ring_r[out_start + i..out_start + i + 4]);
            i += 4;
        }
        while i < chunk {
            out_l[i] += ring_l[out_start + i];
            out_r[i] += ring_r[out_start + i];
            ring_l[out_start + i] = 0.0;
            ring_r[out_start + i] = 0.0;
            i += 1;
        }

        // 2. Record sanitized inputs into per-zone input ring.
        let in_start = self.sample_pos & IN_RING_MASK;
        let mut s = 0;
        while s < chunk {
            let f = &frames[s];
            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                let raw = f[z];
                self.in_ring[z].as_mut_slice()[in_start + s] =
                    if raw.is_finite() { raw } else { 0.0 };
                z += 1;
            }
            s += 1;
        }

        self.sample_pos += chunk;

        // 3. Fire staggered FFT tiers whenever a 64-sample slot completes.
        if (self.sample_pos & (HEAD_TAPS - 1)) == 0 {
            let m = self.sample_pos / HEAD_TAPS;
            self.fire_staggered_tiers(m);
        }
    }

    fn fire_staggered_tiers(&mut self, m: usize) {
        let work = self.time_scratch.as_mut_slice();
        for tier in &mut self.tiers {
            if tier.p_count == 0 {
                continue;
            }
            if tier.split_phase {
                let period_slots = tier.block_size / HEAD_TAPS; // 64
                let sub = m & (period_slots - 1);
                if sub == 0 && m >= period_slots {
                    let s_in_end = self.sample_pos;
                    tier.fdl_head = if tier.fdl_head == 0 {
                        tier.p_count - 1
                    } else {
                        tier.fdl_head - 1
                    };
                    tier.forward_zones_range(&self.in_ring, s_in_end, 0, 3, work);
                    tier.pending_s_in_end = s_in_end;
                    tier.has_pending = true;
                } else if tier.has_pending {
                    let s_in_end = tier.pending_s_in_end;
                    let half_p = tier.p_count / 2;
                    match sub {
                        6 => tier.forward_zones_range(&self.in_ring, s_in_end, 3, 5, work),
                        10 => tier.forward_zones_range(&self.in_ring, s_in_end, 5, 7, work),
                        14 => tier.forward_zones_range(&self.in_ring, s_in_end, 7, 9, work),
                        18 => {
                            tier.acc_l.clear();
                            tier.acc_r.clear();
                            tier.accumulate_partitions(0, half_p);
                        }
                        22 => tier.accumulate_partitions(half_p, tier.p_count),
                        26 => tier.inverse_channel_add(0, &mut self.out_ring_l, s_in_end, work),
                        32 => {
                            tier.inverse_channel_add(1, &mut self.out_ring_r, s_in_end, work);
                            tier.has_pending = false;
                        }
                        _ => {}
                    }
                }
            } else {
                let period_slots = tier.block_size / HEAD_TAPS;
                if m >= period_slots + tier.phi_slots && (m & (period_slots - 1)) == tier.phi_slots
                {
                    let s_in_end = self.sample_pos - tier.phi_slots * HEAD_TAPS;
                    tier.forward_and_push_fdl(&self.in_ring, s_in_end, work);
                    tier.accumulate_and_inverse(
                        &mut self.out_ring_l,
                        &mut self.out_ring_r,
                        s_in_end,
                        work,
                    );
                }
            }
        }
    }

    /// Maximum `64`-aligned chunk size allowed before crossing next slot boundary.
    #[inline]
    pub(crate) fn samples_until_slot_end(&self) -> usize {
        HEAD_TAPS - (self.sample_pos & (HEAD_TAPS - 1))
    }
}
