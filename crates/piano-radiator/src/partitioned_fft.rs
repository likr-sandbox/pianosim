//! Zero-allocation 4-tier `4x`-growth MIMO Staggered NUPOLS engine (`9 zones -> L/R`)
//! for taps `64..ir_len` using `piano_simd::{RealFft, SplitComplexBuf}`.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{AlignedBuf, F32x4};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::fir_head::HEAD_TAPS;
use crate::nupols_tier::{
    NupolsTier, IN_RING_LEN, IN_RING_MASK, MAX_TIER_FFT_LEN, OUT_RING_LEN, OUT_RING_MASK,
    TIER_CONFIGS,
};

/// Zero-allocation Staggered MIMO NUPOLS convolution tail (`taps 64..ir_len`).
#[derive(Clone)]
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

        // 2. Record sanitized inputs sequentially per zone into input rings.
        let in_start = self.sample_pos & IN_RING_MASK;
        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            let dst = &mut self.in_ring[z].as_mut_slice()[in_start..in_start + chunk];
            let mut s = 0;
            while s < chunk {
                let raw = frames[s][z];
                dst[s] = if raw.is_finite() { raw } else { 0.0 };
                s += 1;
            }
            z += 1;
        }

        self.sample_pos += chunk;

        // 3. Fire staggered 4x-sliced FFT tiers whenever a 64-sample slot completes.
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
            match tier.block_size {
                64 => {
                    let s_in_end = self.sample_pos;
                    tier.step_fdl_head();
                    tier.forward_zones_range(
                        &self.in_ring,
                        s_in_end,
                        0,
                        PRESET_RADIATOR_ZONES,
                        work,
                    );
                    tier.accumulate_and_inverse(
                        &mut self.out_ring_l,
                        &mut self.out_ring_r,
                        s_in_end,
                        work,
                    );
                }
                256 => {
                    let sub = m & 3;
                    if sub == 0 && m >= 4 {
                        let s_in_end = self.sample_pos;
                        tier.step_fdl_head();
                        tier.pending_s_in_end = s_in_end;
                        tier.has_pending = true;
                        tier.forward_zones_range(&self.in_ring, s_in_end, 0, 3, work);
                    } else if tier.has_pending {
                        let s_in_end = tier.pending_s_in_end;
                        match sub {
                            1 => tier.forward_zones_range(&self.in_ring, s_in_end, 3, 6, work),
                            2 => {
                                tier.forward_zones_range(&self.in_ring, s_in_end, 6, 9, work);
                                tier.acc_l.clear();
                                tier.acc_r.clear();
                                tier.accumulate_partitions(0, tier.p_count);
                            }
                            3 => {
                                tier.inverse_channel_add(0, &mut self.out_ring_l, s_in_end, work);
                                tier.inverse_channel_add(1, &mut self.out_ring_r, s_in_end, work);
                                tier.has_pending = false;
                            }
                            _ => {}
                        }
                    }
                }
                1_024 => {
                    let sub = m & 15;
                    if sub == 0 && m >= 16 {
                        let s_in_end = self.sample_pos;
                        tier.step_fdl_head();
                        tier.pending_s_in_end = s_in_end;
                        tier.has_pending = true;
                        tier.forward_zones_range(&self.in_ring, s_in_end, 0, 1, work);
                    } else if tier.has_pending {
                        let s_in_end = tier.pending_s_in_end;
                        let p1 = tier.p_count / 3;
                        let p2 = (2 * tier.p_count) / 3;
                        match sub {
                            1..=8 => {
                                tier.forward_zones_range(
                                    &self.in_ring,
                                    s_in_end,
                                    sub,
                                    sub + 1,
                                    work,
                                );
                            }
                            9 => {
                                tier.acc_l.clear();
                                tier.acc_r.clear();
                                tier.accumulate_partitions(0, p1);
                            }
                            10 => tier.accumulate_partitions(p1, p2),
                            11 => tier.accumulate_partitions(p2, tier.p_count),
                            12 => tier.inverse_channel_add(0, &mut self.out_ring_l, s_in_end, work),
                            13 => {
                                tier.inverse_channel_add(1, &mut self.out_ring_r, s_in_end, work);
                                tier.has_pending = false;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    let sub = m & 63;
                    if sub == 0 && m >= 64 {
                        tier.step_fdl_head();
                        tier.pending_s_in_end = self.sample_pos;
                        tier.has_pending = true;
                    } else if tier.has_pending {
                        let s_in_end = tier.pending_s_in_end;
                        let p1 = tier.p_count / 4;
                        let p2 = (2 * tier.p_count) / 4;
                        let p3 = (3 * tier.p_count) / 4;
                        match sub {
                            3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 => {
                                let z = (sub - 3) / 4;
                                tier.forward_zones_range(&self.in_ring, s_in_end, z, z + 1, work);
                            }
                            39 => {
                                tier.acc_l.clear();
                                tier.acc_r.clear();
                                tier.accumulate_partitions(0, p1);
                            }
                            43 => tier.accumulate_partitions(p1, p2),
                            47 => tier.accumulate_partitions(p2, p3),
                            51 => tier.accumulate_partitions(p3, tier.p_count),
                            55 => tier.inverse_channel_add(0, &mut self.out_ring_l, s_in_end, work),
                            59 => {
                                tier.inverse_channel_add(1, &mut self.out_ring_r, s_in_end, work);
                                tier.has_pending = false;
                            }
                            _ => {}
                        }
                    }
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
