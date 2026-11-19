//! Zero-allocation 4-tier `4x`-growth MIMO Staggered NUPOLS engine (`9 zones -> L/R`)
//! for taps `64..ir_len`, plus zero-latency `[active_taps..64)` sub-tiers when
//! `fir_head_taps < 64 && !fir_block_fft`.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{AlignedBuf, F32x4};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::block_fft_head::BLOCK_FFT_SIZE;
use crate::fir_head::{snap_head_taps, FirHead, HEAD_TAPS};
use crate::nupols_collapse::{
    IN_RING_LEN, IN_RING_MASK, MAX_TIER_FFT_LEN, OUT_RING_LEN, OUT_RING_MASK, SUB_TIER_16_CFG,
    SUB_TIER_32_CFG, TIER_CONFIGS,
};
use crate::nupols_tier::NupolsTier;
use crate::nupols_tier_ops::slice_range;
use crate::partitioned_schedule::{fire_tier_1024, fire_tier_256, fire_tier_4096, TierFireCtx};

/// Zero-allocation Staggered MIMO NUPOLS convolution tail (`taps 64..ir_len` + optional sub-tier).
#[derive(Clone)]
pub(crate) struct NupolsTail {
    max_ir_len: usize,
    ir_len: usize,
    split_tap: u16,
    head_taps: u8,
    output_delay: usize,
    sample_pos: usize,
    active_samples_rem: usize,
    tail_clean: bool,
    in_ring: [AlignedBuf; PRESET_RADIATOR_ZONES],
    out_ring_l: AlignedBuf,
    out_ring_r: AlignedBuf,
    time_scratch: AlignedBuf,
    sub_tier_32: NupolsTier,
    sub_tier_16: NupolsTier,
    tiers: Vec<NupolsTier>,
}

impl NupolsTail {
    pub(crate) fn new(max_ir_len: usize) -> Self {
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
            tiers.push(NupolsTier::new(cfg, max_ir_len));
        }
        Self {
            max_ir_len,
            ir_len: max_ir_len,
            split_tap: u16::MAX,
            head_taps: HEAD_TAPS as u8,
            output_delay: 0,
            sample_pos: 0,
            active_samples_rem: 0,
            tail_clean: true,
            in_ring,
            out_ring_l: AlignedBuf::zeroed(OUT_RING_LEN),
            out_ring_r: AlignedBuf::zeroed(OUT_RING_LEN),
            time_scratch: AlignedBuf::zeroed(MAX_TIER_FFT_LEN),
            sub_tier_32: NupolsTier::new(SUB_TIER_32_CFG, HEAD_TAPS),
            sub_tier_16: NupolsTier::new(SUB_TIER_16_CFG, HEAD_TAPS),
            tiers,
        }
    }

    #[inline]
    pub(crate) const fn max_ir_len(&self) -> usize {
        self.max_ir_len
    }

    #[inline]
    pub(crate) const fn active_ir_len(&self) -> usize {
        self.ir_len
    }

    /// Reconfigure active IR length zero-alloc (`<= max_ir_len`).
    pub(crate) fn set_active_ir_len(&mut self, active_ir_len: usize) {
        self.set_stage4_config(
            active_ir_len,
            self.split_tap,
            self.head_taps,
            self.output_delay > 0,
        );
    }

    /// Reconfigure active Stage 4 parameters with zero heap allocations.
    pub(crate) fn set_stage4_config(
        &mut self,
        active_ir_len: usize,
        split_tap: u16,
        head_taps: u8,
        fir_block_fft: bool,
    ) {
        self.set_stage4_config_inner(active_ir_len, split_tap, head_taps, fir_block_fft, true);
    }

    /// Reconfigure active Stage 4 parameters without resetting delay rings/FDLs.
    pub(crate) fn set_stage4_config_no_reset(
        &mut self,
        active_ir_len: usize,
        split_tap: u16,
        head_taps: u8,
        fir_block_fft: bool,
    ) {
        self.set_stage4_config_inner(active_ir_len, split_tap, head_taps, fir_block_fft, false);
    }

    fn set_stage4_config_inner(
        &mut self,
        active_ir_len: usize,
        split_tap: u16,
        head_taps: u8,
        fir_block_fft: bool,
        reset_on_change: bool,
    ) {
        let clamped_ir = active_ir_len.clamp(1, self.max_ir_len);
        let snapped_head = snap_head_taps(head_taps) as u8;
        let new_delay = if fir_block_fft { BLOCK_FFT_SIZE } else { 0 };
        let timing_changed = snapped_head != self.head_taps || new_delay != self.output_delay;

        self.ir_len = clamped_ir;
        self.split_tap = split_tap;
        self.head_taps = snapped_head;
        self.output_delay = new_delay;

        for tier in &mut self.tiers {
            if reset_on_change {
                tier.set_active_layout(clamped_ir, split_tap);
            } else {
                tier.set_active_layout_no_reset(clamped_ir, split_tap);
            }
        }
        if reset_on_change {
            self.sub_tier_32
                .set_active_layout(clamped_ir.min(HEAD_TAPS), split_tap);
            self.sub_tier_16
                .set_active_layout(clamped_ir.min(HEAD_TAPS), split_tap);
            if timing_changed {
                self.reset();
            }
        } else {
            self.sub_tier_32
                .set_active_layout_no_reset(clamped_ir.min(HEAD_TAPS), split_tap);
            self.sub_tier_16
                .set_active_layout_no_reset(clamped_ir.min(HEAD_TAPS), split_tap);
        }
    }

    pub(crate) fn load_ir_flat(&mut self, ir_flat: &[f32], ir_len: usize) {
        if ir_len > self.max_ir_len {
            self.max_ir_len = ir_len;
            self.tiers.clear();
            for &cfg in &TIER_CONFIGS {
                self.tiers.push(NupolsTier::new(cfg, ir_len));
            }
        }
        self.set_active_ir_len(ir_len);
        let work = self.time_scratch.as_mut_slice();
        for tier in &mut self.tiers {
            tier.load_ir(ir_flat, ir_len, work);
        }
        if self.output_delay == 0 {
            if self.head_taps == 32 {
                self.sub_tier_32.load_ir(ir_flat, ir_len, work);
            } else if self.head_taps == 16 {
                self.sub_tier_16.load_ir(ir_flat, ir_len, work);
            }
        }
        if !self.tail_clean {
            self.active_samples_rem = self
                .active_samples_rem
                .max(ir_len + IN_RING_LEN + 2 * MAX_TIER_FFT_LEN + self.output_delay);
        }
    }

    pub(crate) fn load_sub_tier_from_reversed_head_coeffs(&mut self, coeffs: &[f32]) {
        if self.output_delay > 0 {
            return;
        }
        let work = self.time_scratch.as_mut_slice();
        if self.head_taps == 32 {
            self.sub_tier_32
                .load_sub_from_reversed_head_coeffs(coeffs, work);
        } else if self.head_taps == 16 {
            self.sub_tier_16
                .load_sub_from_reversed_head_coeffs(coeffs, work);
        }
    }

    pub(crate) fn load_sub_tier_from_staged_fir_head(&mut self, fir_head: &FirHead) {
        if self.output_delay > 0 {
            return;
        }
        let work = self.time_scratch.as_mut_slice();
        if self.head_taps == 32 {
            self.sub_tier_32
                .load_sub_from_staged_fir_head(fir_head, work);
        } else if self.head_taps == 16 {
            self.sub_tier_16
                .load_sub_from_staged_fir_head(fir_head, work);
        }
    }

    pub(crate) fn ir_spectra_float_len(&self) -> usize {
        self.tiers
            .iter()
            .map(NupolsTier::ir_spectra_float_len)
            .sum()
    }

    pub(crate) fn import_ir_spectra(&mut self, src: &[f32]) -> bool {
        if src.len() != self.ir_spectra_float_len() {
            return false;
        }
        let mut off = 0usize;
        for tier in &mut self.tiers {
            let used = tier.import_ir_spectra(&src[off..]);
            off += used;
        }
        if !self.tail_clean {
            self.active_samples_rem = self
                .active_samples_rem
                .max(self.ir_len + IN_RING_LEN + 2 * MAX_TIER_FFT_LEN + self.output_delay);
        }
        off == src.len()
    }

    pub(crate) fn import_ir_spectra_range(&mut self, tail_off: usize, src: &[f32]) {
        let mut tier_base = 0usize;
        for tier in &mut self.tiers {
            let t_len = tier.ir_spectra_float_len();
            tier.import_ir_spectra_range(tier_base, tail_off, src);
            tier_base += t_len;
        }
        if !self.tail_clean {
            self.active_samples_rem = self
                .active_samples_rem
                .max(self.ir_len + IN_RING_LEN + 2 * MAX_TIER_FFT_LEN + self.output_delay);
        }
    }

    pub(crate) fn import_sub_tier_ir_spectra(&mut self, src: &[f32]) -> bool {
        if self.output_delay > 0 {
            return src.is_empty();
        }
        if self.head_taps == 32 {
            src.len() == self.sub_tier_32.ir_spectra_float_len()
                && self.sub_tier_32.import_ir_spectra(src) == src.len()
        } else if self.head_taps == 16 {
            src.len() == self.sub_tier_16.ir_spectra_float_len()
                && self.sub_tier_16.import_ir_spectra(src) == src.len()
        } else {
            src.is_empty()
        }
    }

    pub(crate) fn import_sub_tier_ir_spectra_range(&mut self, sub_off: usize, src: &[f32]) {
        if self.output_delay > 0 || src.is_empty() {
            return;
        }
        if self.head_taps == 32 {
            self.sub_tier_32.import_ir_spectra_range(0, sub_off, src);
        } else if self.head_taps == 16 {
            self.sub_tier_16.import_ir_spectra_range(0, sub_off, src);
        }
    }

    pub(crate) fn clear_small_state(&mut self) {
        self.sub_tier_32.reset();
        self.sub_tier_16.reset();
        for tier in &mut self.tiers {
            tier.clear_small_state();
        }
        self.sample_pos = 0;
        self.active_samples_rem = 0;
        self.tail_clean = true;
    }

    pub(crate) fn clear_rings_and_fdl_slice(&mut self, slice_idx: usize, total_slices: usize) {
        let (in_s, in_e) = slice_range(IN_RING_LEN, slice_idx, total_slices);
        if in_s < in_e {
            for r in &mut self.in_ring {
                r.as_mut_slice()[in_s..in_e].fill(0.0);
            }
        }
        let (out_s, out_e) = slice_range(OUT_RING_LEN, slice_idx, total_slices);
        if out_s < out_e {
            self.out_ring_l.as_mut_slice()[out_s..out_e].fill(0.0);
            self.out_ring_r.as_mut_slice()[out_s..out_e].fill(0.0);
        }
        for tier in &mut self.tiers {
            tier.clear_fdl_slice(slice_idx, total_slices);
        }
    }

    pub(crate) fn reset(&mut self) {
        for r in &mut self.in_ring {
            r.clear();
        }
        self.out_ring_l.clear();
        self.out_ring_r.clear();
        self.sub_tier_32.reset();
        self.sub_tier_16.reset();
        for tier in &mut self.tiers {
            tier.reset();
        }
        self.sample_pos = 0;
        self.active_samples_rem = 0;
        self.tail_clean = true;
    }

    /// Add scheduled tail outputs into `(out_l, out_r)`, record `frames` into the
    /// input ring, and fire staggered FFT tiers on slot boundaries.
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
        let frame_nz = frames[..chunk]
            .iter()
            .any(|fr| fr.iter().any(|&v| v != 0.0 && v.is_finite()));
        if frame_nz {
            self.active_samples_rem =
                self.ir_len + IN_RING_LEN + 2 * MAX_TIER_FFT_LEN + self.output_delay;
            self.tail_clean = false;
        } else if self.active_samples_rem == 0 && self.tiers.iter().all(|t| !t.has_pending) {
            if !self.tail_clean {
                self.sub_tier_32.fdl_nz.fill(false);
                self.sub_tier_16.fdl_nz.fill(false);
                for tier in &mut self.tiers {
                    tier.fdl_nz.fill(false);
                }
                self.tail_clean = true;
            }
            self.sample_pos += chunk;
            return;
        }

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
        self.active_samples_rem = self.active_samples_rem.saturating_sub(chunk);

        if self.output_delay == 0 {
            let work = self.time_scratch.as_mut_slice();
            let sub_tier = if self.head_taps == 32 && (self.sample_pos & 31) == 0 {
                Some(&mut self.sub_tier_32)
            } else if self.head_taps == 16 && (self.sample_pos & 15) == 0 {
                Some(&mut self.sub_tier_16)
            } else {
                None
            };
            if let Some(st) = sub_tier {
                if st.p_count > 0 {
                    let s_in_end = self.sample_pos;
                    st.step_fdl_head();
                    st.forward_zones_range(&self.in_ring, s_in_end, 0, PRESET_RADIATOR_ZONES, work);
                    st.accumulate_and_inverse(
                        &mut self.out_ring_l,
                        &mut self.out_ring_r,
                        s_in_end,
                        0,
                        work,
                    );
                }
            }
        }

        if (self.sample_pos & (HEAD_TAPS - 1)) == 0 {
            let m = self.sample_pos / HEAD_TAPS;
            self.fire_staggered_tiers(m);
        }
    }

    fn fire_staggered_tiers(&mut self, m: usize) {
        let mut ctx = TierFireCtx {
            in_ring: &self.in_ring,
            out_ring_l: &mut self.out_ring_l,
            out_ring_r: &mut self.out_ring_r,
            sample_pos: self.sample_pos,
            m,
            delay: self.output_delay,
            work: self.time_scratch.as_mut_slice(),
        };
        for tier in &mut self.tiers {
            if tier.p_count == 0 {
                continue;
            }
            match tier.block_size {
                64 => {
                    tier.step_fdl_head();
                    tier.forward_zones_range(
                        ctx.in_ring,
                        ctx.sample_pos,
                        0,
                        PRESET_RADIATOR_ZONES,
                        ctx.work,
                    );
                    tier.accumulate_and_inverse(
                        ctx.out_ring_l,
                        ctx.out_ring_r,
                        ctx.sample_pos,
                        ctx.delay,
                        ctx.work,
                    );
                }
                256 => fire_tier_256(tier, &mut ctx),
                1_024 => fire_tier_1024(tier, &mut ctx),
                _ => fire_tier_4096(tier, &mut ctx),
            }
        }
    }

    #[inline]
    fn sub_slot_size(&self) -> usize {
        if self.output_delay == 0 && self.head_taps < HEAD_TAPS as u8 {
            self.head_taps as usize
        } else {
            HEAD_TAPS
        }
    }

    /// Maximum slot-aligned chunk size allowed before crossing next FFT slot boundary.
    #[inline]
    pub(crate) fn samples_until_slot_end(&self) -> usize {
        let slot = self.sub_slot_size();
        slot - (self.sample_pos & (slot - 1))
    }
}
