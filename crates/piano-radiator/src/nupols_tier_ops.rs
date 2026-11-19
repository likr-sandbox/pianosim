//! Setup, IR loading, spectrum import, and sliced state-clearing helpers for [`NupolsTier`].

use piano_spec::PRESET_RADIATOR_ZONES;

use crate::fir_head::FirHead;
use crate::ir_synth::{path_index, NUM_IR_PATHS};
use crate::nupols_collapse::{collapse_zone_spectra_causal, compute_mimo_p_count, compute_p_count};
use crate::nupols_tier::NupolsTier;

#[inline]
pub(crate) fn slice_range(len: usize, slice_idx: usize, total_slices: usize) -> (usize, usize) {
    if total_slices == 0 || slice_idx >= total_slices || len == 0 {
        return (0, 0);
    }
    let start = ((len as u64 * slice_idx as u64) / total_slices as u64) as usize;
    let end = ((len as u64 * (slice_idx + 1) as u64) / total_slices as u64) as usize;
    (start, end)
}

impl NupolsTier {
    /// Reconfigure active partition count and MIMO split tap with zero allocations.
    pub(crate) fn set_active_layout(&mut self, ir_len: usize, split_tap: u16) {
        self.set_active_layout_inner(ir_len, split_tap, true);
    }

    /// Reconfigure active partition count and MIMO split tap without resetting FDLs.
    pub(crate) fn set_active_layout_no_reset(&mut self, ir_len: usize, split_tap: u16) {
        self.set_active_layout_inner(ir_len, split_tap, false);
    }

    fn set_active_layout_inner(&mut self, ir_len: usize, split_tap: u16, reset_on_change: bool) {
        let new_p = compute_p_count(self.cfg, ir_len).min(self.max_p_count);
        let new_mimo = compute_mimo_p_count(self.cfg, new_p, split_tap);
        if new_p != self.p_count || new_mimo != self.mimo_p_count {
            self.p_count = new_p;
            self.mimo_p_count = new_mimo;
            if reset_on_change {
                self.reset();
            }
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
                collapse_zone_spectra_causal(slice, &mut self.fft, work, 0, cr, ci);
                let (tr, ti) = self.acc_l.planes_mut();
                collapse_zone_spectra_causal(slice, &mut self.fft, work, 1, tr, ti);
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

    pub(crate) fn clear_small_state(&mut self) {
        self.acc_l.clear();
        self.acc_r.clear();
        self.collapsed_x_scratch.clear();
        self.acc_dirty = false;
        self.fdl_head = 0;
        self.has_pending = false;
        self.pending_s_in_end = 0;
        self.fdl_nz.fill(false);
    }

    pub(crate) fn clear_fdl_slice(&mut self, slice_idx: usize, total_slices: usize) {
        let (start, end) = slice_range(self.fdl.len(), slice_idx, total_slices);
        for buf in &mut self.fdl[start..end] {
            buf.clear();
        }
    }

    pub(crate) fn reset(&mut self) {
        for buf in &mut self.fdl {
            buf.clear();
        }
        self.clear_small_state();
    }
}
