//! Opt-in 32-sample block-FFT FIR head (`block_size = 32`, `RealFft(64)`,
//! `ir_offset = 0`, 2 partitions covering `[0..64)`) with 32-sample latency.

extern crate alloc;

use alloc::vec::Vec;
use piano_simd::{RealFft, SplitComplexBuf};
use piano_spec::PRESET_RADIATOR_ZONES;

use crate::fir_head::{FirHead, HEAD_TAPS};
use crate::ir_synth::{path_index, NUM_IR_PATHS};

pub(crate) const BLOCK_FFT_SIZE: usize = 32;
pub(crate) const BLOCK_FFT_PARTITIONS: usize = PARTITIONS;
pub(crate) const BLOCK_FFT_SPEC_LEN: usize = BLOCK_FFT_SIZE + 1;
const FFT_LEN: usize = BLOCK_FFT_SIZE * 2;
const PARTITIONS: usize = 2;

/// Zero-allocation 2-partition 64-pt overlap-save block-FFT head covering taps `[0..64)`.
#[derive(Clone)]
pub(crate) struct BlockFftHead {
    fft: RealFft,
    ir_spectra: Vec<SplitComplexBuf>,
    fdl: Vec<SplitComplexBuf>,
    fdl_nz: [bool; PARTITIONS * PRESET_RADIATOR_ZONES],
    fdl_head: usize,
    in_prev: [[f32; BLOCK_FFT_SIZE]; PRESET_RADIATOR_ZONES],
    in_curr: [[f32; BLOCK_FFT_SIZE]; PRESET_RADIATOR_ZONES],
    out_l: [f32; BLOCK_FFT_SIZE],
    out_r: [f32; BLOCK_FFT_SIZE],
    pos: usize,
    acc_l: SplitComplexBuf,
    acc_r: SplitComplexBuf,
    work: [f32; FFT_LEN],
}

impl BlockFftHead {
    pub(crate) fn new() -> Self {
        let fft = RealFft::new(FFT_LEN);
        let spec_len = fft.spectrum_len();
        let mut ir_spectra = Vec::with_capacity(PARTITIONS * NUM_IR_PATHS);
        for _ in 0..(PARTITIONS * NUM_IR_PATHS) {
            ir_spectra.push(SplitComplexBuf::zeroed(spec_len));
        }
        let mut fdl = Vec::with_capacity(PARTITIONS * PRESET_RADIATOR_ZONES);
        for _ in 0..(PARTITIONS * PRESET_RADIATOR_ZONES) {
            fdl.push(SplitComplexBuf::zeroed(spec_len));
        }
        Self {
            fft,
            ir_spectra,
            fdl,
            fdl_nz: [false; PARTITIONS * PRESET_RADIATOR_ZONES],
            fdl_head: 0,
            in_prev: [[0.0; BLOCK_FFT_SIZE]; PRESET_RADIATOR_ZONES],
            in_curr: [[0.0; BLOCK_FFT_SIZE]; PRESET_RADIATOR_ZONES],
            out_l: [0.0; BLOCK_FFT_SIZE],
            out_r: [0.0; BLOCK_FFT_SIZE],
            pos: 0,
            acc_l: SplitComplexBuf::zeroed(spec_len),
            acc_r: SplitComplexBuf::zeroed(spec_len),
            work: [0.0; FFT_LEN],
        }
    }

    pub(crate) fn reset(&mut self) {
        for buf in &mut self.fdl {
            buf.clear();
        }
        self.fdl_nz.fill(false);
        self.fdl_head = 0;
        self.in_prev = [[0.0; BLOCK_FFT_SIZE]; PRESET_RADIATOR_ZONES];
        self.in_curr = [[0.0; BLOCK_FFT_SIZE]; PRESET_RADIATOR_ZONES];
        self.out_l = [0.0; BLOCK_FFT_SIZE];
        self.out_r = [0.0; BLOCK_FFT_SIZE];
        self.pos = 0;
        self.acc_l.clear();
        self.acc_r.clear();
        self.work = [0.0; FFT_LEN];
    }

    pub(crate) fn load_from_ir_flat(&mut self, ir_flat: &[f32], ir_len: usize) {
        for p in 0..PARTITIONS {
            let tap_start = p * BLOCK_FFT_SIZE;
            for z in 0..PRESET_RADIATOR_ZONES {
                for ch in 0..2 {
                    let ir_base = path_index(z, ch) * ir_len;
                    self.work.fill(0.0);
                    if tap_start < ir_len {
                        let copy_len = BLOCK_FFT_SIZE.min(ir_len - tap_start);
                        self.work[..copy_len].copy_from_slice(
                            &ir_flat[ir_base + tap_start..ir_base + tap_start + copy_len],
                        );
                    }
                    let spec = &mut self.ir_spectra[p * NUM_IR_PATHS + path_index(z, ch)];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(&self.work, re, im);
                }
            }
        }
    }

    pub(crate) fn load_from_reversed_head_coeffs(&mut self, coeffs: &[f32]) {
        for z in 0..PRESET_RADIATOR_ZONES {
            for ch in 0..2 {
                let base = z * HEAD_TAPS * 2 + ch * HEAD_TAPS;
                for p in 0..PARTITIONS {
                    let tap_start = p * BLOCK_FFT_SIZE;
                    self.work.fill(0.0);
                    for k in 0..BLOCK_FFT_SIZE {
                        self.work[k] = coeffs[base + (HEAD_TAPS - 1) - (tap_start + k)];
                    }
                    let spec = &mut self.ir_spectra[p * NUM_IR_PATHS + path_index(z, ch)];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(&self.work, re, im);
                }
            }
        }
    }

    pub(crate) fn load_from_staged_fir_head(&mut self, fir_head: &FirHead) {
        let mut ir64 = [0.0_f32; HEAD_TAPS];
        for z in 0..PRESET_RADIATOR_ZONES {
            for ch in 0..2 {
                fir_head.extract_staged_ir_64(z, ch, &mut ir64);
                for p in 0..PARTITIONS {
                    let tap_start = p * BLOCK_FFT_SIZE;
                    self.work.fill(0.0);
                    self.work[..BLOCK_FFT_SIZE]
                        .copy_from_slice(&ir64[tap_start..tap_start + BLOCK_FFT_SIZE]);
                    let spec = &mut self.ir_spectra[p * NUM_IR_PATHS + path_index(z, ch)];
                    let (re, im) = spec.planes_mut();
                    self.fft.forward_split(&self.work, re, im);
                }
            }
        }
    }

    pub(crate) fn import_ir_spectra(&mut self, src: &[f32]) -> bool {
        let spec_len = self.fft.spectrum_len();
        if src.len() != self.ir_spectra.len() * spec_len * 2 {
            return false;
        }
        let mut off = 0usize;
        for spec in &mut self.ir_spectra {
            let (re, im) = spec.planes_mut();
            re.copy_from_slice(&src[off..off + spec_len]);
            off += spec_len;
            im.copy_from_slice(&src[off..off + spec_len]);
            off += spec_len;
        }
        true
    }

    pub(crate) fn import_ir_spectra_range(&mut self, head_off: usize, src: &[f32]) {
        if src.is_empty() {
            return;
        }
        let spec_len = self.fft.spectrum_len();
        let src_end = head_off.saturating_add(src.len());
        let mut cur = 0usize;
        for spec in &mut self.ir_spectra {
            let (re, im) = spec.planes_mut();
            for plane in [re, im] {
                let p_end = cur + spec_len;
                let lo = cur.max(head_off);
                let hi = p_end.min(src_end);
                if lo < hi {
                    plane[lo - cur..hi - cur].copy_from_slice(&src[lo - head_off..hi - head_off]);
                }
                cur = p_end;
            }
        }
    }

    pub(crate) fn process_frames(
        &mut self,
        frames: &[[f32; PRESET_RADIATOR_ZONES]],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = frames.len().min(out_l.len()).min(out_r.len());
        let mut i = 0;
        while i < n {
            out_l[i] = self.out_l[self.pos];
            out_r[i] = self.out_r[self.pos];
            let fr = &frames[i];
            let mut z = 0;
            while z < PRESET_RADIATOR_ZONES {
                let raw = fr[z];
                self.in_curr[z][self.pos] = if raw.is_finite() { raw } else { 0.0 };
                z += 1;
            }
            self.pos += 1;
            if self.pos == BLOCK_FFT_SIZE {
                self.process_completed_block();
                self.pos = 0;
            }
            i += 1;
        }
    }

    fn process_completed_block(&mut self) {
        self.fdl_head ^= 1;
        let head = self.fdl_head;
        let prev_slot = head ^ 1;
        let mut any_nz = false;

        let mut z = 0;
        while z < PRESET_RADIATOR_ZONES {
            self.work[..BLOCK_FFT_SIZE].copy_from_slice(&self.in_prev[z]);
            self.work[BLOCK_FFT_SIZE..FFT_LEN].copy_from_slice(&self.in_curr[z]);
            self.in_prev[z] = self.in_curr[z];
            let nz = self.work.iter().any(|&v| v != 0.0);
            let slot = head * PRESET_RADIATOR_ZONES + z;
            self.fdl_nz[slot] = nz;
            if nz {
                any_nz = true;
                let (re, im) = self.fdl[slot].planes_mut();
                self.fft.forward_split(&self.work, re, im);
            }
            z += 1;
        }

        let prev_base = prev_slot * PRESET_RADIATOR_ZONES;
        let prev_nz = self.fdl_nz[prev_base..prev_base + PRESET_RADIATOR_ZONES]
            .iter()
            .any(|&b| b);
        if !any_nz && !prev_nz {
            self.out_l.fill(0.0);
            self.out_r.fill(0.0);
            return;
        }

        self.acc_l.clear();
        self.acc_r.clear();
        let mut p = 0;
        while p < PARTITIONS {
            let fdl_slot = head ^ p;
            let fdl_base = fdl_slot * PRESET_RADIATOR_ZONES;
            let ir_base = p * NUM_IR_PATHS;
            let mut z2 = 0;
            while z2 < PRESET_RADIATOR_ZONES {
                if self.fdl_nz[fdl_base + z2] {
                    SplitComplexBuf::fma_accumulate_lr(
                        &mut self.acc_l,
                        &mut self.acc_r,
                        &self.fdl[fdl_base + z2],
                        &self.ir_spectra[ir_base + path_index(z2, 0)],
                        &self.ir_spectra[ir_base + path_index(z2, 1)],
                    );
                }
                z2 += 1;
            }
            p += 1;
        }

        let (re_l, im_l) = self.acc_l.planes();
        self.fft.inverse_split(re_l, im_l, &mut self.work);
        self.out_l
            .copy_from_slice(&self.work[BLOCK_FFT_SIZE..FFT_LEN]);

        let (re_r, im_r) = self.acc_r.planes();
        self.fft.inverse_split(re_r, im_r, &mut self.work);
        self.out_r
            .copy_from_slice(&self.work[BLOCK_FFT_SIZE..FFT_LEN]);
    }
}
