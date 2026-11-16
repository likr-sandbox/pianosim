//! Off-thread spatial IR + NUPOLS FFT spectrum precomputation for zero-stall
//! `AudioWorklet` space switching (`no_std` + `alloc`).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use piano_simd::{AlignedBuf, RealFft};
use piano_spec::{BodyPreset, LidState, ListenerPerspective, PRESET_RADIATOR_ZONES};

use crate::fir_head::{FIR_COEFFS_LEN, HEAD_TAPS};
use crate::ir_synth::{
    ir_len_for_sample_rate, path_index, synthesize_irs, IrSynthScratch, NUM_IR_PATHS,
};
use crate::math::clamp_f64;
use crate::nupols_tier::{MAX_TIER_FFT_LEN, TIER_CONFIGS};
use crate::{empty_fallback_preset, RadiatorBank, CONCERT_GRAND_PRESET_BYTES};

const SPACE_BLOB_MAGIC: u32 = 0x5042_5350; // "PBSP"
const SPACE_BLOB_HEADER_LEN: usize = 4;

/// Computes the exact number of `f32` elements in a serialized spatial NUPOLS blob.
#[must_use]
pub fn space_blob_len_for_sample_rate(sample_rate: f64) -> usize {
    let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
    let ir_len = ir_len_for_sample_rate(sr);
    let mut spectra_floats = 0usize;
    for &cfg in &TIER_CONFIGS {
        let end = ir_len.min(cfg.end_limit);
        let p_count = if end > cfg.ir_offset {
            (end - cfg.ir_offset).div_ceil(cfg.block_size)
        } else {
            0
        };
        let spec_len = cfg.block_size + 1; // (block_size * 2) / 2 + 1
        spectra_floats += p_count * NUM_IR_PATHS * spec_len * 2;
    }
    SPACE_BLOB_HEADER_LEN + FIR_COEFFS_LEN + spectra_floats
}

/// Off-thread precomputation engine producing serialized spatial FIR + NUPOLS blobs
/// without touching real-time audio delay lines.
pub struct SpacePrepContext {
    preset: BodyPreset,
    sample_rate: f64,
    scratch: IrSynthScratch,
    ffts: [RealFft; 4],
    time_scratch: AlignedBuf,
    spec_re: Vec<f32>,
    spec_im: Vec<f32>,
    out_blob: Vec<f32>,
}

impl SpacePrepContext {
    /// Constructs a spatial prep context from the baked concert-grand preset.
    #[must_use]
    pub fn new_concert_grand(sample_rate: f64) -> Self {
        let preset = BodyPreset::decode(CONCERT_GRAND_PRESET_BYTES)
            .unwrap_or_else(|_| empty_fallback_preset());
        let sr = clamp_f64(sample_rate, 8_000.0, 384_000.0);
        let ir_len = ir_len_for_sample_rate(sr);
        let max_spec = MAX_TIER_FFT_LEN / 2 + 1;
        Self {
            preset,
            sample_rate: sr,
            scratch: IrSynthScratch::new(ir_len),
            ffts: [
                RealFft::new(TIER_CONFIGS[0].block_size * 2),
                RealFft::new(TIER_CONFIGS[1].block_size * 2),
                RealFft::new(TIER_CONFIGS[2].block_size * 2),
                RealFft::new(TIER_CONFIGS[3].block_size * 2),
            ],
            time_scratch: AlignedBuf::zeroed(MAX_TIER_FFT_LEN),
            spec_re: vec![0.0; max_spec],
            spec_im: vec![0.0; max_spec],
            out_blob: Vec::with_capacity(space_blob_len_for_sample_rate(sr)),
        }
    }

    /// Synthesizes IRs and pre-FFTs all NUPOLS partitions into a flat POD `f32` slice.
    pub fn prepare_space(
        &mut self,
        perspective: ListenerPerspective,
        lid_state: LidState,
    ) -> &[f32] {
        synthesize_irs(
            &self.preset,
            perspective,
            lid_state,
            self.sample_rate,
            &mut self.scratch,
        );
        self.out_blob.clear();
        self.out_blob.push(f32::from_bits(SPACE_BLOB_MAGIC));
        self.out_blob.push(self.sample_rate as f32);
        self.out_blob.push(f32::from_bits(perspective as u32));
        self.out_blob.push(f32::from_bits(lid_state as u32));

        let ir_len = self.scratch.ir_len;
        let ir_flat = &self.scratch.ir_flat;
        let copy_len = ir_len.min(HEAD_TAPS);
        let mut rev_l = [0.0_f32; HEAD_TAPS];
        let mut rev_r = [0.0_f32; HEAD_TAPS];
        for z in 0..PRESET_RADIATOR_ZONES {
            rev_l.fill(0.0);
            rev_r.fill(0.0);
            let l_start = path_index(z, 0) * ir_len;
            let r_start = path_index(z, 1) * ir_len;
            for k in 0..copy_len {
                let dst = (HEAD_TAPS - 1) - k;
                rev_l[dst] = ir_flat[l_start + k];
                rev_r[dst] = ir_flat[r_start + k];
            }
            self.out_blob.extend_from_slice(&rev_l);
            self.out_blob.extend_from_slice(&rev_r);
        }

        let work_all = self.time_scratch.as_mut_slice();
        for (ti, &cfg) in TIER_CONFIGS.iter().enumerate() {
            let end = ir_len.min(cfg.end_limit);
            let p_count = if end > cfg.ir_offset {
                (end - cfg.ir_offset).div_ceil(cfg.block_size)
            } else {
                0
            };
            if p_count == 0 {
                continue;
            }
            let b = cfg.block_size;
            let fft = &mut self.ffts[ti];
            let spec_len = fft.spectrum_len();
            let work = &mut work_all[..b * 2];
            for p in 0..p_count {
                let tap_start = cfg.ir_offset + p * b;
                for z in 0..PRESET_RADIATOR_ZONES {
                    for ch in 0..2 {
                        let ir_base = path_index(z, ch) * ir_len;
                        work.fill(0.0);
                        if tap_start < ir_len {
                            let c_len = b.min(ir_len - tap_start);
                            work[..c_len].copy_from_slice(
                                &ir_flat[ir_base + tap_start..ir_base + tap_start + c_len],
                            );
                        }
                        let re = &mut self.spec_re[..spec_len];
                        let im = &mut self.spec_im[..spec_len];
                        fft.forward_split(work, re, im);
                        self.out_blob.extend_from_slice(re);
                        self.out_blob.extend_from_slice(im);
                    }
                }
            }
        }
        &self.out_blob
    }

    /// Returns the most recently prepared spatial blob slice.
    #[must_use]
    pub fn out_blob(&self) -> &[f32] {
        &self.out_blob
    }
}

impl RadiatorBank {
    /// Installs a precomputed spatial FIR + NUPOLS spectrum blob with zero allocations
    /// and zero FFT computation on the audio thread.
    pub fn install_precomputed_space(
        &mut self,
        perspective: ListenerPerspective,
        lid_state: LidState,
        blob: &[f32],
    ) -> bool {
        let expected_len =
            SPACE_BLOB_HEADER_LEN + FIR_COEFFS_LEN + self.tail.ir_spectra_float_len();
        if blob.len() != expected_len
            || blob[0].to_bits() != SPACE_BLOB_MAGIC
            || (f64::from(blob[1]) - self.sample_rate).abs() > 0.5
            || blob[2].to_bits() != perspective as u32
            || blob[3].to_bits() != lid_state as u32
        {
            return false;
        }
        let fir_start = SPACE_BLOB_HEADER_LEN;
        let fir_end = fir_start + FIR_COEFFS_LEN;
        self.head.load_coeffs_slice(&blob[fir_start..fir_end]);
        if !self.tail.import_ir_spectra(&blob[fir_end..]) {
            return false;
        }
        self.perspective = perspective;
        self.lid_state = lid_state;
        true
    }

    /// Streams one slice `[blob_off .. blob_off + chunk.len())` of a precomputed spatial blob.
    pub fn install_precomputed_space_chunk(
        &mut self,
        perspective: ListenerPerspective,
        lid_state: LidState,
        blob_off: usize,
        chunk: &[f32],
        total_len: usize,
    ) -> bool {
        let expected_len =
            SPACE_BLOB_HEADER_LEN + FIR_COEFFS_LEN + self.tail.ir_spectra_float_len();
        if total_len != expected_len || blob_off + chunk.len() > total_len {
            return false;
        }
        if blob_off == 0
            && (chunk.len() < SPACE_BLOB_HEADER_LEN
                || chunk[0].to_bits() != SPACE_BLOB_MAGIC
                || (f64::from(chunk[1]) - self.sample_rate).abs() > 0.5
                || chunk[2].to_bits() != perspective as u32
                || chunk[3].to_bits() != lid_state as u32)
        {
            return false;
        }
        let chunk_end = blob_off + chunk.len();
        let fir_start = SPACE_BLOB_HEADER_LEN;
        let fir_end = fir_start + FIR_COEFFS_LEN;
        let f_lo = blob_off.max(fir_start);
        let f_hi = chunk_end.min(fir_end);
        if f_lo < f_hi {
            self.head
                .load_coeffs_range(f_lo - fir_start, &chunk[f_lo - blob_off..f_hi - blob_off]);
        }
        let t_lo = blob_off.max(fir_end);
        let t_hi = chunk_end;
        if t_lo < t_hi {
            self.tail
                .import_ir_spectra_range(t_lo - fir_end, &chunk[t_lo - blob_off..t_hi - blob_off]);
        }
        if chunk_end == total_len {
            self.perspective = perspective;
            self.lid_state = lid_state;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_blob_install_matches_reconfigure_space_bit_exact() {
        let sr = 48_000.0;
        let mut ref_bank = RadiatorBank::new_concert_grand(sr);
        let mut fast_bank = RadiatorBank::new_concert_grand(sr);
        let mut prep = SpacePrepContext::new_concert_grand(sr);

        let cases = [
            (ListenerPerspective::ConcertHall, LidState::Half),
            (ListenerPerspective::BinauralHrtf, LidState::Closed),
            (ListenerPerspective::CloseStereo, LidState::Open),
        ];
        let frames = [[0.25_f32; PRESET_RADIATOR_ZONES]; 128];
        let mut ref_l = [0.0_f32; 128];
        let mut ref_r = [0.0_f32; 128];
        let mut fast_l = [0.0_f32; 128];
        let mut fast_r = [0.0_f32; 128];

        for (persp, lid) in cases {
            ref_bank.reconfigure_space(persp, lid);
            let blob = prep.prepare_space(persp, lid);
            assert_eq!(blob.len(), space_blob_len_for_sample_rate(sr));
            assert!(fast_bank.install_precomputed_space(persp, lid, blob));
            ref_bank.process_block(&frames, &mut ref_l, &mut ref_r);
            fast_bank.process_block(&frames, &mut fast_l, &mut fast_r);
            assert_eq!(ref_l, fast_l);
            assert_eq!(ref_r, fast_r);
        }
        assert!(!fast_bank.install_precomputed_space(
            ListenerPerspective::Player,
            LidState::Open,
            &[0.0; 16],
        ));

        let mut chunk_bank = RadiatorBank::new_concert_grand(sr);
        let blob = prep.prepare_space(ListenerPerspective::ConcertHall, LidState::Closed);
        let total = blob.len();
        let mut off = 0usize;
        while off < total {
            let n = 7_919.min(total - off);
            assert!(chunk_bank.install_precomputed_space_chunk(
                ListenerPerspective::ConcertHall,
                LidState::Closed,
                off,
                &blob[off..off + n],
                total,
            ));
            off += n;
        }
        let mut ref_chunk_bank = RadiatorBank::new_concert_grand(sr);
        ref_chunk_bank.reconfigure_space(ListenerPerspective::ConcertHall, LidState::Closed);
        ref_chunk_bank.process_block(&frames, &mut ref_l, &mut ref_r);
        chunk_bank.process_block(&frames, &mut fast_l, &mut fast_r);
        assert_eq!(ref_l, fast_l);
        assert_eq!(ref_r, fast_r);
    }
}
