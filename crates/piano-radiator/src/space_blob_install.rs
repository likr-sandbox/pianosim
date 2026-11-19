//! Zero-allocation `RadiatorBank` installation of precomputed `SpaceBlob`s
//! (including precomputed `block_fft_head`/`sub_tier` spectra with v2 fallback).

use piano_spec::{LidState, ListenerPerspective};

use crate::fir_head::{FIR_COEFFS_LEN, HEAD_TAPS};
use crate::space_blob::{
    head_spectra_float_len, space_blob_len_for_params, SPACE_BLOB_FLAG_BLOCK_FFT,
    SPACE_BLOB_FLAG_HEAD_SPECTRA, SPACE_BLOB_HEADER_LEN, SPACE_BLOB_MAGIC_V2,
};
use crate::RadiatorBank;

pub(crate) struct ParsedSpaceHeader {
    pub(crate) ir_len: usize,
    pub(crate) head_taps: u8,
    pub(crate) split_tap: u16,
    pub(crate) fir_block_fft: bool,
    pub(crate) has_head_spectra: bool,
    pub(crate) expected_len: usize,
}

fn parse_space_header(
    header: &[f32],
    sample_rate: f64,
    max_ir_len: usize,
    perspective: ListenerPerspective,
    lid_state: LidState,
) -> Option<ParsedSpaceHeader> {
    if header.len() < SPACE_BLOB_HEADER_LEN
        || header[0].to_bits() != SPACE_BLOB_MAGIC_V2
        || (f64::from(header[1]) - sample_rate).abs() > 0.5
        || header[2].to_bits() != perspective as u32
        || header[3].to_bits() != lid_state as u32
    {
        return None;
    }
    let ir_len = header[4].to_bits() as usize;
    let head_taps_u32 = header[5].to_bits();
    let split_tap_u32 = header[6].to_bits();
    let flags = header[7].to_bits();
    if ir_len < 256
        || ir_len > max_ir_len
        || !matches!(head_taps_u32, 16 | 32 | 64)
        || split_tap_u32 < 512
        || split_tap_u32 > u32::from(u16::MAX)
        || (flags & !(SPACE_BLOB_FLAG_BLOCK_FFT | SPACE_BLOB_FLAG_HEAD_SPECTRA)) != 0
    {
        return None;
    }
    let head_taps = head_taps_u32 as u8;
    let split_tap = split_tap_u32 as u16;
    let fir_block_fft = (flags & SPACE_BLOB_FLAG_BLOCK_FFT) != 0;
    let has_head_spectra = (flags & SPACE_BLOB_FLAG_HEAD_SPECTRA) != 0;
    let extra = if has_head_spectra {
        head_spectra_float_len(fir_block_fft, head_taps)
    } else {
        0
    };
    let expected_len = space_blob_len_for_params(ir_len, split_tap) + extra;
    Some(ParsedSpaceHeader {
        ir_len,
        head_taps,
        split_tap,
        fir_block_fft,
        has_head_spectra,
        expected_len,
    })
}

impl RadiatorBank {
    pub(crate) fn apply_blob_stage4_config(
        &mut self,
        hdr: &ParsedSpaceHeader,
        reset_on_change: bool,
    ) {
        let block_toggled = hdr.fir_block_fft != self.fir_block_fft;
        self.fir_block_fft = hdr.fir_block_fft;
        if reset_on_change {
            self.head.set_active_taps(hdr.head_taps);
            self.tail.set_stage4_config(
                hdr.ir_len,
                hdr.split_tap,
                hdr.head_taps,
                hdr.fir_block_fft,
            );
            if block_toggled {
                self.head.reset();
                self.block_fft_head.reset();
            }
        } else {
            self.head.set_active_taps_no_reset(hdr.head_taps);
            self.tail.set_stage4_config_no_reset(
                hdr.ir_len,
                hdr.split_tap,
                hdr.head_taps,
                hdr.fir_block_fft,
            );
        }
        self.scratch.ir_len = self.tail.active_ir_len();
        self.profile.ir_duration_secs = (hdr.ir_len as f64 / self.sample_rate) as f32;
        self.profile.fir_head_taps = hdr.head_taps;
        self.profile.spatial_tail_split_tap = hdr.split_tap;
        self.profile.fir_block_fft = hdr.fir_block_fft;
    }

    /// Installs a precomputed spatial FIR + NUPOLS spectrum blob with zero allocations.
    pub fn install_precomputed_space(
        &mut self,
        perspective: ListenerPerspective,
        lid_state: LidState,
        blob: &[f32],
    ) -> bool {
        let Some(hdr) = parse_space_header(
            blob,
            self.sample_rate,
            self.tail.max_ir_len(),
            perspective,
            lid_state,
        ) else {
            return false;
        };
        if blob.len() != hdr.expected_len {
            return false;
        }
        self.apply_blob_stage4_config(&hdr, true);

        let fir_start = SPACE_BLOB_HEADER_LEN;
        let fir_end = fir_start + FIR_COEFFS_LEN;
        let head_slice = &blob[fir_start..fir_end];
        if hdr.has_head_spectra {
            let tail_end = fir_end + self.tail.ir_spectra_float_len();
            if !self.tail.import_ir_spectra(&blob[fir_end..tail_end]) {
                return false;
            }
            if hdr.fir_block_fft {
                if !self.block_fft_head.import_ir_spectra(&blob[tail_end..]) {
                    return false;
                }
            } else if hdr.head_taps == 64 {
                self.head.load_coeffs_range(0, head_slice);
            } else {
                self.head.load_coeffs_from_full_slice(head_slice);
                if !self.tail.import_sub_tier_ir_spectra(&blob[tail_end..]) {
                    return false;
                }
            }
        } else {
            if hdr.fir_block_fft {
                self.block_fft_head
                    .load_from_reversed_head_coeffs(head_slice);
            } else if hdr.head_taps == 64 {
                self.head.load_coeffs_range(0, head_slice);
            } else {
                self.head.load_coeffs_from_full_slice(head_slice);
                self.tail
                    .load_sub_tier_from_reversed_head_coeffs(head_slice);
            }
            if !self.tail.import_ir_spectra(&blob[fir_end..]) {
                return false;
            }
        }
        self.perspective = perspective;
        self.lid_state = lid_state;
        self.hall_fdn.set_perspective(perspective);
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
        if blob_off == 0 {
            let Some(hdr) = parse_space_header(
                chunk,
                self.sample_rate,
                self.tail.max_ir_len(),
                perspective,
                lid_state,
            ) else {
                return false;
            };
            if total_len != hdr.expected_len || blob_off + chunk.len() > total_len {
                return false;
            }
            self.staged_blob_has_head_spectra = hdr.has_head_spectra;
            self.apply_blob_stage4_config(&hdr, false);
        } else {
            let extra = if self.staged_blob_has_head_spectra {
                head_spectra_float_len(self.fir_block_fft, self.profile.fir_head_taps)
            } else {
                0
            };
            let expected_len = space_blob_len_for_params(
                self.tail.active_ir_len(),
                self.profile.spatial_tail_split_tap,
            ) + extra;
            if total_len != expected_len || blob_off + chunk.len() > total_len {
                return false;
            }
        }

        let chunk_end = blob_off + chunk.len();
        let fir_start = SPACE_BLOB_HEADER_LEN;
        let fir_end = fir_start + FIR_COEFFS_LEN;
        let f_lo = blob_off.max(fir_start);
        let f_hi = chunk_end.min(fir_end);
        if f_lo < f_hi && !(self.fir_block_fft && self.staged_blob_has_head_spectra) {
            self.head
                .load_coeffs_range(f_lo - fir_start, &chunk[f_lo - blob_off..f_hi - blob_off]);
        }
        let tail_end = fir_end + self.tail.ir_spectra_float_len();
        let t_lo = blob_off.max(fir_end);
        let t_hi = chunk_end.min(tail_end);
        if t_lo < t_hi {
            self.tail
                .import_ir_spectra_range(t_lo - fir_end, &chunk[t_lo - blob_off..t_hi - blob_off]);
        }
        if self.staged_blob_has_head_spectra {
            let h_lo = blob_off.max(tail_end);
            let h_hi = chunk_end.min(total_len);
            if h_lo < h_hi {
                let slice = &chunk[h_lo - blob_off..h_hi - blob_off];
                if self.fir_block_fft {
                    self.block_fft_head
                        .import_ir_spectra_range(h_lo - tail_end, slice);
                } else {
                    self.tail
                        .import_sub_tier_ir_spectra_range(h_lo - tail_end, slice);
                }
            }
        }
        if chunk_end == total_len {
            if self.staged_blob_has_head_spectra {
                if !self.fir_block_fft && self.head.active_taps() < HEAD_TAPS {
                    self.head.repack_staged_to_active();
                }
            } else if self.fir_block_fft {
                self.block_fft_head.load_from_staged_fir_head(&self.head);
            } else if self.head.active_taps() < HEAD_TAPS {
                self.tail.load_sub_tier_from_staged_fir_head(&self.head);
                self.head.repack_staged_to_active();
            }
            self.perspective = perspective;
            self.lid_state = lid_state;
            self.hall_fdn.set_perspective(perspective);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{space_blob_len_for_sample_rate, SpacePrepContext};
    use piano_spec::PRESET_RADIATOR_ZONES;

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
