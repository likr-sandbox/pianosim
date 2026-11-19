//! Integration tests for sliced runtime state clearing and precomputed
//! `block_fft_head` / `sub_tier` `SpaceBlob` spectra + legacy v2 fallback.

use piano_radiator::{
    space_blob_len_for_profile, LidState, ListenerPerspective, PerfProfile, RadiatorBank,
    SpacePrepContext, PRESET_RADIATOR_ZONES, SPACE_BLOB_FLAG_HEAD_SPECTRA,
};

fn dirty_bank_with_loud_signal(bank: &mut RadiatorBank) {
    let mut loud = vec![[0.0_f32; PRESET_RADIATOR_ZONES]; 512];
    for (i, frame) in loud.iter_mut().enumerate() {
        for (z, slot) in frame.iter_mut().enumerate() {
            let phase = ((i * 7 + z * 13) % 97) as f32 * 0.05;
            *slot = if i < 256 {
                1.4 * phase.sin()
            } else {
                0.3 * phase.cos()
            };
        }
    }
    let mut l = [0.0_f32; 512];
    let mut r = [0.0_f32; 512];
    for _ in 0..16 {
        bank.process_block(&loud, &mut l, &mut r);
    }
}

#[test]
fn clear_runtime_state_slice_matches_full_reset_across_all_profiles() {
    let sr = 48_000.0;
    let test_frames = [[0.22_f32; PRESET_RADIATOR_ZONES]; 256];
    let mut ref_l = [0.0_f32; 256];
    let mut ref_r = [0.0_f32; 256];
    let mut out_l = [0.0_f32; 256];
    let mut out_r = [0.0_f32; 256];

    // 1. Direct slice-by-slice equivalence with reset(), plus bounds safety.
    let mut reset_bank = RadiatorBank::new_concert_grand(sr);
    let mut sliced_bank = RadiatorBank::new_concert_grand(sr);
    reset_bank.set_reverb(1.8, 0.35);
    sliced_bank.set_reverb(1.8, 0.35);
    dirty_bank_with_loud_signal(&mut reset_bank);
    dirty_bank_with_loud_signal(&mut sliced_bank);
    reset_bank.reset();
    sliced_bank.clear_runtime_state_slice(0, 0);
    sliced_bank.clear_runtime_state_slice(16, 16);
    for s in 0..16 {
        sliced_bank.clear_runtime_state_slice(s, 16);
    }
    reset_bank.process_block(&test_frames, &mut ref_l, &mut ref_r);
    sliced_bank.process_block(&test_frames, &mut out_l, &mut out_r);
    assert_eq!(ref_l, out_l);
    assert_eq!(ref_r, out_r);

    // 2. Streamed profile switches with sliced standby clearing in 2,048-float chunks.
    let transitions = [
        (
            PerfProfile::full(),
            PerfProfile::realtime(),
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
        ),
        (
            PerfProfile::realtime(),
            PerfProfile {
                fir_head_taps: 16,
                ..PerfProfile::balanced()
            },
            ListenerPerspective::ConcertHall,
            LidState::Closed,
        ),
        (
            PerfProfile {
                fir_head_taps: 16,
                ..PerfProfile::balanced()
            },
            PerfProfile::full(),
            ListenerPerspective::CloseStereo,
            LidState::Open,
        ),
    ];
    let mut prep = SpacePrepContext::new_concert_grand(sr);
    for (from_prof, to_prof, persp, lid) in transitions {
        let mut bank = RadiatorBank::new_with_profile(
            sr,
            ListenerPerspective::Player,
            LidState::Open,
            &from_prof,
        );
        bank.set_reverb(1.6, 0.25);
        dirty_bank_with_loud_signal(&mut bank);

        prep.set_perf_profile(&to_prof);
        let blob = prep.prepare_space(persp, lid);
        assert_eq!(blob.len(), space_blob_len_for_profile(sr, &to_prof));

        let total = blob.len();
        let mut off = 0usize;
        let mut slice_idx = 0usize;
        while off < total {
            if slice_idx < 16 {
                bank.clear_runtime_state_slice(slice_idx, 16);
                slice_idx += 1;
            }
            let n = 2_048.min(total - off);
            assert!(bank.install_precomputed_space_chunk(
                persp,
                lid,
                off,
                &blob[off..off + n],
                total,
            ));
            off += n;
        }
        while slice_idx < 16 {
            bank.clear_runtime_state_slice(slice_idx, 16);
            slice_idx += 1;
        }

        let mut ref_bank = RadiatorBank::new_with_profile(sr, persp, lid, &to_prof);
        ref_bank.set_reverb(1.6, 0.25);
        ref_bank.process_block(&test_frames, &mut ref_l, &mut ref_r);
        bank.process_block(&test_frames, &mut out_l, &mut out_r);
        assert_eq!(ref_l, out_l, "L mismatch on transition to {to_prof:?}");
        assert_eq!(ref_r, out_r, "R mismatch on transition to {to_prof:?}");
    }
}

#[test]
fn space_blob_precomputed_head_spectra_and_legacy_v2_fallback() {
    let sr = 48_000.0;
    let mut prep = SpacePrepContext::new_concert_grand(sr);
    let frames = [[0.18_f32; PRESET_RADIATOR_ZONES]; 256];
    let mut ref_l = [0.0_f32; 256];
    let mut ref_r = [0.0_f32; 256];
    let mut out_l = [0.0_f32; 256];
    let mut out_r = [0.0_f32; 256];

    // Full profile must NOT set SPACE_BLOB_FLAG_HEAD_SPECTRA.
    prep.set_perf_profile(&PerfProfile::full());
    let full_blob = prep.prepare_space(ListenerPerspective::Player, LidState::Open);
    assert_eq!(full_blob[7].to_bits() & SPACE_BLOB_FLAG_HEAD_SPECTRA, 0);

    let head_profiles = [
        PerfProfile::realtime(),
        PerfProfile {
            fir_head_taps: 32,
            ..PerfProfile::balanced()
        },
        PerfProfile {
            fir_head_taps: 16,
            ..PerfProfile::balanced()
        },
    ];
    for prof in &head_profiles {
        prep.set_perf_profile(prof);
        let blob = prep
            .prepare_space(ListenerPerspective::BinauralHrtf, LidState::Half)
            .to_vec();
        assert_ne!(blob[7].to_bits() & SPACE_BLOB_FLAG_HEAD_SPECTRA, 0);

        let mut ref_bank = RadiatorBank::new_with_profile(
            sr,
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            prof,
        );
        ref_bank.process_block(&frames, &mut ref_l, &mut ref_r);

        // Reject invalid flag bits cleanly.
        let mut bad = blob.clone();
        bad[7] = f32::from_bits(blob[7].to_bits() | 4);
        let mut bad_bank = RadiatorBank::new_concert_grand(sr);
        assert!(!bad_bank.install_precomputed_space(
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            &bad,
        ));

        // Synthesize legacy v2 blob (without precomputed head spectra) and verify fallback.
        let legacy_len = space_blob_len_for_profile(
            sr,
            &PerfProfile {
                fir_head_taps: 64,
                fir_block_fft: false,
                ..*prof
            },
        );
        assert!(legacy_len < blob.len());
        let mut legacy = blob[..legacy_len].to_vec();
        legacy[7] = f32::from_bits(blob[7].to_bits() & !SPACE_BLOB_FLAG_HEAD_SPECTRA);

        let mut one_shot = RadiatorBank::new_concert_grand(sr);
        assert!(one_shot.install_precomputed_space(
            ListenerPerspective::BinauralHrtf,
            LidState::Half,
            &legacy,
        ));
        one_shot.process_block(&frames, &mut out_l, &mut out_r);
        assert_eq!(ref_l, out_l);
        assert_eq!(ref_r, out_r);

        let mut chunked = RadiatorBank::new_concert_grand(sr);
        let mut off = 0usize;
        while off < legacy_len {
            let n = 2_048.min(legacy_len - off);
            assert!(chunked.install_precomputed_space_chunk(
                ListenerPerspective::BinauralHrtf,
                LidState::Half,
                off,
                &legacy[off..off + n],
                legacy_len,
            ));
            off += n;
        }
        chunked.process_block(&frames, &mut out_l, &mut out_r);
        assert_eq!(ref_l, out_l);
        assert_eq!(ref_r, out_r);
    }
}
