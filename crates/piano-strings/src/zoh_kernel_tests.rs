use super::*;
use crate::StringBank;
use piano_simd::{Complex, RealFft};
use piano_spec::{build_scale_design, VoicingSpec};

#[test]
fn tiled_kernel_matches_single_sample_across_partial_tiles() {
    const FRAMES: usize = 13; // 1 full tile (8) + 1 partial tail (5)
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let map = ChoirMap::build_with_specs(&specs, &voicing);
    let mut tiled_soa = ModalSoa::build(48_000.0, &voicing, &specs, &map);
    let mut ref_soa = tiled_soa.clone();

    for s_idx in [0usize, 120, 229] {
        let r = map.strings()[s_idx].all_vecs();
        inject_excitation(
            &mut tiled_soa.u.as_f32x4_slice_mut()[r.clone()],
            &tiled_soa.b_exc.as_f32x4_slice()[r.clone()],
            1.0,
        );
        inject_excitation(
            &mut ref_soa.u.as_f32x4_slice_mut()[r.clone()],
            &ref_soa.b_exc.as_f32x4_slice()[r],
            1.0,
        );
    }

    let (mut tl, mut tr) = (vec![0.0_f32; FRAMES], vec![0.0_f32; FRAMES]);
    step_all_tiles(&mut tiled_soa, &map, &mut tl, &mut tr);

    for f in 0..FRAMES {
        let (rl, rr) = step_all_strings(&mut ref_soa, &map);
        assert!((tl[f] - rl).abs() < 1e-6 && (tr[f] - rr).abs() < 1e-6);
    }
    flush_subnormals_soa(&mut ref_soa);
    for (a, b) in tiled_soa
        .u
        .as_mut_slice()
        .iter()
        .zip(ref_soa.u.as_mut_slice())
    {
        assert!((*a - *b).abs() < 1e-6);
    }
}

#[test]
fn a4_fundamental_dominates_second_partial() {
    const N: usize = 32_768;
    let mut bank = StringBank::new(48_000.0, &VoicingSpec::default());
    bank.apply_excitation_sample(48, [1.0; 3], 1.0);
    let (mut out_l, mut out_r) = (vec![0.0_f32; N], vec![0.0_f32; N]);
    bank.render_key_subblock(48, &mut out_l, &mut out_r);
    let mono: Vec<f32> = out_l.iter().zip(&out_r).map(|(l, r)| l + r).collect();
    let mut rfft = RealFft::new(N);
    let mut spec = vec![Complex::ZERO; rfft.spectrum_len()];
    rfft.forward(&mono, &mut spec);

    let max_bin = (1..spec.len())
        .max_by(|&a, &b| spec[a].norm_sq().total_cmp(&spec[b].norm_sq()))
        .unwrap();
    let peak_hz = (max_bin as f64) * 48_000.0 / (N as f64);
    let band_max = |lo: usize, hi: usize| {
        spec[lo..=hi]
            .iter()
            .fold(0.0_f32, |m, c| m.max(c.norm_sq()))
    };
    let db_margin = 10.0 * (band_max(290, 310) / band_max(590, 615).max(1e-20)).log10();
    eprintln!("A4 peak_hz={peak_hz:.2} Hz, fund_vs_2nd={db_margin:.2} dB");
    assert!((435.0..=445.0).contains(&peak_hz) && db_margin > 3.9);
}

#[test]
fn subblock_kernel_matches_tiled_128_frames() {
    const FRAMES: usize = 131; // 1 full CHUNK=128 + partial tail=3
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let map = ChoirMap::build_with_specs(&specs, &voicing);
    let mut sub_soa = ModalSoa::build(48_000.0, &voicing, &specs, &map);
    let mut tile_soa = sub_soa.clone();
    for s_idx in [0usize, 120, 229] {
        let r = map.strings()[s_idx].all_vecs();
        inject_excitation(
            &mut sub_soa.u.as_f32x4_slice_mut()[r.clone()],
            &sub_soa.b_exc.as_f32x4_slice()[r.clone()],
            1.5,
        );
        inject_excitation(
            &mut tile_soa.u.as_f32x4_slice_mut()[r.clone()],
            &tile_soa.b_exc.as_f32x4_slice()[r],
            1.5,
        );
    }
    let (mut sl, mut sr_v) = (vec![0.0_f32; FRAMES], vec![0.0_f32; FRAMES]);
    for e in map.strings() {
        step_string_subblock(
            &mut sub_soa,
            e.all_vecs(),
            e.pan_l,
            e.pan_r,
            &mut sl,
            &mut sr_v,
        );
    }
    flush_subnormals_soa(&mut sub_soa);
    let (mut tl, mut tr) = (vec![0.0_f32; FRAMES], vec![0.0_f32; FRAMES]);
    step_all_tiles(&mut tile_soa, &map, &mut tl, &mut tr);
    for f in 0..FRAMES {
        assert!((sl[f] - tl[f]).abs() < 1e-5 && (sr_v[f] - tr[f]).abs() < 1e-5);
    }
    for (a, b) in sub_soa
        .u
        .as_mut_slice()
        .iter()
        .zip(tile_soa.u.as_mut_slice())
    {
        assert!((*a - *b).abs() < 1e-5);
    }
}

#[test]
fn accum_soa_range_matches_single_sample_parity() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let map = ChoirMap::build_with_specs(&specs, &voicing);
    let base_soa = ModalSoa::build(48_000.0, &voicing, &specs, &map);

    for (s_idx, c_len) in [(0usize, 1usize), (10, 3), (48, 7), (120, 127), (229, 128)] {
        let mut soa_fast = base_soa.clone();
        let mut soa_ref = base_soa.clone();
        let vr = map.strings()[s_idx].vert_vecs.clone();
        inject_excitation(
            &mut soa_fast.u.as_f32x4_slice_mut()[vr.clone()],
            &soa_fast.b_exc.as_f32x4_slice()[vr.clone()],
            2.25,
        );
        inject_excitation(
            &mut soa_ref.u.as_f32x4_slice_mut()[vr.clone()],
            &soa_ref.b_exc.as_f32x4_slice()[vr.clone()],
            2.25,
        );

        let mut acc_fast = vec![F32x4::zero(); c_len];
        accum_soa_range_in(&mut soa_fast, vr.clone(), &mut acc_fast);

        let (u, v, d, s, c) = (
            &mut soa_ref.u.as_f32x4_slice_mut()[vr.clone()],
            &mut soa_ref.v.as_f32x4_slice_mut()[vr.clone()],
            &soa_ref.d_coeff.as_f32x4_slice()[vr.clone()],
            &soa_ref.s_coeff.as_f32x4_slice()[vr.clone()],
            &soa_ref.c_bridge.as_f32x4_slice()[vr],
        );
        for acc_slot in &mut acc_fast {
            let mut sum_ref = F32x4::zero();
            for i in 0..u.len() {
                (u[i], v[i]) = zoh_rotate(u[i], v[i], d[i], s[i]);
                sum_ref = v[i].mul_add(c[i], sum_ref);
            }
            for (a, b) in acc_slot.to_array().iter().zip(sum_ref.to_array().iter()) {
                assert!((*a - *b).abs() < 1e-6, "diff={} vs {}", a, b);
            }
        }
    }
}
