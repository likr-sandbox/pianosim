use super::*;
use piano_spec::build_scale_design;

#[test]
fn nostd_math_matches_std_and_poles_preserve_padding_invariants() {
    for i in -32..=32 {
        let a = f64::from(i) * 0.413_789_123;
        let (s, c) = sin_cos_f64(a);
        assert!((s - a.sin()).abs() < 1e-14 && (c - a.cos()).abs() < 1e-14);
    }
    for &y in &[-25.0_f64, -1.5, -1e-6, 0.0, 1.5, 10.0] {
        assert!((exp_f64(y) - y.exp()).abs() <= 1e-13 * y.exp().max(1.0));
    }
    for &alpha in &[1.0e-7_f64, 2.4e-6, 5.0e-4, 2.0e-2] {
        assert!((one_minus_exp_neg_f64(alpha) - (1.0 - (-alpha).exp())).abs() < 1e-15);
    }

    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let map = ChoirMap::build_with_specs(&specs, &voicing);
    let soa = ModalSoa::build(96_000.0, &voicing, &specs, &map);
    assert_eq!(soa.u.as_ptr() as usize % 64, 0);
    assert_eq!(soa.d_coeff.as_ptr() as usize % 64, 0);

    let a0 = &map.strings()[0];
    for slot in (a0.vert_vecs.start * 4 + a0.vert_modes)..(a0.vert_vecs.end * 4) {
        assert_eq!(
            (
                soa.d_coeff[slot],
                soa.s_coeff[slot],
                soa.c_bridge[slot],
                soa.b_exc[slot]
            ),
            (1.0, 0.0, 0.0, 0.0)
        );
    }
    assert!(soa.d_coeff[a0.vert_vecs.start * 4] > 0.0);
    assert!(soa.d_coeff[a0.vert_vecs.start * 4] < 1.0e-3);
    assert!(soa.s_coeff[a0.vert_vecs.start * 4] > 0.0);

    for &sr in &[44_100.0_f64, 48_000.0, 96_000.0] {
        for &freq in &[27.5_f64, 65.406, 130.813, 200.0, 261.626, 440.0] {
            let (d, s) = coupled_pole_f32(freq, sr, f64::INFINITY).expect("valid lossless pole");
            let err = ((1.0 - f64::from(d)).powi(2) + f64::from(s).powi(2) - 1.0).abs();
            let limit = if freq <= 262.0 { 1.0e-10 } else { 5.0e-10 };
            assert!(err <= limit, "sr={sr} freq={freq} err={err:e}");
        }
    }
}
