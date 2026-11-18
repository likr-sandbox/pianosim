use super::*;
use crate::contact::tests::MockAction;
use crate::StringBank;
use piano_spec::{build_scale_design, VoicingSpec, DAMPED_KEYS};

#[test]
fn port_duality_conservation_and_damper_profile_invariants() {
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let map = ChoirMap::build_with_specs(&specs, &voicing);

    for &k in &[0_usize, 15] {
        let sp = specs[k];
        let p1 = damper_phi(k, 1, sp.speaking_length_m, sp.f0_et_hz);
        let p3 = damper_phi(
            k,
            3,
            sp.speaking_length_m,
            partial_frequency_hz(sp.f0_et_hz, sp.inharmonicity_b, 3),
        );
        let p4 = damper_phi(
            k,
            4,
            sp.speaking_length_m,
            partial_frequency_hz(sp.f0_et_hz, sp.inharmonicity_b, 4),
        );
        let (ratio_3, ratio_4) = (p3 / p1, p4 / p1);
        assert!((2.8..=3.5).contains(&ratio_3), "k={k} ratio_3={ratio_3}");
        assert!((2.8..=3.5).contains(&ratio_4), "k={k} ratio_4={ratio_4}");
    }
    let e4 = specs[43];
    let e4_base = damper_phi(43, 1, e4.speaking_length_m, e4.f0_et_hz);
    let e4_4k = damper_phi(43, 10, e4.speaking_length_m, 4_000.0);
    let e4_10k = damper_phi(43, 21, e4.speaking_length_m, 10_000.0);
    assert!(
        e4_base > e4_4k && e4_4k > e4_10k && e4_10k < 0.45 * e4_base,
        "expected monotonic E4 HF damper rolloff: base={e4_base:.4}, 4k={e4_4k:.4}, 10k={e4_10k:.4}"
    );

    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        let mut bank = StringBank::new(sr, &voicing);
        for (s_idx, entry) in map.strings().iter().enumerate() {
            let vr = entry.all_vecs();
            let bs = &bank.ports.b_strike.as_slice()[vr.start * 4..vr.end * 4];
            let bd = &bank.ports.b_damp.as_slice()[vr.start * 4..vr.end * 4];
            let ad = &bank.ports.a_damp.as_slice()[vr.start * 4..vr.end * 4];
            let alpha = f64::from(bank.ports.alpha_vel[s_idx]);
            let r_s_check = alpha * bs.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
            let rel_s = (r_s_check - f64::from(bank.ports.r_port_strike[s_idx])).abs()
                / r_s_check.max(1e-20);
            assert!(rel_s < 1e-5, "sr={sr} s={s_idx} rel_s={rel_s}");

            if (entry.key_idx as usize) >= DAMPED_KEYS {
                assert!(bd.iter().all(|&x| x == 0.0) && ad.iter().all(|&x| x == 0.0));
                assert_eq!(bank.ports.r_port_damp_v[s_idx], 0.0);
                assert_eq!(bank.ports.r_port_damp_h[s_idx], 0.0);
            } else {
                let bd_all = bank.ports.b_damp.as_slice();
                let ad_all = bank.ports.a_damp.as_slice();
                let (vv, hv, lv) = (
                    entry.vert_vecs.clone(),
                    entry.horiz_vecs.clone(),
                    entry.long_vecs.clone(),
                );
                let bd_v = &bd_all[vv.start * 4..vv.end * 4];
                let bd_h = &bd_all[hv.start * 4..hv.end * 4];
                let bd_l = &bd_all[lv.start * 4..lv.end * 4];
                let ad_l = &ad_all[lv.start * 4..lv.end * 4];
                assert!(bd_l.iter().all(|&x| x == 0.0) && ad_l.iter().all(|&x| x == 0.0));

                let rv_check = alpha
                    * bd_v
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>();
                let rh_check = alpha
                    * bd_h
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>();
                let rel_v = (rv_check - f64::from(bank.ports.r_port_damp_v[s_idx])).abs()
                    / rv_check.max(1e-20);
                let rel_h = (rh_check - f64::from(bank.ports.r_port_damp_h[s_idx])).abs()
                    / rh_check.max(1e-20);
                assert!(rel_v < 1e-5, "sr={sr} s={s_idx} rel_v={rel_v}");
                assert!(rel_h < 1e-5, "sr={sr} s={s_idx} rel_h={rel_h}");
            }
        }

        // 1-sample impulse duality check on A0 and C4 (linear energy & smooth velocity duality).
        for &k in &[0usize, 39] {
            bank.reset();
            bank.sav_phantom_enabled = false;
            bank.longitudinal_enabled = false;
            let s_idx = map.keys()[k].string_range.start;
            let expected_beta = f64::from(bank.ports.alpha_vel[s_idx]) / f64::from(sr);
            let rel_beta =
                (bank.ports.beta_s[s_idx] - expected_beta).abs() / bank.ports.beta_s[s_idx];
            assert!(rel_beta < 1e-5, "sr={sr} k={k} rel_beta={rel_beta}");
            let mut drv = MockAction::new(sr);
            drv.hammer_active[k] = true;
            drv.last_impulse_once = Some((k, 10.0));
            let (mut ol, mut or) = ([0.0_f32; 1], [0.0_f32; 1]);
            bank.render_subblock_coupled(&mut drv, &mut ol, &mut or);
            let h_lin =
                bank.ports
                    .linear_energy_joules(s_idx, &bank.soa, map.strings()[s_idx].all_vecs());
            assert!(h_lin > 0.0 && h_lin.is_finite());
            let f_str = 10.0 / (map.keys()[k].string_range.len() as f64);
            let expected_h =
                0.5 * f64::from(bank.ports.r_port_strike[s_idx]) * f_str * f_str / f64::from(sr);
            let rel_h = (h_lin - expected_h).abs() / expected_h.max(1e-20);
            assert!(
                rel_h < 1e-3,
                "sr={sr} k={k} h_lin={h_lin} vs expected={expected_h}"
            );
            let (yh, vh) = (bank.ports.y_h[s_idx], bank.ports.v_h[s_idx]);
            assert!(yh > 0.0 && vh.is_finite());
            bank.reset();
            let mut drv_s = MockAction::new(sr);
            drv_s.strike(k, 1.8);
            let (mut sl, mut sr_buf) = ([0.0_f32; 256], [0.0_f32; 256]);
            bank.render_subblock_coupled(&mut drv_s, &mut sl, &mut sr_buf);
            (drv_s.hammer_active[k], drv_s.y_h[k], drv_s.v_h[k]) = (true, -0.05, 0.0);
            let (mut sl1, mut sr1) = ([0.0_f32; 1], [0.0_f32; 1]);
            bank.render_subblock_coupled(&mut drv_s, &mut sl1, &mut sr1);
            let (yh0, vh0) = (bank.ports.y_h[s_idx], bank.ports.v_h[s_idx]);
            bank.render_subblock_coupled(&mut drv_s, &mut sl1, &mut sr1);
            let (yh1, vh1) = (bank.ports.y_h[s_idx], bank.ports.v_h[s_idx]);
            let (y_dot, v_mid) = ((yh1 - yh0) * sr, 0.5 * (vh0 + vh1));
            let rel = (y_dot - v_mid).abs() / v_mid.abs().max(1e-9);
            assert!(rel < 0.05, "sr={sr} k={k} y_dot={y_dot} vs v_mid={v_mid}");
        }
    }
}

#[test]
fn strike_phi_singularity_guard_near_u_equals_one_on_short_treble_strings() {
    for k in [0usize, 39, 75, 87] {
        let w_h = hammer_contact_width_m(k);
        // Exact u = 1.0 (e.g., short string length L = j * w_h) must never yield NaN/Inf.
        for j in 1..=8 {
            let l_exact = (j as f64) * w_h;
            let p_exact = strike_phi(k, j, l_exact);
            let p_left = strike_phi(k, j, l_exact * 1.000_05);
            let p_right = strike_phi(k, j, l_exact * 0.999_95);
            assert!(p_exact.is_finite() && p_left.is_finite() && p_right.is_finite());
            assert!((p_exact - 0.5 * (p_left + p_right)).abs() < 1e-4);
        }
    }
}
