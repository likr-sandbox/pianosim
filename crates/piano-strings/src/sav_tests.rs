use super::*;
use crate::zoh_kernel::step_string_modes;
use crate::StringBank;
use core::f64::consts::TAU;
use piano_spec::{build_scale_design, key_tuned_f0_hz, partial_frequency_hz, VoicingSpec};

fn blackman_harris_goertzel_power(samples: &[f64], freq_hz: f64, sr: f64) -> f64 {
    let coeff = 2.0 * (TAU * freq_hz / sr).cos();
    let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
    for &x in samples {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

fn measure_f0_hz(mono: &[f32], f_nominal_hz: f64, sr: f64) -> f64 {
    let n = mono.len();
    let denom = (n - 1) as f64;
    let win: Vec<f64> = mono
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let p = TAU * (i as f64) / denom;
            let w =
                0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
            f64::from(x) * w
        })
        .collect();
    let mut best_f = f_nominal_hz;
    let mut best_p = 0.0_f64;
    for step in -200..=200 {
        let f_try = f_nominal_hz * 2.0_f64.powf((step as f64 * 0.5) / 1200.0);
        let p = blackman_harris_goertzel_power(&win, f_try, sr);
        if p > best_p {
            best_p = p;
            best_f = f_try;
        }
    }
    let dc = 0.5_f64;
    let r = 2.0_f64.powf(dc / 1200.0);
    let p_lo = blackman_harris_goertzel_power(&win, best_f / r, sr);
    let p_mid = blackman_harris_goertzel_power(&win, best_f, sr);
    let p_hi = blackman_harris_goertzel_power(&win, best_f * r, sr);
    let denom_q = p_lo - 2.0 * p_mid + p_hi;
    let shift = if denom_q < 0.0 {
        (0.5 * (p_lo - p_hi) / denom_q).clamp(-1.0, 1.0) * dc
    } else {
        0.0
    };
    best_f * 2.0_f64.powf(shift / 1200.0)
}

#[test]
fn lossless_energy_conservation_under_1e5_with_nontrivial_nl() {
    let voicing = VoicingSpec::default();
    let mut bank = StringBank::new_lossless(48_000.0, &voicing);
    let (mut l, mut r) = ([0.0_f32; 32], [0.0_f32; 32]);

    for &key in &[0usize, 15] {
        bank.reset();
        bank.apply_excitation_sample(key, [1.0, 0.0, 0.0], 24.0);
        let h0 = bank.key_sav_energy_joules(key);
        assert!(h0 > 0.0);

        let mut max_rel_err = 0.0_f64;
        let mut max_enl_ratio = 0.0_f64;
        for _ in 0..750 {
            bank.render_key_subblock(key, &mut l, &mut r);
            let h_cur = bank.key_sav_energy_joules(key);
            let e_nl = bank.key_nonlinear_energy_joules(key);
            max_rel_err = max_rel_err.max((h_cur - h0).abs() / h0);
            max_enl_ratio = max_enl_ratio.max(e_nl / h0);
        }
        eprintln!("lossless key={key}: max_rel_err={max_rel_err:e}, max_E_nl/H0={max_enl_ratio:e}");
        assert!(
            max_enl_ratio > 1.0e-3,
            "key={key} nonlinear regime too weak: {max_enl_ratio:e}"
        );
        assert!(
            max_rel_err < 1.0e-5,
            "key={key} energy drift exceeded 1e-5: {max_rel_err:e}"
        );
    }
}

#[test]
fn damped_monotone_passivity_and_drift_regulation_ab() {
    let voicing = VoicingSpec::default();
    let mut bank = StringBank::new(48_000.0, &voicing);
    bank.apply_excitation_sample(0, [1.0, 0.0, 0.0], 20.0);
    let mut h_prev = bank.key_sav_energy_joules(0);
    let h0 = h_prev;
    let (mut l1, mut r1) = ([0.0_f32; 1], [0.0_f32; 1]);
    let mut max_increase = 0.0_f64;
    for _ in 0..4_800 {
        bank.render_key_subblock(0, &mut l1, &mut r1);
        let h_cur = bank.key_sav_energy_joules(0);
        max_increase = max_increase.max(h_cur - h_prev);
        assert!(
            h_cur <= h_prev + 1.0e-8 * h0,
            "passivity violation h_prev={h_prev:e} h_cur={h_cur:e}"
        );
        h_prev = h_cur;
    }
    eprintln!("damped A0 max_increase/H0 = {:e}", max_increase / h0);

    // Risse-Helie-Bilbao drift regulation A/B: lambda_reg = 400 vs lambda_reg = 0.
    let sr = 48_000.0_f64;
    let specs = build_scale_design(&voicing);
    let map = crate::choir_map::ChoirMap::build_with_specs(&specs, &voicing);
    let entry = &map.strings()[0];
    let run_ab = |lambda_reg: f64| -> (f64, f64, f64) {
        let mut soa = ModalSoa::build_with_lossless(sr, &voicing, &specs, &map, false);
        let ports = crate::contact_ports::ContactPorts::build(sr, &voicing, &specs, &map);
        let mut sav = SavBank::build(sr, &voicing, &specs, &map, &ports.gamma_s, &ports.beta_s);
        let r_all = entry.all_vecs();
        crate::zoh_kernel::inject_excitation(
            &mut soa.u.as_f32x4_slice_mut()[r_all.clone()],
            &soa.b_exc.as_f32x4_slice()[r_all.clone()],
            24.0,
        );
        let (mut eps_max, mut eps_sq, steps) = (0.0_f64, 0.0_f64, 36_000usize);
        let mut last_eps = 0.0_f64;
        for step_i in 0..steps {
            if step_i >= 4_800 {
                // Damped string ring-down drives E_nl -> 0 and psi -> 1 to isolate SAV drift.
                for idx in (r_all.start * 4)..(r_all.end * 4) {
                    soa.u.as_mut_slice()[idx] *= 0.999;
                    soa.v.as_mut_slice()[idx] *= 0.999;
                }
            }
            let _ = step_string_modes(
                &mut soa.u.as_f32x4_slice_mut()[r_all.clone()],
                &mut soa.v.as_f32x4_slice_mut()[r_all.clone()],
                &soa.d_coeff.as_f32x4_slice()[r_all.clone()],
                &soa.s_coeff.as_f32x4_slice()[r_all.clone()],
                &soa.c_bridge.as_f32x4_slice()[r_all.clone()],
            );
            sav.step_string_sav_with_lambda(0, entry, &mut soa, true, lambda_reg);
            let psi = sav.eval_psi(0, entry, &soa, true);
            let eps = (sav.r_sav[0] - psi).abs();
            last_eps = eps;
            if step_i % 6000 == 0 || step_i == steps - 1 {
                eprintln!(
                    "  lam={lambda_reg} step={step_i}: r_sav={:.6} psi={psi:.6} eps={eps:e}",
                    sav.r_sav[0]
                );
            }
            eps_max = eps_max.max(eps);
            eps_sq += eps * eps;
        }
        (eps_max, (eps_sq / steps as f64).sqrt(), last_eps)
    };
    let (max_reg, rms_reg, end_reg) = run_ab(LAMBDA_REG_DEFAULT);
    let (max_unreg, rms_unreg, end_unreg) = run_ab(0.0);
    eprintln!(
        "SAV drift A/B: reg(max={max_reg:e}, rms={rms_reg:e}, end={end_reg:e}) vs unreg(max={max_unreg:e}, rms={rms_unreg:e}, end={end_unreg:e})"
    );
    assert!(
        end_reg < 0.05 * end_unreg && rms_reg < 0.25 * rms_unreg,
        "drift regulation failed: reg=({max_reg:e},{rms_reg:e},{end_reg:e}) unreg=({max_unreg:e},{rms_unreg:e},{end_unreg:e})"
    );
}

#[test]
fn hundred_strike_ff_pitch_drift_under_0_5_cents_across_a0_c3() {
    const N_PITCH: usize = 32_768;
    let sr = 48_000.0_f32;
    let sr64 = f64::from(sr);
    let voicing = VoicingSpec {
        unison_detune_scale: 0.0,
        ..VoicingSpec::default()
    };
    let specs = build_scale_design(&voicing);
    let mut bank = StringBank::new(sr, &voicing);
    let (mut l_ring, mut r_ring) = (vec![0.0_f32; N_PITCH], vec![0.0_f32; N_PITCH]);
    let (mut l_mid, mut r_mid) = (vec![0.0_f32; 2_400], vec![0.0_f32; 2_400]);

    for &key in &[0usize, 3, 15, 27] {
        bank.reset();
        let f_nom = partial_frequency_hz(
            key_tuned_f0_hz(key, &specs, &voicing),
            specs[key].inharmonicity_b,
            1,
        );
        let mut f_first = f_nom;
        let mut f_last = f_nom;
        for strike in 0..100 {
            bank.apply_excitation_sample(key, [1.0, 1.0, 1.0], 8.0);
            if strike == 0 || strike == 99 {
                bank.render_key_subblock(key, &mut l_ring, &mut r_ring);
                let mono: Vec<f32> = l_ring
                    .iter()
                    .zip(r_ring.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                assert!(mono.iter().all(|x| x.is_finite()));
                let f_est = measure_f0_hz(&mono, f_nom, sr64);
                if strike == 0 {
                    f_first = f_est;
                } else {
                    f_last = f_est;
                }
            } else {
                bank.render_key_subblock(key, &mut l_mid, &mut r_mid);
            }
        }
        let s0 = bank.choir_map().key(key).unwrap().string_range.start;
        let entry = &bank.choir_map().strings()[s0];
        let psi_end = bank.sav.eval_psi(s0, entry, &bank.soa, true);
        let r_sav_end = bank.sav.r_sav[s0];

        // Engage passive damper for 0.4s via coupled contact ports, then verify quiet probe -> f_nom.
        struct DampDriver {
            active_key: usize,
            engaged: bool,
        }
        impl crate::KeyActionDriver for DampDriver {
            fn is_hammer_active(&self, _key_idx: usize) -> bool {
                false
            }
            fn damper_clearance_m(&self, key_idx: usize) -> f32 {
                if self.engaged && key_idx == self.active_key {
                    -1.0e-4
                } else {
                    5.0e-3
                }
            }
            fn strike_weights(&self, _key_idx: usize) -> [f32; 3] {
                [1.0; 3]
            }
            fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
                self.damper_clearance_m(key_idx)
            }
            fn step_key_clearance_tile(
                &mut self,
                key_idx: usize,
                t_len: usize,
                out_clears: &mut [f32; crate::zoh_kernel::TILE],
            ) -> f32 {
                let c = self.damper_clearance_m(key_idx);
                out_clears[..t_len].fill(c);
                c
            }
            fn step_hammer_sample(&mut self, _: usize, _: f32, _: f32, _: f32) -> f32 {
                0.0
            }
            fn step_damper_sample(
                &mut self,
                _key_idx: usize,
                _slot_idx: usize,
                _y_damp_m: f32,
                v_damp_mps: f32,
                r_port_damp: f32,
                _y_clear_m: f32,
            ) -> f32 {
                -v_damp_mps / (r_port_damp + 0.005)
            }
        }
        let mut drv = DampDriver {
            active_key: key,
            engaged: true,
        };
        let (mut l_rel, mut r_rel) = (vec![0.0_f32; 72_000], vec![0.0_f32; 72_000]);
        bank.render_subblock_coupled(&mut drv, &mut l_rel, &mut r_rel);
        bank.apply_excitation_sample(key, [1.0, 1.0, 1.0], 0.02);
        let r_sav_probe = bank.sav.r_sav[s0];
        bank.render_key_subblock(key, &mut l_ring, &mut r_ring);
        let probe_mono: Vec<f32> = l_ring
            .iter()
            .zip(r_ring.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let f_probe = measure_f0_hz(&probe_mono, f_nom, sr64);
        let probe_cents = (1200.0 * (f_probe / f_nom).log2()).abs();

        // Briefly damp the small probe and fire strike 101 at full ff without resetting bank.
        bank.render_subblock_coupled(&mut drv, &mut l_mid, &mut r_mid);
        bank.apply_excitation_sample(key, [1.0, 1.0, 1.0], 8.0);
        bank.render_key_subblock(key, &mut l_ring, &mut r_ring);
        let ff101_mono: Vec<f32> = l_ring
            .iter()
            .zip(r_ring.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let f_101 = measure_f0_hz(&ff101_mono, f_nom, sr64);
        let ff_drift_cents = (1200.0 * (f_101 / f_first).log2()).abs();
        let accum_glide_cents = 1200.0 * (f_last / f_first).log2();
        eprintln!(
            "100-strike ff key={key}: f_nom={f_nom:.4} f1={f_first:.4} f101={f_101:.4} drift={ff_drift_cents:.4}c probe={probe_cents:.4}c r_sav={r_sav_end:.5} psi={psi_end:.5} r_prb={r_sav_probe:.5} accum_glide={accum_glide_cents:+.1}c"
        );
        assert!(
            ff_drift_cents < 0.5 && probe_cents < 0.05,
            "key={key} drift={ff_drift_cents}c probe={probe_cents}c"
        );
        assert!(
            (r_sav_end - psi_end).abs() < 0.01 && (r_sav_probe - 1.0).abs() < 1e-3,
            "key={key} r_sav_end={r_sav_end} psi_end={psi_end} r_sav_probe={r_sav_probe}"
        );
    }
}
