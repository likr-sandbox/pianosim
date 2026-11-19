use super::*;
use crate::contact_ports::HORIZ_DAMP_SCALE;
use crate::StringBank;
use piano_simd::{Complex, RealFft};
use piano_spec::{build_scale_design, VoicingSpec, DAMPED_KEYS, NUM_KEYS};

pub(crate) struct MockAction {
    sr: f32,
    pub(crate) hammer_active: [bool; NUM_KEYS],
    pub(crate) y_h: [f32; NUM_KEYS],
    pub(crate) v_h: [f32; NUM_KEYS],
    m_h: [f32; NUM_KEYS],
    k_felt: [f32; NUM_KEYS],
    elapsed_samples: [usize; NUM_KEYS],
    y_clear: [f32; NUM_KEYS],
    weights: [[f32; 3]; NUM_KEYS],
    damper_calls: [usize; NUM_KEYS],
    slot_parity: [u8; NUM_KEYS],
    pub(crate) last_impulse_once: Option<(usize, f32)>,
}

impl MockAction {
    pub(crate) fn new(sr: f32) -> Self {
        let mut k_felt = [0.0_f32; NUM_KEYS];
        let mut m_h = [0.0_f32; NUM_KEYS];
        for k in 0..NUM_KEYS {
            let x = (k as f32) / ((NUM_KEYS - 1) as f32);
            let m = 0.0112 - 0.0077 * x;
            let tau_c = (0.0032 * (-0.35 * x - 2.75 * x * x * x).exp()).max(0.00014);
            // Physical indentation peak delta_max ~ v0 * tau_c / 2.2 at mf (1.8 m/s).
            let omega_c = core::f32::consts::PI / tau_c;
            let delta_ref = (1.8 * tau_c / 2.2).max(5.0e-5);
            m_h[k] = m;
            k_felt[k] = m * omega_c * omega_c / delta_ref.powf(1.5);
        }
        Self {
            sr,
            hammer_active: [false; NUM_KEYS],
            y_h: [-2.0e-5; NUM_KEYS],
            v_h: [0.0; NUM_KEYS],
            m_h,
            k_felt,
            elapsed_samples: [0; NUM_KEYS],
            y_clear: [5.0e-3; NUM_KEYS],
            weights: [[1.0; 3]; NUM_KEYS],
            damper_calls: [0; NUM_KEYS],
            slot_parity: [0; NUM_KEYS],
            last_impulse_once: None,
        }
    }

    pub(crate) fn strike(&mut self, k: usize, vel_mps: f32) {
        self.hammer_active[k] = true;
        self.y_h[k] = -2.0e-5;
        self.v_h[k] = vel_mps;
        self.elapsed_samples[k] = 0;
    }
}

impl KeyActionDriver for MockAction {
    fn is_hammer_active(&self, key_idx: usize) -> bool {
        self.hammer_active[key_idx]
    }

    fn damper_clearance_m(&self, key_idx: usize) -> f32 {
        self.y_clear[key_idx]
    }

    fn strike_weights(&self, key_idx: usize) -> [f32; 3] {
        self.weights[key_idx]
    }

    fn step_key_clearance(&mut self, key_idx: usize) -> f32 {
        self.y_clear[key_idx]
    }

    fn step_key_clearance_tile(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; TILE],
    ) -> f32 {
        let c = self.y_clear[key_idx];
        for yc in &mut out_clears[..t_len] {
            *yc = c;
        }
        c
    }

    fn step_hammer_sample(
        &mut self,
        key_idx: usize,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32 {
        if let Some((ik, f_once)) = self.last_impulse_once.take() {
            if ik == key_idx {
                self.hammer_active[key_idx] = false;
                return f_once;
            }
            self.last_impulse_once = Some((ik, f_once));
        }

        const SUB: usize = 8;
        let dt = 1.0 / (self.sr * SUB as f32);
        let inv_m = 1.0 / self.m_h[key_idx];
        let k_f = self.k_felt[key_idx];
        let mut ys = y_str_m;
        let mut vs = v_str_mps;
        let mut f_sum = 0.0_f32;
        for _ in 0..SUB {
            let delta = (self.y_h[key_idx] - ys).max(0.0);
            let f = if delta > 0.0 {
                let rel_v = self.v_h[key_idx] - vs;
                (k_f * delta.powf(2.5) * (1.0 + 0.15 * rel_v).max(0.1)).min(500.0)
            } else {
                0.0
            };
            self.v_h[key_idx] -= f * inv_m * dt;
            self.y_h[key_idx] += self.v_h[key_idx] * dt;
            vs += f * r_port_strike / (SUB as f32);
            ys += vs * dt;
            f_sum += f;
        }
        self.elapsed_samples[key_idx] += 1;
        if (self.y_h[key_idx] < -6.0e-5 && self.v_h[key_idx] < 0.0)
            || self.elapsed_samples[key_idx] > (self.sr * 0.015) as usize
        {
            self.hammer_active[key_idx] = false;
        }
        f_sum / (SUB as f32)
    }

    fn step_damper_sample(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        self.damper_calls[key_idx] += 1;
        self.slot_parity[key_idx] |= 1 << (slot_idx & 1);
        if y_damp_m == 0.0 && v_damp_mps == 0.0 {
            return 0.0;
        }
        let pen = ((y_damp_m.abs() - y_clear_m).max(0.0)) / 1.0e-3;
        if pen <= 0.0 {
            return 0.0;
        }
        // Unilateral passive felt damping matching DamperPadBank bilinear port passivity.
        let c_raw = 80.0 * pen.min(4.0);
        let c_eff = if r_port_damp > 0.0 {
            c_raw.min(2.4 / r_port_damp)
        } else {
            c_raw
        };
        -(c_eff * v_damp_mps) / (1.0 + 0.5 * r_port_damp * c_eff)
    }
}

#[test]
fn mock_mf_strike_loudness_rolloff_and_a4_spectrum() {
    let sr = 48_000.0_f32;
    let mut bank = StringBank::new(sr, &VoicingSpec::default());
    let len = 4_800usize; // 100 ms attack window
    let (mut ol, mut or) = (vec![0.0_f32; len], vec![0.0_f32; len]);
    let mut results = Vec::new();

    for &k in &[0usize, 27, 39, 48, 63, 75, 87] {
        bank.reset();
        let mut drv = MockAction::new(sr);
        drv.strike(k, 1.8);
        bank.render_subblock_coupled(&mut drv, &mut ol, &mut or);
        let energy_sq: f32 = ol.iter().zip(&or).map(|(l, r)| l * l + r * r).sum();
        let stereo_rms = (0.5 * energy_sq / (len as f32)).sqrt();
        let peak = ol
            .iter()
            .zip(&or)
            .fold(0.0_f32, |m, (l, r)| m.max(l.abs()).max(r.abs()));
        eprintln!("mock mf key={k:2}: stereo_rms={stereo_rms:.4}, peak={peak:.4}");
        results.push((k, stereo_rms, peak));
    }
    for (k, stereo_rms, peak) in results {
        assert!(
            (0.05..=0.10).contains(&stereo_rms) && peak < 0.7,
            "key={k} rms={stereo_rms} peak={peak}"
        );
    }

    // A4 440 Hz dominates 880 Hz by >= +5 dB.
    let n = 16_384usize;
    let (mut al, mut ar) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    bank.reset();
    let mut drv = MockAction::new(sr);
    drv.strike(48, 1.8);
    bank.render_subblock_coupled(&mut drv, &mut al, &mut ar);
    let mono: Vec<f32> = al.iter().zip(&ar).map(|(l, r)| l + r).collect();
    let mut rfft = RealFft::new(n);
    let mut spec = vec![Complex::ZERO; rfft.spectrum_len()];
    rfft.forward(&mono, &mut spec);
    let band_max = |lo: usize, hi: usize| {
        spec[lo..=hi]
            .iter()
            .fold(0.0_f32, |m, c| m.max(c.norm_sq()))
    };
    let db = 10.0 * (band_max(142, 158) / band_max(292, 308).max(1e-20)).log10();
    eprintln!("coupled A4 440 vs 880 margin = {db:.2} dB");
    assert!(db >= 5.0, "A4 440 vs 880 margin={db}");
}

#[test]
fn unilateral_damper_passivity_transition_sync_and_una_corda() {
    for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
        for &(nl_on, drive) in &[(true, 0.25_f32), (false, 0.8_f32)] {
            let mut bank = StringBank::new(sr, &VoicingSpec::default());
            bank.set_nonlinear_coupling(nl_on, nl_on);
            let mut drv = MockAction::new(sr);
            drv.y_clear.fill(-3.0e-4); // Resting preload with silent strings -> 0 damper work.
            let (mut ol, mut or) = (vec![0.0_f32; 64], vec![0.0_f32; 64]);
            bank.render_subblock_coupled(&mut drv, &mut ol, &mut or);
            assert!(ol.iter().all(|&x| x == 0.0) && or.iter().all(|&x| x == 0.0));
            assert!(drv.damper_calls.iter().all(|&c| c == 0));

            // Excite via apply_excitation_sample, step lifted (fast path), then drop dampers.
            let keys = [0usize, 39, 63, 69, 70, 75, 87];
            for &k in &keys {
                bank.apply_excitation_sample(k, [1.0; 3], drive);
                drv.y_clear[k] = 5.0e-3;
            }
            let n_step = (sr * 0.02) as usize;
            let (mut bl, mut br) = (vec![0.0_f32; n_step], vec![0.0_f32; n_step]);
            bank.render_subblock_coupled(&mut drv, &mut bl, &mut br);
            assert!(drv.damper_calls.iter().all(|&c| c == 0));

            let pre_rms: Vec<f32> = keys
                .iter()
                .map(|&k| {
                    let (v, h) = bank.measure_key_polarization_rms(k);
                    (v * v + h * h).sqrt()
                })
                .collect();
            for &k in &keys {
                drv.y_clear[k] = -3.0e-4;
            }
            let n_damp = (sr * 0.25) as usize;
            let (mut dl, mut dr) = (vec![0.0_f32; n_damp], vec![0.0_f32; n_damp]);
            bank.render_subblock_coupled(&mut drv, &mut dl, &mut dr);
            assert!(dl.iter().all(|x| x.is_finite()));
            for (&k, &e0) in keys.iter().zip(&pre_rms) {
                let (v, h) = bank.measure_key_polarization_rms(k);
                let ratio = (v * v + h * h).sqrt() / e0;
                assert!(
                    if k < DAMPED_KEYS {
                        ratio < 0.08 && drv.damper_calls[k] > 0 && drv.slot_parity[k] == 0b11
                    } else {
                        ratio > 0.14
                    },
                    "nl={nl_on} sr={sr} k={k} ratio={ratio} parity={:b}",
                    drv.slot_parity[k]
                );
            }
        }
    }

    // Choir Una Corda + monochord weight clamping + subnormal activity clearing.
    let mut bank = StringBank::new(48_000.0, &VoicingSpec::default());
    let mut drv_full = MockAction::new(48_000.0);
    drv_full.strike(0, 1.8);
    let mut drv_uc = MockAction::new(48_000.0);
    drv_uc.weights[0] = [1.0, 0.0, 0.0];
    drv_uc.strike(0, 1.8);
    let (mut l1, mut r1, mut l2, mut r2) = (
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
    );
    let mut bank2 = bank.clone();
    bank.render_subblock_coupled(&mut drv_full, &mut l1, &mut r1);
    bank2.render_subblock_coupled(&mut drv_uc, &mut l2, &mut r2);
    for i in 0..512 {
        assert!((l1[i] - l2[i]).abs() < 1e-6);
    }

    bank.reset();
    bank.mark_key_possibly_active(39);
    assert!(bank
        .choir_map
        .key(39)
        .unwrap()
        .string_range
        .clone()
        .all(|s| bank.ports.may_be_nonzero[s]));
    let mut drv_idle = MockAction::new(48_000.0);
    bank.render_subblock_coupled(&mut drv_idle, &mut l1[..16], &mut r1[..16]);
    assert!(bank
        .choir_map
        .key(39)
        .unwrap()
        .string_range
        .clone()
        .all(|s| !bank.ports.may_be_nonzero[s]));
}

#[test]
fn orthogonal_vh_damper_kills_null_space_decay() {
    let sr = 48_000.0_f32;
    let voicing = VoicingSpec::default();
    let specs = build_scale_design(&voicing);
    let map = ChoirMap::build_with_specs(&specs, &voicing);

    let mut failures = Vec::new();
    for &(nl_on, amp) in &[(false, 200.0_f32), (true, 15.0_f32)] {
        for &k in &[0usize, 15, 27, 39, 51, 63, 69] {
            let mut bank = StringBank::new(sr, &voicing);
            bank.set_nonlinear_coupling(nl_on, nl_on);
            let mut drv = MockAction::new(sr);
            let inv_h = 1.0_f32 / (HORIZ_DAMP_SCALE as f32);

            // Construct the exact anti-phase rank-1 null-space state (V + 0.85*H == 0).
            for s_idx in map.keys()[k].string_range.clone() {
                let e = &map.strings()[s_idx];
                let vb = e.vert_vecs.start * 4;
                let hb = e.horiz_vecs.start * 4;
                let n_modes = e.vert_modes.min(e.horiz_modes).min(12);
                for m in 0..n_modes {
                    let bd_v = bank.ports.b_damp.as_slice()[vb + m];
                    let u_mut = bank.soa.u.as_mut_slice();
                    u_mut[vb + m] = amp * bd_v;
                    u_mut[hb + m] = -amp * inv_h * bd_v;
                }
            }
            bank.mark_key_possibly_active(k);

            let (mut l_pre, mut r_pre) = ([0.0_f32; 32], [0.0_f32; 32]);
            drv.y_clear[k] = 5.0e-3;
            bank.render_subblock_coupled(&mut drv, &mut l_pre, &mut r_pre);
            let (v0, h0) = bank.measure_key_polarization_rms(k);
            assert!(v0 > 1.0e-5 && h0 > 1.0e-5, "k={k} v0={v0} h0={h0}");

            drv.y_clear[k] = -3.0e-4;
            let n_damp = (sr * 0.25) as usize;
            let (mut dl, mut dr) = (vec![0.0_f32; n_damp], vec![0.0_f32; n_damp]);
            bank.render_subblock_coupled(&mut drv, &mut dl, &mut dr);
            let (v1, h1) = bank.measure_key_polarization_rms(k);
            let (rv, rh) = (v1 / v0, h1 / h0);
            let ok_250 = if k == 0 {
                rv < 0.05 && rh < 0.05
            } else {
                rv < 0.01 && rh < 0.01
            };
            if !(ok_250 && drv.slot_parity[k] == 0b11) {
                failures.push((nl_on as usize * 1000 + k, rv, rh, drv.slot_parity[k]));
            }
            if k == 0 {
                bank.render_subblock_coupled(&mut drv, &mut dl, &mut dr);
                let (v2, h2) = bank.measure_key_polarization_rms(k);
                let (rv2, rh2) = (v2 / v0, h2 / h0);
                if rv2 >= 0.01 || rh2 >= 0.01 {
                    failures.push((nl_on as usize * 1000 + 100, rv2, rh2, drv.slot_parity[k]));
                }
            }
        }
    }
    assert!(failures.is_empty(), "failures={failures:?}");
}

#[test]
fn monochord_una_corda_strike_weight_attenuates_force_without_canceling() {
    let mut bank_full = StringBank::new(48_000.0, &VoicingSpec::default());
    let mut bank_uc = bank_full.clone();
    let mut drv_full = MockAction::new(48_000.0);
    drv_full.strike(0, 1.8);
    let mut drv_uc = MockAction::new(48_000.0);
    drv_uc.weights[0] = [0.78, 0.0, 0.0];
    drv_uc.strike(0, 1.8);
    let (mut l1, mut r1, mut l2, mut r2) = (
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
        vec![0.0_f32; 512],
    );
    bank_full.render_subblock_coupled(&mut drv_full, &mut l1, &mut r1);
    bank_uc.render_subblock_coupled(&mut drv_uc, &mut l2, &mut r2);
    let rms_full = (l1.iter().map(|x| x * x).sum::<f32>() / 512.0).sqrt();
    let rms_uc = (l2.iter().map(|x| x * x).sum::<f32>() / 512.0).sqrt();
    let ratio = rms_uc / rms_full.max(1e-12);
    assert!(
        (ratio - 0.78).abs() < 0.05,
        "monochord Una Corda RMS ratio={ratio:.4} expected ~0.78"
    );
}
