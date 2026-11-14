//! Longitudinal bridge inclination readout, smooth ultrasonic anti-aliasing,
//! SAV stiffness frequency compensation, and local hammer-contact axial precursor.

use core::f64::consts::PI;

use piano_simd::{AlignedBuf, F32x4};
use piano_spec::{
    StringPhysicalParams, LONGITUDINAL_MAX_KEY_IDX, NUM_KEYS, STEEL_YOUNGS_MODULUS_PA,
    TOTAL_STRINGS,
};

use crate::choir_map::ChoirMap;
use crate::modal_soa::{exp_f64, sin_cos_f64, sqrt_f64};
use crate::phantom::MAX_PHANTOM_LONG_MODES;
use crate::zoh_kernel::inject_excitation;

pub(crate) const LONG_AA_KNEE_HZ: f64 = 5_200.0;
pub(crate) const LONG_AA_CUTOFF_HZ: f64 = 14_500.0;
pub(crate) const PHANTOM_WEIGHT_SCALE: f64 = 0.42;
pub(crate) const HAMMER_WIDTH_M: f64 = 0.010;
pub(crate) const KAPPA_PREC: f64 = 0.85;
pub(crate) const PREC_T0_CAP_RATIO: f64 = 0.22;

#[inline]
#[must_use]
pub(crate) fn nonlinear_register_weight(key_idx: usize) -> f64 {
    if key_idx >= LONGITUDINAL_MAX_KEY_IDX {
        return 0.0;
    }
    let r = (key_idx as f64) / 30.0;
    let r2 = r * r;
    1.0 / (1.0 + r2 * r2)
}

#[inline]
#[must_use]
pub(crate) fn t60_longitudinal(partial_m: usize) -> f64 {
    let idx = partial_m.saturating_sub(1) as f64;
    (0.15 / (1.0 + 0.20 * idx)).max(0.045)
}

#[inline]
#[must_use]
pub(crate) fn w_aa(freq_hz: f64) -> f64 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 || freq_hz >= LONG_AA_CUTOFF_HZ {
        return 0.0;
    }
    let r = freq_hz / LONG_AA_KNEE_HZ;
    let r2 = r * r;
    1.0 / (1.0 + r2 * r2)
}

#[inline]
#[must_use]
pub(crate) fn w_phantom(freq_hz: f64) -> f64 {
    PHANTOM_WEIGHT_SCALE * w_aa(freq_hz)
}

#[inline]
#[must_use]
pub(crate) fn long_init_freq_hz(key_idx: usize, f_long_hz: f64, partial_m: usize) -> f64 {
    let f_nom = (partial_m as f64) * f_long_hz;
    let w_reg = nonlinear_register_weight(key_idx);
    if w_reg > 0.0 && partial_m <= MAX_PHANTOM_LONG_MODES {
        let wp = w_phantom(f_nom);
        f_nom / sqrt_f64(1.0 + w_reg * wp * wp)
    } else {
        f_nom
    }
}

#[inline]
#[must_use]
pub(crate) fn long_bridge_weight(
    key_idx: usize,
    f0_vert_hz: f64,
    f_long_hz: f64,
    partial_m: usize,
) -> f32 {
    if f0_vert_hz <= 0.0 || f_long_hz <= 0.0 {
        return 0.0;
    }
    let w_reg = nonlinear_register_weight(key_idx);
    if w_reg == 0.0 {
        return 0.0;
    }
    let f_nom = (partial_m as f64) * f_long_hz;
    let kappa_db = 0.048 * exp_f64(-0.9 * (key_idx as f64 / ((NUM_KEYS - 1) as f64))) * w_reg;
    let sign = if partial_m % 2 == 1 { -1.0 } else { 1.0 };
    (sign * kappa_db * (f_long_hz / f0_vert_hz) * w_aa(f_nom)) as f32
}

#[inline]
fn sinc_rad(x: f64) -> f64 {
    if x.abs() < 1.0e-12 {
        1.0
    } else {
        sin_cos_f64(x).0 / x
    }
}

/// Precomputed compact `b_long_prec` projections and per-string axial precursor scalings.
#[derive(Clone, Debug)]
pub(crate) struct LongitudinalBank {
    pub b_long_prec: AlignedBuf,
    pub vec_offset: [usize; TOTAL_STRINGS],
    pub prec_force_scale: [f64; TOTAL_STRINGS],
    pub prec_force_cap_n: [f64; TOTAL_STRINGS],
}

impl LongitudinalBank {
    /// Allocates and precomputes `b_long_prec` over only keys `0..LONGITUDINAL_MAX_KEY_IDX`.
    #[must_use]
    pub fn build(
        sample_rate: f64,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
        gamma_s: &[f64; TOTAL_STRINGS],
    ) -> Self {
        let total_long_vecs: usize = map.strings().iter().map(|e| e.long_vecs.len()).sum();
        let mut bank = Self {
            b_long_prec: AlignedBuf::zeroed(total_long_vecs.saturating_mul(4).max(4)),
            vec_offset: [0; TOTAL_STRINGS],
            prec_force_scale: [0.0; TOTAL_STRINGS],
            prec_force_cap_n: [0.0; TOTAL_STRINGS],
        };

        let ts = 1.0 / sample_rate;
        let mut cursor_vec = 0_usize;
        let b_mut = bank.b_long_prec.as_mut_slice();

        for (s_idx, entry) in map.strings().iter().enumerate() {
            bank.vec_offset[s_idx] = cursor_vec;
            let k = usize::from(entry.key_idx);
            let n_l_vecs = entry.long_vecs.len();
            if k >= LONGITUDINAL_MAX_KEY_IDX || n_l_vecs == 0 {
                cursor_vec += n_l_vecs;
                continue;
            }

            let spec = specs[k];
            let l_k = spec.speaking_length_m;
            let m_str = spec.linear_density_kg_m * l_k;
            let s_core = PI * spec.core_radius_m * spec.core_radius_m;
            let t0 = spec.tension_n.max(1.0);
            let w_reg = nonlinear_register_weight(k);
            bank.prec_force_scale[s_idx] =
                w_reg * KAPPA_PREC * (STEEL_YOUNGS_MODULUS_PA * s_core / (4.0 * t0 * t0));
            bank.prec_force_cap_n[s_idx] = PREC_T0_CAP_RATIO * t0;

            let x = (k as f64) / ((NUM_KEYS - 1) as f64);
            let beta_k = 0.137 - 0.042 * x;
            let gain = gamma_s[s_idx] * (2.0 * ts / m_str);
            let base = cursor_vec * 4;

            for m in 1..=entry.long_modes {
                let f_nom = (m as f64) * spec.f_long_hz;
                let aa = w_aa(f_nom);
                if aa > 0.0 && f_nom < 0.49 * sample_rate {
                    let mf = m as f64;
                    let cos_contact = sin_cos_f64(mf * PI * beta_k).1;
                    let width_sinc = sinc_rad(mf * PI * HAMMER_WIDTH_M / l_k);
                    b_mut[base + (m - 1)] = (gain * cos_contact * width_sinc * aa) as f32;
                }
            }
            cursor_vec += n_l_vecs;
        }
        bank
    }

    /// Injects smooth-clamped local axial hammer-contact precursor into `u_long`.
    #[inline]
    pub fn inject_precursor(
        &self,
        s_idx: usize,
        long_vecs_len: usize,
        f_hammer_n: f32,
        u_long: &mut [F32x4],
    ) -> bool {
        if long_vecs_len == 0 || f_hammer_n <= 0.0 || !f_hammer_n.is_finite() {
            return false;
        }
        let fh = f64::from(f_hammer_n);
        let raw = self.prec_force_scale[s_idx] * fh * fh;
        let cap = self.prec_force_cap_n[s_idx];
        let f_prec = if cap > 0.0 {
            (raw / (1.0 + raw / cap)) as f32
        } else {
            0.0
        };
        if f_prec == 0.0 {
            return false;
        }
        let off = self.vec_offset[s_idx];
        let b_vecs = &self.b_long_prec.as_f32x4_slice()[off..off + long_vecs_len];
        inject_excitation(u_long, b_vecs, f_prec);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choir_map::ChoirMap;
    use crate::contact_ports::ContactPorts;
    use crate::modal_soa::ModalSoa;
    use crate::zoh_kernel::step_string_modes;
    use piano_spec::{build_scale_design, VoicingSpec};

    #[test]
    fn causal_precursor_arrives_at_tau_l_ahead_of_tau_t_on_a0_and_c1() {
        let sr = 96_000.0_f64;
        let voicing = VoicingSpec {
            unison_detune_scale: 0.0,
            ..VoicingSpec::default()
        };
        let specs = build_scale_design(&voicing);
        let map = ChoirMap::build_with_specs(&specs, &voicing);
        let ports = ContactPorts::build(sr, &voicing, &specs, &map);
        let lbank = LongitudinalBank::build(sr, &specs, &map, &ports.gamma_s);

        let base_soa = ModalSoa::build(sr, &voicing, &specs, &map);

        for &key in &[0usize, 3] {
            let mut soa_long = base_soa.clone();
            let mut soa_trans = base_soa.clone();
            let s_idx = map.keys()[key].string_range.start;
            let e = &map.strings()[s_idx];
            let x = (key as f64) / ((NUM_KEYS - 1) as f64);
            let beta_k = 0.137 - 0.042 * x;
            let tau_l_s = (1.0 - beta_k) / (2.0 * specs[key].f_long_hz);
            let tau_t_s = (1.0 - beta_k) / (2.0 * specs[key].f0_et_hz);

            // Half-sine 0.25 ms local force pulse at t = 0: longitudinal arrives at tau_L << tau_T.
            let pulse_samples = (sr * 0.00025) as usize;
            let total_samples = (sr * 0.025) as usize;
            let mut peak_l_before_tau_t = 0.0_f32;
            let mut peak_l_pre_arrival = 0.0_f32;
            let mut max_trans_early = 0.0_f32;
            let mut first_l_peak_t = 0.0_f64;
            let mut first_l_peak_val = 0.0_f32;

            let lv = e.long_vecs.clone();
            let tv_range = e.vert_vecs.clone();
            let bh = &ports.b_strike.as_f32x4_slice()[tv_range.clone()];

            for n in 0..total_samples {
                if n < pulse_samples {
                    let phase = PI * (n as f64 + 0.5) / (pulse_samples as f64);
                    let f_n = (60.0 * phase.sin()) as f32;
                    let u_l = &mut soa_long.u.as_f32x4_slice_mut()[lv.clone()];
                    assert!(lbank.inject_precursor(s_idx, lv.len(), f_n, u_l));
                    let u_t = &mut soa_trans.u.as_f32x4_slice_mut()[tv_range.clone()];
                    inject_excitation(u_t, bh, f_n);
                }

                let y_l = step_string_modes(
                    &mut soa_long.u.as_f32x4_slice_mut()[lv.clone()],
                    &mut soa_long.v.as_f32x4_slice_mut()[lv.clone()],
                    &soa_long.d_coeff.as_f32x4_slice()[lv.clone()],
                    &soa_long.s_coeff.as_f32x4_slice()[lv.clone()],
                    &soa_long.c_bridge.as_f32x4_slice()[lv.clone()],
                )
                .abs();
                let y_t = step_string_modes(
                    &mut soa_trans.u.as_f32x4_slice_mut()[tv_range.clone()],
                    &mut soa_trans.v.as_f32x4_slice_mut()[tv_range.clone()],
                    &soa_trans.d_coeff.as_f32x4_slice()[tv_range.clone()],
                    &soa_trans.s_coeff.as_f32x4_slice()[tv_range.clone()],
                    &soa_trans.c_bridge.as_f32x4_slice()[tv_range.clone()],
                )
                .abs();

                let t_s = (n + 1) as f64 / sr;
                if t_s < 0.65 * tau_l_s {
                    peak_l_pre_arrival = peak_l_pre_arrival.max(y_l);
                }
                if t_s < 0.85 * tau_t_s {
                    peak_l_before_tau_t = peak_l_before_tau_t.max(y_l);
                    max_trans_early = max_trans_early.max(y_t);
                    if y_l > first_l_peak_val {
                        first_l_peak_val = y_l;
                        first_l_peak_t = t_s;
                    }
                }
            }

            eprintln!(
                "key={key} tau_L={:.3}ms tau_T={:.3}ms first_L_peak={:.3}ms pre={peak_l_pre_arrival:.5} l_peak={peak_l_before_tau_t:.5} t_early={max_trans_early:.5}",
                tau_l_s * 1e3,
                tau_t_s * 1e3,
                first_l_peak_t * 1e3
            );
            assert!(
                first_l_peak_t >= 0.70 * tau_l_s && first_l_peak_t <= 0.70 * tau_t_s,
                "key={key} precursor peaked at {:.3}ms (tau_L={:.3}ms, tau_T={:.3}ms)",
                first_l_peak_t * 1e3,
                tau_l_s * 1e3,
                tau_t_s * 1e3
            );
            assert!(
                peak_l_before_tau_t > peak_l_pre_arrival * 4.0 && peak_l_before_tau_t > 1.0e-3,
                "key={key} weak causal onset pre={peak_l_pre_arrival} peak={peak_l_before_tau_t}"
            );
        }
    }

    fn blackman_harris_goertzel_db(samples: &[f32], freq_hz: f64, sr: f64) -> f64 {
        let n = samples.len();
        let denom = (n - 1) as f64;
        let coeff = 2.0 * (core::f64::consts::TAU * freq_hz / sr).cos();
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for (i, &x) in samples.iter().enumerate() {
            let p = core::f64::consts::TAU * (i as f64) / denom;
            let w =
                0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos();
            let s0 = f64::from(x) * w + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(1.0e-30);
        10.0 * power.log10()
    }

    #[test]
    fn phantom_spectral_peaks_emerge_at_doubled_and_sum_frequencies() {
        let sr = 48_000.0_f32;
        let sr64 = f64::from(sr);
        let voicing = VoicingSpec {
            unison_detune_scale: 0.0,
            ..VoicingSpec::default()
        };
        let specs = build_scale_design(&voicing);
        let key = 15usize; // C2: large inharmonic separation between 2*f_n and f_{2n}
        let f0 = piano_spec::key_tuned_f0_hz(key, &specs, &voicing);
        let b = specs[key].inharmonicity_b;
        let f5 = piano_spec::partial_frequency_hz(f0, b, 5);
        let f6 = piano_spec::partial_frequency_hz(f0, b, 6);
        let target_freqs = [
            ("2*f_5", 2.0 * f5),
            ("2*f_6", 2.0 * f6),
            ("f_5+f_6", f5 + f6),
        ];

        const N: usize = 65_536;
        let run_case = |sav_on: bool| -> Vec<f32> {
            let mut bank = crate::StringBank::new(sr, &voicing);
            bank.set_nonlinear_coupling(true, sav_on);
            bank.apply_excitation_sample(key, [1.0, 1.0, 1.0], 4.0);
            let (mut l, mut r) = (vec![0.0_f32; N], vec![0.0_f32; N]);
            bank.render_key_subblock(key, &mut l, &mut r);
            l.iter().zip(r.iter()).map(|(&a, &b)| a + b).collect()
        };

        let mono_on = run_case(true);
        let mono_off = run_case(false);
        for (label, freq) in target_freqs {
            let db_on = blackman_harris_goertzel_db(&mono_on, freq, sr64);
            let db_off = blackman_harris_goertzel_db(&mono_off, freq, sr64);
            eprintln!("C2 phantom {label} ({freq:.2} Hz): on={db_on:.1} dB off={db_off:.1} dB");
            assert!(
                db_on > db_off + 10.0,
                "{label} ({freq:.2} Hz) phantom peak insufficient: on={db_on:.2} off={db_off:.2}"
            );
        }
    }
}
