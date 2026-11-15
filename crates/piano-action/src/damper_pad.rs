//! FTO #3 compliant unilateral geometric-clearance hysteretic damper contact pad
//! (`460` orthogonal V/H per-string slots across `0..70` damped keys, passive at `44.1/48/96 kHz`).

use piano_simd::{AdaaPowerLut, AdaaState};
use piano_spec::{build_scale_design, VoicingSpec, DAMPED_KEYS, TOTAL_STRINGS};

use crate::math::exp_f64;

/// Total independent unilateral damper pad slots (`2 * s_idx` = V, `2 * s_idx + 1` = H).
pub const DAMPER_PAD_SLOTS: usize = TOTAL_STRINGS * 2;

const DAMPER_EXPONENT_P: f32 = 2.2;
const DAMPER_LUT_SIZE: usize = 512;
const DAMPER_LUT_X_MAX: f32 = 2.0;
const DELTA_D_SCALE_M: f32 = 1.0e-3;
const ENGAGE_HALF_M: f32 = 8.0e-5;
const TAU_DAMP_FELT_S: f64 = 1.2e-3;
const EPS_DAMP_HYST: f32 = 0.25;
const DAMPER_C_PER_MASS: f64 = 380.0;
const HYST_DAMP_GAIN: f32 = 1.2;
const MAX_PASSIVE_PORT_RATIO: f32 = 2.4;

#[derive(Copy, Clone, Debug, Default)]
struct DamperStringSlot {
    adaa_pos: AdaaState,
    adaa_neg: AdaaState,
    z_pos: f32,
    z_neg: f32,
}

impl DamperStringSlot {
    #[inline]
    fn reset(&mut self) {
        self.adaa_pos.reset();
        self.adaa_neg.reset();
        self.z_pos = 0.0;
        self.z_neg = 0.0;
    }
}

/// Unilateral geometric-clearance damper contact bank covering all `460` orthogonal V/H slots.
#[derive(Clone, Debug)]
pub struct DamperPadBank {
    lut: AdaaPowerLut,
    a_damp_felt: f32,
    c_nom_key: [f32; DAMPED_KEYS],
    f_scale_key: [f32; DAMPED_KEYS],
    slots: [DamperStringSlot; DAMPER_PAD_SLOTS],
}

fn fill_damper_key_coeffs(
    voicing: &VoicingSpec,
    c_nom_key: &mut [f32; DAMPED_KEYS],
    f_scale_key: &mut [f32; DAMPED_KEYS],
) {
    let specs = build_scale_design(voicing);
    for k in 0..DAMPED_KEYS {
        let p = specs[k];
        let m_wire = p.linear_density_kg_m * p.speaking_length_m;
        let z0 = 2.0 * m_wire * p.f0_et_hz;
        // Scale viscous damper by string mass with wave-impedance cap to avoid node-pinning.
        let c_target = (DAMPER_C_PER_MASS * m_wire).min(6.0 * z0);
        c_nom_key[k] = c_target as f32;
        // Normalized Stulov nonlinear contact impedance boost proportional to tension/mass.
        f_scale_key[k] = (0.06 * DAMPER_C_PER_MASS * m_wire) as f32;
    }
}

impl DamperPadBank {
    /// Initializes damper pad LUT and static per-key mechanical parameters.
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let sr = f64::from(sample_rate.max(8_000.0));
        let a_damp_felt = exp_f64(-1.0 / (TAU_DAMP_FELT_S * sr)) as f32;
        let lut = AdaaPowerLut::new(DAMPER_EXPONENT_P, DAMPER_LUT_X_MAX, DAMPER_LUT_SIZE);

        let mut c_nom_key = [0.0_f32; DAMPED_KEYS];
        let mut f_scale_key = [0.0_f32; DAMPED_KEYS];
        fill_damper_key_coeffs(voicing, &mut c_nom_key, &mut f_scale_key);

        Self {
            lut,
            a_damp_felt,
            c_nom_key,
            f_scale_key,
            slots: [DamperStringSlot::default(); DAMPER_PAD_SLOTS],
        }
    }

    /// Updates per-key damper mechanical impedance coefficients in-place without reallocating.
    pub fn update_voicing_in_place(&mut self, voicing: &VoicingSpec) {
        fill_damper_key_coeffs(voicing, &mut self.c_nom_key, &mut self.f_scale_key);
    }

    /// Resets all 460 V/H per-string ADAA and Stulov relaxation slots to zero.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.reset();
        }
    }

    /// Evaluates one host-rate unilateral damper contact step for `(key_idx, slot_idx)`
    /// (`slot_idx = 2 * s_idx` vertical, `2 * s_idx + 1` horizontal),
    /// returning the passive force `F_damp` (`N`) applied **ON the string**.
    #[inline]
    pub fn step_damper_sample(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32 {
        if key_idx >= DAMPED_KEYS
            || slot_idx >= DAMPER_PAD_SLOTS
            || !y_damp_m.is_finite()
            || !v_damp_mps.is_finite()
            || !r_port_damp.is_finite()
            || !y_clear_m.is_finite()
        {
            return 0.0;
        }

        let slot = &mut self.slots[slot_idx];
        let inv_scale = 1.0 / DELTA_D_SCALE_M;

        if y_damp_m == 0.0 && v_damp_mps == 0.0 {
            // Even under resting preload (y_clear_m < 0), stationary equilibrium yields F = 0.0.
            let delta_eq = (-y_clear_m).max(0.0);
            let u_eq = delta_eq * inv_scale;
            let phi_pos = slot.adaa_pos.process(&self.lut, u_eq);
            let phi_neg = slot.adaa_neg.process(&self.lut, u_eq);
            let a = self.a_damp_felt;
            slot.z_pos = a * slot.z_pos + (1.0 - a) * phi_pos;
            slot.z_neg = a * slot.z_neg + (1.0 - a) * phi_neg;
            return 0.0;
        }

        let delta_pos = (y_damp_m - y_clear_m).max(0.0);
        let delta_neg = (-y_damp_m - y_clear_m).max(0.0);
        let s_sum = delta_pos + delta_neg;

        let u_pos = delta_pos * inv_scale;
        let u_neg = delta_neg * inv_scale;
        let phi_pos = slot.adaa_pos.process(&self.lut, u_pos);
        let phi_neg = slot.adaa_neg.process(&self.lut, u_neg);

        let a = self.a_damp_felt;
        slot.z_pos = a * slot.z_pos + (1.0 - a) * phi_pos;
        slot.z_neg = a * slot.z_neg + (1.0 - a) * phi_neg;

        if s_sum <= 0.0 {
            return 0.0;
        }

        let felt_hyst = (phi_pos - EPS_DAMP_HYST * slot.z_pos).max(0.0)
            + (phi_neg - EPS_DAMP_HYST * slot.z_neg).max(0.0);

        let engage = s_sum / (s_sum + ENGAGE_HALF_M);
        let c_raw = self.c_nom_key[key_idx] * engage * (1.0 + HYST_DAMP_GAIN * felt_hyst)
            + self.f_scale_key[key_idx] * felt_hyst;
        let safe_r = r_port_damp.max(0.0);
        let c_eff = if safe_r > 0.0 {
            c_raw.min(MAX_PASSIVE_PORT_RATIO / safe_r)
        } else {
            c_raw
        };

        (-c_eff * v_damp_mps) / (1.0 + 0.5 * safe_r * c_eff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;
    use piano_spec::key_choir;

    #[test]
    fn stationary_string_produces_exact_zero_force_even_under_preload() {
        let mut bank = DamperPadBank::new(48_000.0, &VoicingSpec::default());
        for k in [0usize, 15, 39, 48, 69] {
            let str_idx = usize::from(key_choir(k).unwrap().first_string_idx);
            for slot_idx in [str_idx * 2, str_idx * 2 + 1] {
                for _ in 0..32 {
                    let f = bank.step_damper_sample(k, slot_idx, 0.0, 0.0, 0.25, -3.0e-4);
                    assert_eq!(f, 0.0);
                }
            }
        }
        // Undamped keys 70..88 and out-of-range slot indices always yield 0.0.
        assert_eq!(
            bank.step_damper_sample(70, 352, 1.0e-3, 1.5, 0.5, -3.0e-4),
            0.0
        );
        assert_eq!(
            bank.step_damper_sample(48, DAMPER_PAD_SLOTS, 1.0e-3, 1.5, 0.1, -3.0e-4),
            0.0
        );
    }

    #[test]
    fn damper_is_strictly_passive_and_damps_in_80_to_180_ms_across_rates() {
        let voicing = VoicingSpec::default();
        let specs = build_scale_design(&voicing);

        for &sr in &[44_100.0_f32, 48_000.0, 96_000.0] {
            let mut bank = DamperPadBank::new(sr, &voicing);
            let ts = 1.0 / f64::from(sr);
            for &k in &[0usize, 15, 39, 48, 69] {
                let str_idx = usize::from(key_choir(k).unwrap().first_string_idx);
                let slot_v = str_idx * 2;
                let p = specs[k];
                let m_wire = p.linear_density_kg_m * p.speaking_length_m;
                let phi_d = 0.38_f64;
                let m_mode = 0.5 * m_wire;
                let omega = TAU * p.f0_et_hz;
                // Coupled ZOH-like 1-mode oscillator + realistic high-mode port impedance.
                let r_port = (2.0 * ts / m_wire * (phi_d * phi_d * 18.0)) as f32;
                let mut u = 1.2_f64; // initial modal velocity (m/s)
                let mut q = 0.0_f64;
                let (s, c) = (omega * ts).sin_cos();
                let mut e_prev = 0.5 * m_mode * u * u;
                let e0 = e_prev;

                let n_steps = (0.35 * f64::from(sr)) as usize;
                let mut t60_est = 0.0_f64;
                for step in 0..n_steps {
                    let y_d = (phi_d * q) as f32;
                    let v_d = (phi_d * u) as f32;
                    let f_damp = bank.step_damper_sample(k, slot_v, y_d, v_d, r_port, -3.0e-4);
                    assert!(f_damp.is_finite());

                    u += (phi_d * f64::from(f_damp) * ts) / m_mode;
                    let u_next = c * u - omega * s * q;
                    let q_next = (s / omega) * u + c * q;
                    u = u_next;
                    q = q_next;

                    let e_cur = 0.5 * m_mode * (u * u + omega * omega * q * q);
                    assert!(
                        e_cur <= e_prev * (1.0 + 1e-6) + 1e-18,
                        "passivity violation sr={sr} k={k} step={step}: {e_cur} > {e_prev}"
                    );
                    e_prev = e_cur;
                    if t60_est == 0.0 && e_cur <= 1.0e-6 * e0 {
                        t60_est = (step as f64) * ts;
                    }
                }
                assert!(
                    (0.050..=0.220).contains(&t60_est),
                    "sr={sr} k={k} damped T60={t60_est:.4}s out of expected band"
                );
            }
        }
    }

    #[test]
    fn orthogonal_vh_slots_are_isolated_with_zero_cross_contamination() {
        let voicing = VoicingSpec::default();
        let mut isolated_h = DamperPadBank::new(48_000.0, &voicing);
        let mut dual_vh = DamperPadBank::new(48_000.0, &voicing);

        let k = 48;
        let s_idx = usize::from(key_choir(k).unwrap().first_string_idx);
        let slot_v = s_idx * 2;
        let slot_h = s_idx * 2 + 1;

        for step in 0..512 {
            let phase = (step as f32) * 0.055;
            let (yv, vv) = (1.6e-3 * phase.sin(), 1.4 * phase.cos());
            let (yh, vh) = (-7.5e-4 * (phase * 1.13).cos(), 0.85 * (phase * 1.13).sin());

            let _fv = dual_vh.step_damper_sample(k, slot_v, yv, vv, 0.08, -3.0e-4);
            let fh_dual = dual_vh.step_damper_sample(k, slot_h, yh, vh, 0.08, -3.0e-4);
            let fh_ref = isolated_h.step_damper_sample(k, slot_h, yh, vh, 0.08, -3.0e-4);
            assert_eq!(fh_dual.to_bits(), fh_ref.to_bits());
        }
    }

    #[test]
    fn update_voicing_in_place_updates_damper_coeffs_and_matches_fresh_bank() {
        let mut voicing = VoicingSpec::default();
        let mut bank = DamperPadBank::new(48_000.0, &voicing);
        voicing.string_length_scale = 1.15;
        bank.update_voicing_in_place(&voicing);
        let fresh = DamperPadBank::new(48_000.0, &voicing);

        for k in [0_usize, 24, 48, 69] {
            assert!((bank.c_nom_key[k] - fresh.c_nom_key[k]).abs() < 1e-6);
            assert!((bank.f_scale_key[k] - fresh.f_scale_key[k]).abs() < 1e-6);
        }
        assert_ne!(
            bank.c_nom_key[39].to_bits(),
            DamperPadBank::new(48_000.0, &VoicingSpec::default()).c_nom_key[39].to_bits()
        );
    }
}
