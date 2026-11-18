//! Shared modal-string fixture and spectral helpers for `piano-action` integration tests.

use piano_simd::{Complex, RealFft};
use piano_spec::{
    build_scale_design, key_tuned_f0_hz, partial_frequency_hz, VoicingSpec, CHOIR_LAYOUTS,
};

use crate::math::{exp_f64, sin_cos_f64};
use crate::ActionBank;

impl ActionBank {
    pub(crate) fn set_shank_flex_enabled(&mut self, key_idx: usize, enabled: bool) {
        if let Some(slot) = self.keys.get_mut(key_idx) {
            slot.hammer.shank.flex_enabled = enabled;
        }
    }

    pub(crate) fn felt_memory(&self, key_idx: usize) -> f32 {
        self.keys
            .get(key_idx)
            .map_or(0.0, |slot| slot.felt_state.z_mem)
    }

    pub(crate) fn lut_table_bytes(&self) -> usize {
        self.p_grid.table_bytes()
    }
}

const TEST_MODES: usize = 24;

struct CoupledMode {
    omega: f64,
    cos_r: f64,
    sin_r: f64,
    inv_omega: f64,
    inv_omega2: f64,
    phi_h: f64,
    phi_d: f64,
    gain: f64,
    y: f64,
    w: f64,
}

pub(super) struct TestModalString {
    modes: Vec<CoupledMode>,
    n_strings: f64,
    pub(super) r_port_strike: f32,
    pub(super) r_port_damp: f32,
}

impl TestModalString {
    pub(super) fn new(key_idx: usize, sample_rate: f64, voicing: &VoicingSpec) -> Self {
        let specs = build_scale_design(voicing);
        let p = specs[key_idx];
        let f0 = key_tuned_f0_hz(key_idx, &specs, voicing);
        let dt = 1.0 / sample_rate;
        let inv_modal_mass = 2.0 / (p.linear_density_kg_m * p.speaking_length_m);
        let alpha_h = 1.0 / (7.6 + 1.9 * ((key_idx as f64) / 87.0));
        let alpha_d = 0.115 + 0.045 * ((key_idx as f64) / 69.0).min(1.0);
        let n_strings = f64::from(CHOIR_LAYOUTS[key_idx].strings_per_key.max(1));

        let mut modes = Vec::with_capacity(TEST_MODES);
        let mut sum_phi_h2 = 0.0_f64;
        let mut sum_phi_d2 = 0.0_f64;

        for m in 1..=TEST_MODES {
            let f_j = partial_frequency_hz(f0, p.inharmonicity_b, m);
            if f_j >= 0.45 * sample_rate {
                break;
            }
            let omega = core::f64::consts::TAU * f_j;
            let t60_j = 10.0 / (1.0 + 0.03 * (m as f64) * (m as f64));
            let sigma_j = 6.907_755_279 / t60_j;
            let r = exp_f64(-sigma_j * dt);
            let (s, c) = sin_cos_f64(omega * dt);

            let phi_h = sin_cos_f64((m as f64) * core::f64::consts::PI * alpha_h).0;
            let phi_d_raw = sin_cos_f64((m as f64) * core::f64::consts::PI * alpha_d).0;
            let phi_d = if phi_d_raw.abs() < 0.28 {
                if phi_d_raw >= 0.0 {
                    0.28
                } else {
                    -0.28
                }
            } else {
                phi_d_raw
            };

            sum_phi_h2 += phi_h * phi_h;
            sum_phi_d2 += phi_d * phi_d;

            modes.push(CoupledMode {
                omega,
                cos_r: r * c,
                sin_r: r * s,
                inv_omega: 1.0 / omega,
                inv_omega2: 1.0 / (omega * omega),
                phi_h,
                phi_d,
                gain: inv_modal_mass,
                y: 0.0,
                w: 0.0,
            });
        }

        let r_port_strike = (inv_modal_mass * dt * sum_phi_h2 / n_strings) as f32;
        let r_port_damp = (inv_modal_mass * dt * sum_phi_d2) as f32;
        Self {
            modes,
            n_strings,
            r_port_strike,
            r_port_damp,
        }
    }

    pub(super) fn strike_port(&self) -> (f32, f32) {
        let mut y = 0.0_f64;
        let mut v = 0.0_f64;
        for m in &self.modes {
            y += m.phi_h * m.y;
            v += m.phi_h * (m.w * m.omega);
        }
        (y as f32, v as f32)
    }

    pub(super) fn damp_port(&self) -> (f32, f32) {
        let mut y = 0.0_f64;
        let mut v = 0.0_f64;
        for m in &self.modes {
            y += m.phi_d * m.y;
            v += m.phi_d * (m.w * m.omega);
        }
        (y as f32, v as f32)
    }

    pub(super) fn excite_fundamental_velocity(&mut self, amp_mps: f64) {
        if let Some(m) = self.modes.first_mut() {
            m.w += amp_mps * m.inv_omega;
        }
    }

    pub(super) fn excite_modes(&mut self, base_mps: f64, slope: f64) {
        for (idx, m) in self.modes.iter_mut().enumerate() {
            let v_m = base_mps / (1.0 + slope * (idx as f64));
            m.w += v_m * m.inv_omega;
        }
    }

    pub(super) fn total_energy(&self) -> f64 {
        let mut e = 0.0_f64;
        for m in &self.modes {
            let v = m.w * m.omega;
            e += 0.5 * (v * v + m.omega * m.omega * m.y * m.y);
        }
        e
    }

    pub(super) fn mode_energy(&self, idx: usize) -> f64 {
        self.modes.get(idx).map_or(0.0, |m| {
            let v = m.w * m.omega;
            0.5 * (v * v + m.omega * m.omega * m.y * m.y)
        })
    }

    pub(super) fn step_forces(&mut self, f_strike_n: f32, f_damp_n: f32) {
        let fs = f64::from(f_strike_n) / self.n_strings;
        let fd = f64::from(f_damp_n);
        for m in &mut self.modes {
            let acc = m.gain * (m.phi_h * fs + m.phi_d * fd);
            let y_eq = acc * m.inv_omega2;
            let y_rel = m.y - y_eq;
            let y_next = m.cos_r * y_rel + m.sin_r * m.w + y_eq;
            let w_next = -m.sin_r * y_rel + m.cos_r * m.w;
            m.y = y_next;
            m.w = w_next;
        }
    }
}

pub(super) fn measure_contact_duration_ms(
    key_idx: usize,
    vel_u16: u16,
    sample_rate: f32,
) -> (f32, f32) {
    let voicing = VoicingSpec::default();
    let mut bank = ActionBank::new(sample_rate, &voicing);
    let mut s = TestModalString::new(key_idx, f64::from(sample_rate), &voicing);
    bank.note_on(key_idx, vel_u16);

    let mut forces = Vec::with_capacity(700);
    for _ in 0..700 {
        let (y, v) = s.strike_port();
        let f = bank.step_hammer_sample(key_idx, y, v, s.r_port_strike);
        forces.push(f);
        s.step_forces(f, 0.0);
        if !bank.is_hammer_active(key_idx) {
            break;
        }
    }

    let peak = forces.iter().copied().fold(0.0_f32, f32::max);
    let thresh = 0.05 * peak;
    let active_samples = forces.iter().filter(|&&f| f > thresh).count();
    let dur_ms = (active_samples as f32) * 1000.0 / sample_rate;
    (dur_ms, peak)
}

pub(super) fn spectral_centroid_and_band(
    samples: &[f32],
    sample_rate: f32,
    low_hz: f32,
    high_hz: f32,
) -> (f32, f32) {
    let n = samples.len();
    let mut fft = RealFft::new(n);
    let mut spec = vec![Complex::default(); n / 2 + 1];
    fft.forward(samples, &mut spec);

    let df = sample_rate / (n as f32);
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut band_energy = 0.0_f64;
    for (bin, c) in spec.iter().enumerate().skip(1) {
        let freq = (bin as f32) * df;
        let mag2 = f64::from(c.re * c.re + c.im * c.im);
        num += f64::from(freq) * mag2;
        den += mag2;
        if freq >= low_hz && freq <= high_hz {
            band_energy += mag2;
        }
    }
    let centroid = if den > 1e-30 { (num / den) as f32 } else { 0.0 };
    (centroid, band_energy as f32)
}
