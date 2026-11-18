//! Exact trapezoidal discrete Stage 3 reduced-admittance simulator and
//! paired minimum-phase inverse `InverseReducedBridge`.

use crate::linalg::{invert_8x8_complex, Complex64};
use piano_bridge::{NUM_BRIDGE_ZONES, NUM_REDUCED_MODES};
use piano_spec::{BodyPreset, PresetReducedPole, PRESET_BRIDGE_ZONES, PRESET_REDUCED_POLES};
use std::f64::consts::PI;

/// Exact trapezoidal discrete inverse `Z_red_norm` paired with `AdmittanceSsm`.
pub struct InverseReducedBridge {
    y_inf: [f32; PRESET_BRIDGE_ZONES],
    poles: [PresetReducedPole; PRESET_REDUCED_POLES],
    sample_rate: f64,
    q: [f64; NUM_REDUCED_MODES],
    q_dot: [f64; NUM_REDUCED_MODES],
    a_br_inv: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
}

impl InverseReducedBridge {
    /// Initializes exact discrete trapezoidal inverse matching `AdmittanceSsm` at `sample_rate`.
    pub fn from_preset(preset: &BodyPreset, sample_rate: f32) -> Self {
        let sr = sample_rate as f64;
        let h = 0.5 / sr;
        let mut a_br = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        for (z, &yi) in preset.y_inf.iter().enumerate() {
            a_br[z][z] = yi as f64;
        }
        for p in &preset.reduced_poles {
            let m = p.modal_mass_kg as f64;
            let w = 2.0 * PI * (p.freq_hz as f64);
            let c = 2.0 * (p.zeta as f64) * w * m;
            let k = m * w * w;
            let y_m = 1.0 / (m / h + c + h * k);
            for (row_a, &gr_f) in a_br.iter_mut().zip(p.zone_gains.iter()) {
                let gr = gr_f as f64;
                for (cell_a, &gc_f) in row_a.iter_mut().zip(p.zone_gains.iter()) {
                    *cell_a += y_m * gr * (gc_f as f64);
                }
            }
        }
        let a_br_c: [[Complex64; 8]; 8] =
            std::array::from_fn(|r| std::array::from_fn(|c| Complex64::new(a_br[r][c], 0.0)));
        let inv_c = invert_8x8_complex(&a_br_c).expect("SPD A_br matrix");
        let a_br_inv = std::array::from_fn(|r| std::array::from_fn(|c| inv_c[r][c].re));

        Self {
            y_inf: preset.y_inf,
            poles: preset.reduced_poles,
            sample_rate: sr,
            q: [0.0; NUM_REDUCED_MODES],
            q_dot: [0.0; NUM_REDUCED_MODES],
            a_br_inv,
        }
    }

    /// Steps one sample: given Stage 3 midpoint velocity `v`, returns `y_norm = diag(y_inf) * f_recovered`.
    pub fn step(&mut self, v: &[f64; NUM_BRIDGE_ZONES]) -> [f64; NUM_BRIDGE_ZONES] {
        let ts = 1.0 / self.sample_rate;
        let inv_h = 2.0 * self.sample_rate;
        let h = 0.5 * ts;
        let mut y_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut qd_free_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut v_free_z = [0.0_f64; NUM_BRIDGE_ZONES];

        for (i, p) in self.poles.iter().enumerate() {
            let m = p.modal_mass_kg as f64;
            let w = 2.0 * PI * (p.freq_hz as f64);
            let c = 2.0 * (p.zeta as f64) * w * m;
            let k = m * w * w;
            let ym = 1.0 / (m * inv_h + c + h * k);
            y_m[i] = ym;
            let rhs = (m * inv_h) * self.q_dot[i] - k * self.q[i];
            let qdf = ym * rhs;
            qd_free_m[i] = qdf;
            for (z, &g) in p.zone_gains.iter().enumerate() {
                v_free_z[z] += (g as f64) * qdf;
            }
        }

        let mut dv = [0.0_f64; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            dv[z] = v[z] - v_free_z[z];
        }
        let mut f_rec = [0.0_f64; NUM_BRIDGE_ZONES];
        for (f_slot, row_inv) in f_rec.iter_mut().zip(self.a_br_inv.iter()) {
            let mut sum = 0.0;
            for (&a_rc, &dv_c) in row_inv.iter().zip(dv.iter()) {
                sum += a_rc * dv_c;
            }
            *f_slot = sum;
        }

        for (i, p) in self.poles.iter().enumerate() {
            let mut f_m = 0.0;
            for (z, &g) in p.zone_gains.iter().enumerate() {
                f_m += (g as f64) * f_rec[z];
            }
            let qd_mid = qd_free_m[i] + y_m[i] * f_m;
            let qd_next = 2.0 * qd_mid - self.q_dot[i];
            self.q[i] += ts * qd_mid;
            self.q_dot[i] = qd_next;
        }

        std::array::from_fn(|z| (self.y_inf[z] as f64) * f_rec[z])
    }
}
