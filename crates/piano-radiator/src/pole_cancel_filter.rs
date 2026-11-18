//! Exact discrete trapezoidal Stage 3 reduced-pole cancellation filter
//! (`H_total(z) = Z_red_norm(z)^T * H_rad(z)` for bridge zones `0..8`).

use piano_spec::{BodyPreset, PRESET_BRIDGE_ZONES, PRESET_REDUCED_POLES};

use crate::math::{invert_10x10, PI};
use crate::modal_synth::path_index;

const NUM_REDUCED_POLES: usize = PRESET_REDUCED_POLES;

/// Exact discrete trapezoidal pole cancellation `H_total(z) = Z_red_norm(z)^T * H_rad(z)`
/// for bridge zones `0..8` at host `sample_rate`.
pub(crate) fn apply_stage3_pole_cancellation(
    preset: &BodyPreset,
    sample_rate: f64,
    ir_len: usize,
    ir_flat: &mut [f32],
    ch: usize,
) {
    let mut y_inf_inv = [0.0_f64; PRESET_BRIDGE_ZONES];
    let mut z = 0;
    while z < PRESET_BRIDGE_ZONES {
        let y = preset.y_inf[z] as f64;
        if !y.is_finite() || y <= 1.0e-12 {
            return;
        }
        y_inf_inv[z] = 1.0 / y;
        z += 1;
    }

    let h = 0.5 / sample_rate;
    let mut k_stiff = [0.0_f64; NUM_REDUCED_POLES];
    let mut d_plus = [1.0_f64; NUM_REDUCED_POLES];
    let mut d_minus = [1.0_f64; NUM_REDUCED_POLES];
    let mut phi = [[0.0_f64; PRESET_BRIDGE_ZONES]; NUM_REDUCED_POLES];
    let mut any_active = false;

    let mut m = 0;
    while m < NUM_REDUCED_POLES {
        let pole = &preset.reduced_poles[m];
        let freq = pole.freq_hz as f64;
        let zeta = pole.zeta as f64;
        let mass = pole.modal_mass_kg as f64;
        if freq.is_finite()
            && zeta.is_finite()
            && mass.is_finite()
            && freq > 0.5
            && mass > 1.0e-12
            && zeta >= 0.0
        {
            any_active = true;
            let omega = 2.0 * PI * freq;
            let cm = 2.0 * zeta * omega * mass;
            let km = omega * omega * mass;
            k_stiff[m] = km;
            d_plus[m] = mass + h * cm + h * h * km;
            d_minus[m] = mass - h * cm - h * h * km;
            let mut bz = 0;
            while bz < PRESET_BRIDGE_ZONES {
                let g = pole.zone_gains[bz] as f64;
                phi[m][bz] = if g.is_finite() { g } else { 0.0 };
                bz += 1;
            }
        }
        m += 1;
    }
    if !any_active {
        return;
    }

    let mut a_inv = [[0.0_f64; NUM_REDUCED_POLES]; NUM_REDUCED_POLES];
    let mut i = 0;
    while i < NUM_REDUCED_POLES {
        a_inv[i][i] = d_plus[i];
        let mut j = 0;
        while j < NUM_REDUCED_POLES {
            let mut sum = 0.0_f64;
            let mut bz = 0;
            while bz < PRESET_BRIDGE_ZONES {
                sum += phi[i][bz] * y_inf_inv[bz] * phi[j][bz];
                bz += 1;
            }
            a_inv[i][j] += h * sum;
            j += 1;
        }
        i += 1;
    }

    let Some(a_inv_inv) = invert_10x10(a_inv) else {
        return;
    };

    let mut eta = [0.0_f64; NUM_REDUCED_POLES];
    let mut prev_eta_dot = [0.0_f64; NUM_REDUCED_POLES];
    let mut prev_f = [0.0_f64; PRESET_BRIDGE_ZONES];

    let mut n = 0;
    while n < ir_len {
        let mut x = [0.0_f64; PRESET_BRIDGE_ZONES];
        let mut bz = 0;
        while bz < PRESET_BRIDGE_ZONES {
            x[bz] = ir_flat[path_index(bz, ch) * ir_len + n] as f64;
            bz += 1;
        }

        let mut rhs = [0.0_f64; NUM_REDUCED_POLES];
        let mut mode_i = 0;
        while mode_i < NUM_REDUCED_POLES {
            let mut phi_sum = 0.0_f64;
            let mut bz2 = 0;
            while bz2 < PRESET_BRIDGE_ZONES {
                phi_sum += phi[mode_i][bz2] * (x[bz2] + prev_f[bz2]);
                bz2 += 1;
            }
            rhs[mode_i] = h * phi_sum + d_minus[mode_i] * prev_eta_dot[mode_i]
                - 2.0 * h * k_stiff[mode_i] * eta[mode_i];
            mode_i += 1;
        }

        let mut eta_dot = [0.0_f64; NUM_REDUCED_POLES];
        let mut r = 0;
        while r < NUM_REDUCED_POLES {
            let mut acc = 0.0_f64;
            let mut c = 0;
            while c < NUM_REDUCED_POLES {
                acc += a_inv_inv[r][c] * rhs[c];
                c += 1;
            }
            eta_dot[r] = acc;
            r += 1;
        }

        let mut bz3 = 0;
        while bz3 < PRESET_BRIDGE_ZONES {
            let mut phi_eta_dot = 0.0_f64;
            let mut mi = 0;
            while mi < NUM_REDUCED_POLES {
                phi_eta_dot += phi[mi][bz3] * eta_dot[mi];
                mi += 1;
            }
            let f_val = x[bz3] - y_inf_inv[bz3] * phi_eta_dot;
            ir_flat[path_index(bz3, ch) * ir_len + n] = f_val as f32;
            prev_f[bz3] = f_val;
            bz3 += 1;
        }

        let mut mi2 = 0;
        while mi2 < NUM_REDUCED_POLES {
            eta[mi2] += h * (eta_dot[mi2] + prev_eta_dot[mi2]);
            prev_eta_dot[mi2] = eta_dot[mi2];
            mi2 += 1;
        }

        n += 1;
    }
}
