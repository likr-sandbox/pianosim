//! Smooth live-body switching projection and offline port-resistance matrix assembly
//! for [`crate::AdmittanceSsm`].

use piano_spec::{PresetReducedPole, TOTAL_STRINGS};

use crate::admittance_ssm::AdmittanceSsm;
use crate::bridge_presets::{sanitize_reduced_poles, sanitize_y_inf};
use crate::math::{identity_8x8, invert_8x8, mat8_vec, matmul_8x8};
use crate::zone_proj::{ZoneWeight, NUM_BRIDGE_ZONES};
use crate::NUM_REDUCED_MODES;

impl AdmittanceSsm {
    /// Replaces reduced soundboard poles while projecting active 8-zone restoring force and
    /// velocity smoothly onto the new modal coordinates so live body switching does not pop.
    pub fn set_reduced_model_smooth(
        &mut self,
        y_inf: &[f32; NUM_BRIDGE_ZONES],
        reduced_poles: &[PresetReducedPole; NUM_REDUCED_MODES],
    ) {
        if self.is_quiescent() {
            self.set_reduced_model(y_inf, reduced_poles);
            return;
        }
        let mut v_cur = [0.0_f64; NUM_BRIDGE_ZONES];
        let mut f_rest = [0.0_f64; NUM_BRIDGE_ZONES];
        for m in 0..NUM_REDUCED_MODES {
            let qd = if self.q_dot[m].is_finite() {
                self.q_dot[m]
            } else {
                0.0
            };
            let kq = if self.q[m].is_finite() {
                self.k_m[m] * self.q[m]
            } else {
                0.0
            };
            for z in 0..NUM_BRIDGE_ZONES {
                let p = self.phi[m][z];
                v_cur[z] += p * qd;
                f_rest[z] += p * kq;
            }
        }
        self.apply_model(sanitize_y_inf(y_inf), sanitize_reduced_poles(reduced_poles));
        self.recompute_junction_matrices();

        let mut g_k = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        let mut g_m = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        for m in 0..NUM_REDUCED_MODES {
            let km = self.k_m[m];
            let inv_mm = 1.0 / self.m_m[m].max(1.0e-9);
            let pm = &self.phi[m];
            for i in 0..NUM_BRIDGE_ZONES {
                for j in 0..NUM_BRIDGE_ZONES {
                    let pp = pm[i] * pm[j];
                    g_k[i][j] += km * pp;
                    g_m[i][j] += inv_mm * pp;
                }
            }
        }
        let (mut max_gk, mut max_gm) = (0.0_f64, 0.0_f64);
        for z in 0..NUM_BRIDGE_ZONES {
            max_gk = max_gk.max(g_k[z][z]);
            max_gm = max_gm.max(g_m[z][z]);
        }
        let eps_k = 1.0e-4 * max_gk + 1.0;
        let eps_m = 1.0e-4 * max_gm + 1.0e-9;
        for z in 0..NUM_BRIDGE_ZONES {
            g_k[z][z] += eps_k;
            g_m[z][z] += eps_m;
        }
        let alpha = mat8_vec(&invert_8x8(&g_k), &f_rest);
        let beta = mat8_vec(&invert_8x8(&g_m), &v_cur);
        for m in 0..NUM_REDUCED_MODES {
            let inv_mm = 1.0 / self.m_m[m].max(1.0e-9);
            let (mut qm, mut qdm) = (0.0_f64, 0.0_f64);
            for z in 0..NUM_BRIDGE_ZONES {
                let p = self.phi[m][z];
                qm += p * alpha[z];
                qdm += p * beta[z];
            }
            let qd_val = qdm * inv_mm;
            self.q[m] = if qm.is_finite() { qm } else { 0.0 };
            self.q_dot[m] = if qd_val.is_finite() { qd_val } else { 0.0 };
        }
    }

    pub(crate) fn recompute_junction_matrices(&mut self) {
        let mut sys = identity_8x8();
        for (i, sys_row) in sys.iter_mut().enumerate() {
            for (j, sys_cell) in sys_row.iter_mut().enumerate() {
                let mut sum = 0.0_f64;
                for (l, g_row) in self.g_total.iter().enumerate() {
                    sum += self.a_br[i][l] * g_row[j];
                }
                *sys_cell += sum;
            }
        }

        let inv = invert_8x8(&sys);
        let solve_abr = matmul_8x8(&inv, &self.a_br);
        for (dst_row, phi_row) in self.solve_phi.iter_mut().zip(self.phi.iter()) {
            for (dst, inv_row) in dst_row.iter_mut().zip(inv.iter()) {
                let mut sum = 0.0_f64;
                for (&inv_ij, &phi_j) in inv_row.iter().zip(phi_row.iter()) {
                    sum += inv_ij * phi_j;
                }
                *dst = sum;
            }
        }
        self.solve_8x8 = inv;
        self.solve_abr = solve_abr;
    }

    /// Recomputes `G_total = G_str + diag(r_extra_diag)`, `(I_8 + A_br * G_total)^-1`,
    /// and `solve_abr = solve_8x8 * A_br`. Only called during `init` / `set_port_resistances`.
    pub fn set_port_resistances(
        &mut self,
        r0: &[f32; TOTAL_STRINGS],
        zone_weights: &[ZoneWeight; TOTAL_STRINGS],
        r_extra_diag: &[f64; NUM_BRIDGE_ZONES],
    ) {
        let mut g_tot = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        for (s_idx, &(z0, w0, z1, w1)) in zone_weights.iter().enumerate() {
            let r = f64::from(if r0[s_idx].is_finite() {
                r0[s_idx].max(0.0)
            } else {
                0.0
            });
            let (i0, i1) = (usize::from(z0), usize::from(z1));
            let (fw0, fw1) = (f64::from(w0), f64::from(w1));
            g_tot[i0][i0] += fw0 * fw0 * r;
            g_tot[i0][i1] += fw0 * fw1 * r;
            g_tot[i1][i0] += fw1 * fw0 * r;
            g_tot[i1][i1] += fw1 * fw1 * r;
        }
        for (z, &extra) in r_extra_diag.iter().enumerate() {
            if extra.is_finite() && extra > 0.0 {
                g_tot[z][z] += extra;
            }
        }

        self.g_total = g_tot;
        self.recompute_junction_matrices();
    }
}
