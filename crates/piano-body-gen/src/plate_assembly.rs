//! Separable 2D tensor-product sum-factorization for `15x20` Rayleigh-Ritz
//! stiffness and mass matrix assembly on a `48x64` quadrature grid.

use crate::presets::BodySpec;
use crate::ribs_rim::{evaluate_plate_site_for, PlateSiteProps};
use std::f64::consts::PI;

/// Ritz sine basis count along width (`x`) and length (`y`): `15 x 20 = 300` modes.
pub const RITZ_NX: usize = 15;
pub const RITZ_NY: usize = 20;
pub const RITZ_N: usize = RITZ_NX * RITZ_NY;
/// Gauss/midpoint quadrature grid across soundboard domain: `48 x 64 = 3,072` points.
pub const QUAD_NX: usize = 48;
pub const QUAD_NY: usize = 64;

pub(crate) struct SineBasis1D {
    pub val: Vec<f64>,
    pub d2: Vec<f64>,
    pub d1: Vec<f64>,
    pub k: Vec<f64>,
}

pub(crate) fn build_1d_sine_table(n_modes: usize, n_quad: usize, span: f64) -> SineBasis1D {
    let norm = (2.0 / span).sqrt();
    let mut val = vec![0.0; n_modes * n_quad];
    let mut d2 = vec![0.0; n_modes * n_quad];
    let mut d1 = vec![0.0; n_modes * n_quad];
    let mut k_vec = vec![0.0; n_modes];

    for (m, k_slot) in k_vec.iter_mut().enumerate().take(n_modes) {
        let km = ((m + 1) as f64) * PI / span;
        *k_slot = km;
        for q in 0..n_quad {
            let u = ((q as f64) + 0.5) / (n_quad as f64) * span;
            let arg = km * u;
            let s = norm * arg.sin();
            let c = norm * arg.cos();
            let idx = m * n_quad + q;
            val[idx] = s;
            d1[idx] = km * c;
            d2[idx] = -km * km * s;
        }
    }
    SineBasis1D {
        val,
        d2,
        d1,
        k: k_vec,
    }
}

fn rotated_flexural_rigidity(spec: &BodySpec, site: &PlateSiteProps) -> [[f64; 3]; 3] {
    let h3_12 = site.thickness_m.powi(3) / 12.0;
    let nu_rl = spec.spruce_nu_lr * (spec.spruce_e_r_pa / spec.spruce_e_l_pa);
    let denom = 1.0 - spec.spruce_nu_lr * nu_rl;
    let q_11 = spec.spruce_e_l_pa / denom;
    let q_22 = spec.spruce_e_r_pa / denom;
    let q_12 = spec.spruce_nu_lr * spec.spruce_e_r_pa / denom;
    let q_66 = spec.spruce_g_lr_pa;

    let c = spec.grain_angle_rad.cos();
    let s = spec.grain_angle_rad.sin();
    let t = [
        [c * c, s * s, c * s],
        [s * s, c * c, -c * s],
        [-2.0 * c * s, 2.0 * c * s, c * c - s * s],
    ];
    let q_0 = [[q_11, q_12, 0.0], [q_12, q_22, 0.0], [0.0, 0.0, q_66]];

    let mut d = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sum += t[k][i] * q_0[k][l] * t[l][j];
                }
            }
            d[i][j] = h3_12 * sum;
        }
    }

    let v_rib = [s * s, c * c, -s * c];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] += site.rib_bending_nm * v_rib[i] * v_rib[j];
        }
    }

    let cb = site.bridge_angle_rad.cos();
    let sb = site.bridge_angle_rad.sin();
    let v_br = [cb * cb, sb * sb, cb * sb];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] += site.bridge_bending_nm * v_br[i] * v_br[j];
        }
    }

    d
}

struct YContractedTensors {
    s_d00: Vec<f64>,
    s_d01: Vec<f64>,
    s_d02: Vec<f64>,
    s_x00_k: Vec<f64>,
    s_d12: Vec<f64>,
    s_d22: Vec<f64>,
    s_mel: Vec<f64>,
    s_min: Vec<f64>,
    integrals: PlateSurfaceIntegrals,
}

pub(crate) struct PlateSurfaceIntegrals {
    pub int_w: [f64; RITZ_N],
    pub int_xw: [f64; RITZ_N],
    pub int_yw: [f64; RITZ_N],
    pub inside_area_m2: f64,
    pub inside_mass_kg: f64,
}

fn build_y_contracted_tensors(
    spec: &BodySpec,
    bx: &SineBasis1D,
    by: &SineBasis1D,
    dx: f64,
    dy: f64,
) -> YContractedTensors {
    let d_area = dx * dy;
    let e_eff = (spec.spruce_e_l_pa * spec.spruce_e_r_pa).sqrt();
    let inv_r2 = 1.0 / (spec.crown_radius_m * spec.crown_radius_m);
    let stride_q = RITZ_NY * RITZ_NY;
    let total = QUAD_NX * stride_q;

    let mut s_d00 = vec![0.0_f64; total];
    let mut s_d01 = vec![0.0_f64; total];
    let mut s_d02 = vec![0.0_f64; total];
    let mut s_x00_k = vec![0.0_f64; total];
    let mut s_d12 = vec![0.0_f64; total];
    let mut s_d22 = vec![0.0_f64; total];
    let mut s_mel = vec![0.0_f64; total];
    let mut s_min = vec![0.0_f64; total];

    let mut int_w = [0.0_f64; RITZ_N];
    let mut int_xw = [0.0_f64; RITZ_N];
    let mut int_yw = [0.0_f64; RITZ_N];
    let mut inside_area_m2 = 0.0_f64;
    let mut inside_mass_kg = 0.0_f64;

    let mut y0_row = [0.0_f64; RITZ_NY];
    let mut y1_row = [0.0_f64; RITZ_NY];
    let mut y2_row = [0.0_f64; RITZ_NY];

    for qx in 0..QUAD_NX {
        let x = spec.x_min_m + ((qx as f64) + 0.5) * dx;
        let q_off = qx * stride_q;
        let mut sum_y0 = [0.0_f64; RITZ_NY];
        let mut sum_yy = [0.0_f64; RITZ_NY];
        for qy in 0..QUAD_NY {
            let y = ((qy as f64) + 0.5) * dy;
            let site = evaluate_plate_site_for(spec, x, y);
            let dm = rotated_flexural_rigidity(spec, &site);
            let inside = if site.rim_signed_dist_m > 0.0 {
                1.0
            } else {
                0.0
            };
            let k_crown =
                (0.44 * e_eff * site.thickness_m * inv_r2 + spec.crown_prestress) * inside;
            let k_el = (k_crown + site.rim_stiffness_n_m3) * d_area;
            let m_el = site.areal_mass_kg_m2 * d_area;
            let m_in = if site.rim_signed_dist_m > 0.0 {
                m_el
            } else {
                0.0
            };
            let d00 = dm[0][0] * d_area;
            let d01 = dm[0][1] * d_area;
            let d02 = 2.0 * dm[0][2] * d_area;
            let d11 = dm[1][1] * d_area;
            let d12 = 2.0 * dm[1][2] * d_area;
            let d22 = 4.0 * dm[2][2] * d_area;

            for iy in 0..RITZ_NY {
                let idx = iy * QUAD_NY + qy;
                y0_row[iy] = by.val[idx];
                y1_row[iy] = by.d1[idx];
                y2_row[iy] = by.d2[idx];
            }

            if inside > 0.0 {
                inside_area_m2 += d_area;
                inside_mass_kg += m_in;
                let y_rel = y - spec.multipole_y_center_m;
                for iy in 0..RITZ_NY {
                    let w_y = y0_row[iy] * d_area;
                    sum_y0[iy] += w_y;
                    sum_yy[iy] += y_rel * w_y;
                }
            }

            for iy1 in 0..RITZ_NY {
                let (a0, a1, a2) = (y0_row[iy1], y1_row[iy1], y2_row[iy1]);
                let row = q_off + iy1 * RITZ_NY;
                for iy2 in 0..RITZ_NY {
                    let (b0, b1, b2) = (y0_row[iy2], y1_row[iy2], y2_row[iy2]);
                    let cell = row + iy2;
                    s_d00[cell] += d00 * a0 * b0;
                    s_d01[cell] += d01 * a0 * b2;
                    s_d02[cell] += d02 * a0 * b1;
                    s_x00_k[cell] += d11 * a2 * b2 + k_el * a0 * b0;
                    s_d12[cell] += d12 * a2 * b1;
                    s_d22[cell] += d22 * a1 * b1;
                    s_mel[cell] += m_el * a0 * b0;
                    s_min[cell] += m_in * a0 * b0;
                }
            }
        }

        for ix in 0..RITZ_NX {
            let px = bx.val[ix * QUAD_NX + qx];
            let p_base = ix * RITZ_NY;
            for iy in 0..RITZ_NY {
                let p = p_base + iy;
                let sy0 = px * sum_y0[iy];
                int_w[p] += sy0;
                int_xw[p] += x * sy0;
                int_yw[p] += px * sum_yy[iy];
            }
        }
    }

    YContractedTensors {
        s_d00,
        s_d01,
        s_d02,
        s_x00_k,
        s_d12,
        s_d22,
        s_mel,
        s_min,
        integrals: PlateSurfaceIntegrals {
            int_w,
            int_xw,
            int_yw,
            inside_area_m2,
            inside_mass_kg,
        },
    }
}

pub(crate) fn assemble_ritz_matrices(
    spec: &BodySpec,
    bx: &SineBasis1D,
    by: &SineBasis1D,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, PlateSurfaceIntegrals) {
    let dx = spec.domain_lx_m / (QUAD_NX as f64);
    let dy = spec.domain_ly_m / (QUAD_NY as f64);
    let t = build_y_contracted_tensors(spec, bx, by, dx, dy);

    let mut k_mat = vec![0.0_f64; RITZ_N * RITZ_N];
    let mut m_mat = vec![0.0_f64; RITZ_N * RITZ_N];
    let mut m_int_mat = vec![0.0_f64; RITZ_N * RITZ_N];
    let stride_q = RITZ_NY * RITZ_NY;

    let mut k_blk = [0.0_f64; RITZ_NY * RITZ_NY];
    let mut m_blk = [0.0_f64; RITZ_NY * RITZ_NY];
    let mut mi_blk = [0.0_f64; RITZ_NY * RITZ_NY];

    for ix1 in 0..RITZ_NX {
        let x1_0 = &bx.val[ix1 * QUAD_NX..(ix1 + 1) * QUAD_NX];
        let x1_1 = &bx.d1[ix1 * QUAD_NX..(ix1 + 1) * QUAD_NX];
        let x1_2 = &bx.d2[ix1 * QUAD_NX..(ix1 + 1) * QUAD_NX];

        for ix2 in ix1..RITZ_NX {
            let x2_0 = &bx.val[ix2 * QUAD_NX..(ix2 + 1) * QUAD_NX];
            let x2_1 = &bx.d1[ix2 * QUAD_NX..(ix2 + 1) * QUAD_NX];
            let x2_2 = &bx.d2[ix2 * QUAD_NX..(ix2 + 1) * QUAD_NX];

            k_blk.fill(0.0);
            m_blk.fill(0.0);
            mi_blk.fill(0.0);

            for qx in 0..QUAD_NX {
                let (a0, a1, a2) = (x1_0[qx], x1_1[qx], x1_2[qx]);
                let (b0, b1, b2) = (x2_0[qx], x2_1[qx], x2_2[qx]);
                let (x00, x22) = (a0 * b0, a2 * b2);
                let (x20, x02) = (a2 * b0, a0 * b2);
                let (x21, x12) = (a2 * b1, a1 * b2);
                let (x01, x10, x11) = (a0 * b1, a1 * b0, a1 * b1);
                let q_off = qx * stride_q;

                for iy1 in 0..RITZ_NY {
                    let row = iy1 * RITZ_NY;
                    let q_row = q_off + row;
                    for iy2 in 0..RITZ_NY {
                        let c = q_row + iy2;
                        let c_t = q_off + iy2 * RITZ_NY + iy1;
                        let cell = row + iy2;
                        k_blk[cell] += x22 * t.s_d00[c]
                            + x00 * t.s_x00_k[c]
                            + x20 * t.s_d01[c]
                            + x02 * t.s_d01[c_t]
                            + x21 * t.s_d02[c]
                            + x12 * t.s_d02[c_t]
                            + x01 * t.s_d12[c]
                            + x10 * t.s_d12[c_t]
                            + x11 * t.s_d22[c];
                        m_blk[cell] += x00 * t.s_mel[c];
                        mi_blk[cell] += x00 * t.s_min[c];
                    }
                }
            }

            let row_base = ix1 * RITZ_NY;
            let col_base = ix2 * RITZ_NY;
            for iy1 in 0..RITZ_NY {
                let p1 = row_base + iy1;
                for iy2 in 0..RITZ_NY {
                    let p2 = col_base + iy2;
                    let val_k = k_blk[iy1 * RITZ_NY + iy2];
                    let val_m = m_blk[iy1 * RITZ_NY + iy2];
                    let val_mi = mi_blk[iy1 * RITZ_NY + iy2];
                    k_mat[p1 * RITZ_N + p2] = val_k;
                    m_mat[p1 * RITZ_N + p2] = val_m;
                    m_int_mat[p1 * RITZ_N + p2] = val_mi;
                    k_mat[p2 * RITZ_N + p1] = val_k;
                    m_mat[p2 * RITZ_N + p1] = val_m;
                    m_int_mat[p2 * RITZ_N + p1] = val_mi;
                }
            }
        }
    }

    (k_mat, m_mat, m_int_mat, t.integrals)
}
