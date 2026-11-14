//! Bank-Sujbert exact 78-pair trigonometric modal convolution for nonlinear
//! transverse-to-longitudinal phantom coupling (`no_std`, zero allocations).

use core::f64::consts::PI;

use crate::modal_soa::sqrt_f64;

/// Maximum SIMD4 vector count per transverse polarization coupled via SAV/phantom.
pub(crate) const MAX_PHANTOM_TRANSVERSE_VECS: usize = 3;
/// Maximum transverse modes per polarization coupled via SAV/phantom (`<= 12`).
pub(crate) const MAX_PHANTOM_TRANSVERSE_MODES: usize = MAX_PHANTOM_TRANSVERSE_VECS * 4;
/// Maximum SIMD4 vector count of longitudinal modes coupled via SAV/phantom.
pub(crate) const MAX_PHANTOM_LONG_VECS: usize = 6;
/// Maximum longitudinal modes coupled via SAV/phantom (`<= 24`).
pub(crate) const MAX_PHANTOM_LONG_MODES: usize = MAX_PHANTOM_LONG_VECS * 4;
/// 1-based longitudinal projection buffer length (`0` unused, `1..=24` active).
pub(crate) const PHANTOM_ARRAY_LEN: usize = MAX_PHANTOM_LONG_MODES + 1;

/// Maps modal oscillator state `v_{p,n}` to regularized physical modal slope `d_{p,n}`.
#[allow(dead_code)]
#[inline]
#[must_use]
pub(crate) fn slope_factor(partial_n: usize, length_m: f64, gamma_s: f64, omega_rad_s: f64) -> f64 {
    if partial_n == 0 || length_m <= 0.0 || gamma_s <= 0.0 || omega_rad_s <= 0.0 {
        return 0.0;
    }
    let nm1 = (partial_n - 1) as f64;
    let rolloff = sqrt_f64(1.0 + 0.015 * nm1 * nm1);
    ((partial_n as f64) * PI) / (length_m * gamma_s * omega_rad_s * rolloff)
}

/// Evaluates exact tension-stretching integral `S(v)` and 78-pair modal convolution
/// projections `Phi_tilde_m(v)` for `m = 1..=m_ph`.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
#[must_use]
pub(crate) fn convolve_transverse_slopes(
    length_m: f64,
    d_v: &[f64; MAX_PHANTOM_TRANSVERSE_MODES],
    d_h: &[f64; MAX_PHANTOM_TRANSVERSE_MODES],
    m_ph: usize,
) -> (f64, [f64; PHANTOM_ARRAY_LEN]) {
    let mut s_v = 0.0_f64;
    let mut s_h = 0.0_f64;
    let mut phi_tilde = [0.0_f64; PHANTOM_ARRAY_LEN];

    if m_ph == 0 {
        for a in 0..MAX_PHANTOM_TRANSVERSE_MODES {
            s_v += d_v[a] * d_v[a];
            s_h += d_h[a] * d_h[a];
        }
        return (0.25 * length_m * (s_v + s_h), phi_tilde);
    }

    let mut phi_v = [0.0_f64; PHANTOM_ARRAY_LEN];
    let mut phi_h = [0.0_f64; PHANTOM_ARRAY_LEN];

    for a in 0..MAX_PHANTOM_TRANSVERSE_MODES {
        let dv_a = d_v[a];
        let dh_a = d_h[a];
        let dv_sq = dv_a * dv_a;
        let dh_sq = dh_a * dh_a;
        s_v += dv_sq;
        s_h += dh_sq;

        let n = a + 1;
        phi_v[2 * n] += dv_sq;
        phi_h[2 * n] += dh_sq;

        for b in (a + 1)..MAX_PHANTOM_TRANSVERSE_MODES {
            let r = b + 1;
            let two_v = 2.0 * dv_a * d_v[b];
            let two_h = 2.0 * dh_a * d_h[b];
            phi_v[n + r] += two_v;
            phi_v[r - n] += two_v;
            phi_h[n + r] += two_h;
            phi_h[r - n] += two_h;
        }
    }

    let s_val = 0.25 * length_m * (s_v + s_h);
    let m_lim = m_ph.min(MAX_PHANTOM_LONG_MODES);
    let scale = 0.125 * length_m;
    for m in 1..=m_lim {
        phi_tilde[m] = scale * (phi_v[m] + phi_h[m]);
    }
    (s_val, phi_tilde)
}

/// Exact analytical transpose accumulating `DF(d)^T (w_s, w_phi)` into `(grad_v, grad_h)`.
/// Caller must ensure `w_phi[m] == 0.0` for `m == 0` and `m > m_ph`.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub(crate) fn accum_phantom_grad_d(
    length_m: f64,
    d_v: &[f64; MAX_PHANTOM_TRANSVERSE_MODES],
    d_h: &[f64; MAX_PHANTOM_TRANSVERSE_MODES],
    w_s: f64,
    w_phi: &[f64; PHANTOM_ARRAY_LEN],
    grad_v: &mut [f64; MAX_PHANTOM_TRANSVERSE_MODES],
    grad_h: &mut [f64; MAX_PHANTOM_TRANSVERSE_MODES],
) {
    let ws_half_l = 0.5 * length_m * w_s;
    let qtr_l = 0.25 * length_m;

    for a in 0..MAX_PHANTOM_TRANSVERSE_MODES {
        let n = a + 1;
        let w_diag = ws_half_l + qtr_l * w_phi[2 * n];
        grad_v[a] += w_diag * d_v[a];
        grad_h[a] += w_diag * d_h[a];

        for b in (a + 1)..MAX_PHANTOM_TRANSVERSE_MODES {
            let r = b + 1;
            let w_pair = qtr_l * (w_phi[n + r] + w_phi[r - n]);
            grad_v[a] += w_pair * d_v[b];
            grad_v[b] += w_pair * d_v[a];
            grad_h[a] += w_pair * d_h[b];
            grad_h[b] += w_pair * d_h[a];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phantom_78_pair_adjoint_matches_directional_derivative_under_1e12() {
        let length_m = 1.42_f64;
        let mut d_v = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        let mut d_h = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        let mut dd_v = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        let mut dd_h = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        for i in 0..MAX_PHANTOM_TRANSVERSE_MODES {
            let f = (i + 1) as f64;
            d_v[i] = 0.015 * (0.7 * f).sin() / (1.0 + 0.2 * f);
            d_h[i] = -0.011 * (1.1 * f).cos() / (1.0 + 0.15 * f);
            dd_v[i] = 0.008 * (1.3 * f + 0.4).cos();
            dd_h[i] = 0.006 * (0.9 * f - 0.3).sin();
        }

        for &m_ph in &[6usize, 15, MAX_PHANTOM_LONG_MODES] {
            let w_s = 1.37_f64;
            let mut w_phi = [0.0_f64; PHANTOM_ARRAY_LEN];
            for (m, w) in w_phi[1..=m_ph].iter_mut().enumerate() {
                *w = 0.85 * ((m as f64 + 1.0) * 0.43).sin() + 0.25;
            }

            // Exact directional derivative via central finite difference & analytical adjoint.
            // Both S(d) and Phi(d) are homogeneous quadratics, so central difference has zero
            // truncation error for any eps != 0; eps=1e-2 avoids finite-diff cancellation.
            let eps = 1.0e-2_f64;
            let mut d_vp = d_v;
            let mut d_hp = d_h;
            let mut d_vm = d_v;
            let mut d_hm = d_h;
            for i in 0..MAX_PHANTOM_TRANSVERSE_MODES {
                d_vp[i] += eps * dd_v[i];
                d_hp[i] += eps * dd_h[i];
                d_vm[i] -= eps * dd_v[i];
                d_hm[i] -= eps * dd_h[i];
            }
            let (s_p, phi_p) = convolve_transverse_slopes(length_m, &d_vp, &d_hp, m_ph);
            let (s_m, phi_m) = convolve_transverse_slopes(length_m, &d_vm, &d_hm, m_ph);
            let mut lhs = w_s * (s_p - s_m) / (2.0 * eps);
            for m in 1..=m_ph {
                lhs += w_phi[m] * (phi_p[m] - phi_m[m]) / (2.0 * eps);
            }

            let mut grad_v = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
            let mut grad_h = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
            accum_phantom_grad_d(length_m, &d_v, &d_h, w_s, &w_phi, &mut grad_v, &mut grad_h);
            let mut rhs = 0.0_f64;
            for i in 0..MAX_PHANTOM_TRANSVERSE_MODES {
                rhs += grad_v[i] * dd_v[i] + grad_h[i] * dd_h[i];
            }
            let diff = (lhs - rhs).abs();
            let rel = diff / lhs.abs().max(1e-20);
            assert!(rel < 1.0e-12 && diff < 1.0e-12, "m_ph={m_ph} rel={rel:e}");
        }

        // Verify analytical difference/sum and doubled frequencies for two modes (n=2, r=5 -> 4, 10, 7, 3).
        let mut d_pair = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        let zero_h = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
        d_pair[1] = 0.5;
        d_pair[4] = 0.3;
        let (s_pair, phi_pair) =
            convolve_transverse_slopes(1.0, &d_pair, &zero_h, MAX_PHANTOM_LONG_MODES);
        assert!((s_pair - 0.25 * (0.25 + 0.09)).abs() < 1e-14);
        assert!((phi_pair[4] - 0.125 * 0.25).abs() < 1e-14);
        assert!((phi_pair[10] - 0.125 * 0.09).abs() < 1e-14);
        assert!((phi_pair[7] - 0.125 * 0.30).abs() < 1e-14);
        assert!((phi_pair[3] - 0.125 * 0.30).abs() < 1e-14);
    }
}
