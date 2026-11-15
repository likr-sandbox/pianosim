//! Pure-Rust offline linear algebra for generalized plate eigenproblems and complex `8x8` mobility inversion.

use core::ops::{Add, AddAssign, Div, Mul, Sub};

/// Minimal `f64` complex scalar for frequency-domain `8x8` admittance and pole-cancellation math.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    #[must_use]
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[must_use]
    #[inline]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[must_use]
    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[must_use]
    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div for Complex64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let den = rhs.norm_sq().max(1.0e-300);
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / den,
            (self.im * rhs.re - self.re * rhs.im) / den,
        )
    }
}

/// Solves `A * X = I_8` for a complex `8x8` matrix via Gauss-Jordan with partial pivoting.
#[must_use]
pub fn invert_8x8_complex(mat: &[[Complex64; 8]; 8]) -> Option<[[Complex64; 8]; 8]> {
    let mut a = *mat;
    let mut inv = [[Complex64::ZERO; 8]; 8];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = Complex64::ONE;
    }

    for col in 0..8 {
        let mut pivot = col;
        let mut best = a[col][col].norm_sq();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            let cand = a_row[col].norm_sq();
            if cand > best {
                best = cand;
                pivot = row;
            }
        }
        if !best.is_finite() || best < 1.0e-36 {
            return None;
        }
        if pivot != col {
            a.swap(col, pivot);
            inv.swap(col, pivot);
        }
        let inv_diag = Complex64::ONE / a[col][col];
        for j in 0..8 {
            a[col][j] = a[col][j] * inv_diag;
            inv[col][j] = inv[col][j] * inv_diag;
        }
        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor.norm_sq() == 0.0 {
                continue;
            }
            for j in 0..8 {
                a[row][j] = a[row][j] - factor * a[col][j];
                inv[row][j] = inv[row][j] - factor * inv[col][j];
            }
        }
    }
    Some(inv)
}

/// Computes lower-triangular Cholesky factor `L` (`M = L * L^T`) for SPD `n x n` row-major `mat`.
#[must_use]
pub fn cholesky_spd(n: usize, mat: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 1.0e-18 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Cyclic Jacobi symmetric eigensolver returning `(eigenvalues, eigenvectors_row_major_cols)`
/// sorted in ascending eigenvalue order (`eigenvector k` is column `k`: `v[row * n + k]`).
#[must_use]
pub fn jacobi_symmetric_eigen(n: usize, sym_mat: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = sym_mat.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_diag = (0..n).map(|i| a[i * n + i].abs()).fold(1.0_f64, f64::max);
    let tol = 1.0e-13 * max_diag;

    for sweep in 0..32 {
        let mut max_off = 0.0_f64;
        let thresh = if sweep < 4 {
            0.2 * max_diag / ((n * n) as f64)
        } else {
            0.0
        };
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                let abs_apq = apq.abs();
                max_off = max_off.max(abs_apq);
                if abs_apq <= thresh || abs_apq <= tol {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let rho = s / (1.0 + c);

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..p {
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = arp - s * (arq + rho * arp);
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = arq + s * (arp - rho * arq);
                    a[q * n + r] = a[r * n + q];
                }
                for r in (p + 1)..q {
                    let apr = a[p * n + r];
                    let arq = a[r * n + q];
                    a[p * n + r] = apr - s * (arq + rho * apr);
                    a[r * n + p] = a[p * n + r];
                    a[r * n + q] = arq + s * (apr - rho * arq);
                    a[q * n + r] = a[r * n + q];
                }
                for r in (q + 1)..n {
                    let apr = a[p * n + r];
                    let aqr = a[q * n + r];
                    a[p * n + r] = apr - s * (aqr + rho * apr);
                    a[r * n + p] = a[p * n + r];
                    a[q * n + r] = aqr + s * (apr - rho * aqr);
                    a[r * n + q] = a[q * n + r];
                }
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = vrp - s * (vrq + rho * vrp);
                    v[r * n + q] = vrq + s * (vrp - rho * vrq);
                }
            }
        }
        if max_off <= tol {
            break;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[i * n + i]
            .partial_cmp(&a[j * n + j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut evals = Vec::with_capacity(n);
    let mut sorted_v = vec![0.0_f64; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        evals.push(a[old_col * n + old_col]);
        for row in 0..n {
            sorted_v[row * n + new_col] = v[row * n + old_col];
        }
    }
    (evals, sorted_v)
}

/// Solves generalized symmetric eigenproblem `K v = lambda M v` (`M` SPD) returning
/// `(eigenvalues, eigenvectors_col_major)` sorted ascending with `v_k^T M v_k == 1`.
#[must_use]
pub fn solve_generalized_symmetric_eigen(
    n: usize,
    k_mat: &[f64],
    m_mat: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let l = cholesky_spd(n, m_mat)?;

    // Compute Y = L^{-1} * K (row-by-row forward solve on columns of K).
    let mut y = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = k_mat[row * n + col];
            for k in 0..row {
                sum -= l[row * n + k] * y[k * n + col];
            }
            y[row * n + col] = sum / l[row * n + row];
        }
    }

    // Compute symmetric H = Y * L^{-T} (forward solve on rows of Y).
    let mut h = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..=row {
            let mut sum = y[row * n + col];
            for k in 0..col {
                sum -= h[row * n + k] * l[col * n + k];
            }
            let val = sum / l[col * n + col];
            h[row * n + col] = val;
            h[col * n + row] = val;
        }
    }

    let (evals, w_vecs) = jacobi_symmetric_eigen(n, &h);

    // Back-transform v = L^{-T} * w via backward substitution on each eigenvector column.
    let mut v_vecs = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in (0..n).rev() {
            let mut sum = w_vecs[row * n + col];
            for k in (row + 1)..n {
                sum -= l[k * n + row] * v_vecs[k * n + col];
            }
            v_vecs[row * n + col] = sum / l[row * n + row];
        }
    }
    Some((evals, v_vecs))
}

/// Computes all complex eigenvalues of a real square `n x n` matrix via Householder
/// Hessenberg reduction and implicit Francis double-shift QR (`(re, im)` pairs).
#[must_use]
pub fn real_eigenvalues_qr(n: usize, mat: &[f64]) -> Vec<Complex64> {
    if n == 0 {
        return Vec::new();
    }
    let mut h = mat.to_vec();

    // 1. Reduce to upper Hessenberg form via Householder reflections.
    for k in 0..n.saturating_sub(2) {
        let mut norm_sq = 0.0_f64;
        for i in (k + 1)..n {
            let v = h[i * n + k];
            norm_sq += v * v;
        }
        if norm_sq < 1.0e-30 {
            continue;
        }
        let alpha = if h[(k + 1) * n + k] >= 0.0 {
            -norm_sq.sqrt()
        } else {
            norm_sq.sqrt()
        };
        let mut u = vec![0.0_f64; n - (k + 1)];
        for (idx, i) in ((k + 1)..n).enumerate() {
            u[idx] = h[i * n + k];
        }
        u[0] -= alpha;
        let u_norm_sq: f64 = u.iter().map(|x| x * x).sum();
        if u_norm_sq < 1.0e-30 {
            continue;
        }
        let beta = 2.0 / u_norm_sq;
        for j in k..n {
            let mut dot = 0.0_f64;
            for (idx, &ui) in u.iter().enumerate() {
                dot += ui * h[(k + 1 + idx) * n + j];
            }
            for (idx, &ui) in u.iter().enumerate() {
                h[(k + 1 + idx) * n + j] -= beta * dot * ui;
            }
        }
        for i in 0..n {
            let mut dot = 0.0_f64;
            for (idx, &ui) in u.iter().enumerate() {
                dot += h[i * n + (k + 1 + idx)] * ui;
            }
            for (idx, &ui) in u.iter().enumerate() {
                h[i * n + (k + 1 + idx)] -= beta * dot * ui;
            }
        }
    }

    // 2. Francis double-shift QR iterations.
    let mut out = Vec::with_capacity(n);
    let mut hi = n;
    let mut iter = 0_usize;
    let eps = 1.0e-13_f64;
    while hi > 0 {
        if hi == 1 {
            out.push(Complex64::new(h[0], 0.0));
            break;
        }
        let mut lo = hi - 1;
        while lo > 0 {
            let scale = h[(lo - 1) * n + (lo - 1)].abs() + h[lo * n + lo].abs();
            if h[lo * n + (lo - 1)].abs() <= eps * scale.max(1.0e-12) {
                h[lo * n + (lo - 1)] = 0.0;
                break;
            }
            lo -= 1;
        }
        if lo == hi - 1 {
            out.push(Complex64::new(h[(hi - 1) * n + (hi - 1)], 0.0));
            hi -= 1;
            iter = 0;
            continue;
        }
        if lo == hi - 2 {
            let (a00, a01) = (h[lo * n + lo], h[lo * n + lo + 1]);
            let (a10, a11) = (h[(lo + 1) * n + lo], h[(lo + 1) * n + lo + 1]);
            let tr = 0.5 * (a00 + a11);
            let d_sq = 0.25 * (a00 - a11) * (a00 - a11) + a01 * a10;
            if d_sq < 0.0 {
                let im = (-d_sq).sqrt();
                out.push(Complex64::new(tr, im));
                out.push(Complex64::new(tr, -im));
            } else {
                let s = d_sq.sqrt();
                out.push(Complex64::new(tr + s, 0.0));
                out.push(Complex64::new(tr - s, 0.0));
            }
            hi -= 2;
            iter = 0;
            continue;
        }

        iter += 1;
        if iter > 250 {
            out.push(Complex64::new(h[(hi - 1) * n + (hi - 1)], 0.0));
            hi -= 1;
            iter = 0;
            continue;
        }
        let (a00, a01) = (h[(hi - 2) * n + (hi - 2)], h[(hi - 2) * n + (hi - 1)]);
        let (a10, a11) = (h[(hi - 1) * n + (hi - 2)], h[(hi - 1) * n + (hi - 1)]);
        let (s, t) = if iter.is_multiple_of(10) {
            let w = h[(hi - 1) * n + (hi - 2)].abs() + h[(hi - 2) * n + (hi - 3)].abs();
            (1.5 * w, w * w)
        } else {
            (a00 + a11, a00 * a11 - a01 * a10)
        };

        let mut p =
            h[lo * n + lo] * (h[lo * n + lo] - s) + t + h[lo * n + lo + 1] * h[(lo + 1) * n + lo];
        let mut q = h[(lo + 1) * n + lo] * (h[lo * n + lo] + h[(lo + 1) * n + lo + 1] - s);
        let mut r = if lo + 2 < hi {
            h[(lo + 1) * n + lo] * h[(lo + 2) * n + lo + 1]
        } else {
            0.0
        };

        for k in lo..(hi - 1) {
            let size = (hi - k).min(3);
            let mut v = [p, q, r];
            let norm =
                (v[0] * v[0] + v[1] * v[1] + if size == 3 { v[2] * v[2] } else { 0.0 }).sqrt();
            if norm > 1.0e-30 {
                let sign = if v[0] >= 0.0 { norm } else { -norm };
                v[0] += sign;
                let v2_sum = v[0] * v[0] + v[1] * v[1] + if size == 3 { v[2] * v[2] } else { 0.0 };
                if v2_sum > 1.0e-30 {
                    let inv_v2 = 2.0 / v2_sum;
                    let col_start = k.saturating_sub(1).max(lo);
                    for j in col_start..hi {
                        let mut dot = v[0] * h[k * n + j] + v[1] * h[(k + 1) * n + j];
                        if size == 3 {
                            dot += v[2] * h[(k + 2) * n + j];
                        }
                        h[k * n + j] -= inv_v2 * dot * v[0];
                        h[(k + 1) * n + j] -= inv_v2 * dot * v[1];
                        if size == 3 {
                            h[(k + 2) * n + j] -= inv_v2 * dot * v[2];
                        }
                    }
                    let row_end = (k + 4).min(hi);
                    for i in lo..row_end {
                        let mut dot = h[i * n + k] * v[0] + h[i * n + (k + 1)] * v[1];
                        if size == 3 {
                            dot += h[i * n + (k + 2)] * v[2];
                        }
                        h[i * n + k] -= inv_v2 * dot * v[0];
                        h[i * n + (k + 1)] -= inv_v2 * dot * v[1];
                        if size == 3 {
                            h[i * n + (k + 2)] -= inv_v2 * dot * v[2];
                        }
                    }
                }
            }
            if k + 2 < hi {
                p = h[(k + 1) * n + k];
                q = h[(k + 2) * n + k];
                r = if k + 3 < hi { h[(k + 3) * n + k] } else { 0.0 };
            }
        }
    }
    out
}
