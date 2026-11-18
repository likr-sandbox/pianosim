//! Pure-Rust symmetric and generalized symmetric eigensolvers
//! (Householder tridiagonalization + implicit QL and cyclic Jacobi).

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

    sort_eigenpairs_ascending(n, &(0..n).map(|i| a[i * n + i]).collect::<Vec<_>>(), &v)
}

/// Fast `O(n^3)` symmetric eigensolver using Householder tridiagonalization and
/// implicit QL with Wilkinson shifts, returning ascending `(eigenvalues, col_major_vecs)`.
#[must_use]
pub fn householder_tridiagonal_ql(n: usize, sym_mat: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![sym_mat[0]], vec![1.0]);
    }

    let mut z = sym_mat.to_vec();
    let mut d = vec![0.0_f64; n];
    let mut e = vec![0.0_f64; n];

    // 1. Householder reduction to symmetric tridiagonal form.
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0_f64;
        let mut scale = 0.0_f64;
        if l > 0 {
            for k in 0..=l {
                scale += z[i * n + k].abs();
            }
            if scale == 0.0 {
                e[i] = z[i * n + l];
            } else {
                let inv_scale = 1.0 / scale;
                for k in 0..=l {
                    let scaled = z[i * n + k] * inv_scale;
                    z[i * n + k] = scaled;
                    h += scaled * scaled;
                }
                let f = z[i * n + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                z[i * n + l] = f - g;
                let mut f_acc = 0.0_f64;
                for j in 0..=l {
                    z[j * n + i] = z[i * n + j] / h;
                    let mut g_acc = 0.0_f64;
                    for k in 0..=j {
                        g_acc += z[j * n + k] * z[i * n + k];
                    }
                    for k in (j + 1)..=l {
                        g_acc += z[k * n + j] * z[i * n + k];
                    }
                    e[j] = g_acc / h;
                    f_acc += e[j] * z[i * n + j];
                }
                let hh = f_acc / (h + h);
                for j in 0..=l {
                    let f_j = z[i * n + j];
                    let g_j = e[j] - hh * f_j;
                    e[j] = g_j;
                    for k in 0..=j {
                        z[j * n + k] -= f_j * e[k] + g_j * z[i * n + k];
                    }
                }
            }
        } else {
            e[i] = z[i * n + l];
        }
        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;
    for i in 0..n {
        if d[i] != 0.0 {
            for j in 0..i {
                let mut g = 0.0_f64;
                for k in 0..i {
                    g += z[i * n + k] * z[k * n + j];
                }
                for k in 0..i {
                    z[k * n + j] -= g * z[k * n + i];
                }
            }
        }
        d[i] = z[i * n + i];
        z[i * n + i] = 1.0;
        for j in 0..i {
            z[j * n + i] = 0.0;
            z[i * n + j] = 0.0;
        }
    }

    // 2. Implicit QL iterations on tridiagonal (d, e).
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter = 0_usize;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1.0e-15 * dd.max(1.0e-300) {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 80 {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let mut s = 1.0_f64;
            let mut c = 1.0_f64;
            let mut p = 0.0_f64;
            let mut broke = false;

            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    broke = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..n {
                    let f_k = z[k * n + (i + 1)];
                    let z_ki = z[k * n + i];
                    z[k * n + (i + 1)] = s * z_ki + c * f_k;
                    z[k * n + i] = c * z_ki - s * f_k;
                }
            }
            if broke {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    sort_eigenpairs_ascending(n, &d, &z)
}

fn sort_eigenpairs_ascending(n: usize, evals_unsorted: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        evals_unsorted[i]
            .partial_cmp(&evals_unsorted[j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut evals = Vec::with_capacity(n);
    let mut sorted_v = vec![0.0_f64; n * n];
    for (new_col, &old_col) in order.iter().enumerate() {
        evals.push(evals_unsorted[old_col]);
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
    let l = crate::linalg::cholesky_spd(n, m_mat)?;

    // Compute Y = L^{-1} * K (row-by-row forward solve on columns of K).
    let mut y = vec![0.0_f64; n * n];
    for col in 0..n {
        for row in 0..n {
            let mut sum = k_mat[row * n + col];
            let l_row = &l[row * n..row * n + row];
            for k in 0..row {
                sum -= l_row[k] * y[k * n + col];
            }
            y[row * n + col] = sum / l[row * n + row];
        }
    }

    // Compute symmetric H = Y * L^{-T} (forward solve on rows of Y).
    let mut h = vec![0.0_f64; n * n];
    for row in 0..n {
        for col in 0..=row {
            let mut sum = y[row * n + col];
            let h_row = &h[row * n..row * n + col];
            let l_col = &l[col * n..col * n + col];
            for k in 0..col {
                sum -= h_row[k] * l_col[k];
            }
            let val = sum / l[col * n + col];
            h[row * n + col] = val;
            h[col * n + row] = val;
        }
    }

    let (evals, w_vecs) = if n <= 12 {
        jacobi_symmetric_eigen(n, &h)
    } else {
        householder_tridiagonal_ql(n, &h)
    };

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
