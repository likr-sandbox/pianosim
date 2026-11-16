//! Reduced-order concert grand soundboard/bridge admittance state-space model (`AdmittanceSsm`).
//!
//! Models 10 passive reduced soundboard modes (`~54..1,130 Hz`) plus diagonal local
//! high-frequency resistive mobility `y_inf` across the 8 bridge zones, integrated
//! with exact discrete trapezoidal energy balance and precomputed `8x8` WDF junction solve.

use core::f64::consts::TAU;

use piano_spec::{build_scale_design, VoicingSpec, CHOIR_LAYOUTS, TOTAL_STRINGS};

use crate::zone_proj::{ZoneWeight, NUM_BRIDGE_ZONES};

/// Number of reduced bidirectional soundboard/bridge modes (`~54..1,130 Hz`).
pub const NUM_REDUCED_MODES: usize = 10;

/// Publicly exported reduced bridge pole descriptor for M6 `piano-body-gen` pole cancellation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReducedBridgePole {
    /// Modal natural frequency in Hz.
    pub freq_hz: f32,
    /// Viscous damping ratio `zeta` (`0.0` in lossless test mode).
    pub zeta: f32,
    /// Modal quality factor `Q = 1 / (2 * zeta)`.
    pub q_factor: f32,
    /// Effective modal mass in kilograms.
    pub modal_mass_kg: f32,
    /// Dimensionless spatial mode gains at each of the 8 bridge zones.
    pub zone_gains: [f32; NUM_BRIDGE_ZONES],
}

/// Default high-frequency passive local resistive mobility `y_inf` `[m / (N*s)]`
/// across bridge zones `0..8`, preventing any `1.13 kHz` brick wall on treble/longitudinal signals.
pub const DEFAULT_Y_INF: [f32; NUM_BRIDGE_ZONES] = [
    0.000_14, 0.000_16, 0.000_28, 0.000_28, 0.000_28, 0.000_24, 0.000_22, 0.000_16,
];

const MODE_FREQS_HZ: [f32; NUM_REDUCED_MODES] = [
    54.0, 89.0, 142.0, 212.0, 285.0, 405.0, 530.0, 730.0, 930.0, 1130.0,
];
const MODE_ZETAS: [f32; NUM_REDUCED_MODES] = [
    0.055, 0.042, 0.030, 0.020, 0.020, 0.019, 0.024, 0.021, 0.020, 0.019,
];
const MODE_MASSES_KG: [f32; NUM_REDUCED_MODES] =
    [3.60, 2.50, 1.25, 0.68, 0.48, 0.56, 0.90, 2.20, 2.90, 3.40];

const MODE_ZONE_GAINS: [[f32; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES] = [
    [0.12, 0.18, 0.30, 0.36, 0.34, 0.26, 0.18, 0.10],
    [0.16, 0.18, 0.18, 0.06, -0.16, -0.28, -0.26, -0.16],
    [0.05, 0.06, -0.26, -0.32, -0.28, -0.12, 0.18, 0.22],
    [0.04, -0.05, -0.34, -0.36, -0.20, 0.26, 0.20, -0.16],
    [-0.04, 0.05, 0.28, 0.40, 0.26, -0.20, -0.26, 0.18],
    [0.03, -0.04, 0.18, 0.36, -0.24, -0.32, 0.14, 0.24],
    [-0.04, 0.06, -0.12, 0.30, 0.30, -0.20, -0.16, 0.14],
    [-0.03, 0.05, -0.10, 0.14, -0.14, -0.14, 0.16, -0.12],
    [0.03, -0.04, 0.08, -0.12, 0.14, -0.12, -0.16, 0.16],
    [-0.02, 0.04, -0.08, 0.10, -0.14, 0.16, -0.14, 0.16],
];

/// Builds default physical per-string WDF port resistances `[kg/s]` for standalone initialization
/// before `StringBank::bridge_port_resistances()` is supplied.
#[must_use]
pub fn default_string_port_resistances(sample_rate: f32) -> [f32; TOTAL_STRINGS] {
    let safe_sr = f64::from(if sample_rate.is_finite() {
        sample_rate.clamp(8_000.0, 384_000.0)
    } else {
        48_000.0
    });
    let specs = build_scale_design(&VoicingSpec::default());
    let mut r0 = [1.0_f32; TOTAL_STRINGS];
    for (k, layout) in CHOIR_LAYOUTS.iter().enumerate() {
        let p = specs[k];
        let m_str = p.linear_density_kg_m * p.speaking_length_m;
        let z0 = 2.0 * p.f0_et_hz * m_str;
        let scale = (11_000.0 / safe_sr).clamp(0.05, 0.6);
        let r_val = (z0 * scale).max(0.05) as f32;
        let start = usize::from(layout.first_string_idx);
        let count = usize::from(layout.strings_per_key);
        for slot in &mut r0[start..start + count] {
            *slot = r_val;
        }
    }
    r0
}

/// Passive 10-mode reduced soundboard/bridge state-space admittance with exact
/// trapezoidal integration and precomputed `8x8` WDF junction solve matrices.
#[derive(Clone, Debug)]
pub struct AdmittanceSsm {
    poles: [ReducedBridgePole; NUM_REDUCED_MODES],
    y_inf: [f32; NUM_BRIDGE_ZONES],
    y_inf_f64: [f64; NUM_BRIDGE_ZONES],
    ts: f64,
    half_ts: f64,
    m_m: [f64; NUM_REDUCED_MODES],
    c_m: [f64; NUM_REDUCED_MODES],
    k_m: [f64; NUM_REDUCED_MODES],
    y_m: [f64; NUM_REDUCED_MODES],
    phi: [[f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES],
    q: [f64; NUM_REDUCED_MODES],
    q_dot: [f64; NUM_REDUCED_MODES],
    diss_energy_j: f64,
    a_br: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    g_total: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    solve_8x8: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    solve_abr: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    solve_phi: [[f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES],
}

impl AdmittanceSsm {
    /// Constructs the passive reduced soundboard admittance at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::build(sample_rate, false)
    }

    /// Constructs the lossless test admittance (`zeta = 0`, `y_inf = 0`).
    #[must_use]
    pub fn new_lossless(sample_rate: f32) -> Self {
        Self::build(sample_rate, true)
    }

    fn build(sample_rate: f32, lossless: bool) -> Self {
        let safe_sr = f64::from(if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        });
        let ts = 1.0 / safe_sr;
        let h = 0.5 * ts;

        let y_inf = if lossless {
            [0.0_f32; NUM_BRIDGE_ZONES]
        } else {
            DEFAULT_Y_INF
        };
        let mut y_inf_f64 = [0.0_f64; NUM_BRIDGE_ZONES];
        for (dst, &src) in y_inf_f64.iter_mut().zip(y_inf.iter()) {
            *dst = f64::from(src);
        }

        let placeholder = ReducedBridgePole {
            freq_hz: 0.0,
            zeta: 0.0,
            q_factor: 0.0,
            modal_mass_kg: 1.0,
            zone_gains: [0.0; NUM_BRIDGE_ZONES],
        };
        let mut poles = [placeholder; NUM_REDUCED_MODES];
        let mut m_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut c_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut k_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut y_m = [0.0_f64; NUM_REDUCED_MODES];
        let mut phi = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES];

        for m in 0..NUM_REDUCED_MODES {
            let freq_hz = MODE_FREQS_HZ[m];
            let zeta = if lossless { 0.0 } else { MODE_ZETAS[m] };
            let q_factor = if zeta > 0.0 {
                0.5 / zeta
            } else {
                f32::INFINITY
            };
            let modal_mass_kg = MODE_MASSES_KG[m];
            let zone_gains = MODE_ZONE_GAINS[m];
            poles[m] = ReducedBridgePole {
                freq_hz,
                zeta,
                q_factor,
                modal_mass_kg,
                zone_gains,
            };

            let mass = f64::from(modal_mass_kg);
            let omega = TAU * f64::from(freq_hz);
            let stiff = mass * omega * omega;
            let damp = 2.0 * f64::from(zeta) * mass * omega;
            let z_mode = mass / h + damp + h * stiff;
            m_m[m] = mass;
            c_m[m] = damp;
            k_m[m] = stiff;
            y_m[m] = 1.0 / z_mode;
            for (dst, &src) in phi[m].iter_mut().zip(zone_gains.iter()) {
                *dst = f64::from(src);
            }
        }

        let mut a_br = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            a_br[z][z] = y_inf_f64[z];
            for zp in 0..NUM_BRIDGE_ZONES {
                let mut sum = 0.0_f64;
                for m in 0..NUM_REDUCED_MODES {
                    sum += y_m[m] * phi[m][z] * phi[m][zp];
                }
                a_br[z][zp] += sum;
            }
        }

        let ident = identity_8x8();
        Self {
            poles,
            y_inf,
            y_inf_f64,
            ts,
            half_ts: h,
            m_m,
            c_m,
            k_m,
            y_m,
            phi,
            q: [0.0; NUM_REDUCED_MODES],
            q_dot: [0.0; NUM_REDUCED_MODES],
            diss_energy_j: 0.0,
            a_br,
            g_total: [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
            solve_8x8: ident,
            solve_abr: a_br,
            solve_phi: phi,
        }
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

        let mut sys = identity_8x8();
        for (i, sys_row) in sys.iter_mut().enumerate() {
            for (j, sys_cell) in sys_row.iter_mut().enumerate() {
                let mut sum = 0.0_f64;
                for (l, g_row) in g_tot.iter().enumerate() {
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
        self.g_total = g_tot;
        self.solve_8x8 = inv;
        self.solve_abr = solve_abr;
    }

    /// Resets modal displacements, velocities, and accumulated dissipation energy.
    pub fn reset(&mut self) {
        self.q.fill(0.0);
        self.q_dot.fill(0.0);
        self.diss_energy_j = 0.0;
    }

    /// Returns the 10 reduced soundboard poles for M6 `piano-body-gen` pole cancellation.
    #[must_use]
    #[inline]
    pub const fn poles(&self) -> &[ReducedBridgePole; NUM_REDUCED_MODES] {
        &self.poles
    }

    /// Returns the diagonal high-frequency resistive bridge mobility `[m/(N*s)]`.
    #[must_use]
    #[inline]
    pub const fn y_inf(&self) -> &[f32; NUM_BRIDGE_ZONES] {
        &self.y_inf
    }

    /// Returns precomputed `A_br` `[m/(N*s)]`.
    #[must_use]
    #[inline]
    pub const fn a_br(&self) -> &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
        &self.a_br
    }

    /// Returns precomputed `G_total = G_str + diag(R_dup)`.
    #[must_use]
    #[inline]
    pub const fn g_total(&self) -> &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
        &self.g_total
    }

    /// Returns precomputed `(I_8 + A_br * G_total)^-1`.
    #[must_use]
    #[inline]
    pub const fn solve_8x8(&self) -> &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
        &self.solve_8x8
    }

    /// Returns precomputed `(I_8 + A_br * G_total)^-1 * A_br`.
    #[must_use]
    #[inline]
    pub const fn solve_abr(&self) -> &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
        &self.solve_abr
    }

    /// Evaluates free modal midpoint velocities and free zone velocities `v_zone_free`.
    #[must_use]
    #[inline]
    pub fn free_velocities(&self) -> ([f64; NUM_REDUCED_MODES], [f64; NUM_BRIDGE_ZONES]) {
        let mut qd_free = [0.0_f64; NUM_REDUCED_MODES];
        let mut v_free = [0.0_f64; NUM_BRIDGE_ZONES];
        let inv_h = 1.0 / self.half_ts;
        for (m, qdf_slot) in qd_free.iter_mut().enumerate() {
            let rhs = (self.m_m[m] * inv_h) * self.q_dot[m] - self.k_m[m] * self.q[m];
            let qdf = self.y_m[m] * rhs;
            *qdf_slot = qdf;
            for (vf, &phi_mz) in v_free.iter_mut().zip(self.phi[m].iter()) {
                *vf += phi_mz * qdf;
            }
        }
        (qd_free, v_free)
    }

    /// Evaluates free modal midpoint velocities and pre-solved linear zone velocity
    /// `v0_free = solve_8x8 * v_free` using precomputed `solve_phi = solve_8x8 * phi`.
    #[must_use]
    #[inline]
    pub fn free_v0(&self) -> ([f64; NUM_REDUCED_MODES], [f64; NUM_BRIDGE_ZONES]) {
        let mut qd_free = [0.0_f64; NUM_REDUCED_MODES];
        let mut v0_free = [0.0_f64; NUM_BRIDGE_ZONES];
        let inv_h = 1.0 / self.half_ts;
        for (m, qdf_slot) in qd_free.iter_mut().enumerate() {
            let rhs = (self.m_m[m] * inv_h) * self.q_dot[m] - self.k_m[m] * self.q[m];
            let qdf = self.y_m[m] * rhs;
            *qdf_slot = qdf;
            for (vf, &sp_mz) in v0_free.iter_mut().zip(self.solve_phi[m].iter()) {
                *vf += sp_mz * qdf;
            }
        }
        (qd_free, v0_free)
    }

    /// Returns `true` when all modal displacements and velocities are zero.
    #[must_use]
    #[inline]
    pub fn is_quiescent(&self) -> bool {
        self.q.iter().all(|&v| v == 0.0) && self.q_dot.iter().all(|&v| v == 0.0)
    }

    /// Commits one exact trapezoidal discrete integration step using `f_zone_ssm` (`[N]`).
    #[inline]
    pub fn commit_step(
        &mut self,
        q_dot_free: &[f64; NUM_REDUCED_MODES],
        f_zone_ssm: &[f64; NUM_BRIDGE_ZONES],
    ) {
        let mut diss = 0.0_f64;
        for (m, &qdf) in q_dot_free.iter().enumerate() {
            let pm = &self.phi[m];
            let p01 = pm[0] * f_zone_ssm[0] + pm[1] * f_zone_ssm[1];
            let p23 = pm[2] * f_zone_ssm[2] + pm[3] * f_zone_ssm[3];
            let p45 = pm[4] * f_zone_ssm[4] + pm[5] * f_zone_ssm[5];
            let p67 = pm[6] * f_zone_ssm[6] + pm[7] * f_zone_ssm[7];
            let gen_force = (p01 + p23) + (p45 + p67);
            let qd_mid = qdf + self.y_m[m] * gen_force;
            let qd_next = 2.0 * qd_mid - self.q_dot[m];
            let q_next = self.q[m] + self.ts * qd_mid;
            diss += self.ts * self.c_m[m] * qd_mid * qd_mid;
            self.q_dot[m] = qd_next;
            self.q[m] = q_next;
        }
        for (&y_z, &fz) in self.y_inf_f64.iter().zip(f_zone_ssm.iter()) {
            diss += self.ts * y_z * fz * fz;
        }
        self.diss_energy_j += diss;
    }

    /// Returns stored kinetic + potential energy across all 10 reduced bridge modes in Joules.
    #[must_use]
    pub fn energy_joules(&self) -> f64 {
        let mut e = 0.0_f64;
        for m in 0..NUM_REDUCED_MODES {
            let qd = self.q_dot[m];
            let q = self.q[m];
            e += 0.5 * (self.m_m[m] * qd * qd + self.k_m[m] * q * q);
        }
        e
    }

    /// Returns cumulative dissipated energy across modal damping and `y_inf` in Joules.
    #[must_use]
    #[inline]
    pub const fn dissipated_energy_joules(&self) -> f64 {
        self.diss_energy_j
    }
}

#[inline]
const fn identity_8x8() -> [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
    let mut id = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    let mut i = 0;
    while i < NUM_BRIDGE_ZONES {
        id[i][i] = 1.0;
        i += 1;
    }
    id
}

#[inline]
fn matmul_8x8(
    a: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    b: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
) -> [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
    let mut out = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
    for i in 0..NUM_BRIDGE_ZONES {
        for j in 0..NUM_BRIDGE_ZONES {
            let mut sum = 0.0_f64;
            for k in 0..NUM_BRIDGE_ZONES {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }
    out
}

/// Pure-Rust `no_std` Gauss-Jordan matrix inversion with partial pivoting (`8x8`, zero panic).
#[must_use]
pub fn invert_8x8(
    mat: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
) -> [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES] {
    let mut a = *mat;
    let mut inv = identity_8x8();

    for col in 0..NUM_BRIDGE_ZONES {
        let mut pivot_row = col;
        let mut max_abs = a[col][col].abs();
        for (row, a_row) in a.iter().enumerate().skip(col + 1) {
            let cand = a_row[col].abs();
            if cand > max_abs {
                max_abs = cand;
                pivot_row = row;
            }
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }
        let diag = a[col][col];
        if !diag.is_finite() || diag.abs() < 1.0e-18 {
            continue;
        }
        let inv_diag = 1.0 / diag;
        for j in 0..NUM_BRIDGE_ZONES {
            a[col][j] *= inv_diag;
            inv[col][j] *= inv_diag;
        }
        for row in 0..NUM_BRIDGE_ZONES {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor != 0.0 {
                for j in 0..NUM_BRIDGE_ZONES {
                    a[row][j] -= factor * a[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }
    inv
}
