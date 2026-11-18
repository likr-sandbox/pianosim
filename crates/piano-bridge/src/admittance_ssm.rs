//! Reduced-order concert grand soundboard/bridge admittance state-space model (`AdmittanceSsm`).
//!
//! Models 10 passive reduced soundboard modes (`~54..1,130 Hz`) plus diagonal local
//! high-frequency resistive mobility `y_inf` across the 8 bridge zones, integrated
//! with exact discrete trapezoidal energy balance and precomputed `8x8` WDF junction solve.

use core::f64::consts::TAU;

use piano_spec::PresetReducedPole;

use crate::bridge_presets::{build_default_poles, sanitize_reduced_poles, sanitize_y_inf};
pub use crate::bridge_presets::{
    default_string_port_resistances, ReducedBridgePole, DEFAULT_Y_INF, MODE_ZONE_GAINS,
    NUM_REDUCED_MODES,
};
pub use crate::math::invert_8x8;
use crate::math::{identity_8x8, sanitize_sample_rate};
use crate::zone_proj::NUM_BRIDGE_ZONES;

/// Passive 10-mode reduced soundboard/bridge state-space admittance with exact
/// trapezoidal integration and precomputed `8x8` WDF junction solve matrices.
#[derive(Clone, Debug)]
pub struct AdmittanceSsm {
    pub(crate) poles: [ReducedBridgePole; NUM_REDUCED_MODES],
    pub(crate) y_inf: [f32; NUM_BRIDGE_ZONES],
    pub(crate) y_inf_f64: [f64; NUM_BRIDGE_ZONES],
    pub(crate) ts: f64,
    pub(crate) half_ts: f64,
    pub(crate) m_m: [f64; NUM_REDUCED_MODES],
    pub(crate) c_m: [f64; NUM_REDUCED_MODES],
    pub(crate) k_m: [f64; NUM_REDUCED_MODES],
    pub(crate) y_m: [f64; NUM_REDUCED_MODES],
    pub(crate) phi: [[f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES],
    pub(crate) q: [f64; NUM_REDUCED_MODES],
    pub(crate) q_dot: [f64; NUM_REDUCED_MODES],
    pub(crate) diss_energy_j: f64,
    pub(crate) a_br: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    pub(crate) g_total: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    pub(crate) solve_8x8: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    pub(crate) solve_abr: [[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    pub(crate) solve_phi: [[f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES],
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

    /// Constructs a passive reduced soundboard admittance from external `.pbdy`
    /// `(y_inf, reduced_poles)` data, clamping parameters to strictly positive bounds.
    #[must_use]
    pub fn from_reduced_model(
        sample_rate: f32,
        y_inf: &[f32; NUM_BRIDGE_ZONES],
        reduced_poles: &[PresetReducedPole; NUM_REDUCED_MODES],
    ) -> Self {
        Self::build_with_model(
            sample_rate,
            sanitize_y_inf(y_inf),
            sanitize_reduced_poles(reduced_poles),
        )
    }

    fn build(sample_rate: f32, lossless: bool) -> Self {
        let y_inf = if lossless {
            [0.0_f32; NUM_BRIDGE_ZONES]
        } else {
            DEFAULT_Y_INF
        };
        let poles = build_default_poles(lossless);
        Self::build_with_model(sample_rate, y_inf, poles)
    }

    fn build_with_model(
        sample_rate: f32,
        y_inf: [f32; NUM_BRIDGE_ZONES],
        poles: [ReducedBridgePole; NUM_REDUCED_MODES],
    ) -> Self {
        let safe_sr = f64::from(sanitize_sample_rate(sample_rate));
        let ts = 1.0 / safe_sr;
        let h = 0.5 * ts;
        let mut ssm = Self {
            poles,
            y_inf,
            y_inf_f64: [0.0_f64; NUM_BRIDGE_ZONES],
            ts,
            half_ts: h,
            m_m: [0.0_f64; NUM_REDUCED_MODES],
            c_m: [0.0_f64; NUM_REDUCED_MODES],
            k_m: [0.0_f64; NUM_REDUCED_MODES],
            y_m: [0.0_f64; NUM_REDUCED_MODES],
            phi: [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES],
            q: [0.0; NUM_REDUCED_MODES],
            q_dot: [0.0; NUM_REDUCED_MODES],
            diss_energy_j: 0.0,
            a_br: [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
            g_total: [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
            solve_8x8: identity_8x8(),
            solve_abr: [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
            solve_phi: [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES],
        };
        ssm.apply_model(y_inf, poles);
        ssm.solve_abr = ssm.a_br;
        ssm.solve_phi = ssm.phi;
        ssm
    }

    pub(crate) fn apply_model(
        &mut self,
        y_inf: [f32; NUM_BRIDGE_ZONES],
        poles: [ReducedBridgePole; NUM_REDUCED_MODES],
    ) {
        self.poles = poles;
        self.y_inf = y_inf;
        for (dst, &src) in self.y_inf_f64.iter_mut().zip(y_inf.iter()) {
            *dst = f64::from(src);
        }

        let h = self.half_ts;
        for (m, pole) in poles.iter().enumerate() {
            let mass = f64::from(pole.modal_mass_kg);
            let omega = TAU * f64::from(pole.freq_hz);
            let stiff = mass * omega * omega;
            let damp = 2.0 * f64::from(pole.zeta) * mass * omega;
            let z_mode = mass / h + damp + h * stiff;
            self.m_m[m] = mass;
            self.c_m[m] = damp;
            self.k_m[m] = stiff;
            self.y_m[m] = 1.0 / z_mode;
            for (dst, &src) in self.phi[m].iter_mut().zip(pole.zone_gains.iter()) {
                *dst = f64::from(src);
            }
        }

        let mut a_br = [[0.0_f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES];
        for (z, a_row) in a_br.iter_mut().enumerate() {
            a_row[z] = self.y_inf_f64[z];
            for (zp, cell) in a_row.iter_mut().enumerate() {
                let mut sum = 0.0_f64;
                for m in 0..NUM_REDUCED_MODES {
                    sum += self.y_m[m] * self.phi[m][z] * self.phi[m][zp];
                }
                *cell += sum;
            }
        }
        self.a_br = a_br;
        self.reset();
    }

    /// Updates `poles`, `y_inf`, `y_inf_f64`, modal matrices `(m_m, c_m, k_m, y_m, phi, a_br)`,
    /// and junction solves `(solve_8x8, solve_abr, solve_phi)` from `.pbdy` data, resetting
    /// `(q, q_dot, diss_energy_j)` to rest equilibrium.
    pub fn set_reduced_model(
        &mut self,
        y_inf: &[f32; NUM_BRIDGE_ZONES],
        reduced_poles: &[PresetReducedPole; NUM_REDUCED_MODES],
    ) {
        self.apply_model(sanitize_y_inf(y_inf), sanitize_reduced_poles(reduced_poles));
        self.recompute_junction_matrices();
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

    /// Returns precomputed `solve_8x8 * phi` modal velocity projection matrix.
    #[must_use]
    #[inline]
    pub const fn solve_phi(&self) -> &[[f64; NUM_BRIDGE_ZONES]; NUM_REDUCED_MODES] {
        &self.solve_phi
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

    /// Flushes `< 1e-20` denormals/subnormals and non-finite modal states `(q, q_dot)` to `0.0`.
    pub fn flush_subnormals(&mut self) {
        for m in 0..NUM_REDUCED_MODES {
            if !self.q[m].is_finite() || self.q[m].abs() < 1.0e-20 {
                self.q[m] = 0.0;
            }
            if !self.q_dot[m].is_finite() || self.q_dot[m].abs() < 1.0e-20 {
                self.q_dot[m] = 0.0;
            }
        }
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
