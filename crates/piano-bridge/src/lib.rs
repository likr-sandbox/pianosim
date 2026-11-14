//! Stage 3: reduced-mode bidirectional bridge coupling, Hertz contact compliance,
//! 8-zone spatial projection (with 9th keybed zone reserved), and 64-mode duplex scale bank.
//!
//! Operates strictly in `no_std` real-time DSP mode with zero heap allocations, zero locks,
//! and zero panics in [`BridgeBank::step_bridge_sample`].

#![cfg_attr(not(test), no_std)]

pub mod admittance_ssm;
pub mod duplex;
pub mod hertz_contact;
mod math;
pub mod zone_proj;

#[cfg(test)]
#[path = "bridge_tests.rs"]
mod tests;

use piano_spec::TOTAL_STRINGS;

pub use admittance_ssm::{
    default_string_port_resistances, AdmittanceSsm, ReducedBridgePole, DEFAULT_Y_INF,
    NUM_REDUCED_MODES,
};
pub use duplex::{DuplexBank, DUPLEX_VEC_ZONES, NUM_DUPLEX_MODES, NUM_DUPLEX_VECS};
pub use hertz_contact::{
    hertz_force_f32, hertz_potential_f32, HertzBridgeContact, C_HERTZ_SAV, DEFAULT_HERTZ_K3,
    DEFAULT_HERTZ_K4,
};
pub use zone_proj::{
    build_zone_weights, gather_zones_to_strings, ZoneWeight, KEYBED_ZONE_IDX, NUM_BRIDGE_ZONES,
    NUM_RADIATOR_ZONES, TREBLE_DUPLEX_ZONE_START,
};

#[inline]
const fn sanitize_sample_rate(sample_rate: f32) -> f32 {
    if sample_rate.is_finite() {
        if sample_rate < 8_000.0 {
            8_000.0
        } else if sample_rate > 384_000.0 {
            384_000.0
        } else {
            sample_rate
        }
    } else {
        48_000.0
    }
}

/// Integrated Stage 3 bidirectional bridge bank coupling all 230 strings, 10 reduced
/// soundboard/bridge modes, nonlinear Hertz bridge-pin contact, and 64 treble duplex modes
/// across 8 physical bridge zones.
#[derive(Clone, Debug)]
pub struct BridgeBank {
    sample_rate: f32,
    ts: f64,
    zone_weights: [ZoneWeight; TOTAL_STRINGS],
    r0: [f32; TOTAL_STRINGS],
    ssm: AdmittanceSsm,
    hertz: HertzBridgeContact,
    duplex: DuplexBank,
}

impl BridgeBank {
    /// Initializes the Stage 3 bridge bank with physical concert-grand defaults at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        let r0 = default_string_port_resistances(safe_sr);
        let zone_weights = build_zone_weights();
        let mut ssm = AdmittanceSsm::new(safe_sr);
        let hertz = HertzBridgeContact::new(safe_sr);
        let duplex = DuplexBank::new(safe_sr);
        ssm.set_port_resistances(&r0, &zone_weights, duplex.port_resistances());
        Self {
            sample_rate: safe_sr,
            ts: 1.0 / f64::from(safe_sr),
            zone_weights,
            r0,
            ssm,
            hertz,
            duplex,
        }
    }

    /// Initializes the Stage 3 bridge bank in exact lossless mode (`zeta = 0`, `y_inf = 0`,
    /// `duplex_gain = 0`) while keeping conservative nonlinear `HertzBridgeContact` active.
    #[must_use]
    pub fn lossless_for_test(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        let r0 = default_string_port_resistances(safe_sr);
        let zone_weights = build_zone_weights();
        let mut ssm = AdmittanceSsm::new_lossless(safe_sr);
        let hertz = HertzBridgeContact::new(safe_sr);
        let duplex = DuplexBank::new_with_params(safe_sr, 0.0, true);
        ssm.set_port_resistances(&r0, &zone_weights, duplex.port_resistances());
        Self {
            sample_rate: safe_sr,
            ts: 1.0 / f64::from(safe_sr),
            zone_weights,
            r0,
            ssm,
            hertz,
            duplex,
        }
    }

    #[cfg(test)]
    pub(crate) fn lossless_with_duplex_for_test(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        let r0 = default_string_port_resistances(safe_sr);
        let zone_weights = build_zone_weights();
        let mut ssm = AdmittanceSsm::new_lossless(safe_sr);
        let hertz = HertzBridgeContact::new(safe_sr);
        let duplex = DuplexBank::new_with_params(safe_sr, 1.0, true);
        ssm.set_port_resistances(&r0, &zone_weights, duplex.port_resistances());
        Self {
            sample_rate: safe_sr,
            ts: 1.0 / f64::from(safe_sr),
            zone_weights,
            r0,
            ssm,
            hertz,
            duplex,
        }
    }

    /// Updates per-string WDF bridge port resistances `r0` (`[kg/s]`) and recomputes the
    /// precomputed `8x8` WDF junction matrices offline (`init` / reconfiguration only).
    pub fn set_port_resistances(&mut self, r0: &[f32; TOTAL_STRINGS]) {
        for (dst, &src) in self.r0.iter_mut().zip(r0.iter()) {
            *dst = if src.is_finite() { src.max(0.0) } else { 0.0 };
        }
        self.ssm
            .set_port_resistances(&self.r0, &self.zone_weights, self.duplex.port_resistances());
    }

    /// Resets dynamic bridge, Hertz contact, and duplex states to rest equilibrium.
    pub fn reset(&mut self) {
        self.ssm.reset();
        self.hertz.reset();
        self.duplex.reset();
    }

    /// Configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Configured per-string WDF bridge port resistances `[kg/s]`.
    #[must_use]
    #[inline]
    pub const fn port_resistances(&self) -> &[f32; TOTAL_STRINGS] {
        &self.r0
    }

    /// Partition-of-unity mapping `(z0, w0, z1, w1)` for all 230 strings onto Zones `0..8`.
    #[must_use]
    #[inline]
    pub const fn zone_weights(&self) -> &[ZoneWeight; TOTAL_STRINGS] {
        &self.zone_weights
    }

    /// Public 10 reduced soundboard/bridge poles for M6 `piano-body-gen` pole cancellation.
    #[must_use]
    #[inline]
    pub const fn reduced_poles(&self) -> &[ReducedBridgePole; NUM_REDUCED_MODES] {
        self.ssm.poles()
    }

    /// Direct immutable view of the reduced soundboard admittance model.
    #[must_use]
    #[inline]
    pub const fn ssm(&self) -> &AdmittanceSsm {
        &self.ssm
    }

    /// Direct immutable view of the nonlinear Hertz bridge-pin contact model.
    #[must_use]
    #[inline]
    pub const fn hertz(&self) -> &HertzBridgeContact {
        &self.hertz
    }

    /// Direct immutable view of the 64-mode treble duplex scale resonator bank.
    #[must_use]
    #[inline]
    pub const fn duplex(&self) -> &DuplexBank {
        &self.duplex
    }

    /// Instantaneous stored energy in the 10 reduced soundboard/bridge modes [J].
    #[must_use]
    #[inline]
    pub fn ssm_energy_joules(&self) -> f64 {
        self.ssm.energy_joules()
    }

    /// Instantaneous SAV energy stored in the nonlinear Hertz bridge-pin contact [J].
    #[must_use]
    #[inline]
    pub fn hertz_energy_joules(&self) -> f64 {
        self.hertz.energy_joules()
    }

    /// Instantaneous stored energy across all 64 treble duplex scale modes [J].
    #[must_use]
    #[inline]
    pub fn duplex_energy_joules(&self) -> f64 {
        self.duplex.energy_joules()
    }

    /// Total instantaneous stored mechanical energy across bridge, Hertz contact, and duplex [J].
    #[must_use]
    #[inline]
    pub fn total_energy_joules(&self) -> f64 {
        self.ssm.energy_joules() + self.hertz.energy_joules() + self.duplex.energy_joules()
    }

    /// Cumulative dissipated energy in the reduced soundboard admittance [J].
    #[must_use]
    #[inline]
    pub fn dissipated_energy_joules(&self) -> f64 {
        self.ssm.dissipated_energy_joules()
    }

    /// Advances the Stage 3 bidirectional bridge junction by one audio sample.
    ///
    /// - `a_waves`: incident string velocity waves `[m/s]` entering the bridge from each string.
    /// - `zone_str_audio`: broadband (`>= 5.5 kHz` + longitudinal precursor) zone force `[N]`;
    ///   strictly bypasses the passive WDF state and `b_waves_out` so WDF passivity holds cleanly.
    /// - `b_waves_out`: reflected string velocity waves `[m/s]` (`2 * v_br_s - a_s`).
    /// - `zone_out`: physical 8-zone radiator velocities `[m/s]` (`v_zone_wdf + y_inf * zone_str_audio`).
    /// - Returns `(duplex_left, duplex_right)` stereo acoustic contribution from Zones `2..7`.
    #[inline]
    pub fn step_bridge_sample(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; NUM_BRIDGE_ZONES],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; NUM_BRIDGE_ZONES],
    ) -> (f32, f32) {
        let mut f_inc_str = [0.0_f64; NUM_BRIDGE_ZONES];
        for ((&a_s, &(z0, w0, z1, w1)), &r_s) in a_waves
            .iter()
            .zip(self.zone_weights.iter())
            .zip(self.r0.iter())
        {
            let safe_a = if a_s.is_finite() { f64::from(a_s) } else { 0.0 };
            let inc = f64::from(r_s) * safe_a;
            let i0 = usize::from(z0) & (NUM_BRIDGE_ZONES - 1);
            let i1 = usize::from(z1) & (NUM_BRIDGE_ZONES - 1);
            f_inc_str[i0] += f64::from(w0) * inc;
            f_inc_str[i1] += f64::from(w1) * inc;
        }

        let f_dup_free = self.duplex.free_forces();
        let mut f_inc_wdf = [0.0_f64; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            f_inc_wdf[z] = f_inc_str[z] + f_dup_free[z];
        }

        let (qd_free, v_free) = self.ssm.free_velocities();
        let g = self.hertz.eval_sav_gradient();
        let r_h = self.hertz.r_sav();

        let a_br = self.ssm.a_br();
        let solve_8x8 = self.ssm.solve_8x8();
        let solve_abr = self.ssm.solve_abr();
        let g_total = self.ssm.g_total();

        let mut f_rhs_eff = [0.0_f64; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            f_rhs_eff[z] = f_inc_wdf[z] - r_h * g[z];
        }

        let mut rhs = [0.0_f64; NUM_BRIDGE_ZONES];
        for i in 0..NUM_BRIDGE_ZONES {
            let mut acc = v_free[i];
            for j in 0..NUM_BRIDGE_ZONES {
                acc += a_br[i][j] * f_rhs_eff[j];
            }
            rhs[i] = acc;
        }

        let mut v0 = [0.0_f64; NUM_BRIDGE_ZONES];
        for i in 0..NUM_BRIDGE_ZONES {
            let mut acc = 0.0_f64;
            for j in 0..NUM_BRIDGE_ZONES {
                acc += solve_8x8[i][j] * rhs[j];
            }
            v0[i] = acc;
        }

        let half_ts = 0.5 * self.ts;
        let mut u_sm = [0.0_f64; NUM_BRIDGE_ZONES];
        let mut denom = 1.0_f64;
        let mut g_dot_v0 = 0.0_f64;
        for i in 0..NUM_BRIDGE_ZONES {
            let mut acc = 0.0_f64;
            for j in 0..NUM_BRIDGE_ZONES {
                acc += solve_abr[i][j] * g[j];
            }
            let ui = half_ts * acc;
            u_sm[i] = ui;
            denom += g[i] * ui;
            g_dot_v0 += g[i] * v0[i];
        }
        let alpha_sm = if denom > 1.0e-12 {
            g_dot_v0 / denom
        } else {
            0.0
        };
        let mut v_zone_f64 = [0.0_f64; NUM_BRIDGE_ZONES];
        for i in 0..NUM_BRIDGE_ZONES {
            v_zone_f64[i] = v0[i] - alpha_sm * u_sm[i];
        }

        let f_hertz_on_br = self.hertz.commit_step(&g, &v_zone_f64);
        let mut f_zone_ssm = [0.0_f64; NUM_BRIDGE_ZONES];
        for i in 0..NUM_BRIDGE_ZONES {
            let mut g_v = 0.0_f64;
            for j in 0..NUM_BRIDGE_ZONES {
                g_v += g_total[i][j] * v_zone_f64[j];
            }
            f_zone_ssm[i] = f_inc_wdf[i] + f_hertz_on_br[i] - g_v;
        }
        self.ssm.commit_step(&qd_free, &f_zone_ssm);

        let mut v_zone = [0.0_f32; NUM_BRIDGE_ZONES];
        for (dst, &src) in v_zone.iter_mut().zip(v_zone_f64.iter()) {
            let vz = src as f32;
            *dst = if vz.is_finite() { vz } else { 0.0 };
        }
        let duplex_stereo = self.duplex.step_with_zone_velocities(&v_zone);

        for ((b_out, &a_s), &(z0, w0, z1, w1)) in b_waves_out
            .iter_mut()
            .zip(a_waves.iter())
            .zip(self.zone_weights.iter())
        {
            let safe_a = if a_s.is_finite() { f64::from(a_s) } else { 0.0 };
            let i0 = usize::from(z0) & (NUM_BRIDGE_ZONES - 1);
            let i1 = usize::from(z1) & (NUM_BRIDGE_ZONES - 1);
            let v_br_s = f64::from(w0) * v_zone_f64[i0] + f64::from(w1) * v_zone_f64[i1];
            let b_val = (2.0 * v_br_s - safe_a) as f32;
            *b_out = if b_val.is_finite() { b_val } else { 0.0 };
        }

        let y_inf = self.ssm.y_inf();
        for z in 0..NUM_BRIDGE_ZONES {
            let f_audio = if zone_str_audio[z].is_finite() {
                zone_str_audio[z]
            } else {
                0.0
            };
            let out_v = v_zone[z] + y_inf[z] * f_audio;
            zone_out[z] = if out_v.is_finite() { out_v } else { 0.0 };
        }

        duplex_stereo
    }
}
