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

const MAX_ZONE_RUNS: usize = 16;
const _: () = assert!(TOTAL_STRINGS <= u8::MAX as usize);

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ZoneRun(u8, u8, u8, u8);

impl ZoneRun {
    #[inline]
    fn bounds(self) -> (usize, usize, usize, usize) {
        let end = usize::from(self.1).min(TOTAL_STRINGS);
        let mask = NUM_BRIDGE_ZONES - 1;
        (
            usize::from(self.0).min(end),
            end,
            usize::from(self.2) & mask,
            usize::from(self.3) & mask,
        )
    }
}

#[derive(Clone, Debug)]
struct ZoneRunCache {
    runs: [ZoneRun; MAX_ZONE_RUNS],
    run_count: usize,
    rw0: [f64; TOTAL_STRINGS],
    rw1: [f64; TOTAL_STRINGS],
    w0_f64: [f64; TOTAL_STRINGS],
    w1_f64: [f64; TOTAL_STRINGS],
}

impl ZoneRunCache {
    fn build(zone_weights: &[ZoneWeight; TOTAL_STRINGS], r0: &[f32; TOTAL_STRINGS]) -> Self {
        let mut runs = [ZoneRun::default(); MAX_ZONE_RUNS];
        let mut run_count = 0_usize;
        let mut rw0 = [0.0_f64; TOTAL_STRINGS];
        let mut rw1 = [0.0_f64; TOTAL_STRINGS];
        let mut w0_f64 = [0.0_f64; TOTAL_STRINGS];
        let mut w1_f64 = [0.0_f64; TOTAL_STRINGS];

        for (s, (&(raw_z0, w0, raw_z1, w1), &r_s)) in zone_weights.iter().zip(r0).enumerate() {
            let z0 = raw_z0 & ((NUM_BRIDGE_ZONES - 1) as u8);
            let z1 = raw_z1 & ((NUM_BRIDGE_ZONES - 1) as u8);
            let fw0 = f64::from(w0);
            let fw1 = f64::from(w1);
            let fr = f64::from(r_s);
            w0_f64[s] = fw0;
            w1_f64[s] = fw1;
            rw0[s] = fr * fw0;
            rw1[s] = fr * fw1;

            let end = (s + 1) as u8;
            if run_count > 0 && runs[run_count - 1].2 == z0 && runs[run_count - 1].3 == z1 {
                runs[run_count - 1].1 = end;
            } else if run_count < MAX_ZONE_RUNS {
                runs[run_count] = ZoneRun(s as u8, end, z0, z1);
                run_count += 1;
            }
        }

        Self {
            runs,
            run_count,
            rw0,
            rw1,
            w0_f64,
            w1_f64,
        }
    }

    fn update_resistances(&mut self, r0: &[f32; TOTAL_STRINGS]) {
        for (s, &r_s) in r0.iter().enumerate() {
            let fr = f64::from(r_s);
            self.rw0[s] = fr * self.w0_f64[s];
            self.rw1[s] = fr * self.w1_f64[s];
        }
    }
}

#[cold]
#[inline(never)]
fn scatter_sanitized_run(
    cache: &ZoneRunCache,
    run: ZoneRun,
    a_waves: &[f32; TOTAL_STRINGS],
) -> (f64, f64) {
    let (start, end, _, _) = run.bounds();
    let (mut acc0, mut acc1) = (0.0_f64, 0.0_f64);
    for ((&a_s, &c0), &c1) in a_waves[start..end]
        .iter()
        .zip(&cache.rw0[start..end])
        .zip(&cache.rw1[start..end])
    {
        let safe_a = if a_s.is_finite() { f64::from(a_s) } else { 0.0 };
        acc0 += c0 * safe_a;
        acc1 += c1 * safe_a;
    }
    (acc0, acc1)
}

#[cold]
#[inline(never)]
fn gather_sanitized_run(
    cache: &ZoneRunCache,
    run: ZoneRun,
    b_waves: &mut [f32; TOTAL_STRINGS],
    a_waves: &[f32; TOTAL_STRINGS],
    two_vz0: f64,
    two_vz1: f64,
) {
    let (start, end, _, _) = run.bounds();
    for (((b_out, &a_s), &w0), &w1) in b_waves[start..end]
        .iter_mut()
        .zip(&a_waves[start..end])
        .zip(&cache.w0_f64[start..end])
        .zip(&cache.w1_f64[start..end])
    {
        if !b_out.is_finite() {
            let safe_a = if a_s.is_finite() { f64::from(a_s) } else { 0.0 };
            let fb = (w0 * two_vz0 + w1 * two_vz1 - safe_a) as f32;
            *b_out = if fb.is_finite() { fb } else { 0.0 };
        }
    }
}

#[inline]
fn mat8_vec(
    m: &[[f64; NUM_BRIDGE_ZONES]; NUM_BRIDGE_ZONES],
    v: &[f64; NUM_BRIDGE_ZONES],
) -> [f64; NUM_BRIDGE_ZONES] {
    let mut out = [0.0_f64; NUM_BRIDGE_ZONES];
    for (dst, row) in out.iter_mut().zip(m.iter()) {
        let mut acc = 0.0_f64;
        for (&m_ij, &v_j) in row.iter().zip(v.iter()) {
            acc += m_ij * v_j;
        }
        *dst = acc;
    }
    out
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
    zone_cache: ZoneRunCache,
    ssm: AdmittanceSsm,
    hertz: HertzBridgeContact,
    duplex: DuplexBank,
}

impl BridgeBank {
    fn build(sample_rate: f32, mut ssm: AdmittanceSsm, duplex: DuplexBank) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        let r0 = default_string_port_resistances(safe_sr);
        let zone_weights = build_zone_weights();
        let hertz = HertzBridgeContact::new(safe_sr);
        ssm.set_port_resistances(&r0, &zone_weights, duplex.port_resistances());
        let zone_cache = ZoneRunCache::build(&zone_weights, &r0);
        Self {
            sample_rate: safe_sr,
            ts: 1.0 / f64::from(safe_sr),
            zone_weights,
            r0,
            zone_cache,
            ssm,
            hertz,
            duplex,
        }
    }

    /// Initializes the Stage 3 bridge bank with physical concert-grand defaults at `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        Self::build(
            safe_sr,
            AdmittanceSsm::new(safe_sr),
            DuplexBank::new(safe_sr),
        )
    }

    /// Initializes the Stage 3 bridge bank in exact lossless mode (`zeta = 0`, `y_inf = 0`,
    /// `duplex_gain = 0`) while keeping conservative nonlinear `HertzBridgeContact` active.
    #[must_use]
    pub fn lossless_for_test(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        Self::build(
            safe_sr,
            AdmittanceSsm::new_lossless(safe_sr),
            DuplexBank::new_with_params(safe_sr, 0.0, true),
        )
    }

    #[cfg(test)]
    pub(crate) fn lossless_with_duplex_for_test(sample_rate: f32) -> Self {
        let safe_sr = sanitize_sample_rate(sample_rate);
        Self::build(
            safe_sr,
            AdmittanceSsm::new_lossless(safe_sr),
            DuplexBank::new_with_params(safe_sr, 1.0, true),
        )
    }

    /// Updates per-string WDF bridge port resistances `r0` (`[kg/s]`) and recomputes the
    /// precomputed `8x8` WDF junction matrices offline (`init` / reconfiguration only).
    pub fn set_port_resistances(&mut self, r0: &[f32; TOTAL_STRINGS]) {
        for (dst_r, &src) in self.r0.iter_mut().zip(r0.iter()) {
            *dst_r = if src.is_finite() { src.max(0.0) } else { 0.0 };
        }
        self.zone_cache.update_resistances(&self.r0);
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
        let cache = &self.zone_cache;
        let mut f_inc_str = [0.0_f64; NUM_BRIDGE_ZONES];
        for &run in &cache.runs[..cache.run_count] {
            let (start, end, z0, z1) = run.bounds();
            let (mut acc0, mut acc1) = (0.0_f64, 0.0_f64);
            for ((&a_s, &rw0), &rw1) in a_waves[start..end]
                .iter()
                .zip(&cache.rw0[start..end])
                .zip(&cache.rw1[start..end])
            {
                let a = f64::from(a_s);
                acc0 += rw0 * a;
                acc1 += rw1 * a;
            }
            if !(acc0.is_finite() && acc1.is_finite()) {
                (acc0, acc1) = scatter_sanitized_run(cache, run, a_waves);
            }
            f_inc_str[z0] += acc0;
            f_inc_str[z1] += acc1;
        }

        let f_dup_free = self.duplex.free_forces();
        let (qd_free, v_free) = self.ssm.free_velocities();
        let g = self.hertz.eval_sav_gradient();
        let r_h = self.hertz.r_sav();

        let mut f_inc_wdf = [0.0_f64; NUM_BRIDGE_ZONES];
        let mut f_rhs_eff = [0.0_f64; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            let f_wdf = f_inc_str[z] + f_dup_free[z];
            f_inc_wdf[z] = f_wdf;
            f_rhs_eff[z] = f_wdf - r_h * g[z];
        }

        let a_f = mat8_vec(self.ssm.a_br(), &f_rhs_eff);
        let mut rhs = [0.0_f64; NUM_BRIDGE_ZONES];
        for i in 0..NUM_BRIDGE_ZONES {
            rhs[i] = v_free[i] + a_f[i];
        }
        let v0 = mat8_vec(self.ssm.solve_8x8(), &rhs);
        let s_g = mat8_vec(self.ssm.solve_abr(), &g);

        let half_ts = 0.5 * self.ts;
        let mut u_sm = [0.0_f64; NUM_BRIDGE_ZONES];
        let (mut denom, mut g_dot_v0) = (1.0_f64, 0.0_f64);
        for i in 0..NUM_BRIDGE_ZONES {
            let ui = half_ts * s_g[i];
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
        let g_v = mat8_vec(self.ssm.g_total(), &v_zone_f64);
        let mut f_zone_ssm = [0.0_f64; NUM_BRIDGE_ZONES];
        for i in 0..NUM_BRIDGE_ZONES {
            f_zone_ssm[i] = f_inc_wdf[i] + f_hertz_on_br[i] - g_v[i];
        }
        self.ssm.commit_step(&qd_free, &f_zone_ssm);

        let mut v_zone = [0.0_f32; NUM_BRIDGE_ZONES];
        for (dst, &src) in v_zone.iter_mut().zip(v_zone_f64.iter()) {
            let vz = src as f32;
            *dst = if vz.is_finite() { vz } else { 0.0 };
        }
        let duplex_stereo = self.duplex.step_with_zone_velocities(&v_zone);

        for &run in &cache.runs[..cache.run_count] {
            let (start, end, z0, z1) = run.bounds();
            let (vz0, vz1) = (v_zone_f64[z0], v_zone_f64[z1]);
            let two_vz0 = if vz0.is_finite() { 2.0 * vz0 } else { 0.0 };
            let two_vz1 = if vz1.is_finite() { 2.0 * vz1 } else { 0.0 };
            let mut all_finite = true;
            for (((b_out, &a_s), &w0), &w1) in b_waves_out[start..end]
                .iter_mut()
                .zip(&a_waves[start..end])
                .zip(&cache.w0_f64[start..end])
                .zip(&cache.w1_f64[start..end])
            {
                let b = (w0 * two_vz0 + w1 * two_vz1 - f64::from(a_s)) as f32;
                *b_out = b;
                all_finite &= b.is_finite();
            }
            if !all_finite {
                gather_sanitized_run(cache, run, b_waves_out, a_waves, two_vz0, two_vz1);
            }
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
