//! Conservative passive nonlinear bridge-pin / downbearing Hertz contact stiffness
//! (`F32`/`F32x4` zone arithmetic + Sherman-Morrison SAV step).
//!
//! Potential energy `V_hertz(x) = (k3/3) * max(x, 0)^3 + (k4/4) * x^4 >= 0` models
//! nonlinear asymmetric bridge-pin compression stiffening around static equilibrium
//! without shifting the small-signal linear soundboard pole frequencies.

use piano_simd::F32x4;

use crate::math::sqrt_f64;
use crate::zone_proj::NUM_BRIDGE_ZONES;

/// Gauge constant `C_H` for Hertz contact SAV (`r = sqrt(2 * V_hertz(x) + C_H)`).
pub const C_HERTZ_SAV: f64 = 1.0e-3;

/// Default asymmetric downward bridge-pin compression coefficient `[N/m^2]`.
pub const DEFAULT_HERTZ_K3: f32 = 4.0e7;

/// Default symmetric high-amplitude bridge-cap hardening coefficient `[N/m^3]`.
pub const DEFAULT_HERTZ_K4: f32 = 8.0e11;

/// Evaluates single-zone Hertz contact potential energy `V_hertz(x) >= 0` in `f32`.
#[must_use]
#[inline]
pub fn hertz_potential_f32(x: f32, k3: f32, k4: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let xp = x.max(0.0);
    let x2 = x * x;
    (1.0_f32 / 3.0_f32) * k3 * (xp * xp * xp) + 0.25_f32 * k4 * (x2 * x2)
}

/// Evaluates single-zone nonlinear Hertz restoring force `dV/dx` in `f32`.
#[must_use]
#[inline]
pub fn hertz_force_f32(x: f32, k3: f32, k4: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    let xp = x.max(0.0);
    k3 * (xp * xp) + k4 * (x * x * x)
}

/// Passive energy-conserving nonlinear Hertz contact compliance across the 8 bridge zones.
#[derive(Clone, Debug)]
pub struct HertzBridgeContact {
    x: [F32x4; 2],
    v_prev: [F32x4; 2],
    r_sav: f64,
    ts: f64,
    k3: f32,
    k4: f32,
}

impl HertzBridgeContact {
    /// Initializes Hertz bridge-pin contact at rest for `sample_rate`.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_stiffness(sample_rate, DEFAULT_HERTZ_K3, DEFAULT_HERTZ_K4)
    }

    /// Initializes Hertz contact with explicit `(k3, k4)` stiffness parameters.
    #[must_use]
    pub fn new_with_stiffness(sample_rate: f32, k3: f32, k4: f32) -> Self {
        let safe_sr = f64::from(if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        });
        Self {
            x: [F32x4::zero(); 2],
            v_prev: [F32x4::zero(); 2],
            r_sav: sqrt_f64(C_HERTZ_SAV),
            ts: 1.0 / safe_sr,
            k3: if k3.is_finite() { k3.max(0.0) } else { 0.0 },
            k4: if k4.is_finite() { k4.max(0.0) } else { 0.0 },
        }
    }

    /// Resets zone displacements and SAV auxiliary state back to static equilibrium.
    pub fn reset(&mut self) {
        self.x = [F32x4::zero(); 2];
        self.v_prev = [F32x4::zero(); 2];
        self.r_sav = sqrt_f64(C_HERTZ_SAV);
    }

    /// Returns `true` when zone compressions and prior velocities are identically zero.
    #[must_use]
    #[inline]
    pub fn is_quiescent(&self) -> bool {
        self.x[0].to_array() == [0.0; 4]
            && self.x[1].to_array() == [0.0; 4]
            && self.v_prev[0].to_array() == [0.0; 4]
            && self.v_prev[1].to_array() == [0.0; 4]
    }

    /// Evaluates the midpoint-predicted normalized SAV gradient vector `g = dV/dx / r_sav`
    /// using `F32x4` lane operations across all 8 zones.
    #[must_use]
    #[inline]
    pub fn eval_sav_gradient(&self) -> [f64; NUM_BRIDGE_ZONES] {
        if self.k3 == 0.0 && self.k4 == 0.0 {
            return [0.0; NUM_BRIDGE_ZONES];
        }
        let half_ts = F32x4::splat((0.5 * self.ts) as f32);
        let k3_v = F32x4::splat(self.k3);
        let k4_v = F32x4::splat(self.k4);
        let zero = F32x4::zero();
        let inv_r = F32x4::splat((1.0 / self.r_sav.max(1.0e-12)) as f32);

        let mut g = [0.0_f64; NUM_BRIDGE_ZONES];
        for half in 0..2 {
            let x_star = self.v_prev[half].mul_add(half_ts, self.x[half]);
            let xp = x_star.max(zero);
            let f_nl = (k3_v * (xp * xp)) + (k4_v * (x_star * x_star * x_star));
            let g_vec = (f_nl * inv_r).to_array();
            for lane in 0..4 {
                let val = g_vec[lane];
                g[half * 4 + lane] = if val.is_finite() { f64::from(val) } else { 0.0 };
            }
        }
        g
    }

    /// Current SAV auxiliary scalar `r_sav`.
    #[must_use]
    #[inline]
    pub const fn r_sav(&self) -> f64 {
        self.r_sav
    }

    /// Updates zone compression `x` and SAV scalar `r_sav` with solved `v_zone`,
    /// returning the exact energy-conserving force `F_hertz_on_br` (`[N]`) applied to the bridge.
    #[inline]
    pub fn commit_step(
        &mut self,
        g: &[f64; NUM_BRIDGE_ZONES],
        v_zone: &[f64; NUM_BRIDGE_ZONES],
    ) -> [f64; NUM_BRIDGE_ZONES] {
        let mut gv = 0.0_f64;
        for (&gz, &vz) in g.iter().zip(v_zone.iter()) {
            if vz.is_finite() {
                gv += gz * vz;
            }
        }
        let r_prev = self.r_sav;
        let r_next = r_prev + self.ts * gv;
        let r_mid = 0.5 * (r_prev + r_next);
        self.r_sav = if r_next.is_finite() {
            r_next
        } else {
            sqrt_f64(C_HERTZ_SAV)
        };

        let ts_v = F32x4::splat(self.ts as f32);
        let v0 = F32x4::from_array([
            v_zone[0] as f32,
            v_zone[1] as f32,
            v_zone[2] as f32,
            v_zone[3] as f32,
        ]);
        let v1 = F32x4::from_array([
            v_zone[4] as f32,
            v_zone[5] as f32,
            v_zone[6] as f32,
            v_zone[7] as f32,
        ]);
        self.x[0] = v0.mul_add(ts_v, self.x[0]);
        self.x[1] = v1.mul_add(ts_v, self.x[1]);
        self.v_prev[0] = v0;
        self.v_prev[1] = v1;

        let mut f_on_br = [0.0_f64; NUM_BRIDGE_ZONES];
        for (dst, &gz) in f_on_br.iter_mut().zip(g.iter()) {
            *dst = -r_mid * gz;
        }
        f_on_br
    }

    /// Evaluates the exact instantaneous physical Hertz potential `sum_z V_hertz(x_z) >= 0.0` [J].
    #[must_use]
    pub fn potential_energy_joules(&self) -> f64 {
        let zero = F32x4::zero();
        let c3 = F32x4::splat((1.0_f32 / 3.0_f32) * self.k3);
        let c4 = F32x4::splat(0.25_f32 * self.k4);
        let mut total = 0.0_f64;
        for &xv in &self.x {
            let xp = xv.max(zero);
            let x2 = xv * xv;
            let v_vec = c3 * (xp * xp * xp) + c4 * (x2 * x2);
            for lane in v_vec.to_array() {
                if lane.is_finite() && lane > 0.0 {
                    total += f64::from(lane);
                }
            }
        }
        total
    }

    /// Returns the exact discrete-conserved SAV energy `0.5 * (r_sav^2 - C_HERTZ_SAV)` [J].
    #[must_use]
    #[inline]
    pub fn energy_joules(&self) -> f64 {
        0.5 * (self.r_sav * self.r_sav - C_HERTZ_SAV)
    }

    /// Returns the 8-zone bridge compression displacements `[m]`.
    #[must_use]
    pub fn displacements_m(&self) -> [f32; NUM_BRIDGE_ZONES] {
        let a = self.x[0].to_array();
        let b = self.x[1].to_array();
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
}
