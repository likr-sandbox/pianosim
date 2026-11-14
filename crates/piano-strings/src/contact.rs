//! Stage 1 <-> Stage 2 physical SI dual ports (`Newtons`, `meters`, `m/s`) and
//! single-pass coupled L1-tiled SIMD integration for hammer strikes and dampers.

use core::f64::consts::{PI, TAU};
use core::ops::Range;

use piano_simd::{AlignedBuf, F32x4};
use piano_spec::{
    effective_string_f0_hz, partial_frequency_hz, Polarization, StringPhysicalParams, VoicingSpec,
    NUM_KEYS, TOTAL_STRINGS,
};

use crate::choir_map::{ChoirMap, KeyMapEntry, StringMapEntry};
use crate::modal_soa::{exp_f64, sin_cos_f64, sqrt_f64, C_HORIZ, C_VERT, NYQUIST_CUTOFF_RATIO};
use crate::zoh_kernel::{flush_subnormals_range, step_string_tile, zoh_rotate, TILE};

const HORIZ_STRIKE_SCALE: f64 = 0.65;
const HORIZ_DAMP_SCALE: f64 = 0.78;
const LONG_DAMP_SCALE: f64 = 0.45;
const DAMPER_CLEAR_THRESHOLD_M: f32 = 4.5e-3;

/// Stage 1 key-action driver driving nonlinear hammer strike and physical damper
/// ports per sample with exact SI duality (`Newtons`, `meters`, `m/s`).
pub trait KeyActionDriver {
    /// Returns `true` while hammer `key_idx` is still in flight or rebounding toward rest.
    fn is_hammer_active(&self, key_idx: usize) -> bool;
    /// Instantaneous geometric clearance `y_clear` (m) of damper pad `key_idx`.
    fn damper_clearance_m(&self, key_idx: usize) -> f32;
    /// Returns per-choir-string hammer strike weights `[w0, w1, w2]` (Una Corda support).
    fn strike_weights(&self, key_idx: usize) -> [f32; 3];
    /// Advances damper rail smoother by 1 sample for `key_idx` and returns `y_clear` (m).
    fn step_key_clearance(&mut self, key_idx: usize) -> f32;
    /// Evaluates hammer force `F_strike` (N, `+` up) from `(y_str_m, v_str_mps, r_port_strike)`.
    fn step_hammer_sample(
        &mut self,
        key_idx: usize,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32;
    /// Evaluates unilateral damper force `F_damp` (N, on string) from port state & `y_clear_m`.
    fn step_damper_sample(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32;
}

/// Precomputed SI port projection arrays and per-string dual-port state cache.
#[derive(Clone, Debug)]
pub(crate) struct ContactPorts {
    pub b_strike: AlignedBuf,
    pub a_strike: AlignedBuf,
    pub b_damp: AlignedBuf,
    pub a_damp: AlignedBuf,
    pub alpha_vel: [f32; TOTAL_STRINGS],
    pub r_port_strike: [f32; TOTAL_STRINGS],
    pub r_port_damp: [f32; TOTAL_STRINGS],
    pub y_h: [f32; TOTAL_STRINGS],
    pub v_h: [f32; TOTAL_STRINGS],
    pub y_d: [f32; TOTAL_STRINGS],
    pub v_d: [f32; TOTAL_STRINGS],
    pub may_be_nonzero: [bool; TOTAL_STRINGS],
    pub ports_synced: [bool; TOTAL_STRINGS],
}

impl ContactPorts {
    /// Precomputes exact SI strike/damper modal projections and port mobilities.
    #[must_use]
    pub fn build(
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        map: &ChoirMap,
    ) -> Self {
        let len = map.total_padded_modes();
        let mut ports = Self {
            b_strike: AlignedBuf::zeroed(len),
            a_strike: AlignedBuf::zeroed(len),
            b_damp: AlignedBuf::zeroed(len),
            a_damp: AlignedBuf::zeroed(len),
            alpha_vel: [0.0; TOTAL_STRINGS],
            r_port_strike: [0.0; TOTAL_STRINGS],
            r_port_damp: [0.0; TOTAL_STRINGS],
            y_h: [0.0; TOTAL_STRINGS],
            v_h: [0.0; TOTAL_STRINGS],
            y_d: [0.0; TOTAL_STRINGS],
            v_d: [0.0; TOTAL_STRINGS],
            may_be_nonzero: [false; TOTAL_STRINGS],
            ports_synced: [true; TOTAL_STRINGS],
        };
        for (s_idx, entry) in map.strings().iter().enumerate() {
            populate_string_ports(&mut ports, s_idx, entry, sample_rate, voicing, specs);
        }
        ports
    }

    pub fn reset_resting(&mut self) {
        self.y_h.fill(0.0);
        self.v_h.fill(0.0);
        self.y_d.fill(0.0);
        self.v_d.fill(0.0);
        self.may_be_nonzero.fill(false);
        self.ports_synced.fill(true);
    }

    #[inline]
    pub fn clear_string_state(&mut self, s_idx: usize) {
        self.may_be_nonzero[s_idx] = false;
        self.y_h[s_idx] = 0.0;
        self.v_h[s_idx] = 0.0;
        self.y_d[s_idx] = 0.0;
        self.v_d[s_idx] = 0.0;
        self.ports_synced[s_idx] = true;
    }
}

#[inline]
pub(crate) fn strike_phi(key_idx: usize, partial_index: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    let beta = 0.137 - 0.042 * x;
    let s = sin_cos_f64((partial_index as f64) * PI * beta).0;
    s / (1.0 + (0.25 + 4.40 * x) * ((partial_index - 1) as f64))
}

#[inline]
pub(crate) fn damper_phi(partial_index: usize) -> f64 {
    let jf = partial_index as f64;
    let (x_d, w) = (0.125_f64, 0.082_f64);
    let pw = jf * PI * w;
    let sinc_w = sin_cos_f64(pw).0 / pw;
    let cos_2xd = sin_cos_f64(2.0 * jf * PI * x_d).1;
    let env_rms = sqrt_f64((0.5 - 0.5 * cos_2xd * sinc_w).max(0.0));
    env_rms * (1.0 + 0.44 * (1.0 - exp_f64(-0.48 * ((partial_index - 1) as f64))))
}

fn populate_string_ports(
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
    sample_rate: f64,
    voicing: &VoicingSpec,
    specs: &[StringPhysicalParams; NUM_KEYS],
) {
    let k = entry.key_idx as usize;
    let s_choir = entry.string_in_choir as usize;
    let spec = specs[k];
    let m_str = spec.linear_density_kg_m * spec.speaking_length_m;
    let ts = 1.0 / sample_rate;
    let nyquist = NYQUIST_CUTOFF_RATIO * sample_rate;
    let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
    let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);

    let x = (k as f64) / ((NUM_KEYS - 1) as f64);
    let m_h = 0.0112 - 0.0077 * x;
    let tau_c = (0.0032 * exp_f64(-0.35 * x - 2.75 * x * x * x)).max(0.00014);
    let z_choir = (entry.choir_size as f64) * 2.0 * f0_v * m_str;
    let i_mf = (1.55 * m_h * 1.8).min(1.45 * z_choir * tau_c * 1.8);

    let mut s_eff_sq = 0.0_f64;
    let (vb, vm) = (entry.vert_vecs.start * 4, entry.vert_modes);
    let (hb, hm) = (entry.horiz_vecs.start * 4, entry.horiz_modes);
    let pols = [
        (vb, vm, f0_v, 1.0, 1.0, f64::from(C_VERT)),
        (
            hb,
            hm,
            f0_h,
            HORIZ_STRIKE_SCALE,
            HORIZ_DAMP_SCALE,
            f64::from(C_HORIZ),
        ),
    ];
    for &(_, modes, f0, hs, _, c_b) in &pols {
        for j in 1..=modes {
            let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
            if f_j > 0.0 && f_j < nyquist {
                let sinc = 1.0 / sqrt_f64(1.0 + (2.0 * f_j * tau_c) * (2.0 * f_j * tau_c));
                let w = hs * strike_phi(k, j) * c_b * sinc;
                s_eff_sq += w * w;
            }
        }
    }

    let target_rms = 0.062 * (1.0 + 0.28 * x * x);
    let gamma_k = if s_eff_sq > 0.0 {
        target_rms * m_str / (i_mf * sqrt_f64(s_eff_sq))
    } else {
        1.0
    };
    let b_gain = gamma_k * (2.0 * ts / m_str);
    ports.alpha_vel[s_idx] = (m_str / (2.0 * ts * gamma_k * gamma_k)) as f32;

    let (mut sum_bh_sq, mut sum_bd_sq) = (0.0_f64, 0.0_f64);
    let (bh_s, ah_s) = (ports.b_strike.as_mut_slice(), ports.a_strike.as_mut_slice());
    let (bd_s, ad_s) = (ports.b_damp.as_mut_slice(), ports.a_damp.as_mut_slice());

    for &(base, modes, f0, hs, ds, _) in &pols {
        for j in 1..=modes {
            let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
            if f_j > 0.0 && f_j < nyquist {
                let omega = TAU * f_j;
                let ph = hs * strike_phi(k, j);
                let (bh, ah) = ((b_gain * ph) as f32, (ph / (omega * gamma_k)) as f32);
                bh_s[base + (j - 1)] = bh;
                ah_s[base + (j - 1)] = ah;
                sum_bh_sq += f64::from(bh) * f64::from(bh);
                if entry.has_damper {
                    let pd = ds * damper_phi(j);
                    let (bd, ad) = ((b_gain * pd) as f32, (pd / (omega * gamma_k)) as f32);
                    bd_s[base + (j - 1)] = bd;
                    ad_s[base + (j - 1)] = ad;
                    sum_bd_sq += f64::from(bd) * f64::from(bd);
                }
            }
        }
    }
    if entry.has_damper {
        let long_base = entry.long_vecs.start * 4;
        for m in 1..=entry.long_modes {
            let f_m = (m as f64) * spec.f_long_hz;
            if f_m > 0.0 && f_m < nyquist {
                let pd = LONG_DAMP_SCALE * damper_phi(m);
                let (bd, ad) = ((b_gain * pd) as f32, (pd / (TAU * f_m * gamma_k)) as f32);
                bd_s[long_base + (m - 1)] = bd;
                ad_s[long_base + (m - 1)] = ad;
                sum_bd_sq += f64::from(bd) * f64::from(bd);
            }
        }
    }

    let alpha64 = f64::from(ports.alpha_vel[s_idx]);
    ports.r_port_strike[s_idx] = (alpha64 * sum_bh_sq) as f32;
    ports.r_port_damp[s_idx] = (alpha64 * sum_bd_sq) as f32;
}

/// Recomputes cached `(y_h, v_h, y_d, v_d)` from current `soa.(u, v)` after fast-tile steps.
#[inline]
pub(crate) fn refresh_string_ports(
    soa: &crate::modal_soa::ModalSoa,
    ports: &mut ContactPorts,
    s_idx: usize,
    vr: Range<usize>,
    has_damper: bool,
) {
    let u = &soa.u.as_f32x4_slice()[vr.clone()];
    let v = &soa.v.as_f32x4_slice()[vr.clone()];
    let bh = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let ah = &ports.a_strike.as_f32x4_slice()[vr.clone()];
    let bd = &ports.b_damp.as_f32x4_slice()[vr.clone()];
    let ad = &ports.a_damp.as_f32x4_slice()[vr];

    let (mut ayh, mut avh, mut ayd, mut avd) =
        (F32x4::zero(), F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in 0..u.len() {
        ayh = v[i].mul_add(ah[i], ayh);
        avh = u[i].mul_add(bh[i], avh);
        if has_damper {
            ayd = v[i].mul_add(ad[i], ayd);
            avd = u[i].mul_add(bd[i], avd);
        }
    }
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = ayh.reduce_sum();
    ports.v_h[s_idx] = alpha * avh.reduce_sum();
    (ports.y_d[s_idx], ports.v_d[s_idx]) = if has_damper {
        (ayd.reduce_sum(), alpha * avd.reduce_sum())
    } else {
        (0.0, 0.0)
    };
    ports.ports_synced[s_idx] = true;
}

#[inline]
fn step_string_coupled<const HAS_DAMPER: bool>(
    soa: &mut crate::modal_soa::ModalSoa,
    ports: &mut ContactPorts,
    s_idx: usize,
    vr: Range<usize>,
    f_h: f32,
    f_d: f32,
) -> f32 {
    let u_s = &mut soa.u.as_f32x4_slice_mut()[vr.clone()];
    let v_s = &mut soa.v.as_f32x4_slice_mut()[vr.clone()];
    let d_s = &soa.d_coeff.as_f32x4_slice()[vr.clone()];
    let s_s = &soa.s_coeff.as_f32x4_slice()[vr.clone()];
    let c_s = &soa.c_bridge.as_f32x4_slice()[vr.clone()];
    let bh_s = &ports.b_strike.as_f32x4_slice()[vr.clone()];
    let ah_s = &ports.a_strike.as_f32x4_slice()[vr.clone()];
    let bd_s = &ports.b_damp.as_f32x4_slice()[vr.clone()];
    let ad_s = &ports.a_damp.as_f32x4_slice()[vr];

    let (fh_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(f_d));
    let (mut acc_b, mut acc_yh, mut acc_vh) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    let (mut acc_yd, mut acc_vd) = (F32x4::zero(), F32x4::zero());

    for i in 0..u_s.len() {
        let u_inj = if HAS_DAMPER {
            fd_v.mul_add(bd_s[i], fh_v.mul_add(bh_s[i], u_s[i]))
        } else {
            fh_v.mul_add(bh_s[i], u_s[i])
        };
        let (un, vn) = zoh_rotate(u_inj, v_s[i], d_s[i], s_s[i]);
        u_s[i] = un;
        v_s[i] = vn;
        acc_b = vn.mul_add(c_s[i], acc_b);
        acc_yh = vn.mul_add(ah_s[i], acc_yh);
        acc_vh = un.mul_add(bh_s[i], acc_vh);
        if HAS_DAMPER {
            acc_yd = vn.mul_add(ad_s[i], acc_yd);
            acc_vd = un.mul_add(bd_s[i], acc_vd);
        }
    }

    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = acc_yh.reduce_sum();
    ports.v_h[s_idx] = alpha * acc_vh.reduce_sum();
    if HAS_DAMPER {
        ports.y_d[s_idx] = acc_yd.reduce_sum();
        ports.v_d[s_idx] = alpha * acc_vd.reduce_sum();
    }
    if f_h != 0.0 || f_d != 0.0 {
        ports.may_be_nonzero[s_idx] = true;
    }
    ports.ports_synced[s_idx] = true;
    acc_b.reduce_sum()
}

#[inline]
fn eval_hammer_choir(
    driver: &mut impl KeyActionDriver,
    ports: &ContactPorts,
    key_idx: usize,
    sr: Range<usize>,
    choir_size: usize,
) -> [f32; 3] {
    let w = driver.strike_weights(key_idx);
    let (mut w_eff, mut w_sum) = ([0.0_f32; 3], 0.0_f32);
    for c in 0..choir_size {
        let wc = if w[c].is_finite() { w[c].max(0.0) } else { 0.0 };
        w_eff[c] = wc;
        w_sum += wc;
    }
    if w_sum <= 1.0e-6 {
        return [0.0; 3];
    }
    let inv_w = 1.0 / w_sum;
    let (mut y_eff, mut v_eff, mut r_eff) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (c, s_idx) in sr.enumerate() {
        let wc = w_eff[c];
        y_eff += wc * ports.y_h[s_idx];
        v_eff += wc * ports.v_h[s_idx];
        r_eff += wc * wc * ports.r_port_strike[s_idx];
    }
    let f_tot =
        driver.step_hammer_sample(key_idx, y_eff * inv_w, v_eff * inv_w, r_eff * inv_w * inv_w);
    let mut f_out = [0.0_f32; 3];
    if f_tot.is_finite() {
        for (c, out) in f_out[..choir_size].iter_mut().enumerate() {
            *out = f_tot * w_eff[c] * inv_w;
        }
    }
    f_out
}

struct ActiveKeyCtx<'a> {
    k: usize,
    key: &'a KeyMapEntry,
    hammer_active: bool,
    damper_engaged: bool,
    y_clear_tile: &'a [f32],
}

fn step_active_key_tile(
    soa: &mut crate::modal_soa::ModalSoa,
    ports: &mut ContactPorts,
    strings: &[StringMapEntry],
    driver: &mut impl KeyActionDriver,
    ctx: &ActiveKeyCtx<'_>,
    out: (&mut [f32], &mut [f32]),
) {
    let (k, key, hammer_active, damper_engaged) =
        (ctx.k, ctx.key, ctx.hammer_active, ctx.damper_engaged);
    let (l_tile, r_tile) = out;
    let sr = key.string_range.clone();
    for s_idx in sr.clone() {
        if ports.may_be_nonzero[s_idx] && !ports.ports_synced[s_idx] {
            refresh_string_ports(soa, ports, s_idx, strings[s_idx].all_vecs(), key.has_damper);
        }
    }
    let choir_size = usize::from(key.choir_size);
    for (t, &yc) in ctx.y_clear_tile.iter().enumerate() {
        let fh = if hammer_active {
            eval_hammer_choir(driver, ports, k, sr.clone(), choir_size)
        } else {
            [0.0; 3]
        };
        for (c, s_idx) in sr.clone().enumerate() {
            let mut fd = 0.0_f32;
            if damper_engaged && ports.may_be_nonzero[s_idx] {
                let (yd, vd, rd) = (ports.y_d[s_idx], ports.v_d[s_idx], ports.r_port_damp[s_idx]);
                let raw = driver.step_damper_sample(k, s_idx, yd, vd, rd, yc);
                if raw.is_finite() {
                    fd = raw;
                }
            }
            let e = &strings[s_idx];
            let term = if key.has_damper {
                step_string_coupled::<true>(soa, ports, s_idx, e.all_vecs(), fh[c], fd)
            } else {
                step_string_coupled::<false>(soa, ports, s_idx, e.all_vecs(), fh[c], 0.0)
            };
            l_tile[t] += term * e.pan_l;
            r_tile[t] += term * e.pan_r;
        }
    }
}

/// Steps all 88 keys / 230 strings unconditionally across `[out_l, out_r]` with closed-loop
/// hammer and damper coupling on active tiles and `TILE = 8` fast SIMD stepping elsewhere.
pub(crate) fn render_subblock_coupled_core(
    soa: &mut crate::modal_soa::ModalSoa,
    ports: &mut ContactPorts,
    map: &ChoirMap,
    driver: &mut impl KeyActionDriver,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    out_l[..total].fill(0.0);
    out_r[..total].fill(0.0);
    let strings = map.strings();
    let mut base = 0usize;
    while base < total {
        let t_len = (total - base).min(TILE);
        let l_tile = &mut out_l[base..base + t_len];
        let r_tile = &mut out_r[base..base + t_len];
        for (k, key) in map.keys().iter().enumerate() {
            let (mut y_clear, mut min_clear) = ([0.0_f32; TILE], f32::MAX);
            if key.has_damper {
                for yc in &mut y_clear[..t_len] {
                    let c = driver.step_key_clearance(k);
                    *yc = c;
                    min_clear = min_clear.min(c);
                }
            }
            let sr = key.string_range.clone();
            let hammer_active = driver.is_hammer_active(k);
            let damper_engaged = key.has_damper && min_clear < DAMPER_CLEAR_THRESHOLD_M;
            let any_nz = ports.may_be_nonzero[sr.clone()].iter().any(|&b| b);
            if !hammer_active && (!damper_engaged || !any_nz) {
                for s_idx in sr {
                    let e = &strings[s_idx];
                    step_string_tile(soa, e.all_vecs(), e.pan_l, e.pan_r, l_tile, r_tile);
                    if ports.may_be_nonzero[s_idx] {
                        ports.ports_synced[s_idx] = false;
                    }
                }
            } else {
                let ctx = ActiveKeyCtx {
                    k,
                    key,
                    hammer_active,
                    damper_engaged,
                    y_clear_tile: &y_clear[..t_len],
                };
                step_active_key_tile(
                    soa,
                    ports,
                    strings,
                    driver,
                    &ctx,
                    (&mut *l_tile, &mut *r_tile),
                );
            }
        }
        base += t_len;
    }
    for (s_idx, entry) in strings.iter().enumerate() {
        if ports.may_be_nonzero[s_idx] && !flush_subnormals_range(soa, entry.all_vecs()) {
            ports.clear_string_state(s_idx);
        }
    }
}

#[cfg(test)]
#[path = "contact_tests.rs"]
mod tests;
