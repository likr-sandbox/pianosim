//! Stage 1 <-> Stage 2 closed-loop SI contact stepping (`Newtons`, `meters`, `m/s`)
//! with independent vertical/horizontal damper slots and M2-speed `TILE = 8` SIMD kernels.

use core::ops::Range;

use piano_simd::F32x4;

use crate::choir_map::{ChoirMap, KeyMapEntry, StringMapEntry};
use crate::contact_ports::{ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::modal_soa::ModalSoa;
use crate::zoh_kernel::{
    accum_tile_stereo, flush_subnormals_range, step_string_modes, step_string_tile, zoh_rotate,
    TILE,
};

/// Stage 1 key-action driver driving nonlinear hammer strike and orthogonal V/H
/// unilateral damper ports per sample with exact SI duality (`Newtons`, `meters`, `m/s`).
pub trait KeyActionDriver {
    /// Returns `true` while hammer `key_idx` is still in flight or rebounding toward rest.
    fn is_hammer_active(&self, key_idx: usize) -> bool;
    /// Instantaneous geometric clearance `y_clear` (m) of damper pad `key_idx`.
    fn damper_clearance_m(&self, key_idx: usize) -> f32;
    /// Returns per-choir-string hammer strike weights `[w0, w1, w2]` (Una Corda support).
    fn strike_weights(&self, key_idx: usize) -> [f32; 3];
    /// Advances damper rail smoother by 1 sample for `key_idx` and returns `y_clear` (m).
    fn step_key_clearance(&mut self, key_idx: usize) -> f32;
    /// Advances damper rail smoother by `t_len` samples (<= 8), fills `out_clears[..t_len]`,
    /// and returns the minimum clearance across the tile.
    fn step_key_clearance_tile(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; TILE],
    ) -> f32;
    /// Evaluates hammer force `F_strike` (N, `+` up) from `(y_str_m, v_str_mps, r_port_strike)`.
    fn step_hammer_sample(
        &mut self,
        key_idx: usize,
        y_str_m: f32,
        v_str_mps: f32,
        r_port_strike: f32,
    ) -> f32;
    /// Evaluates unilateral damper force `F_damp` (N, on string) for damper `slot_idx`
    /// (`s_idx * 2` = vertical, `s_idx * 2 + 1` = horizontal) from port state & `y_clear_m`.
    fn step_damper_sample(
        &mut self,
        key_idx: usize,
        slot_idx: usize,
        y_damp_m: f32,
        v_damp_mps: f32,
        r_port_damp: f32,
        y_clear_m: f32,
    ) -> f32;
}

#[inline]
fn accum_pol_ports(
    soa: &ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    has_damper: bool,
) -> (f32, f32, f32, f32) {
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
    (
        ayh.reduce_sum(),
        avh.reduce_sum(),
        ayd.reduce_sum(),
        avd.reduce_sum(),
    )
}

/// Recomputes cached `(y_h, v_h, y_dv, v_dv, y_dh, v_dh)` from `soa.(u, v)` after fast-tile steps.
#[inline]
pub(crate) fn refresh_string_ports(
    soa: &ModalSoa,
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
) {
    let (yh_v, vh_v, ydv, vdv) =
        accum_pol_ports(soa, ports, entry.vert_vecs.clone(), entry.has_damper);
    let (yh_h, vh_h, ydh, vdh) =
        accum_pol_ports(soa, ports, entry.horiz_vecs.clone(), entry.has_damper);
    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = yh_v + yh_h;
    ports.v_h[s_idx] = alpha * (vh_v + vh_h);
    if entry.has_damper {
        ports.y_dv[s_idx] = ydv;
        ports.v_dv[s_idx] = alpha * vdv;
        ports.y_dh[s_idx] = ydh;
        ports.v_dh[s_idx] = alpha * vdh;
    } else {
        ports.y_dv[s_idx] = 0.0;
        ports.v_dv[s_idx] = 0.0;
        ports.y_dh[s_idx] = 0.0;
        ports.v_dh[s_idx] = 0.0;
    }
    ports.ports_synced[s_idx] = true;
}

#[inline]
fn step_transverse_pol<const HAS_DAMPER: bool>(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    f_h: f32,
    f_d: f32,
) -> (f32, f32, f32, f32, f32) {
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
    (
        acc_b.reduce_sum(),
        acc_yh.reduce_sum(),
        acc_vh.reduce_sum(),
        acc_yd.reduce_sum(),
        acc_vd.reduce_sum(),
    )
}

#[inline]
fn step_string_coupled<const HAS_DAMPER: bool>(
    soa: &mut ModalSoa,
    ports: &mut ContactPorts,
    s_idx: usize,
    entry: &StringMapEntry,
    f_h: f32,
    fd_v: f32,
    fd_h: f32,
) -> f32 {
    let (tv, yhv, vhv, ydv, vdv) =
        step_transverse_pol::<HAS_DAMPER>(soa, ports, entry.vert_vecs.clone(), f_h, fd_v);
    let (th, yhh, vhh, ydh, vdh) =
        step_transverse_pol::<HAS_DAMPER>(soa, ports, entry.horiz_vecs.clone(), f_h, fd_h);
    let rl = entry.long_vecs.clone();
    let tl = if rl.is_empty() {
        0.0
    } else {
        step_string_modes(
            &mut soa.u.as_f32x4_slice_mut()[rl.clone()],
            &mut soa.v.as_f32x4_slice_mut()[rl.clone()],
            &soa.d_coeff.as_f32x4_slice()[rl.clone()],
            &soa.s_coeff.as_f32x4_slice()[rl.clone()],
            &soa.c_bridge.as_f32x4_slice()[rl],
        )
    };

    let alpha = ports.alpha_vel[s_idx];
    ports.y_h[s_idx] = yhv + yhh;
    ports.v_h[s_idx] = alpha * (vhv + vhh);
    if HAS_DAMPER {
        ports.y_dv[s_idx] = ydv;
        ports.v_dv[s_idx] = alpha * vdv;
        ports.y_dh[s_idx] = ydh;
        ports.v_dh[s_idx] = alpha * vdh;
    }
    if f_h != 0.0 || fd_v != 0.0 || fd_h != 0.0 {
        ports.may_be_nonzero[s_idx] = true;
    }
    ports.ports_synced[s_idx] = true;
    tv + th + tl
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
    soa: &mut ModalSoa,
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
            refresh_string_ports(soa, ports, s_idx, &strings[s_idx]);
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
            let (mut fd_v, mut fd_h) = (0.0_f32, 0.0_f32);
            if damper_engaged && ports.may_be_nonzero[s_idx] {
                let rv = driver.step_damper_sample(
                    k,
                    s_idx * 2,
                    ports.y_dv[s_idx],
                    ports.v_dv[s_idx],
                    ports.r_port_damp_v[s_idx],
                    yc,
                );
                if rv.is_finite() {
                    fd_v = rv;
                }
                let rh = driver.step_damper_sample(
                    k,
                    s_idx * 2 + 1,
                    ports.y_dh[s_idx],
                    ports.v_dh[s_idx],
                    ports.r_port_damp_h[s_idx],
                    yc,
                );
                if rh.is_finite() {
                    fd_h = rh;
                }
            }
            let e = &strings[s_idx];
            let term = if key.has_damper {
                step_string_coupled::<true>(soa, ports, s_idx, e, fh[c], fd_v, fd_h)
            } else {
                step_string_coupled::<false>(soa, ports, s_idx, e, fh[c], 0.0, 0.0)
            };
            l_tile[t] += term * e.pan_l;
            r_tile[t] += term * e.pan_r;
        }
    }
}

/// Steps all 88 keys / 230 strings unconditionally across `[out_l, out_r]` with closed-loop
/// hammer and orthogonal V/H damper coupling on active tiles and `TILE = 8` fast SIMD stepping elsewhere.
pub(crate) fn render_subblock_coupled_core(
    soa: &mut ModalSoa,
    ports: &mut ContactPorts,
    map: &ChoirMap,
    driver: &mut impl KeyActionDriver,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let total = out_l.len().min(out_r.len());
    let strings = map.strings();
    let mut base = 0usize;
    while base < total {
        let t_len = (total - base).min(TILE);
        let (mut tile_l, mut tile_r) = ([0.0_f32; TILE], [0.0_f32; TILE]);
        for (k, key) in map.keys().iter().enumerate() {
            let mut y_clear = [0.0_f32; TILE];
            let min_clear = if key.has_damper {
                driver.step_key_clearance_tile(k, t_len, &mut y_clear)
            } else {
                f32::MAX
            };
            let sr = key.string_range.clone();
            let hammer_active = driver.is_hammer_active(k);
            let damper_engaged = key.has_damper && min_clear < DAMPER_CLEAR_THRESHOLD_M;
            let any_nz = ports.may_be_nonzero[sr.clone()].iter().any(|&b| b);
            if !hammer_active && (!damper_engaged || !any_nz) {
                for s_idx in sr {
                    let e = &strings[s_idx];
                    let acc = step_string_tile(soa, e.all_vecs(), t_len);
                    accum_tile_stereo(&acc, e.pan_l, e.pan_r, t_len, &mut tile_l, &mut tile_r);
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
                    (&mut tile_l[..t_len], &mut tile_r[..t_len]),
                );
            }
        }
        out_l[base..base + t_len].copy_from_slice(&tile_l[..t_len]);
        out_r[base..base + t_len].copy_from_slice(&tile_r[..t_len]);
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
