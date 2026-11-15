//! Stage 1 <-> Stage 2 closed-loop SI contact stepping (`Newtons`, `meters`, `m/s`)
//! with independent vertical/horizontal damper slots and M2-speed `TILE = 8` SIMD kernels.

use core::ops::Range;

use piano_simd::F32x4;

use crate::choir_map::{ChoirMap, KeyMapEntry, StringMapEntry};
use crate::contact_ports::{refresh_string_ports, ContactPorts, DAMPER_CLEAR_THRESHOLD_M};
use crate::longitudinal::LongitudinalBank;
use crate::modal_soa::ModalSoa;
use crate::sav_drift::SavBank;
use crate::sav_pair::SavPairQueue;
use crate::zoh_kernel::{
    flush_subnormals_range, step_string_modes, step_string_subblock, step_string_tile, zoh_rotate,
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
    /// Returns `true` if `key_idx` has a sample-accurate NoteOn/NoteOff inside `[chunk_base, chunk_base + sub)`.
    #[inline]
    fn has_mid_block_event(&self, _key_idx: usize, _chunk_base: usize, _sub: usize) -> bool {
        false
    }
    /// Applies queued NoteOn/NoteOff events at `sample_idx`, returning a bitmask of NoteOn keys.
    #[inline]
    fn apply_sample_events_at(&mut self, _sample_idx: usize) -> u128 {
        0
    }
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
pub(crate) fn step_transverse_pol<const HAS_DAMPER: bool>(
    soa: &mut ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
    f_h: f32,
    f_d: f32,
) -> (f32, f32, f32) {
    let (u_s, v_s) = (soa.u.as_f32x4_slice_mut(), soa.v.as_f32x4_slice_mut());
    let (d_s, s_s, c_s) = (
        soa.d_coeff.as_f32x4_slice(),
        soa.s_coeff.as_f32x4_slice(),
        soa.c_bridge.as_f32x4_slice(),
    );
    let (bh_s, ah_s) = (
        ports.b_strike.as_f32x4_slice(),
        ports.a_strike.as_f32x4_slice(),
    );
    let (bd_s, ad_s) = (ports.b_damp.as_f32x4_slice(), ports.a_damp.as_f32x4_slice());
    let (fh_v, fd_v) = (F32x4::splat(f_h), F32x4::splat(f_d));
    let (mut acc_b, mut acc_yh, mut acc_yd) = (F32x4::zero(), F32x4::zero(), F32x4::zero());
    for i in vr {
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
        if HAS_DAMPER {
            acc_yd = vn.mul_add(ad_s[i], acc_yd);
        }
    }
    (acc_b.reduce_sum(), acc_yh.reduce_sum(), acc_yd.reduce_sum())
}

#[inline]
fn accum_pol_vel<const HAS_DAMPER: bool>(
    soa: &ModalSoa,
    ports: &ContactPorts,
    vr: Range<usize>,
) -> (f32, f32) {
    let (u_s, bh_s, bd_s) = (
        soa.u.as_f32x4_slice(),
        ports.b_strike.as_f32x4_slice(),
        ports.b_damp.as_f32x4_slice(),
    );
    let (mut acc_vh, mut acc_vd) = (F32x4::zero(), F32x4::zero());
    for i in vr {
        acc_vh = u_s[i].mul_add(bh_s[i], acc_vh);
        if HAS_DAMPER {
            acc_vd = u_s[i].mul_add(bd_s[i], acc_vd);
        }
    }
    (acc_vh.reduce_sum(), acc_vd.reduce_sum())
}

struct CoupledRun<'a> {
    soa: &'a mut ModalSoa,
    ports: &'a mut ContactPorts,
    long_bank: &'a LongitudinalBank,
    sav: &'a mut SavBank,
    keys: &'a [KeyMapEntry],
    strings: &'a [StringMapEntry],
    long_on: bool,
    sav_on: bool,
}

#[inline]
fn step_string_coupled<const HAS_DAMPER: bool>(
    r: &mut CoupledRun<'_>,
    s_idx: usize,
    f_h: f32,
    fd_v: f32,
    fd_h: f32,
) -> f32 {
    let entry = &r.strings[s_idx];
    let (tv, yhv, ydv) =
        step_transverse_pol::<HAS_DAMPER>(r.soa, r.ports, entry.vert_vecs.clone(), f_h, fd_v);
    let (th, yhh, ydh) =
        step_transverse_pol::<HAS_DAMPER>(r.soa, r.ports, entry.horiz_vecs.clone(), f_h, fd_h);
    let rl = entry.long_vecs.clone();
    let tl = if rl.is_empty() {
        0.0
    } else {
        if r.long_on && f_h > 0.0 {
            let lv = &mut r.soa.u.as_f32x4_slice_mut()[rl.clone()];
            r.long_bank.inject_precursor(s_idx, rl.len(), f_h, lv);
        }
        step_string_modes(
            &mut r.soa.u.as_f32x4_slice_mut()[rl.clone()],
            &mut r.soa.v.as_f32x4_slice_mut()[rl.clone()],
            &r.soa.d_coeff.as_f32x4_slice()[rl.clone()],
            &r.soa.s_coeff.as_f32x4_slice()[rl.clone()],
            &r.soa.c_bridge.as_f32x4_slice()[rl],
        )
    };
    if r.sav_on
        && r.sav.string_n_vh(s_idx) > 0
        && (r.ports.may_be_nonzero[s_idx] || f_h != 0.0 || fd_v != 0.0 || fd_h != 0.0)
    {
        r.sav.step_string_sav(s_idx, entry, r.soa, r.long_on);
    }
    let (vhv, vdv) = accum_pol_vel::<HAS_DAMPER>(r.soa, r.ports, entry.vert_vecs.clone());
    let (vhh, vdh) = accum_pol_vel::<HAS_DAMPER>(r.soa, r.ports, entry.horiz_vecs.clone());
    let alpha = r.ports.alpha_vel[s_idx];
    let (p, s) = (&mut *r.ports, s_idx);
    (p.y_h[s], p.v_h[s]) = (yhv + yhh, alpha * (vhv + vhh));
    if HAS_DAMPER {
        (p.y_dv[s], p.v_dv[s], p.y_dh[s], p.v_dh[s]) = (ydv, alpha * vdv, ydh, alpha * vdh);
    }
    let excited = f_h != 0.0 || fd_v != 0.0 || fd_h != 0.0;
    p.may_be_nonzero[s] |= excited;
    p.tail_nonzero[s] |= excited;
    p.ports_synced[s] = true;
    tv + th + if r.long_on { tl } else { 0.0 }
}

#[inline]
pub(crate) fn eval_hammer_choir(
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
    let inv_choir = 1.0 / (choir_size as f32);
    let mut f_out = [0.0_f32; 3];
    if f_tot.is_finite() {
        for (c, out) in f_out[..choir_size].iter_mut().enumerate() {
            *out = f_tot * w_eff[c] * inv_choir;
        }
    }
    f_out
}

#[inline]
pub(crate) fn sample_damper(
    drv: &mut impl KeyActionDriver,
    p: &ContactPorts,
    k: usize,
    s: usize,
    yc: f32,
) -> (f32, f32) {
    let rv = drv.step_damper_sample(k, s * 2, p.y_dv[s], p.v_dv[s], p.r_port_damp_v[s], yc);
    let rh = drv.step_damper_sample(k, s * 2 + 1, p.y_dh[s], p.v_dh[s], p.r_port_damp_h[s], yc);
    (
        if rv.is_finite() { rv } else { 0.0 },
        if rh.is_finite() { rh } else { 0.0 },
    )
}

#[inline(never)]
fn step_active_key_tile(
    r: &mut CoupledRun<'_>,
    driver: &mut impl KeyActionDriver,
    k: usize,
    key: &KeyMapEntry,
    flags: (bool, bool),
    y_clear_tile: &[f32],
    out: (&mut [f32], &mut [f32]),
) {
    let (hammer_active, damper_engaged) = flags;
    let (l_tile, r_tile) = out;
    let sr = key.string_range.clone();
    for s_idx in sr.clone() {
        if r.ports.may_be_nonzero[s_idx] && !r.ports.ports_synced[s_idx] {
            refresh_string_ports(r.soa, r.ports, s_idx, &r.strings[s_idx]);
        }
    }
    let choir_size = usize::from(key.choir_size);
    let step_fn = if key.has_damper {
        step_string_coupled::<true>
    } else {
        step_string_coupled::<false>
    };
    for (t, &yc) in y_clear_tile.iter().enumerate() {
        let fh = if hammer_active {
            eval_hammer_choir(driver, r.ports, k, sr.clone(), choir_size)
        } else {
            [0.0; 3]
        };
        for (c, s_idx) in sr.clone().enumerate() {
            let (fd_v, fd_h) = if damper_engaged && r.ports.may_be_nonzero[s_idx] {
                sample_damper(driver, r.ports, k, s_idx, yc)
            } else {
                (0.0, 0.0)
            };
            let term = step_fn(r, s_idx, fh[c], fd_v, fd_h);
            let e = &r.strings[s_idx];
            l_tile[t] += term * e.pan_l;
            r_tile[t] += term * e.pan_r;
        }
    }
}

#[inline]
pub(crate) fn step_clearance_subblock(
    driver: &mut impl KeyActionDriver,
    k: usize,
    sub: usize,
    mut dst: Option<&mut [f32]>,
) -> f32 {
    let (mut min_c, mut b, mut tmp) = (f32::MAX, 0usize, [0.0_f32; TILE]);
    while b < sub {
        let t_len = (sub - b).min(TILE);
        min_c = min_c.min(driver.step_key_clearance_tile(k, t_len, &mut tmp));
        if let Some(ref mut out) = dst {
            out[b..b + t_len].copy_from_slice(&tmp[..t_len]);
        }
        b += t_len;
    }
    min_c
}

#[inline]
fn step_uncoupled_key(
    q: &mut SavPairQueue,
    run: &mut CoupledRun<'_>,
    sr: Range<usize>,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let n = out_l.len();
    for s_idx in sr {
        if run.ports.may_be_nonzero[s_idx] {
            q.push_ringing(run.soa, run.sav, run.strings, s_idx, out_l, out_r);
            run.ports.ports_synced[s_idx] = false;
        } else {
            let e = &run.strings[s_idx];
            if n <= TILE {
                let acc = step_string_tile(run.soa, e.all_vecs(), n);
                for ((a, l), r) in acc[..n].iter().zip(out_l.iter_mut()).zip(out_r.iter_mut()) {
                    let f = a.reduce_sum();
                    *l += f * e.pan_l;
                    *r += f * e.pan_r;
                }
            } else {
                step_string_subblock(run.soa, e.all_vecs(), e.pan_l, e.pan_r, out_l, out_r);
            }
        }
    }
}

#[inline]
fn step_coupled_tiles(
    run: &mut CoupledRun<'_>,
    driver: &mut impl KeyActionDriver,
    act: &[u8],
    sub_l: &mut [f32],
    sub_r: &mut [f32],
) {
    let (sub, mut base) = (sub_l.len(), 0usize);
    while base < sub {
        let t_len = (sub - base).min(TILE);
        let (mut tile_l, mut tile_r) = ([0.0_f32; TILE], [0.0_f32; TILE]);
        let (tl, tr) = (&mut tile_l[..t_len], &mut tile_r[..t_len]);
        let mut sav_tile = SavPairQueue::new(run.long_on, run.sav_on);
        for &k_u8 in act {
            let (k, key) = (usize::from(k_u8), &run.keys[usize::from(k_u8)]);
            let sr = key.string_range.clone();
            let hammer = driver.is_hammer_active(k);
            let ys = &run.ports.y_clear_scratch[k][base..base + t_len];
            let min_c = if key.has_damper {
                ys.iter().copied().fold(f32::MAX, f32::min)
            } else {
                f32::MAX
            };
            let damp = key.has_damper && min_c < DAMPER_CLEAR_THRESHOLD_M;
            let any_nz = run.ports.may_be_nonzero[sr.clone()].iter().any(|&b| b);
            if !hammer && (!damp || !any_nz) {
                step_uncoupled_key(&mut sav_tile, run, sr, tl, tr);
            } else {
                let mut y_tile = [0.0_f32; TILE];
                y_tile[..t_len].copy_from_slice(ys);
                step_active_key_tile(
                    run,
                    driver,
                    k,
                    key,
                    (hammer, damp),
                    &y_tile[..t_len],
                    (tl, tr),
                );
            }
        }
        sav_tile.flush(run.soa, run.sav, run.strings, tl, tr);
        for t in 0..t_len {
            sub_l[base + t] += tl[t];
            sub_r[base + t] += tr[t];
        }
        base += t_len;
    }
}

/// Steps all 88 keys / 230 strings unconditionally across `[out_l, out_r]` with closed-loop
/// hammer and orthogonal V/H damper coupling on active tiles and subblock SAV/ZOH elsewhere.
#[allow(clippy::too_many_arguments)]
pub(crate) fn render_subblock_coupled_core(
    soa: &mut ModalSoa,
    ports: &mut ContactPorts,
    long_bank: &LongitudinalBank,
    sav: &mut SavBank,
    map: &ChoirMap,
    driver: &mut impl KeyActionDriver,
    long_on: bool,
    sav_on: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    const SUB: usize = 128;
    let total = out_l.len().min(out_r.len());
    out_l[..total].fill(0.0);
    out_r[..total].fill(0.0);
    let (keys, strings) = (map.keys(), map.strings());
    let mut run = CoupledRun {
        soa,
        ports,
        long_bank,
        sav,
        keys,
        strings,
        long_on,
        sav_on,
    };
    let mut chunk_base = 0usize;
    while chunk_base < total {
        let sub = (total - chunk_base).min(SUB);
        let (sub_l, sub_r) = (
            &mut out_l[chunk_base..chunk_base + sub],
            &mut out_r[chunk_base..chunk_base + sub],
        );
        let (mut coupled_keys, mut flush_keys) =
            ([0_u8; piano_spec::NUM_KEYS], [0_u8; piano_spec::NUM_KEYS]);
        let (mut n_coupled, mut n_flush) = (0usize, 0usize);
        let mut sav_pairs = SavPairQueue::new(run.long_on, run.sav_on);
        for (k, key) in keys.iter().enumerate() {
            let sr = key.string_range.clone();
            let hammer = driver.is_hammer_active(k);
            let any_nz = run.ports.may_be_nonzero[sr.clone()].iter().any(|&b| b);
            if !hammer && !any_nz {
                if key.has_damper {
                    let _ = step_clearance_subblock(driver, k, sub, None);
                }
                for s_idx in sr {
                    let e = &strings[s_idx];
                    step_string_subblock(run.soa, e.all_vecs(), e.pan_l, e.pan_r, sub_l, sub_r);
                }
                continue;
            }
            flush_keys[n_flush] = k as u8;
            n_flush += 1;
            let dst = &mut run.ports.y_clear_scratch[k][..sub];
            let min_c = if key.has_damper {
                step_clearance_subblock(driver, k, sub, Some(dst))
            } else {
                f32::MAX
            };
            if !hammer && min_c >= DAMPER_CLEAR_THRESHOLD_M {
                step_uncoupled_key(&mut sav_pairs, &mut run, sr, sub_l, sub_r);
            } else {
                coupled_keys[n_coupled] = k as u8;
                n_coupled += 1;
            }
        }
        sav_pairs.flush(run.soa, run.sav, strings, sub_l, sub_r);
        if n_coupled > 0 {
            step_coupled_tiles(&mut run, driver, &coupled_keys[..n_coupled], sub_l, sub_r);
        }
        for &k_u8 in &flush_keys[..n_flush] {
            for s_idx in keys[usize::from(k_u8)].string_range.clone() {
                if run.ports.may_be_nonzero[s_idx]
                    && !flush_subnormals_range(run.soa, strings[s_idx].all_vecs())
                {
                    run.ports.clear_string_state(s_idx);
                    run.sav.reset_string(s_idx);
                }
            }
        }
        chunk_base += sub;
    }
}

#[cfg(test)]
#[path = "contact_tests.rs"]
mod tests;
