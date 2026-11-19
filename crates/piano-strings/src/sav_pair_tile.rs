//! Two-string SAV tile steppers (`step_no_phantom_pair_tile` / `step_phantom_pair_tile`).

use crate::choir_map::StringMapEntry;
use crate::modal_soa::ModalSoa;
use crate::phantom::BASE_PHANTOM_LONG_VECS;
use crate::phantom_sparse::SPARSE_PHANTOM_LONG_VECS;
use crate::sav_drift::{load_nl_state_n, load_vh_state, store_nl_state_n, SavBank};
use crate::sav_pair::{eval_no_phantom_pair, eval_phantom_pair};
use crate::sav_tile::{step_2_slices, step_nl_slices, step_tails, store_vh_state};

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_no_phantom_pair_tile(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    longitudinal_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    for e in [e0, e1] {
        let (vb, hb) = (e.vert_vecs.start, e.horiz_vecs.start);
        let (r1, r2) = if longitudinal_enabled {
            (hb + 2..e.long_vecs.end, 0..0)
        } else {
            (hb + 2..e.horiz_vecs.end, e.long_vecs.clone())
        };
        step_tails(soa, vb + 2..e.vert_vecs.end, r1, r2, e, out_l, out_r);
    }
    let (vb0, hb0, vb1, hb1) = (
        e0.vert_vecs.start,
        e0.horiz_vecs.start,
        e1.vert_vecs.start,
        e1.horiz_vecs.start,
    );
    let (mut uv0, mut vv0) = load_vh_state(soa, vb0, 2);
    let (mut uh0, mut vh0) = load_vh_state(soa, hb0, 2);
    let (mut uv1, mut vv1) = load_vh_state(soa, vb1, 2);
    let (mut uh1, mut vh1) = load_vh_state(soa, hb1, 2);
    let (sp0, sp1, ts) = (sav.spec(s0), sav.spec(s1), sav.ts());
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    let steps = out_l.len().min(out_r.len());
    {
        let (d_s, s_s, c_s) = (
            soa.d_coeff.as_f32x4_slice(),
            soa.s_coeff.as_f32x4_slice(),
            soa.c_bridge.as_f32x4_slice(),
        );
        let min_len = d_s.len().min(s_s.len()).min(c_s.len());
        assert!(
            vb0 + 2 <= min_len && hb0 + 2 <= min_len && vb1 + 2 <= min_len && hb1 + 2 <= min_len
        );
        for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
            let av0 = step_2_slices(&mut uv0, &mut vv0, vb0, d_s, s_s, c_s);
            let ah0 = step_2_slices(&mut uh0, &mut vh0, hb0, d_s, s_s, c_s);
            let av1 = step_2_slices(&mut uv1, &mut vv1, vb1, d_s, s_s, c_s);
            let ah1 = step_2_slices(&mut uh1, &mut vh1, hb1, d_s, s_s, c_s);
            eval_no_phantom_pair(
                sp0, sp1, ts, &mut r0, &mut r1, &mut uv0, &vv0, &mut uh0, &vh0, &mut uv1, &vv1,
                &mut uh1, &vh1,
            );
            let f0 = (av0 + ah0).reduce_sum();
            let f1 = (av1 + ah1).reduce_sum();
            *lo += f0 * e0.pan_l + f1 * e1.pan_l;
            *ro += f0 * e0.pan_r + f1 * e1.pan_r;
        }
    }
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_state(soa, vb0, 2, &uv0, &vv0);
    store_vh_state(soa, hb0, 2, &uh0, &vh0);
    store_vh_state(soa, vb1, 2, &uv1, &vv1);
    store_vh_state(soa, hb1, 2, &uh1, &vh1);
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn step_phantom_pair_tile<const NL_MAX: usize>(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    nl: usize,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    let n = if NL_MAX == 1 { 1 } else { nl.min(NL_MAX) };
    for e in [e0, e1] {
        let (vb, hb, lb) = (e.vert_vecs.start, e.horiz_vecs.start, e.long_vecs.start);
        step_tails(
            soa,
            vb + 2..e.vert_vecs.end,
            hb + 2..e.horiz_vecs.end,
            lb + n..e.long_vecs.end,
            e,
            out_l,
            out_r,
        );
    }
    let (vb0, hb0, lb0) = (e0.vert_vecs.start, e0.horiz_vecs.start, e0.long_vecs.start);
    let (vb1, hb1, lb1) = (e1.vert_vecs.start, e1.horiz_vecs.start, e1.long_vecs.start);
    let (mut uv0, mut vv0) = load_vh_state(soa, vb0, 2);
    let (mut uh0, mut vh0) = load_vh_state(soa, hb0, 2);
    let (mut ul0, mut vl0) = load_nl_state_n::<NL_MAX>(soa, lb0, n);
    let (mut uv1, mut vv1) = load_vh_state(soa, vb1, 2);
    let (mut uh1, mut vh1) = load_vh_state(soa, hb1, 2);
    let (mut ul1, mut vl1) = load_nl_state_n::<NL_MAX>(soa, lb1, n);
    let (sp0, sp1, ts) = (sav.spec(s0), sav.spec(s1), sav.ts());
    let (mut r0, mut r1) = (sav.r_sav[s0], sav.r_sav[s1]);
    let steps = out_l.len().min(out_r.len());
    {
        let (d_s, s_s, c_s) = (
            soa.d_coeff.as_f32x4_slice(),
            soa.s_coeff.as_f32x4_slice(),
            soa.c_bridge.as_f32x4_slice(),
        );
        let min_len = d_s.len().min(s_s.len()).min(c_s.len());
        assert!(
            vb0 + 2 <= min_len
                && hb0 + 2 <= min_len
                && lb0 + n <= min_len
                && vb1 + 2 <= min_len
                && hb1 + 2 <= min_len
                && lb1 + n <= min_len
        );
        for (lo, ro) in out_l[..steps].iter_mut().zip(out_r[..steps].iter_mut()) {
            let av0 = step_2_slices(&mut uv0, &mut vv0, vb0, d_s, s_s, c_s);
            let ah0 = step_2_slices(&mut uh0, &mut vh0, hb0, d_s, s_s, c_s);
            let al0 = step_nl_slices::<NL_MAX>(&mut ul0, &mut vl0, lb0, n, d_s, s_s, c_s);
            let av1 = step_2_slices(&mut uv1, &mut vv1, vb1, d_s, s_s, c_s);
            let ah1 = step_2_slices(&mut uh1, &mut vh1, hb1, d_s, s_s, c_s);
            let al1 = step_nl_slices::<NL_MAX>(&mut ul1, &mut vl1, lb1, n, d_s, s_s, c_s);
            eval_phantom_pair::<NL_MAX>(
                sp0, sp1, ts, n, &mut r0, &mut r1, &mut uv0, &vv0, &mut uh0, &vh0, &mut ul0, &vl0,
                &mut uv1, &vv1, &mut uh1, &vh1, &mut ul1, &vl1,
            );
            let f0 = (av0 + ah0 + al0).reduce_sum();
            let f1 = (av1 + ah1 + al1).reduce_sum();
            *lo += f0 * e0.pan_l + f1 * e1.pan_l;
            *ro += f0 * e0.pan_r + f1 * e1.pan_r;
        }
    }
    sav.r_sav[s0] = r0;
    sav.r_sav[s1] = r1;
    store_vh_state(soa, vb0, 2, &uv0, &vv0);
    store_vh_state(soa, hb0, 2, &uh0, &vh0);
    store_nl_state_n::<NL_MAX>(soa, lb0, n, &ul0, &vl0);
    store_vh_state(soa, vb1, 2, &uv1, &vv1);
    store_vh_state(soa, hb1, 2, &uh1, &vh1);
    store_nl_state_n::<NL_MAX>(soa, lb1, n, &ul1, &vl1);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_sav_pair_tile(
    soa: &mut ModalSoa,
    sav: &mut SavBank,
    s0: usize,
    e0: &StringMapEntry,
    s1: usize,
    e1: &StringMapEntry,
    bucket: usize,
    longitudinal_enabled: bool,
    out_l: &mut [f32],
    out_r: &mut [f32],
) {
    if bucket == 0 {
        step_no_phantom_pair_tile(soa, sav, s0, e0, s1, e1, longitudinal_enabled, out_l, out_r);
    } else if bucket == 1 {
        step_phantom_pair_tile::<1>(soa, sav, s0, e0, s1, e1, 1, out_l, out_r);
    } else if bucket == SPARSE_PHANTOM_LONG_VECS {
        step_phantom_pair_tile::<SPARSE_PHANTOM_LONG_VECS>(
            soa, sav, s0, e0, s1, e1, bucket, out_l, out_r,
        );
    } else {
        step_phantom_pair_tile::<BASE_PHANTOM_LONG_VECS>(
            soa, sav, s0, e0, s1, e1, bucket, out_l, out_r,
        );
    }
}
