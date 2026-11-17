//! Transverse string T60 damping laws (`no_std`).

use piano_spec::NUM_KEYS;

use crate::math::exp_f64;

#[inline]
pub(crate) fn t60_vert_fundamental(key_idx: usize) -> f64 {
    let x = (key_idx as f64) / ((NUM_KEYS - 1) as f64);
    30.0 * exp_f64(-0.45 * x - 4.35 * x * x + 1.50 * x * x * x)
}

#[inline]
pub(crate) fn t60_wire_fundamental(key_idx: usize) -> f64 {
    t60_vert_fundamental(key_idx).max(1.85)
}

#[inline]
pub(crate) fn t60_transverse_partial(
    t60_v_base: f64,
    f_j: f64,
    j: usize,
    hold_mult: f64,
    bridge_coupled: bool,
) -> f64 {
    let f_norm = f_j / 20_000.0;
    if bridge_coupled {
        let loss_scale = 1.0 + 0.04 * ((j - 1) as f64) + 1.5 * f_norm * f_norm;
        (t60_v_base / loss_scale).max(0.45) * hold_mult
    } else {
        let loss_scale = 1.0 + 0.05 * ((j - 1) as f64) + 1.5 * f_norm * f_norm;
        (t60_v_base / loss_scale).max(0.25) * hold_mult
    }
}
