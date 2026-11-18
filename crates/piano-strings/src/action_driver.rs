//! Stage 1 <-> Stage 2 closed-loop SI action driver trait (`Newtons`, `meters`, `m/s`).

use crate::zoh_kernel::TILE;

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
    /// Returns the instantaneous damper peel-off adhesion force (`N`, `>= 0`) for `key_idx`
    /// produced by the most recent clearance step.
    #[inline]
    fn damper_peel_force_n(&self, _key_idx: usize) -> f32 {
        0.0
    }
    /// Advances damper rail smoother by `t_len` samples (<= 8), fills `out_clears[..t_len]`
    /// and per-sample peel-off adhesion forces `out_peel_n[..t_len]` (`N`), and returns
    /// `(min_clear_m, any_peel_nonzero)`.
    #[inline]
    fn step_key_clearance_tile_with_peel(
        &mut self,
        key_idx: usize,
        t_len: usize,
        out_clears: &mut [f32; TILE],
        out_peel_n: &mut [f32; TILE],
    ) -> (f32, bool) {
        let min_c = self.step_key_clearance_tile(key_idx, t_len, out_clears);
        let peel = self.damper_peel_force_n(key_idx);
        let n = t_len.min(TILE);
        out_peel_n[..n].fill(peel);
        (min_c, peel != 0.0)
    }
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
    /// Evaluates both V (`str_idx * 2`) and H (`str_idx * 2 + 1`) unilateral damper forces
    /// for `(key_idx, str_idx)`, sharing geometric clearance calculations.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn step_damper_pair(
        &mut self,
        key_idx: usize,
        str_idx: usize,
        y_dv_m: f32,
        v_dv_mps: f32,
        r_dv: f32,
        y_dh_m: f32,
        v_dh_mps: f32,
        r_dh: f32,
        y_clear_m: f32,
    ) -> (f32, f32) {
        let fv = self.step_damper_sample(key_idx, str_idx * 2, y_dv_m, v_dv_mps, r_dv, y_clear_m);
        let fh =
            self.step_damper_sample(key_idx, str_idx * 2 + 1, y_dh_m, v_dh_mps, r_dh, y_clear_m);
        (fv, fh)
    }
    /// Returns exact closed-damper linear-equilibrium V/H velocity gains when felt memory has settled.
    #[inline]
    fn symp_linear_gains_if_settled(
        &mut self,
        _key_idx: usize,
        _str_idx: usize,
        _r_dv: f32,
        _r_dh: f32,
        _y_clear_m: f32,
    ) -> Option<(f32, f32)> {
        None
    }
    /// Advances quiescent damper felt memory over `sub` samples and returns chunk linear V/H gains.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn symp_linear_gains_for_chunk(
        &mut self,
        _key_idx: usize,
        _str_idx: usize,
        _r_dv: f32,
        _r_dh: f32,
        _yc0: f32,
        _yc_end: f32,
        _sub: usize,
    ) -> Option<(f32, f32)> {
        None
    }
    /// Relaxes damper felt hysteresis states across `sub` open-gap samples when contact-free.
    #[inline]
    fn relax_open_gap_chunk(&mut self, _str_idx: usize, _sub: usize) {}
}
