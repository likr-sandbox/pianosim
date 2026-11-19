//! Per-sample bidirectional Stage 3 bridge junction stepping (`BridgeBank`).

use piano_spec::TOTAL_STRINGS;

use crate::math::{mat8_vec, mat8_vec2};
use crate::zone_proj::{
    gather_sanitized_run, scatter_sanitized_run, MAX_ZONE_RUNS, NUM_BRIDGE_ZONES,
};
use crate::BridgeBank;

impl BridgeBank {
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
        let mut unused_dup_zones = [0.0_f32; NUM_BRIDGE_ZONES];
        self.step_bridge_sample_with_duplex_zones(
            a_waves,
            zone_str_audio,
            b_waves_out,
            zone_out,
            &mut unused_dup_zones,
        )
    }

    /// Advances the Stage 3 bidirectional bridge junction by one audio sample and writes
    /// unpanned per-zone treble duplex contributions (`Zones 2..8`) into `dup_zone_out`
    /// (`Zones 0..2` strictly `0.0`) for Stage 4 convolution without stereo-pan bypass.
    #[inline(never)]
    pub fn step_bridge_sample_with_duplex_zones(
        &mut self,
        a_waves: &[f32; TOTAL_STRINGS],
        zone_str_audio: &[f32; NUM_BRIDGE_ZONES],
        b_waves_out: &mut [f32; TOTAL_STRINGS],
        zone_out: &mut [f32; NUM_BRIDGE_ZONES],
        dup_zone_out: &mut [f32; NUM_BRIDGE_ZONES],
    ) -> (f32, f32) {
        let cache = &self.zone_cache;
        let n_runs = cache.run_count.min(MAX_ZONE_RUNS);
        let mut run_a_finite = [true; MAX_ZONE_RUNS];
        let mut f_inc_str = [0.0_f64; NUM_BRIDGE_ZONES];
        for (&run, slot_ok) in cache.runs[..n_runs]
            .iter()
            .zip(run_a_finite[..n_runs].iter_mut())
        {
            let (start, end, z0, z1) = run.bounds();
            let (mut v0_acc, mut v1_acc) = (piano_simd::F32x4::zero(), piano_simd::F32x4::zero());
            let mut i = start;
            while i + 4 <= end {
                let aw = piano_simd::F32x4::from_slice(&a_waves[i..i + 4]);
                v0_acc =
                    piano_simd::F32x4::from_slice(&cache.rw0_f32[i..i + 4]).mul_add(aw, v0_acc);
                v1_acc =
                    piano_simd::F32x4::from_slice(&cache.rw1_f32[i..i + 4]).mul_add(aw, v1_acc);
                i += 4;
            }
            let (mut s0, mut s1) = (v0_acc.reduce_sum(), v1_acc.reduce_sum());
            while i < end {
                s0 += cache.rw0_f32[i] * a_waves[i];
                s1 += cache.rw1_f32[i] * a_waves[i];
                i += 1;
            }
            let is_finite = s0.is_finite() && s1.is_finite();
            *slot_ok = is_finite;
            let (acc0, acc1) = if is_finite {
                (f64::from(s0), f64::from(s1))
            } else {
                scatter_sanitized_run(cache, run, a_waves)
            };
            f_inc_str[z0] += acc0;
            f_inc_str[z1] += acc1;
        }

        if self.fixed_end {
            for (dst, &a) in b_waves_out.iter_mut().zip(a_waves.iter()) {
                *dst = if a.is_finite() { -a } else { 0.0 };
            }
            dup_zone_out.fill(0.0);
            let y_inf = self.ssm.y_inf();
            for z in 0..NUM_BRIDGE_ZONES {
                let f_audio = if zone_str_audio[z].is_finite() {
                    zone_str_audio[z]
                } else {
                    0.0
                };
                let out_v = y_inf[z] * (f_inc_str[z] as f32 + f_audio);
                zone_out[z] = if out_v.is_finite() { out_v } else { 0.0 };
            }
            return (0.0, 0.0);
        }

        if run_a_finite[..n_runs].iter().all(|&ok| ok)
            && f_inc_str.iter().all(|&f| f == 0.0)
            && self.ssm.is_quiescent()
            && self.duplex.is_quiescent()
            && self.hertz.is_quiescent()
        {
            if a_waves.iter().all(|&a| a == 0.0) {
                b_waves_out.fill(0.0);
            } else {
                for (dst, &a) in b_waves_out.iter_mut().zip(a_waves.iter()) {
                    *dst = -a;
                }
            }
            let y_inf = self.ssm.y_inf();
            for z in 0..NUM_BRIDGE_ZONES {
                let f_audio = if zone_str_audio[z].is_finite() {
                    zone_str_audio[z]
                } else {
                    0.0
                };
                let out_v = y_inf[z] * f_audio;
                zone_out[z] = if out_v.is_finite() { out_v } else { 0.0 };
            }
            dup_zone_out.fill(0.0);
            return (0.0, 0.0);
        }

        let f_dup_free = self.duplex.free_forces();
        let (qd_free, v0_free) = self.ssm.free_v0();

        let mut f_inc_wdf = [0.0_f64; NUM_BRIDGE_ZONES];
        for z in 0..NUM_BRIDGE_ZONES {
            f_inc_wdf[z] = f_inc_str[z] + f_dup_free[z];
        }

        let (v_zone_f64, f_hertz_on_br) = if self.hertz.is_enabled() {
            let g = self.hertz.eval_sav_gradient();
            let g_nz = g.iter().any(|&gz| gz != 0.0);
            let v_zone = if g_nz {
                let r_h = self.hertz.r_sav();
                let (s_wdf, s_g) = mat8_vec2(self.ssm.solve_abr(), &f_inc_wdf, &g);
                let mut v0 = [0.0_f64; NUM_BRIDGE_ZONES];
                for i in 0..NUM_BRIDGE_ZONES {
                    v0[i] = v0_free[i] + s_wdf[i] - r_h * s_g[i];
                }
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
                let mut out = [0.0_f64; NUM_BRIDGE_ZONES];
                for i in 0..NUM_BRIDGE_ZONES {
                    out[i] = v0[i] - alpha_sm * u_sm[i];
                }
                out
            } else {
                let s_wdf = mat8_vec(self.ssm.solve_abr(), &f_inc_wdf);
                let mut out = [0.0_f64; NUM_BRIDGE_ZONES];
                for i in 0..NUM_BRIDGE_ZONES {
                    out[i] = v0_free[i] + s_wdf[i];
                }
                out
            };
            let f_h = self.hertz.commit_step(&g, &v_zone);
            (v_zone, f_h)
        } else {
            let s_wdf = mat8_vec(self.ssm.solve_abr(), &f_inc_wdf);
            let mut out = [0.0_f64; NUM_BRIDGE_ZONES];
            for i in 0..NUM_BRIDGE_ZONES {
                out[i] = v0_free[i] + s_wdf[i];
            }
            (out, [0.0_f64; NUM_BRIDGE_ZONES])
        };

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
        let duplex_stereo = self
            .duplex
            .step_with_zone_velocities_and_zones(&v_zone, dup_zone_out);

        for (&run, &a_ok) in cache.runs[..n_runs]
            .iter()
            .zip(run_a_finite[..n_runs].iter())
        {
            let (start, end, z0, z1) = run.bounds();
            let (vz0, vz1) = (v_zone_f64[z0], v_zone_f64[z1]);
            let two_vz0 = if vz0.is_finite() { 2.0 * vz0 } else { 0.0 };
            let two_vz1 = if vz1.is_finite() { 2.0 * vz1 } else { 0.0 };
            let (two_vz0_f32, two_vz1_f32) = (two_vz0 as f32, two_vz1 as f32);
            let tv0 = piano_simd::F32x4::splat(two_vz0_f32);
            let tv1 = piano_simd::F32x4::splat(two_vz1_f32);
            let mut i = start;
            while i + 4 <= end {
                let w0 = piano_simd::F32x4::from_slice(&cache.w0_f32[i..i + 4]);
                let w1 = piano_simd::F32x4::from_slice(&cache.w1_f32[i..i + 4]);
                let aw = piano_simd::F32x4::from_slice(&a_waves[i..i + 4]);
                (w0.mul_add(tv0, w1 * tv1) - aw).write_to_slice(&mut b_waves_out[i..i + 4]);
                i += 4;
            }
            while i < end {
                b_waves_out[i] =
                    (cache.w0_f32[i] * two_vz0_f32 + cache.w1_f32[i] * two_vz1_f32) - a_waves[i];
                i += 1;
            }
            if !a_ok
                || !vz0.is_finite()
                || !vz1.is_finite()
                || !two_vz0_f32.is_finite()
                || !two_vz1_f32.is_finite()
            {
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
