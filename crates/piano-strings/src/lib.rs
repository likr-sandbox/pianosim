//! Stage 2: 230-string ZOH modal bank, SAV nonlinearity, and longitudinal coupling.
//!
//! M2 implements all 88 keys, 230 physical strings, and ~26,184 SIMD4-padded
//! modes using Exact Coupled-Form Incremental ZOH updated unconditionally on
//! every audio sample with zero heap allocations after [`StringBank::new`].

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod choir_map;
pub mod modal_soa;
pub mod zoh_kernel;

use piano_spec::{build_scale_design, VoicingSpec, NUM_KEYS, TOTAL_STRINGS};

use crate::choir_map::ChoirMap;
use crate::modal_soa::ModalSoa;
use crate::zoh_kernel::{
    flush_subnormals_soa, inject_excitation, render_subblock_core, step_all_strings, KeyPulseSpec,
    StringPulse,
};

/// All-88-key, 230-string, ~26k-mode continuous SIMD coupled-form ZOH bank.
#[derive(Clone, Debug)]
pub struct StringBank {
    sample_rate: f32,
    choir_map: ChoirMap,
    soa: ModalSoa,
    key_hold: [f32; NUM_KEYS],
    key_pulses: [KeyPulseSpec; NUM_KEYS],
    string_pulses: [StringPulse; TOTAL_STRINGS],
}

impl StringBank {
    /// Allocates and precomputes all SoA mode tables for `sample_rate` and `voicing`.
    ///
    /// Initial damper state is resting (`hold_01 = 0.0`: dampers resting on keys
    /// `0..70`, undamped keys `70..88` free-ringing).
    #[must_use]
    pub fn new(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let safe_rate = if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        };
        let specs = build_scale_design(voicing);
        let choir_map = ChoirMap::build_with_specs(&specs, voicing);
        let soa = ModalSoa::build(f64::from(safe_rate), voicing, &specs, &choir_map);
        let key_pulses = core::array::from_fn(|k| {
            KeyPulseSpec::for_key(
                piano_spec::key_tuned_f0_hz(k, &specs, voicing),
                f64::from(safe_rate),
            )
        });

        Self {
            sample_rate: safe_rate,
            choir_map,
            soa,
            key_hold: [0.0; NUM_KEYS],
            key_pulses,
            string_pulses: [StringPulse::default(); TOTAL_STRINGS],
        }
    }

    /// Resets all modal state buffers, active pulses, and damper positions without reallocating.
    pub fn reset(&mut self) {
        self.soa.reset_resting();
        self.key_hold.fill(0.0);
        self.string_pulses.fill(StringPulse::default());
    }

    /// Returns the configured sample rate in Hz.
    #[must_use]
    #[inline]
    pub const fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the total SIMD4-padded modal slot count (`~26,184`).
    #[must_use]
    #[inline]
    pub const fn total_modes(&self) -> usize {
        self.choir_map.total_padded_modes()
    }

    /// Returns the total unpadded active mode count (`~25,184`).
    #[must_use]
    #[inline]
    pub const fn active_unpadded_modes(&self) -> usize {
        self.choir_map.active_unpadded_modes()
    }

    /// Returns the total `F32x4` vector count (`~6,546`).
    #[must_use]
    #[inline]
    pub const fn total_vecs(&self) -> usize {
        self.choir_map.total_vecs()
    }

    /// Returns the 88-key / 230-string / vector-range mapping.
    #[must_use]
    #[inline]
    pub const fn choir_map(&self) -> &ChoirMap {
        &self.choir_map
    }

    /// Returns the current damper hold level `[0.0, 1.0]` for `key_idx`.
    #[must_use]
    #[inline]
    pub fn key_hold(&self, key_idx: usize) -> f32 {
        self.key_hold.get(key_idx).copied().unwrap_or(0.0)
    }

    /// Sets continuous damper hold (`0.0` = fully damped, `1.0` = undamped/held)
    /// for `key_idx` using exact pitch-invariant affine pole interpolation.
    pub fn set_key_hold(&mut self, key_idx: usize, hold_01: f32) {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return;
        };
        let clamped = if hold_01.is_finite() {
            hold_01.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if self.key_hold[key_idx].to_bits() == clamped.to_bits() {
            return;
        }
        self.key_hold[key_idx] = clamped;
        if key_entry.has_damper {
            self.soa
                .set_vec_range_hold(key_entry.vec_range.clone(), clamped);
        }
    }

    /// Triggers a key-period-adapted half-sine hammer excitation pulse on `key_idx`
    /// weighted per choir string (`string_weights`, e.g. for una corda shifting).
    pub fn trigger_key_pulse(&mut self, key_idx: usize, string_weights: [f32; 3], amplitude: f32) {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return;
        };
        if !amplitude.is_finite() || amplitude == 0.0 {
            return;
        }
        let spec = &self.key_pulses[key_idx];
        for (str_in_choir, str_idx) in key_entry.string_range.clone().enumerate() {
            let w = string_weights[str_in_choir];
            let eff = if w.is_finite() { amplitude * w } else { 0.0 };
            self.string_pulses[str_idx].trigger(spec, eff);
        }
    }

    /// Injects a single-sample force impulse into `key_idx` across its choir strings
    /// weighted by `string_weights`. Intended for sample-accurate external action forces.
    pub fn apply_excitation_sample(
        &mut self,
        key_idx: usize,
        string_weights: [f32; 3],
        force: f32,
    ) {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return;
        };
        if !force.is_finite() || force == 0.0 {
            return;
        }
        let total_vecs = self.choir_map.total_vecs();
        let u_vecs = &mut self.soa.u.as_f32x4_slice_mut()[..total_vecs];
        let b_vecs = &self.soa.b_exc.as_f32x4_slice()[..total_vecs];
        for (str_in_choir, entry) in self.choir_map.strings()[key_entry.string_range.clone()]
            .iter()
            .enumerate()
        {
            let w = string_weights[str_in_choir];
            if w.is_finite() && w != 0.0 {
                let range = entry.all_vecs();
                inject_excitation(&mut u_vecs[range.clone()], &b_vecs[range], force * w);
            }
        }
    }

    /// Steps all 230 strings unconditionally for one audio frame and returns `(L, R)`.
    /// Call [`StringBank::flush_subnormals`] once per subblock after sample loops.
    #[must_use]
    #[inline]
    pub fn render_frame(&mut self) -> (f32, f32) {
        step_all_strings(&mut self.soa, &self.choir_map)
    }

    /// Flushes subnormals (`< 1e-20`) in `u` and `v` once at subblock end.
    #[inline]
    pub fn flush_subnormals(&mut self) {
        flush_subnormals_soa(&mut self.soa);
    }

    /// Renders a stereo subblock across all 230 strings, stepping any queued
    /// key pulses and flushing subnormals once at the end of the subblock.
    pub fn render_subblock(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        let frames = out_l.len().min(out_r.len());
        if out_l.len() > frames {
            out_l[frames..].fill(0.0);
        }
        if out_r.len() > frames {
            out_r[frames..].fill(0.0);
        }
        render_subblock_core(
            &mut self.soa,
            &self.choir_map,
            &mut self.string_pulses,
            &mut out_l[..frames],
            &mut out_r[..frames],
        );
    }

    /// Diagnostic helper returning `(vert_rms, horiz_rms)` of modal bridge velocity
    /// `v * c_bridge` across `key_idx`'s strings at the current instant.
    #[must_use]
    pub fn measure_key_polarization_rms(&self, key_idx: usize) -> (f32, f32) {
        let Some(key_entry) = self.choir_map.key(key_idx) else {
            return (0.0, 0.0);
        };
        let v = self.soa.v.as_f32x4_slice();
        let c = self.soa.c_bridge.as_f32x4_slice();
        let strings = &self.choir_map.strings()[key_entry.string_range.clone()];
        let calc = |sel: fn(&choir_map::StringMapEntry) -> core::ops::Range<usize>| {
            let (mut sum, mut cnt) = (0.0_f64, 0usize);
            for entry in strings {
                for idx in sel(entry) {
                    for val in (v[idx] * c[idx]).to_array() {
                        sum += f64::from(val) * f64::from(val);
                    }
                    cnt += 4;
                }
            }
            if cnt > 0 {
                crate::modal_soa::sqrt_f64(sum / cnt as f64) as f32
            } else {
                0.0
            }
        };
        (
            calc(|e| e.vert_vecs.clone()),
            calc(|e| e.horiz_vecs.clone()),
        )
    }

    #[cfg(test)]
    pub(crate) fn render_key_subblock(
        &mut self,
        key_idx: usize,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let key_entry = self.choir_map.key(key_idx).expect("valid key");
        zoh_kernel::step_key_subblock(
            &mut self.soa,
            key_entry,
            self.choir_map.strings(),
            &mut self.string_pulses,
            out_l,
            out_r,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;
    use piano_simd::RealFft;
    use piano_spec::{
        build_scale_design, count_transverse_partials, key_tuned_f0_hz, partial_frequency_hz,
        MAX_AUDIBLE_FREQ_HZ,
    };

    const GATE1_KEYS: [usize; 10] = [0, 3, 12, 15, 27, 39, 48, 63, 75, 87];
    const GATE1_RATES: [f32; 3] = [44_100.0, 48_000.0, 96_000.0];

    fn no_detune() -> VoicingSpec {
        VoicingSpec {
            unison_detune_scale: 0.0,
            ..VoicingSpec::default()
        }
    }

    fn rms(slice: &[f32]) -> f32 {
        (slice.iter().map(|x| x * x).sum::<f32>() / (slice.len().max(1) as f32)).sqrt()
    }

    fn pol_rms(bank: &StringBank, key: usize) -> f32 {
        let (v, h) = bank.measure_key_polarization_rms(key);
        (v * v + h * h).sqrt()
    }

    fn goertzel_power(samples: &[f64], freq_hz: f64, sr: f64) -> f64 {
        let coeff = 2.0 * (TAU * freq_hz / sr).cos();
        let (mut s1, mut s2) = (0.0_f64, 0.0_f64);
        for &x in samples {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - coeff * s1 * s2
    }

    fn parabolic_step_hz(samples: &[f64], f_center: f64, delta_cents: f64, sr: f64) -> f64 {
        let r = 2.0_f64.powf(delta_cents / 1200.0);
        let p_lo = goertzel_power(samples, f_center / r, sr);
        let p_mid = goertzel_power(samples, f_center, sr);
        let p_hi = goertzel_power(samples, f_center * r, sr);
        let denom = p_lo - 2.0 * p_mid + p_hi;
        let shift = if denom < 0.0 {
            (0.5 * (p_lo - p_hi) / denom).clamp(-1.0, 1.0) * delta_cents
        } else {
            0.0
        };
        f_center * 2.0_f64.powf(shift / 1200.0)
    }

    #[test]
    fn gate1_coupled_zoh_frequency_accuracy_under_0_05_cents_across_sample_rates() {
        const N: usize = 65_536;
        let voicing = no_detune();
        let specs = build_scale_design(&voicing);
        let denom = (N - 1) as f64;
        let window: Vec<f64> = (0..N)
            .map(|n| {
                let p = TAU * (n as f64) / denom;
                0.35875 - 0.48829 * p.cos() + 0.14128 * (2.0 * p).cos() - 0.01168 * (3.0 * p).cos()
            })
            .collect();
        let mut rfft = RealFft::new(N);
        let mut spectrum = vec![piano_simd::Complex::ZERO; rfft.spectrum_len()];
        let (mut out_l, mut out_r) = (vec![0.0_f32; N], vec![0.0_f32; N]);
        let (mut mono, mut win) = (vec![0.0_f32; N], vec![0.0_f64; N]);
        let mut max_cents_err = 0.0_f64;

        for &sr in &GATE1_RATES {
            let mut bank = StringBank::new(sr, &voicing);
            let sr64 = f64::from(sr);
            assert!((25_000..=26_500).contains(&bank.total_modes()));
            for &key in &GATE1_KEYS {
                bank.reset();
                bank.set_key_hold(key, 1.0);
                bank.apply_excitation_sample(key, [1.0, 1.0, 1.0], 1.0);
                bank.render_key_subblock(key, &mut out_l, &mut out_r);
                for i in 0..N {
                    let m = out_l[i] + out_r[i];
                    assert!(m.is_finite());
                    mono[i] = m;
                    win[i] = f64::from(m) * window[i];
                }
                rfft.forward(&mono, &mut spectrum);

                let f0 = key_tuned_f0_hz(key, &specs, &voicing);
                let b = specs[key].inharmonicity_b;
                let n_p = count_transverse_partials(f0, b, MAX_AUDIBLE_FREQ_HZ);
                let mut partials: Vec<usize> = (1..=n_p.min(8)).collect();
                if n_p > 8 {
                    let step = ((n_p - 8) / 8).max(1);
                    partials.extend(((8 + step)..n_p).step_by(step));
                    partials.push(n_p);
                }
                for j in partials {
                    let f_exp = partial_frequency_hz(f0, b, j);
                    assert!(f_exp < 0.49 * sr64);
                    let bin = ((f_exp * N as f64) / sr64).round() as usize;
                    let energy = (bin.saturating_sub(2)..=(bin + 2).min(spectrum.len() - 1))
                        .map(|idx| spectrum[idx].norm_sq())
                        .fold(0.0_f32, f32::max);
                    assert!(energy > 0.0, "missing FFT peak sr={sr} k={key} j={j}");

                    let f1 = parabolic_step_hz(&win, f_exp, 0.25, sr64);
                    let f_peak = parabolic_step_hz(&win, f1, 0.03, sr64);
                    let cents = (1200.0 * (f_peak / f_exp).log2()).abs();
                    max_cents_err = max_cents_err.max(cents);
                    assert!(cents < 0.05, "sr={sr} key={key} j={j} err={cents:.5} cents");
                }
            }
        }
        eprintln!("Gate1 max_cents_err = {max_cents_err:.6} cents");
        assert!(max_cents_err < 0.05);
    }

    #[test]
    fn full_render_subblock_c6_to_c8_dampers_and_frame_api() {
        let mut full_bank = StringBank::new(48_000.0, &VoicingSpec::default());
        for &k in &[0, 39, 87] {
            full_bank.set_key_hold(k, 1.0);
            full_bank.trigger_key_pulse(k, [1.0; 3], 1.0);
        }
        let mut slice_bank = full_bank.clone();
        let (mut fl, mut fr) = (vec![0.0_f32; 2_048], vec![0.0_f32; 2_048]);
        let (mut sl, mut sr_v) = (vec![0.0_f32; 2_048], vec![0.0_f32; 2_048]);
        let (mut tl, mut tr) = (vec![0.0_f32; 2_048], vec![0.0_f32; 2_048]);
        full_bank.render_subblock(&mut fl, &mut fr);
        for &k in &[0, 39, 87] {
            slice_bank.render_key_subblock(k, &mut tl, &mut tr);
            for i in 0..2_048 {
                sl[i] += tl[i];
                sr_v[i] += tr[i];
            }
        }
        for i in 0..2_048 {
            assert!((fl[i] - sl[i]).abs() < 1e-6 && (fr[i] - sr_v[i]).abs() < 1e-6);
        }

        // C6, C7, C8 pulse attack RMS + sustained T60 (~1..4 s).
        let (mut buf_l, mut buf_r) = (vec![0.0_f32; 12_000], vec![0.0_f32; 12_000]);
        for &(key, lo, hi) in &[(63, 2.5, 6.0), (75, 1.3, 3.5), (87, 0.7, 2.0)] {
            full_bank.reset();
            full_bank.set_key_hold(key, 1.0);
            full_bank.trigger_key_pulse(key, [1.0; 3], 1.0);
            full_bank.render_subblock(&mut buf_l[..960], &mut buf_r[..960]);
            let atk = rms(&buf_l[..960]);
            assert!(atk > 0.01 && atk.is_finite(), "k={key} atk={atk}");
            let (v0, _) = full_bank.measure_key_polarization_rms(key);
            full_bank.render_key_subblock(key, &mut buf_l, &mut buf_r); // +0.25s
            let (v1, _) = full_bank.measure_key_polarization_rms(key);
            let t60 = (0.25 * 3.0) / (v0 / v1).log10();
            assert!((lo..=hi).contains(&t60), "k={key} t60={t60}");
        }

        // Damper release across 0..70 vs undamped 70..88 + stereo pan + half-pedal.
        let keys = [0usize, 39, 63, 69, 70, 75, 87];
        full_bank.reset();
        for &k in &keys {
            full_bank.set_key_hold(k, 1.0);
            full_bank.trigger_key_pulse(k, [1.0, 0.5, 1.0], 1.0);
        }
        full_bank.render_subblock(&mut buf_l[..960], &mut buf_r[..960]);
        let atk_pol: Vec<f32> = keys.iter().map(|&k| pol_rms(&full_bank, k)).collect();
        for &k in &keys {
            full_bank.set_key_hold(k, 0.0);
        }
        full_bank.render_subblock(&mut buf_l, &mut buf_r);
        let damped_end = piano_spec::DAMPED_KEYS;
        for (&k, &e0) in keys.iter().zip(atk_pol.iter()) {
            let r = pol_rms(&full_bank, k) / e0;
            assert!(
                if k < damped_end { r < 1e-3 } else { r > 0.14 },
                "k={k} r={r}"
            );
        }

        // render_frame + flush_subnormals + out-of-range/NaN resilience.
        full_bank.reset();
        full_bank.set_key_hold(99, f32::NAN);
        full_bank.set_key_hold(39, 0.5);
        full_bank.apply_excitation_sample(39, [1.0, 0.0, 1.0], 1.0);
        let (l0, r0) = full_bank.render_frame();
        assert!(l0.is_finite() && r0.is_finite() && l0.abs() + r0.abs() > 0.0);
        full_bank.flush_subnormals();
    }

    #[test]
    fn gate3_two_polarization_decay_and_trichord_beating() {
        let mut bank = StringBank::new(48_000.0, &no_detune());
        bank.set_key_hold(63, 1.0);
        bank.apply_excitation_sample(63, [1.0; 3], 1.0);
        let (mut ol, mut or) = (vec![0.0_f32; 4_800], vec![0.0_f32; 4_800]);
        bank.render_key_subblock(63, &mut ol, &mut or);
        let (v_early, h_early) = bank.measure_key_polarization_rms(63);
        for _ in 0..35 {
            bank.render_key_subblock(63, &mut ol, &mut or);
        }
        let (v_late, h_late) = bank.measure_key_polarization_rms(63);
        assert!(v_early > h_early && (h_late / h_early) > (v_late / v_early) * 2.0);

        // Gate #3B: C4 trichord beating modulation across 8.0 s.
        let mut detuned = StringBank::new(48_000.0, &VoicingSpec::default());
        let mut flat = StringBank::new(48_000.0, &no_detune());
        for b in [&mut detuned, &mut flat] {
            b.set_key_hold(39, 1.0);
            b.apply_excitation_sample(39, [1.0; 3], 1.0);
        }
        let win = 2_400usize;
        let (mut dl, mut dr) = (vec![0.0_f32; win], vec![0.0_f32; win]);
        let (mut fl, mut fr) = (vec![0.0_f32; win], vec![0.0_f32; win]);
        let (mut prev_d, mut rebounds, mut ratio_var) = (0.0_f32, 0usize, 0.0_f32);
        for w in 0..160 {
            detuned.render_key_subblock(39, &mut dl, &mut dr);
            flat.render_key_subblock(39, &mut fl, &mut fr);
            let (ed, ef) = (rms(&dl), rms(&fl));
            if w >= 2 {
                if ed > prev_d * 1.01 {
                    rebounds += 1;
                }
                ratio_var = ratio_var.max((ed / ef.max(1e-9) - 1.0).abs());
            }
            prev_d = ed;
        }
        assert!(rebounds >= 3 && ratio_var > 0.30);
    }
}
