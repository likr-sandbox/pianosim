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
        for &(nl_on, force, cap6) in &[(false, 1.0_f32, false), (true, 0.02_f32, true)] {
            let mut bank = StringBank::new(sr, &voicing);
            bank.set_nonlinear_coupling(nl_on, nl_on);
            let sr64 = f64::from(sr);
            assert!((25_000..=26_500).contains(&bank.total_modes()));
            for &key in &GATE1_KEYS {
                bank.reset();
                bank.apply_excitation_sample(key, [1.0, 1.0, 1.0], force);
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
                let mut partials: Vec<usize> = (1..=n_p.min(if cap6 { 6 } else { 8 })).collect();
                if !cap6 && n_p > 8 {
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
                    assert!(
                        cents < 0.05,
                        "nl={nl_on} sr={sr} key={key} j={j} err={cents:.5} cents"
                    );
                }
            }
        }
    }
    eprintln!("Gate1 max_cents_err = {max_cents_err:.6} cents");
    assert!(max_cents_err < 0.05);
}

#[test]
fn full_render_subblock_c6_to_c8_and_frame_api() {
    let mut full_bank = StringBank::new(48_000.0, &VoicingSpec::default());
    for &k in &[0, 39, 87] {
        full_bank.apply_excitation_sample(k, [1.0; 3], 1.0);
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

    let vert_env_rms = |bank: &StringBank, key: usize| -> f64 {
        let ke = bank.choir_map.key(key).unwrap();
        let (u, v, c) = (
            bank.soa.u.as_f32x4_slice(),
            bank.soa.v.as_f32x4_slice(),
            bank.soa.c_bridge.as_f32x4_slice(),
        );
        let mut sum = 0.0_f64;
        for entry in &bank.choir_map.strings()[ke.string_range.clone()] {
            for i in entry.vert_vecs.clone() {
                let e = ((u[i] * u[i] + v[i] * v[i]) * (c[i] * c[i])).reduce_sum();
                sum += f64::from(e);
            }
        }
        sum.sqrt()
    };

    let (mut buf_l, mut buf_r) = (vec![0.0_f32; 12_000], vec![0.0_f32; 12_000]);
    for &(key, lo, hi) in &[(63, 2.5, 6.0), (75, 1.3, 3.5), (87, 0.7, 2.0)] {
        full_bank.reset();
        full_bank.apply_excitation_sample(key, [1.0; 3], 1.0);
        full_bank.render_subblock(&mut buf_l[..960], &mut buf_r[..960]);
        let atk = rms(&buf_l[..960]);
        assert!(atk > 0.01 && atk.is_finite(), "k={key} atk={atk}");
        let e0 = vert_env_rms(&full_bank, key);
        full_bank.render_key_subblock(key, &mut buf_l, &mut buf_r); // +0.25s
        let e1 = vert_env_rms(&full_bank, key);
        let t60 = (0.25 * 3.0) / (e0 / e1).log10();
        assert!((lo..=hi).contains(&t60), "k={key} t60={t60}");
    }

    // render_frame + flush_subnormals + out-of-range/NaN resilience.
    full_bank.reset();
    full_bank.mark_key_possibly_active(99);
    full_bank.apply_excitation_sample(39, [1.0, f32::NAN, 1.0], 1.0);
    let (l0, r0) = full_bank.render_frame();
    assert!(l0.is_finite() && r0.is_finite() && l0.abs() + r0.abs() > 0.0);
    full_bank.flush_subnormals();
}

#[test]
fn gate3_two_polarization_decay_and_trichord_beating() {
    let mut bank = StringBank::new(48_000.0, &no_detune());
    bank.apply_excitation_sample(63, [1.0; 3], 1.0);
    let (mut ol, mut or) = (vec![0.0_f32; 4_800], vec![0.0_f32; 4_800]);
    bank.render_key_subblock(63, &mut ol, &mut or);
    let (v_early, h_early) = bank.measure_key_polarization_rms(63);
    for _ in 0..35 {
        bank.render_key_subblock(63, &mut ol, &mut or);
    }
    let (v_late, h_late) = bank.measure_key_polarization_rms(63);
    assert!(v_early > h_early && (h_late / h_early) > (v_late / v_early) * 2.0);

    let mut detuned = StringBank::new(48_000.0, &VoicingSpec::default());
    let mut flat = StringBank::new(48_000.0, &no_detune());
    for b in [&mut detuned, &mut flat] {
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

impl StringBank {
    pub(crate) fn render_key_subblock(
        &mut self,
        key_idx: usize,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let key_entry = self.choir_map.key(key_idx).expect("valid key");
        let total = out_l.len().min(out_r.len());
        let strings = self.choir_map.strings();
        let mut base = 0usize;
        while base < total {
            let t_len = (total - base).min(TILE);
            let (mut tile_l, mut tile_r) = ([0.0_f32; TILE], [0.0_f32; TILE]);
            for s_idx in key_entry.string_range.clone() {
                let entry = &strings[s_idx];
                if self.ports.may_be_nonzero[s_idx] {
                    step_ringing_string_tile(
                        &mut self.soa,
                        &mut self.sav,
                        s_idx,
                        entry,
                        self.longitudinal_enabled,
                        self.sav_phantom_enabled,
                        &mut tile_l[..t_len],
                        &mut tile_r[..t_len],
                    );
                    self.ports.ports_synced[s_idx] = false;
                } else {
                    let acc = step_string_tile(&mut self.soa, entry.all_vecs(), t_len);
                    accum_tile_stereo(
                        &acc,
                        entry.pan_l,
                        entry.pan_r,
                        t_len,
                        &mut tile_l,
                        &mut tile_r,
                    );
                }
            }
            out_l[base..base + t_len].copy_from_slice(&tile_l[..t_len]);
            out_r[base..base + t_len].copy_from_slice(&tile_r[..t_len]);
            base += t_len;
        }
    }
}
