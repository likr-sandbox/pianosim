//! 64-mode analytical coupled-form ZOH duplex scale bank (`16` `F32x4` SIMD vectors).
//!
//! Passively and bidirectionally coupled ONLY to treble/tenor bridge **Zones 2..7**
//! via exact WDF port scattering so bass Zones 0..1 remain uncoupled and `E_duplex >= 0` holds.

use core::f64::consts::LN_2;

use piano_simd::F32x4;
use piano_spec::{build_scale_design, railsback_stretch_cents_table, VoicingSpec, NUM_KEYS};

use crate::math::{coupled_pole_f32, exp_f64};
use crate::zone_proj::{zone_stereo_pan, NUM_BRIDGE_ZONES, TREBLE_DUPLEX_ZONE_START};

/// Total analytical duplex scale resonant modes (`16 * 4 = 64`).
pub const NUM_DUPLEX_MODES: usize = 64;

/// Number of `F32x4` SIMD vectors in `DuplexBank`.
pub const NUM_DUPLEX_VECS: usize = 16;

/// Zone assignment (`2..=7`) for each of the 16 `F32x4` duplex mode vectors.
pub const DUPLEX_VEC_ZONES: [usize; NUM_DUPLEX_VECS] =
    [2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7];

/// Physical `(anchor_key_idx, harmonic_mult, fine_aliquot_detune)` for all 64 duplex wire modes.
///
/// Each mode models a rear non-speaking wire segment of speaking-length ratio
/// `L(k) / L_dup(m) = harmonic_mult * (1.0 + fine_aliquot_detune)` anchored to a key
/// in that physical bridge zone (`26..32` Z2, `33..44` Z3, `45..56` Z4, `57..68` Z5,
/// `69..80` Z6, `81..87` Z7), computed strictly offline without runtime harmonic branching.
const DUPLEX_MODE_LAYOUT: [(u8, u8, f64); NUM_DUPLEX_MODES] = [
    // Zone 2 (8 modes, keys 26..=32)
    (27, 12, 0.000),
    (28, 11, 0.002),
    (29, 12, -0.002),
    (30, 12, 0.000),
    (31, 12, 0.002),
    (32, 12, -0.002),
    (31, 15, 0.001),
    (32, 16, 0.000),
    // Zone 3 (8 modes, keys 33..=44)
    (33, 10, 0.000),
    (35, 10, 0.002),
    (37, 10, -0.002),
    (39, 9, 0.000),
    (40, 9, 0.002),
    (41, 10, -0.001),
    (43, 10, 0.002),
    (44, 11, 0.000),
    // Zone 4 (12 modes, keys 45..=56)
    (45, 7, 0.000),
    (46, 7, 0.002),
    (47, 7, -0.002),
    (48, 6, 0.000),
    (49, 6, 0.002),
    (50, 6, -0.002),
    (51, 6, 0.000),
    (52, 6, 0.002),
    (53, 7, -0.001),
    (54, 7, 0.002),
    (55, 7, 0.000),
    (56, 8, -0.002),
    // Zone 5 (12 modes, keys 57..=68)
    (57, 5, 0.000),
    (58, 5, 0.002),
    (59, 5, -0.002),
    (60, 5, 0.000),
    (61, 5, 0.002),
    (62, 5, -0.002),
    (63, 4, 0.000),
    (63, 6, 0.001),
    (64, 5, 0.000),
    (65, 5, 0.002),
    (66, 5, -0.002),
    (67, 5, 0.000),
    // Zone 6 (12 modes, keys 69..=80)
    (69, 4, 0.000),
    (70, 4, 0.002),
    (71, 4, -0.002),
    (72, 4, 0.000),
    (73, 4, 0.002),
    (74, 4, -0.002),
    (75, 4, 0.000),
    (76, 4, 0.002),
    (77, 4, -0.002),
    (78, 4, 0.000),
    (79, 3, 0.002),
    (80, 3, -0.002),
    // Zone 7 (12 modes, keys 81..=87)
    (81, 3, 0.000),
    (82, 3, 0.002),
    (83, 3, -0.002),
    (84, 3, 0.000),
    (85, 3, 0.001),
    (87, 2, 0.000),
    (86, 2, 0.002),
    (85, 2, -0.002),
    (84, 2, 0.001),
    (83, 2, -0.001),
    (82, 2, 0.002),
    (81, 2, -0.002),
];

const DUPLEX_TARGET_R_PER_ZONE: f64 = 0.018;
/// Stage 4 duplex coupling readout gain (`+20 dB` over initial `0.30` calibration).
const DUPLEX_AUDIO_GAIN: f32 = 3.0;
const SUBNORMAL_FLUSH_THRESHOLD: f32 = 1.0e-20;

/// 64-mode SIMD coupled-form ZOH duplex scale resonator bank (`Zones 2..7`).
#[derive(Clone, Debug)]
pub struct DuplexBank {
    u: [F32x4; NUM_DUPLEX_VECS],
    v: [F32x4; NUM_DUPLEX_VECS],
    d: [F32x4; NUM_DUPLEX_VECS],
    s: [F32x4; NUM_DUPLEX_VECS],
    beta: [F32x4; NUM_DUPLEX_VECS],
    c_audio: [F32x4; NUM_DUPLEX_VECS],
    freqs_hz: [f32; NUM_DUPLEX_MODES],
    r_dup: [f64; NUM_BRIDGE_ZONES],
    alpha_dup: f64,
    ts: f64,
    sample_rate_f64: f64,
    lossless: bool,
    voicing: VoicingSpec,
    tuning_initialized: bool,
    last_stereo: (f32, f32),
    active: bool,
}

#[inline]
fn tuning_matches(a: &VoicingSpec, b: &VoicingSpec) -> bool {
    if a.a4_freq_hz != b.a4_freq_hz
        || a.stretch_amount != b.stretch_amount
        || a.string_length_scale != b.string_length_scale
    {
        return false;
    }
    for k in 0..NUM_KEYS {
        if a.note_offsets[k].pitch_cents != b.note_offsets[k].pitch_cents
            || a.note_offsets[k].inharmonicity_scale != b.note_offsets[k].inharmonicity_scale
        {
            return false;
        }
    }
    true
}

impl DuplexBank {
    /// Constructs default passive `DuplexBank` (`duplex_gain = 1.0`, damped, default voicing).
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self::new_with_voicing(sample_rate, &VoicingSpec::default())
    }

    /// Constructs passive `DuplexBank` (`duplex_gain = 1.0`, damped) tuned for `voicing`.
    #[must_use]
    pub fn new_with_voicing(sample_rate: f32, voicing: &VoicingSpec) -> Self {
        let mut bank = Self::new_with_params(sample_rate, 1.0, false);
        bank.retune_voicing(voicing);
        bank
    }

    /// Constructs `DuplexBank` with explicit coupling gain and lossless flag (`zeta = 0`).
    #[must_use]
    pub fn new_with_params(sample_rate: f32, duplex_gain: f32, lossless: bool) -> Self {
        let safe_sr = f64::from(if sample_rate.is_finite() {
            sample_rate.clamp(8_000.0, 384_000.0)
        } else {
            48_000.0
        });
        let ts = 1.0 / safe_sr;
        let alpha_dup = 1.0 / ts;
        let gain = f64::from(if duplex_gain.is_finite() {
            duplex_gain.max(0.0)
        } else {
            0.0
        });

        let mut counts_per_zone = [0usize; NUM_BRIDGE_ZONES];
        for &z in &DUPLEX_VEC_ZONES {
            counts_per_zone[z] += 4;
        }

        let mut beta = [F32x4::zero(); NUM_DUPLEX_VECS];
        let mut c_audio = [F32x4::zero(); NUM_DUPLEX_VECS];
        let mut r_dup = [0.0_f64; NUM_BRIDGE_ZONES];
        let mut seen_in_zone = [0usize; NUM_BRIDGE_ZONES];

        for (vec_idx, &z) in DUPLEX_VEC_ZONES.iter().enumerate() {
            let total_in_z = counts_per_zone[z];
            let beta_mag = if gain > 0.0 {
                let per_mode = 2.0 * DUPLEX_TARGET_R_PER_ZONE * ts / (total_in_z as f64);
                gain * crate::math::sqrt_f64(per_mode)
            } else {
                0.0
            };

            let (mut b_arr, mut c_arr) = ([0.0_f32; 4], [0.0_f32; 4]);
            for lane in 0..4 {
                let m_in_z = seen_in_zone[z];
                seen_in_zone[z] += 1;
                let sign = if (m_in_z & 1) == 0 { 1.0 } else { -1.0 };
                let b_val = (sign * beta_mag) as f32;
                b_arr[lane] = b_val;
                c_arr[lane] = (sign * f64::from(DUPLEX_AUDIO_GAIN)) as f32;
                r_dup[z] += 0.5 * alpha_dup * f64::from(b_val) * f64::from(b_val);
            }
            beta[vec_idx] = F32x4::from_array(b_arr);
            c_audio[vec_idx] = F32x4::from_array(c_arr);
        }

        let default_voicing = VoicingSpec::default();
        let mut bank = Self {
            u: [F32x4::zero(); NUM_DUPLEX_VECS],
            v: [F32x4::zero(); NUM_DUPLEX_VECS],
            d: [F32x4::splat(1.0); NUM_DUPLEX_VECS],
            s: [F32x4::zero(); NUM_DUPLEX_VECS],
            beta,
            c_audio,
            freqs_hz: [0.0_f32; NUM_DUPLEX_MODES],
            r_dup,
            alpha_dup,
            ts,
            sample_rate_f64: safe_sr,
            lossless,
            voicing: default_voicing.clone(),
            tuning_initialized: false,
            last_stereo: (0.0, 0.0),
            active: false,
        };
        bank.retune_voicing(&default_voicing);
        bank
    }

    /// Retunes all 64 non-speaking duplex wire-segment resonances in place for `voicing`
    /// (`a4_freq_hz`, Railsback `stretch_amount`, `string_length_scale`, and per-note `pitch_cents`)
    /// with zero heap allocations. No-op when tuning fields are unchanged.
    pub fn retune_voicing(&mut self, voicing: &VoicingSpec) {
        let v = voicing.sanitized();
        if self.tuning_initialized && tuning_matches(&v, &self.voicing) {
            return;
        }
        self.voicing = v.clone();
        self.tuning_initialized = true;

        let specs = build_scale_design(&v);
        let stretch_table = railsback_stretch_cents_table(&specs);
        let stretch_scale = v.effective_stretch_amount() as f64;
        let cents_to_ln = LN_2 / 1200.0;
        let max_nyq_safe = (0.47 * self.sample_rate_f64).min(12_500.0);

        for vec_idx in 0..NUM_DUPLEX_VECS {
            let (mut d_arr, mut s_arr) = ([1.0_f32; 4], [0.0_f32; 4]);
            for lane in 0..4 {
                let mode_idx = vec_idx * 4 + lane;
                let (key_u8, mult_u8, detune) = DUPLEX_MODE_LAYOUT[mode_idx];
                let k = usize::from(key_u8).min(NUM_KEYS - 1);
                let cents =
                    stretch_table[k] * stretch_scale + f64::from(v.effective_pitch_cents(k));
                let f0_tuned = if cents == 0.0 {
                    specs[k].f0_et_hz
                } else {
                    specs[k].f0_et_hz * exp_f64(cents * cents_to_ln)
                };
                let ratio = f64::from(mult_u8) * (1.0 + detune);
                let f_hz = (f0_tuned * ratio).clamp(1_350.0_f64.min(max_nyq_safe), max_nyq_safe);
                self.freqs_hz[mode_idx] = f_hz as f32;

                let t60 = if self.lossless {
                    f64::INFINITY
                } else {
                    0.55 / (1.0 + 0.12 * (f_hz * 1.0e-3))
                };
                if let Some((df, sf)) = coupled_pole_f32(f_hz, self.sample_rate_f64, t60) {
                    d_arr[lane] = df;
                    s_arr[lane] = sf;
                }
            }
            self.d[vec_idx] = F32x4::from_array(d_arr);
            self.s[vec_idx] = F32x4::from_array(s_arr);
        }
    }

    /// Retunes all 64 non-speaking duplex wire-segment resonances in place for `a4_freq_hz`
    /// (`[415.0, 466.0]` Hz) with zero heap allocations. No-op when `a4_freq_hz` is unchanged.
    pub fn retune_a4(&mut self, a4_freq_hz: f32) {
        let mut next = self.voicing.clone();
        next.a4_freq_hz = a4_freq_hz;
        self.retune_voicing(&next);
    }

    /// Returns the currently configured concert pitch `A4` reference in Hz.
    #[must_use]
    #[inline]
    pub const fn a4_freq_hz(&self) -> f32 {
        self.voicing.a4_freq_hz
    }

    /// Returns the active [`VoicingSpec`] used to tune the duplex bank.
    #[must_use]
    #[inline]
    pub const fn voicing(&self) -> &VoicingSpec {
        &self.voicing
    }

    /// Flushes `< 1e-20` denormals/subnormals and non-finite resonator states to `0.0`,
    /// clearing `active` so Stage 3 idle bypass re-engages cleanly once decayed.
    pub fn flush_subnormals(&mut self) {
        if !self.active {
            return;
        }
        let mut any_nz = false;
        for vec_idx in 0..NUM_DUPLEX_VECS {
            let mut ua = self.u[vec_idx].to_array();
            let mut va = self.v[vec_idx].to_array();
            for lane in 0..4 {
                if !ua[lane].is_finite() || ua[lane].abs() < SUBNORMAL_FLUSH_THRESHOLD {
                    ua[lane] = 0.0;
                } else {
                    any_nz = true;
                }
                if !va[lane].is_finite() || va[lane].abs() < SUBNORMAL_FLUSH_THRESHOLD {
                    va[lane] = 0.0;
                } else {
                    any_nz = true;
                }
            }
            self.u[vec_idx] = F32x4::from_array(ua);
            self.v[vec_idx] = F32x4::from_array(va);
        }
        if !any_nz {
            self.active = false;
            self.last_stereo = (0.0, 0.0);
        }
    }

    /// Resets all 64 duplex resonator states and cached stereo output to zero.
    pub fn reset(&mut self) {
        self.u = [F32x4::zero(); NUM_DUPLEX_VECS];
        self.v = [F32x4::zero(); NUM_DUPLEX_VECS];
        self.last_stereo = (0.0, 0.0);
        self.active = false;
    }

    /// Returns `true` while all 64 duplex modes remain strictly unexcited at zero.
    #[must_use]
    #[inline]
    pub const fn is_quiescent(&self) -> bool {
        !self.active
    }

    /// Returns passive WDF port resistances on the 8 bridge zones (`0.0` on bass Zones 0..2).
    #[must_use]
    #[inline]
    pub const fn port_resistances(&self) -> &[f64; NUM_BRIDGE_ZONES] {
        &self.r_dup
    }

    /// Computes instantaneous free forces `[N]` exerted by duplex modes onto Zones 2..7.
    #[must_use]
    #[inline]
    pub fn free_forces(&self) -> [f64; NUM_BRIDGE_ZONES] {
        let mut f_free = [0.0_f64; NUM_BRIDGE_ZONES];
        if !self.active {
            return f_free;
        }
        let mut z_acc = [F32x4::zero(); 6];
        for (vec_idx, &z) in DUPLEX_VEC_ZONES.iter().enumerate() {
            let slot = z - TREBLE_DUPLEX_ZONE_START;
            z_acc[slot] = self.v[vec_idx].mul_add(self.beta[vec_idx], z_acc[slot]);
        }
        for (slot, acc) in z_acc.iter().enumerate() {
            f_free[TREBLE_DUPLEX_ZONE_START + slot] = self.alpha_dup * f64::from(acc.reduce_sum());
        }
        f_free
    }

    /// Steps all 64 SIMD duplex modes coupled to solved bridge zone velocities `v_zone[2..8]`,
    /// writes finite per-zone duplex contributions into `dup_zone_out` (with bass Zones `0..2`
    /// strictly `0.0`), and returns the new `(left, right)` stereo output.
    #[inline]
    pub fn step_with_zone_velocities_and_zones(
        &mut self,
        v_zone: &[f32; NUM_BRIDGE_ZONES],
        dup_zone_out: &mut [f32; NUM_BRIDGE_ZONES],
    ) -> (f32, f32) {
        if !self.active
            && v_zone[TREBLE_DUPLEX_ZONE_START..]
                .iter()
                .all(|&vz| vz == 0.0)
        {
            dup_zone_out.fill(0.0);
            self.last_stereo = (0.0, 0.0);
            return (0.0, 0.0);
        }
        self.active = true;
        let mut v_br_z = [F32x4::zero(); 6];
        for (slot, &vz) in v_zone[TREBLE_DUPLEX_ZONE_START..].iter().enumerate() {
            v_br_z[slot] = F32x4::splat(if vz.is_finite() { vz } else { 0.0 });
        }
        let mut zone_vec_acc = [F32x4::zero(); 6];
        for (vec_idx, &z) in DUPLEX_VEC_ZONES.iter().enumerate() {
            let slot = z - TREBLE_DUPLEX_ZONE_START;
            let v_coupled = self.v[vec_idx] - self.beta[vec_idx] * v_br_z[slot];

            let u0 = self.u[vec_idx];
            let d0 = self.d[vec_idx];
            let s0 = self.s[vec_idx];
            let du = d0.mul_add(u0, s0 * v_coupled);
            let dv = s0 * u0 - d0 * v_coupled;
            let un = u0 - du;
            let vn = v_coupled + dv;
            self.u[vec_idx] = un;
            self.v[vec_idx] = vn;

            zone_vec_acc[slot] = vn.mul_add(self.c_audio[vec_idx], zone_vec_acc[slot]);
        }

        dup_zone_out[..TREBLE_DUPLEX_ZONE_START].fill(0.0);
        let (mut out_l, mut out_r) = (0.0_f32, 0.0_f32);
        for (slot, acc_v) in zone_vec_acc.iter().enumerate() {
            let z = TREBLE_DUPLEX_ZONE_START + slot;
            let acc = acc_v.reduce_sum();
            let safe_acc = if acc.is_finite() { acc } else { 0.0 };
            dup_zone_out[z] = safe_acc;
            let (pan_l, pan_r) = zone_stereo_pan(z);
            out_l += safe_acc * pan_l;
            out_r += safe_acc * pan_r;
        }
        self.last_stereo = (out_l, out_r);
        (out_l, out_r)
    }

    /// Steps all 64 SIMD duplex modes coupled to solved bridge zone velocities `v_zone[2..8]`
    /// and returns the new `(left, right)` stereo output.
    #[inline]
    pub fn step_with_zone_velocities(&mut self, v_zone: &[f32; NUM_BRIDGE_ZONES]) -> (f32, f32) {
        let mut unused_zones = [0.0_f32; NUM_BRIDGE_ZONES];
        self.step_with_zone_velocities_and_zones(v_zone, &mut unused_zones)
    }

    /// Returns the most recent `(left, right)` stereo duplex output.
    #[must_use]
    #[inline]
    pub const fn stereo_out(&self) -> (f32, f32) {
        self.last_stereo
    }

    /// Returns total instantaneous duplex modal energy `E_duplex >= 0` in Joules.
    #[must_use]
    pub fn energy_joules(&self) -> f64 {
        let mut sum_sq = 0.0_f64;
        for vec_idx in 0..NUM_DUPLEX_VECS {
            let ua = self.u[vec_idx].to_array();
            let va = self.v[vec_idx].to_array();
            for lane in 0..4 {
                let u64v = f64::from(ua[lane]);
                let v64v = f64::from(va[lane]);
                sum_sq += u64v * u64v + v64v * v64v;
            }
        }
        0.5 * self.alpha_dup * self.ts * sum_sq
    }

    /// Returns analytical resonant frequency in Hz for mode `mode_idx` (`0..64`).
    #[must_use]
    pub fn mode_freq_hz(&self, mode_idx: usize) -> f32 {
        self.freqs_hz.get(mode_idx).copied().unwrap_or(0.0)
    }
}
