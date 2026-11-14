//! Precomputed per-string SAV / phantom slope and stiffness constants (`no_std`).

use core::f64::consts::{PI, TAU};

use piano_simd::F32x4;
use piano_spec::{
    effective_string_f0_hz, partial_frequency_hz, Polarization, StringPhysicalParams, VoicingSpec,
    NUM_KEYS, STEEL_YOUNGS_MODULUS_PA,
};

use crate::choir_map::StringMapEntry;
use crate::longitudinal::{
    long_init_freq_hz, nonlinear_register_weight, w_phantom, LONG_AA_CUTOFF_HZ,
};
use crate::modal_soa::NYQUIST_CUTOFF_RATIO;
use crate::phantom::{
    slope_factor, MAX_PHANTOM_LONG_MODES, MAX_PHANTOM_LONG_VECS, MAX_PHANTOM_TRANSVERSE_MODES,
    MAX_PHANTOM_TRANSVERSE_VECS,
};

fn pack_vec4<const N: usize>(vals: &[f64]) -> [F32x4; N] {
    let mut out = [F32x4::zero(); N];
    for (i, dst) in out.iter_mut().enumerate() {
        let base = 4 * i;
        let lane = |k: usize| vals.get(base + k).copied().unwrap_or(0.0) as f32;
        *dst = F32x4::from_array([lane(0), lane(1), lane(2), lane(3)]);
    }
    out
}

fn build_mask4<const N: usize>(count: usize) -> [F32x4; N] {
    let mut out = [F32x4::zero(); N];
    for (i, dst) in out.iter_mut().enumerate() {
        let base = 4 * i;
        let bit = |k: usize| if base + k < count { 1.0_f32 } else { 0.0_f32 };
        *dst = F32x4::from_array([bit(0), bit(1), bit(2), bit(3)]);
    }
    out
}

fn build_scaled_mask4<const N: usize>(count: usize, scale: f64) -> [F32x4; N] {
    let s = scale as f32;
    let mut out = [F32x4::zero(); N];
    for (i, dst) in out.iter_mut().enumerate() {
        let base = 4 * i;
        let bit = |k: usize| if base + k < count { s } else { 0.0_f32 };
        *dst = F32x4::from_array([bit(0), bit(1), bit(2), bit(3)]);
    }
    out
}

fn fill_pol_slopes(
    modes: usize,
    f0: f64,
    nyquist: f64,
    spec: &StringPhysicalParams,
    gamma_s: f64,
) -> (
    [f64; MAX_PHANTOM_TRANSVERSE_MODES],
    [f64; MAX_PHANTOM_TRANSVERSE_MODES],
    usize,
) {
    let mut sig = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
    let mut omsig = [0.0_f64; MAX_PHANTOM_TRANSVERSE_MODES];
    let mut n_c = 0usize;
    for j in 1..=modes.min(MAX_PHANTOM_TRANSVERSE_MODES) {
        let f_j = partial_frequency_hz(f0, spec.inharmonicity_b, j);
        if f_j > 0.0 && f_j < nyquist {
            let om = TAU * f_j;
            let s = slope_factor(j, spec.speaking_length_m, gamma_s, om);
            sig[j - 1] = s;
            omsig[j - 1] = om * s;
            n_c = j;
        }
    }
    (sig, omsig, n_c)
}

#[derive(Clone, Debug)]
pub(crate) struct SavEnergySlopes {
    pub nv_c: usize,
    pub nh_c: usize,
    pub sigma_v: [f64; MAX_PHANTOM_TRANSVERSE_MODES],
    pub sigma_h: [f64; MAX_PHANTOM_TRANSVERSE_MODES],
    pub alpha_l: [f64; MAX_PHANTOM_LONG_MODES],
}

impl SavEnergySlopes {
    #[must_use]
    pub fn zeroed() -> Self {
        Self {
            nv_c: 0,
            nh_c: 0,
            sigma_v: [0.0; MAX_PHANTOM_TRANSVERSE_MODES],
            sigma_h: [0.0; MAX_PHANTOM_TRANSVERSE_MODES],
            alpha_l: [0.0; MAX_PHANTOM_LONG_MODES],
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SavStringSpec {
    pub n_vh: usize,
    pub m_ph: usize,
    pub nv_vecs: usize,
    pub nh_vecs: usize,
    pub nl_vecs: usize,
    pub length_m: f64,
    pub quarter_l: f64,
    pub kappa_0: f64,
    pub kappa_ph: f64,
    pub sigma_v4: [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    pub omega_sigma_v4: [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    pub mask_v4: [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    pub sigma_h4: [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    pub omega_sigma_h4: [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    pub mask_h4: [F32x4; MAX_PHANTOM_TRANSVERSE_VECS],
    pub alpha_l4: [F32x4; MAX_PHANTOM_LONG_VECS],
    pub half_ql_mask4: [F32x4; MAX_PHANTOM_LONG_VECS],
    pub mask_l4: [F32x4; MAX_PHANTOM_LONG_VECS],
    pub omega_alpha_l_norm4: [F32x4; MAX_PHANTOM_LONG_VECS],
}

impl SavStringSpec {
    #[must_use]
    pub fn zeroed() -> Self {
        let zt = [F32x4::zero(); MAX_PHANTOM_TRANSVERSE_VECS];
        let zl = [F32x4::zero(); MAX_PHANTOM_LONG_VECS];
        Self {
            n_vh: 0,
            m_ph: 0,
            nv_vecs: 0,
            nh_vecs: 0,
            nl_vecs: 0,
            length_m: 0.0,
            quarter_l: 0.0,
            kappa_0: 0.0,
            kappa_ph: 0.0,
            sigma_v4: zt,
            omega_sigma_v4: zt,
            mask_v4: zt,
            sigma_h4: zt,
            omega_sigma_h4: zt,
            mask_h4: zt,
            alpha_l4: zl,
            half_ql_mask4: zl,
            mask_l4: zl,
            omega_alpha_l_norm4: zl,
        }
    }

    #[must_use]
    pub fn build(
        entry: &StringMapEntry,
        sample_rate: f64,
        voicing: &VoicingSpec,
        specs: &[StringPhysicalParams; NUM_KEYS],
        gamma_s: f64,
        beta_s: f64,
    ) -> (Self, SavEnergySlopes) {
        let k = entry.key_idx as usize;
        let w_reg = nonlinear_register_weight(k);
        if w_reg == 0.0 {
            return (Self::zeroed(), SavEnergySlopes::zeroed());
        }
        let s_choir = entry.string_in_choir as usize;
        let spec = specs[k];
        let length_m = spec.speaking_length_m;
        let nyquist = NYQUIST_CUTOFF_RATIO * sample_rate;
        let f0_v = effective_string_f0_hz(k, s_choir, Polarization::Vertical, specs, voicing);
        let f0_h = effective_string_f0_hz(k, s_choir, Polarization::Horizontal, specs, voicing);
        let s_core = PI * spec.core_radius_m * spec.core_radius_m;
        let kappa_0 = if length_m > 0.0 && beta_s > 0.0 {
            w_reg * STEEL_YOUNGS_MODULUS_PA * s_core / (beta_s * length_m)
        } else {
            0.0
        };

        let (sigma_v, omega_sigma_v, nv_c) =
            fill_pol_slopes(entry.vert_modes, f0_v, nyquist, &spec, gamma_s);
        let (sigma_h, omega_sigma_h, nh_c) =
            fill_pol_slopes(entry.horiz_modes, f0_h, nyquist, &spec, gamma_s);

        let quarter_l = 0.25 * length_m;
        let inv_q = if quarter_l > 0.0 {
            1.0 / quarter_l
        } else {
            0.0
        };
        let mut alpha_l = [0.0_f64; MAX_PHANTOM_LONG_MODES];
        let mut omega_alpha_l_norm = [0.0_f64; MAX_PHANTOM_LONG_MODES];
        let mut m_ph = 0usize;
        let c_l = 2.0 * length_m * spec.f_long_hz;
        let alpha_pref = if gamma_s > 0.0 && c_l > 0.0 {
            -length_m / (2.0 * gamma_s * c_l)
        } else {
            0.0
        };
        for m in 1..=entry.long_modes.min(MAX_PHANTOM_LONG_MODES) {
            let f_nom = (m as f64) * spec.f_long_hz;
            let f_init = long_init_freq_hz(k, spec.f_long_hz, m);
            if f_init > 0.0 && f_init < nyquist && f_nom < LONG_AA_CUTOFF_HZ {
                m_ph = m;
                let al = alpha_pref * w_phantom(f_nom);
                alpha_l[m - 1] = al;
                omega_alpha_l_norm[m - 1] = (TAU * f_init) * al * inv_q;
            }
        }

        (
            Self {
                n_vh: nv_c.max(nh_c),
                m_ph,
                nv_vecs: nv_c.div_ceil(4).min(entry.vert_vecs.len()),
                nh_vecs: nh_c.div_ceil(4).min(entry.horiz_vecs.len()),
                nl_vecs: m_ph.div_ceil(4).min(entry.long_vecs.len()),
                length_m,
                quarter_l,
                kappa_0,
                kappa_ph: 2.0 * kappa_0,
                sigma_v4: pack_vec4(&sigma_v),
                omega_sigma_v4: pack_vec4(&omega_sigma_v),
                mask_v4: build_mask4(nv_c),
                sigma_h4: pack_vec4(&sigma_h),
                omega_sigma_h4: pack_vec4(&omega_sigma_h),
                mask_h4: build_mask4(nh_c),
                alpha_l4: pack_vec4(&alpha_l),
                half_ql_mask4: build_scaled_mask4(m_ph, 0.5 * quarter_l),
                mask_l4: build_mask4(m_ph),
                omega_alpha_l_norm4: pack_vec4(&omega_alpha_l_norm),
            },
            SavEnergySlopes {
                nv_c,
                nh_c,
                sigma_v,
                sigma_h,
                alpha_l,
            },
        )
    }
}
