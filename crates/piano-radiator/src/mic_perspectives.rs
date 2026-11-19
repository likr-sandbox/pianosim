//! 3D physical Green's function projection from 9 physical radiator zones
//! (`preset.zone_pos_m[0..9]`) to `(L, R)` microphone/ear channels.
//!
//! Strictly FTO #5 compliant: every zone projects independently from its own
//! 3D coordinates without group-summed shared post-delays.

use piano_spec::{BodyPreset, LidState, ListenerPerspective, PRESET_RADIATOR_ZONES};

use crate::binaural_hrtf::{
    compute_binaural_params, BinauralEarParams, Biquad, SPEED_OF_SOUND_MPS,
};
use crate::math::{asin_f64, atan2_f64, clamp_f64, max_f64, min_f64, sin_cos_f64, sqrt_f64};

/// Acoustic path descriptors for `(L, R)` projections of one radiator zone.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChannelPathSpec {
    /// Sample delay relative to the nearer ear/mic of this zone (`0.0` on nearer channel).
    pub delay_samples: f64,
    /// Specular lid mirror reflection sample delay (`0.0` if disabled).
    pub lid_refl_delay_samples: f64,
    /// Specular lid reflection weight relative to direct modal radiation.
    pub lid_refl_gain: f32,
    /// Secondary hall boundary reflection delay (`ConcertHall` only, `0.0` otherwise).
    pub hall_refl_delay_samples: f64,
    /// Secondary hall boundary reflection relative weight.
    pub hall_refl_gain: f32,
    /// Weight applied to `preset.direct_zone_gain[z]` for `h[0]` direct radiation.
    pub direct_gain: f32,
    /// Spatial projection weights for `BodyModeRecord::rad_coeffs[0..5]`.
    pub mode_proj: [f32; 5],
    /// Air absorption shelf.
    pub air_filter: Biquad,
    /// Lid baffle occlusion lowpass.
    pub lid_lowpass: Biquad,
    /// Lid baffle HF shelf.
    pub lid_shelf: Biquad,
    /// Binaural HRTF ear filters (identity for non-binaural perspectives).
    pub hrtf_head_shadow: Biquad,
    pub hrtf_concha: Biquad,
    pub hrtf_pinna_notch: Biquad,
}

/// Pair of stereo acoustic path descriptors `(left, right)` for one zone.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ZonePathPair {
    pub left: ChannelPathSpec,
    pub right: ChannelPathSpec,
}

#[derive(Clone, Copy)]
struct PianoBounds {
    x_min: f64,
    x_center: f64,
    y_front: f64,
    z_board: f64,
}

#[derive(Clone, Copy)]
struct CapsuleRig {
    pos: [f64; 3],
    aim: [f64; 3],
    cardioid: f64,
    r_ref: f64,
    persp_trim: f64,
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(v: [f64; 3]) -> f64 {
    sqrt_f64(dot3(v, v))
}

#[inline]
fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = norm3(v);
    if n < 1.0e-12 {
        [0.0, 1.0, 0.0]
    } else {
        let inv = 1.0 / n;
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }
}

fn piano_bounds(preset: &BodyPreset) -> PianoBounds {
    let mut x_min = preset.zone_pos_m[0][0] as f64;
    let mut x_max = x_min;
    let mut y_min = preset.zone_pos_m[0][1] as f64;
    let mut z_sum = 0.0_f64;
    let mut z = 0;
    while z < 8 {
        let px = preset.zone_pos_m[z][0] as f64;
        let py = preset.zone_pos_m[z][1] as f64;
        let pz = preset.zone_pos_m[z][2] as f64;
        x_min = min_f64(x_min, px);
        x_max = max_f64(x_max, px);
        y_min = min_f64(y_min, py);
        z_sum += pz;
        z += 1;
    }
    PianoBounds {
        x_min,
        x_center: 0.5 * (x_min + x_max),
        y_front: y_min,
        z_board: z_sum / 8.0,
    }
}

fn stereo_rigs(perspective: ListenerPerspective, b: &PianoBounds) -> (CapsuleRig, CapsuleRig) {
    match perspective {
        ListenerPerspective::Player => (
            CapsuleRig {
                pos: [b.x_center - 0.17, b.y_front - 0.30, b.z_board + 0.42],
                aim: normalize3([-0.58, 0.75, -0.32]),
                cardioid: 0.52,
                r_ref: 0.90,
                persp_trim: 1.25,
            },
            CapsuleRig {
                pos: [b.x_center + 0.17, b.y_front - 0.30, b.z_board + 0.42],
                aim: normalize3([0.58, 0.75, -0.32]),
                cardioid: 0.52,
                r_ref: 0.90,
                persp_trim: 1.25,
            },
        ),
        ListenerPerspective::CloseStereo => (
            CapsuleRig {
                pos: [b.x_center - 0.25, b.y_front + 0.42, b.z_board + 0.48],
                aim: normalize3([-0.24, 0.28, -0.93]),
                cardioid: 0.50,
                r_ref: 0.72,
                persp_trim: 1.05,
            },
            CapsuleRig {
                pos: [b.x_center + 0.28, b.y_front + 0.42, b.z_board + 0.48],
                aim: normalize3([0.24, 0.28, -0.93]),
                cardioid: 0.50,
                r_ref: 0.72,
                persp_trim: 1.05,
            },
        ),
        ListenerPerspective::ConcertHall => (
            CapsuleRig {
                pos: [b.x_center - 0.42, b.y_front - 1.55, b.z_board + 0.95],
                aim: normalize3([0.14, 0.95, -0.28]),
                cardioid: 0.24,
                r_ref: 1.90,
                persp_trim: 1.32,
            },
            CapsuleRig {
                pos: [b.x_center + 0.78, b.y_front - 1.35, b.z_board + 0.95],
                aim: normalize3([-0.18, 0.94, -0.28]),
                cardioid: 0.24,
                r_ref: 1.90,
                persp_trim: 1.32,
            },
        ),
        ListenerPerspective::BinauralHrtf => (
            CapsuleRig {
                pos: [b.x_center - 0.0875, b.y_front - 0.38, b.z_board + 0.45],
                aim: normalize3([-0.25, 0.91, -0.32]),
                cardioid: 0.12,
                r_ref: 0.92,
                persp_trim: 1.00,
            },
            CapsuleRig {
                pos: [b.x_center + 0.0875, b.y_front - 0.38, b.z_board + 0.45],
                aim: normalize3([0.25, 0.91, -0.32]),
                cardioid: 0.12,
                r_ref: 0.92,
                persp_trim: 1.00,
            },
        ),
    }
}

/// Mirror `zone_pos` across the grand piano lid plane hinged on the bass `-x` rim.
fn reflect_across_lid(zone_pos: [f64; 3], b: &PianoBounds, lid_state: LidState) -> [f64; 3] {
    let angle_rad = match lid_state {
        LidState::Closed => 0.02,
        LidState::Half => 0.31, // ~18 deg
        LidState::Open => 0.63, // ~36 deg
    };
    let (sin_a, cos_a) = sin_cos_f64(angle_rad);
    // Outward normal from lid underside pointing into the case (`-x`, `-z`).
    let normal = [-sin_a, 0.0, -cos_a];
    let hinge = [b.x_min - 0.05, zone_pos[1], b.z_board + 0.14];
    let rel = sub3(zone_pos, hinge);
    let signed_dist = dot3(rel, normal);
    [
        zone_pos[0] - 2.0 * signed_dist * normal[0],
        zone_pos[1],
        zone_pos[2] - 2.0 * signed_dist * normal[2],
    ]
}

fn lid_occlusion_filters(lid_state: LidState, zone: usize, sample_rate: f64) -> (Biquad, Biquad) {
    let is_keybed = zone == 8;
    match lid_state {
        LidState::Open => (Biquad::identity(), Biquad::identity()),
        LidState::Half => {
            if is_keybed {
                (
                    Biquad::lowpass_2nd(8_800.0, 0.707, sample_rate),
                    Biquad::high_shelf(2_200.0, -2.5, 0.85, sample_rate),
                )
            } else {
                (
                    Biquad::lowpass_2nd(5_800.0, 0.707, sample_rate),
                    Biquad::high_shelf(1_800.0, -4.5, 0.85, sample_rate),
                )
            }
        }
        LidState::Closed => {
            if is_keybed {
                (
                    Biquad::lowpass_2nd(2_600.0, 0.707, sample_rate),
                    Biquad::high_shelf(1_300.0, -4.5, 0.85, sample_rate),
                )
            } else {
                (
                    Biquad::lowpass_2nd(1_150.0, 0.68, sample_rate),
                    Biquad::high_shelf(850.0, -9.5, 0.85, sample_rate),
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_channel_spec(
    zone: usize,
    zone_pos: [f64; 3],
    rig: &CapsuleRig,
    r_dir: f64,
    r_near: f64,
    img_pos: [f64; 3],
    b: &PianoBounds,
    perspective: ListenerPerspective,
    lid_state: LidState,
    hrtf: Option<&BinauralEarParams>,
    sample_rate: f64,
) -> ChannelPathSpec {
    let delta = sub3(rig.pos, zone_pos);
    let u_hat = normalize3(delta);
    let cos_polar = dot3([-u_hat[0], -u_hat[1], -u_hat[2]], rig.aim);
    let polar_w = max_f64(0.18, (1.0 - rig.cardioid) + rig.cardioid * cos_polar);

    let lid_open_f = lid_state.openness_f32() as f64;
    let inv_dist = rig.r_ref / max_f64(r_dir, 0.22);
    let base_gain = rig.persp_trim * inv_dist * polar_w;

    let delay_samples = if let Some(ear) = hrtf {
        ear.delay_samples
    } else {
        max_f64(0.0, ((r_dir - r_near) / SPEED_OF_SOUND_MPS) * sample_rate)
    };

    let (lid_refl_delay_samples, lid_refl_gain) = match lid_state {
        LidState::Closed => (0.0, 0.0_f32),
        LidState::Half | LidState::Open => {
            let r_refl = norm3(sub3(rig.pos, img_pos));
            let dt = max_f64(
                0.0001 * sample_rate,
                ((max_f64(r_refl, r_near) - r_near) / SPEED_OF_SOUND_MPS) * sample_rate
                    + 0.0001 * sample_rate,
            );
            let scale = if matches!(lid_state, LidState::Open) {
                0.34
            } else {
                0.18
            };
            let g = scale * polar_w * (rig.r_ref / max_f64(r_refl, 0.28));
            (dt, g as f32)
        }
    };

    let (hall_refl_delay_samples, hall_refl_gain) =
        if matches!(perspective, ListenerPerspective::ConcertHall) {
            let floor_img = [zone_pos[0], zone_pos[1], b.z_board - 1.45];
            let r_floor = norm3(sub3(rig.pos, floor_img));
            let dt = max_f64(
                0.0002 * sample_rate,
                ((max_f64(r_floor, r_near) - r_near) / SPEED_OF_SOUND_MPS) * sample_rate
                    + 0.0002 * sample_rate,
            );
            let g = 0.16 * (0.45 + 0.55 * lid_open_f) * (rig.r_ref / max_f64(r_floor, 0.5));
            (dt, g as f32)
        } else {
            (0.0, 0.0_f32)
        };

    let (lid_lowpass, lid_shelf) = lid_occlusion_filters(lid_state, zone, sample_rate);
    let air_db = -0.65 * max_f64(0.0, r_dir - 0.3);
    let air_filter = Biquad::high_shelf(6_500.0, air_db, 0.75, sample_rate);

    let lid_mode_weight = 0.20 + 0.80 * lid_open_f;
    let mode_proj = [
        base_gain as f32,
        (base_gain * 0.45 * u_hat[0]) as f32,
        (base_gain * 0.45 * u_hat[1]) as f32,
        (base_gain * 0.45 * u_hat[2]) as f32,
        (base_gain * lid_mode_weight) as f32,
    ];

    let direct_gain = (base_gain * 0.65) as f32;

    let (hrtf_head_shadow, hrtf_concha, hrtf_pinna_notch) = match hrtf {
        Some(e) => (e.head_shadow, e.concha, e.pinna_notch),
        None => (Biquad::identity(), Biquad::identity(), Biquad::identity()),
    };

    ChannelPathSpec {
        delay_samples,
        lid_refl_delay_samples,
        lid_refl_gain,
        hall_refl_delay_samples,
        hall_refl_gain,
        direct_gain,
        mode_proj,
        air_filter,
        lid_lowpass,
        lid_shelf,
        hrtf_head_shadow,
        hrtf_concha,
        hrtf_pinna_notch,
    }
}

/// Compute 3D physical acoustic Green's function paths `(left, right)` for all 9 zones.
pub(crate) fn compute_all_zone_paths(
    preset: &BodyPreset,
    perspective: ListenerPerspective,
    lid_state: LidState,
    sample_rate: f64,
) -> [ZonePathPair; PRESET_RADIATOR_ZONES] {
    let b = piano_bounds(preset);
    let (rig_l, rig_r) = stereo_rigs(perspective, &b);
    let head_center = [b.x_center, b.y_front - 0.38, b.z_board + 0.45];
    let head_fwd = normalize3([0.0, 0.92, -0.39]);
    let head_right = [1.0, 0.0, 0.0];
    // Right-handed head basis up = right x forward
    let head_up = [
        head_right[1] * head_fwd[2] - head_right[2] * head_fwd[1],
        head_right[2] * head_fwd[0] - head_right[0] * head_fwd[2],
        head_right[0] * head_fwd[1] - head_right[1] * head_fwd[0],
    ];

    let default_pair = ZonePathPair {
        left: build_channel_spec(
            0,
            [0.0; 3],
            &rig_l,
            1.0,
            1.0,
            [0.0; 3],
            &b,
            perspective,
            lid_state,
            None,
            sample_rate,
        ),
        right: build_channel_spec(
            0,
            [0.0; 3],
            &rig_r,
            1.0,
            1.0,
            [0.0; 3],
            &b,
            perspective,
            lid_state,
            None,
            sample_rate,
        ),
    };
    let mut out = [default_pair; PRESET_RADIATOR_ZONES];

    let mut z = 0;
    while z < PRESET_RADIATOR_ZONES {
        let pos = [
            preset.zone_pos_m[z][0] as f64,
            preset.zone_pos_m[z][1] as f64,
            preset.zone_pos_m[z][2] as f64,
        ];
        let r_l = norm3(sub3(rig_l.pos, pos));
        let r_r = norm3(sub3(rig_r.pos, pos));
        let r_near = min_f64(r_l, r_r);
        let img = reflect_across_lid(pos, &b, lid_state);

        let binaural = if matches!(perspective, ListenerPerspective::BinauralHrtf) {
            let rel = sub3(pos, head_center);
            let dx_loc = dot3(rel, head_right);
            let dy_loc = dot3(rel, head_fwd);
            let dz_loc = dot3(rel, head_up);
            let r_h = max_f64(norm3(rel), 0.15);
            let az = atan2_f64(dx_loc, dy_loc);
            let el = asin_f64(clamp_f64(dz_loc / r_h, -1.0, 1.0));
            Some(compute_binaural_params(az, el, sample_rate))
        } else {
            None
        };

        out[z] = ZonePathPair {
            left: build_channel_spec(
                z,
                pos,
                &rig_l,
                r_l,
                r_near,
                img,
                &b,
                perspective,
                lid_state,
                binaural.as_ref().map(|p| &p.left),
                sample_rate,
            ),
            right: build_channel_spec(
                z,
                pos,
                &rig_r,
                r_r,
                r_near,
                img,
                &b,
                perspective,
                lid_state,
                binaural.as_ref().map(|p| &p.right),
                sample_rate,
            ),
        };
        z += 1;
    }

    out
}
