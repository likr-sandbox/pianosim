//! Parameter smoothing and spatial crossfade transitions for real-time engine control.

use core::f32::consts::PI;
use piano_spec::voicing::{
    HAMMER_HARDNESS_DEFAULT, HAMMER_HARDNESS_MAX, HAMMER_HARDNESS_MIN, STRETCH_AMOUNT_DEFAULT,
    STRETCH_AMOUNT_MAX, STRETCH_AMOUNT_MIN, STRING_LENGTH_SCALE_DEFAULT, STRING_LENGTH_SCALE_MAX,
    STRING_LENGTH_SCALE_MIN, UNISON_DETUNE_SCALE_DEFAULT, UNISON_DETUNE_SCALE_MAX,
    UNISON_DETUNE_SCALE_MIN,
};
use piano_spec::{LidState, ListenerPerspective, VoicingSpec};

/// Time constant for voicing parameter smoothing (`20 ms`).
pub const VOICING_SMOOTH_TAU_S: f32 = 0.020;
/// Convergence epsilon tolerance below which smoothed parameters snap to target (`1e-4`).
pub const VOICING_SETTLE_TOL: f32 = 1.0e-4;
/// Duration of the raised-cosine crossfade when changing listener perspective or lid (`8 ms`).
pub const SPACE_CROSSFADE_S: f32 = 0.008;
/// Signal envelope threshold below which spatial switching occurs instantly without crossfade.
pub const SPACE_SILENCE_ENV_THRESHOLD: f32 = 1.0e-7;

const DEFAULTS: [f32; 4] = [
    HAMMER_HARDNESS_DEFAULT,
    UNISON_DETUNE_SCALE_DEFAULT,
    STRETCH_AMOUNT_DEFAULT,
    STRING_LENGTH_SCALE_DEFAULT,
];
const MINS: [f32; 4] = [
    HAMMER_HARDNESS_MIN,
    UNISON_DETUNE_SCALE_MIN,
    STRETCH_AMOUNT_MIN,
    STRING_LENGTH_SCALE_MIN,
];
const MAXS: [f32; 4] = [
    HAMMER_HARDNESS_MAX,
    UNISON_DETUNE_SCALE_MAX,
    STRETCH_AMOUNT_MAX,
    STRING_LENGTH_SCALE_MAX,
];

/// Smoothing interpolation curve for parameter transitions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum SmoothCurve {
    #[default]
    Exponential,
    Linear,
}

/// Global 4-parameter voicing smoother with O(1) idle detection.
#[derive(Clone, Debug, PartialEq)]
pub struct VoicingSmoother {
    curve: SmoothCurve,
    active: bool,
    current: [f32; 4],
    target: [f32; 4],
}

impl Default for VoicingSmoother {
    fn default() -> Self {
        Self::new()
    }
}

impl VoicingSmoother {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            curve: SmoothCurve::Exponential,
            active: false,
            current: DEFAULTS,
            target: DEFAULTS,
        }
    }

    #[must_use]
    pub fn new_with_voicing(spec: &VoicingSpec) -> Self {
        let mut s = Self::new();
        s.set_immediate(
            spec.effective_global_hammer_hardness(),
            spec.effective_unison_detune_scale(),
            spec.effective_stretch_amount(),
            spec.effective_string_length_scale(),
        );
        s
    }

    pub fn set_curve(&mut self, curve: SmoothCurve) {
        self.curve = curve;
    }

    #[inline]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    pub fn set_target(&mut self, hammer: f32, unison: f32, stretch: f32, length: f32) {
        let vals = [hammer, unison, stretch, length];
        let mut any_diff = false;
        for i in 0..4 {
            let t = if !vals[i].is_finite() {
                DEFAULTS[i]
            } else {
                vals[i].clamp(MINS[i], MAXS[i])
            };
            self.target[i] = t;
            if (self.current[i] - t).abs() > VOICING_SETTLE_TOL {
                any_diff = true;
            } else {
                self.current[i] = t;
            }
        }
        self.active = any_diff;
    }

    pub fn set_immediate(&mut self, hammer: f32, unison: f32, stretch: f32, length: f32) {
        self.set_target(hammer, unison, stretch, length);
        self.current = self.target;
        self.active = false;
    }

    pub fn step_block(&mut self, frames: usize, sample_rate: f32) -> Option<VoicingSpec> {
        if !self.active || frames == 0 || sample_rate <= 0.0 {
            return None;
        }
        let dt = (frames as f32) / sample_rate;
        match self.curve {
            SmoothCurve::Exponential => {
                let alpha = (-dt / VOICING_SMOOTH_TAU_S).exp();
                for i in 0..4 {
                    self.current[i] = self.target[i] + (self.current[i] - self.target[i]) * alpha;
                }
            }
            SmoothCurve::Linear => {
                let step_rate = dt / VOICING_SMOOTH_TAU_S;
                for i in 0..4 {
                    let diff = self.target[i] - self.current[i];
                    let max_step = step_rate * (MAXS[i] - MINS[i]);
                    if diff.abs() <= max_step {
                        self.current[i] = self.target[i];
                    } else {
                        self.current[i] += max_step * diff.signum();
                    }
                }
            }
        }
        for i in 0..4 {
            if (self.current[i] - self.target[i]).abs() <= VOICING_SETTLE_TOL {
                self.current[i] = self.target[i];
            }
        }
        if self.current == self.target {
            self.active = false;
        }
        let mut spec = VoicingSpec::new();
        spec.hammer_hardness = self.current[0];
        spec.unison_detune_scale = self.current[1];
        spec.stretch_amount = self.current[2];
        spec.string_length_scale = self.current[3];
        Some(spec)
    }

    #[inline]
    #[must_use]
    pub const fn current_hammer_hardness(&self) -> f32 {
        self.current[0]
    }
    #[inline]
    #[must_use]
    pub const fn current_unison_detune(&self) -> f32 {
        self.current[1]
    }
    #[inline]
    #[must_use]
    pub const fn current_stretch(&self) -> f32 {
        self.current[2]
    }
    #[inline]
    #[must_use]
    pub const fn current_length_scale(&self) -> f32 {
        self.current[3]
    }
}

/// Action for the engine when a space transition is requested.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SpaceTransitionAction {
    None,
    InstantSwitch {
        target_idx: usize,
        perspective: ListenerPerspective,
        lid: LidState,
    },
    StartCrossfade {
        fade_out_idx: usize,
        fade_in_idx: usize,
        perspective: ListenerPerspective,
        lid: LidState,
        total_samples: usize,
    },
}

/// Raised-cosine spatial crossfade guard across perspective and lid changes.
#[derive(Clone, Debug, PartialEq)]
pub struct SpaceCrossfadeGuard {
    current_perspective: ListenerPerspective,
    current_lid: LidState,
    target_perspective: ListenerPerspective,
    target_lid: LidState,
    active_radiator_idx: usize,
    elapsed_samples: usize,
    total_samples: usize,
    crossfading: bool,
    recent_envelope: f32,
}

impl SpaceCrossfadeGuard {
    #[must_use]
    pub const fn new(perspective: ListenerPerspective, lid: LidState) -> Self {
        Self {
            current_perspective: perspective,
            current_lid: lid,
            target_perspective: perspective,
            target_lid: lid,
            active_radiator_idx: 0,
            elapsed_samples: 0,
            total_samples: 0,
            crossfading: false,
            recent_envelope: 0.0,
        }
    }

    #[inline]
    pub fn update_envelope(&mut self, env: f32) {
        self.recent_envelope = if env.is_nan() { 0.0 } else { env.abs() };
    }

    pub fn request_space(
        &mut self,
        perspective: ListenerPerspective,
        lid: LidState,
        sample_rate: f32,
    ) -> SpaceTransitionAction {
        if perspective == self.current_perspective && lid == self.current_lid && !self.crossfading {
            return SpaceTransitionAction::None;
        }
        self.target_perspective = perspective;
        self.target_lid = lid;

        if self.recent_envelope <= SPACE_SILENCE_ENV_THRESHOLD {
            self.current_perspective = perspective;
            self.current_lid = lid;
            self.crossfading = false;
            self.elapsed_samples = 0;
            SpaceTransitionAction::InstantSwitch {
                target_idx: self.active_radiator_idx,
                perspective,
                lid,
            }
        } else {
            let total = ((SPACE_CROSSFADE_S * sample_rate).round() as usize).max(1);
            self.total_samples = total;
            self.elapsed_samples = 0;
            self.crossfading = true;
            SpaceTransitionAction::StartCrossfade {
                fade_out_idx: self.active_radiator_idx,
                fade_in_idx: 1 - self.active_radiator_idx,
                perspective,
                lid,
                total_samples: total,
            }
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_crossfading(&self) -> bool {
        self.crossfading
    }
    #[inline]
    #[must_use]
    pub const fn active_radiator_idx(&self) -> usize {
        self.active_radiator_idx
    }
    #[inline]
    #[must_use]
    pub const fn elapsed_samples(&self) -> usize {
        self.elapsed_samples
    }
    #[inline]
    #[must_use]
    pub const fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub fn step_block(&mut self, frames: usize) {
        if !self.crossfading {
            return;
        }
        self.elapsed_samples += frames;
        if self.elapsed_samples >= self.total_samples {
            self.active_radiator_idx = 1 - self.active_radiator_idx;
            self.current_perspective = self.target_perspective;
            self.current_lid = self.target_lid;
            self.crossfading = false;
        }
    }
}

#[inline]
#[must_use]
pub fn raised_cosine_weights(sample_idx: usize, total_samples: usize) -> (f32, f32) {
    if total_samples == 0 {
        return (0.0, 1.0);
    }
    let t = (sample_idx as f32 / total_samples as f32).clamp(0.0, 1.0);
    let cos_val = (PI * t).cos();
    (0.5 * (1.0 + cos_val), 0.5 * (1.0 - cos_val))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_raised_cosine_crossfade(
    out_old_l: &[f32],
    out_old_r: &[f32],
    out_new_l: &[f32],
    out_new_r: &[f32],
    dest_l: &mut [f32],
    dest_r: &mut [f32],
    start_sample: usize,
    total_samples: usize,
) {
    let frames = out_old_l
        .len()
        .min(out_old_r.len())
        .min(out_new_l.len())
        .min(out_new_r.len())
        .min(dest_l.len())
        .min(dest_r.len());

    for i in 0..frames {
        let (w_out, w_in) = raised_cosine_weights(start_sample + i, total_samples);
        dest_l[i] = out_old_l[i] * w_out + out_new_l[i] * w_in;
        dest_r[i] = out_old_r[i] * w_out + out_new_r[i] * w_in;
    }
}

pub trait IntoListenerPerspective {
    fn into_perspective(self) -> Option<ListenerPerspective>;
}
impl IntoListenerPerspective for ListenerPerspective {
    fn into_perspective(self) -> Option<ListenerPerspective> {
        Some(self)
    }
}
impl IntoListenerPerspective for u8 {
    fn into_perspective(self) -> Option<ListenerPerspective> {
        ListenerPerspective::from_u8(self)
    }
}
impl IntoListenerPerspective for u32 {
    fn into_perspective(self) -> Option<ListenerPerspective> {
        u8::try_from(self)
            .ok()
            .and_then(ListenerPerspective::from_u8)
    }
}
impl IntoListenerPerspective for i32 {
    fn into_perspective(self) -> Option<ListenerPerspective> {
        u8::try_from(self)
            .ok()
            .and_then(ListenerPerspective::from_u8)
    }
}

pub trait IntoLidState {
    fn into_lid_state(self) -> Option<LidState>;
}
impl IntoLidState for LidState {
    fn into_lid_state(self) -> Option<LidState> {
        Some(self)
    }
}
impl IntoLidState for u8 {
    fn into_lid_state(self) -> Option<LidState> {
        LidState::from_u8(self)
    }
}
impl IntoLidState for u32 {
    fn into_lid_state(self) -> Option<LidState> {
        u8::try_from(self).ok().and_then(LidState::from_u8)
    }
}
impl IntoLidState for i32 {
    fn into_lid_state(self) -> Option<LidState> {
        u8::try_from(self).ok().and_then(LidState::from_u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voicing_smoother_settles_and_idles() {
        let mut sm = VoicingSmoother::new();
        sm.set_target(0.8, 1.5, 1.2, 0.9);
        let sr = 48_000.0;
        let mut steps = 0;
        while sm.is_active() && steps < 1_000 {
            let spec = sm.step_block(128, sr).expect("spec expected");
            assert!(spec.hammer_hardness >= 0.5 && spec.hammer_hardness <= 0.8);
            steps += 1;
        }
        assert!(!sm.is_active());
        assert_eq!(sm.current_hammer_hardness(), 0.8);
        assert_eq!(sm.current_unison_detune(), 1.5);
        assert_eq!(sm.current_stretch(), 1.2);
        assert_eq!(sm.current_length_scale(), 0.9);
        assert!(sm.step_block(128, sr).is_none());
    }

    #[test]
    fn spatial_crossfade_instant_on_silence() {
        let mut guard = SpaceCrossfadeGuard::new(ListenerPerspective::Player, LidState::Open);
        guard.update_envelope(0.0);
        let act = guard.request_space(ListenerPerspective::ConcertHall, LidState::Half, 48_000.0);
        assert_eq!(
            act,
            SpaceTransitionAction::InstantSwitch {
                target_idx: 0,
                perspective: ListenerPerspective::ConcertHall,
                lid: LidState::Half,
            }
        );
        guard.update_envelope(0.05);
        let act = guard.request_space(ListenerPerspective::Player, LidState::Open, 48_000.0);
        assert!(matches!(act, SpaceTransitionAction::StartCrossfade { .. }));
        assert!(guard.is_crossfading());

        let (w_out_0, w_in_0) = raised_cosine_weights(0, 384);
        assert_eq!(w_out_0, 1.0);
        assert_eq!(w_in_0, 0.0);
        let (w_out_mid, w_in_mid) = raised_cosine_weights(192, 384);
        assert!((w_out_mid - 0.5).abs() < 1e-6);
        assert!((w_in_mid - 0.5).abs() < 1e-6);
        let (w_out_end, w_in_end) = raised_cosine_weights(384, 384);
        assert_eq!(w_out_end, 0.0);
        assert_eq!(w_in_end, 1.0);
    }
}
