//! Spatial raised-cosine crossfade guard and perspective/lid conversion traits.

use core::f32::consts::PI;
use piano_radiator::BodyArchetype;
use piano_spec::{LidState, ListenerPerspective};

/// Duration of the raised-cosine crossfade when changing listener perspective or lid (`8 ms`).
pub const SPACE_CROSSFADE_S: f32 = 0.008;
/// Signal envelope threshold below which spatial switching occurs instantly without crossfade.
pub const SPACE_SILENCE_ENV_THRESHOLD: f32 = 1.0e-7;

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

/// Raised-cosine spatial crossfade guard across body, perspective, and lid changes.
#[derive(Clone, Debug, PartialEq)]
pub struct SpaceCrossfadeGuard {
    current_body: BodyArchetype,
    current_perspective: ListenerPerspective,
    current_lid: LidState,
    target_body: BodyArchetype,
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
            current_body: BodyArchetype::ConcertGrand,
            current_perspective: perspective,
            current_lid: lid,
            target_body: BodyArchetype::ConcertGrand,
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

    /// Resets envelope to silence and immediately completes any in-flight crossfade.
    pub fn on_silence_reset(&mut self) {
        self.recent_envelope = 0.0;
        if self.crossfading {
            self.active_radiator_idx = 1 - self.active_radiator_idx;
            self.current_body = self.target_body;
            self.current_perspective = self.target_perspective;
            self.current_lid = self.target_lid;
            self.crossfading = false;
            self.elapsed_samples = 0;
        }
    }

    #[inline]
    #[must_use]
    pub fn would_instant_switch(&self) -> bool {
        self.recent_envelope <= SPACE_SILENCE_ENV_THRESHOLD
    }

    #[inline]
    #[must_use]
    pub fn target_radiator_idx(&self) -> usize {
        if self.would_instant_switch() {
            self.active_radiator_idx
        } else {
            1 - self.active_radiator_idx
        }
    }

    pub fn request_space(
        &mut self,
        perspective: ListenerPerspective,
        lid: LidState,
        sample_rate: f32,
    ) -> SpaceTransitionAction {
        self.request_body_space(self.current_body, perspective, lid, sample_rate)
    }

    pub fn request_body_space(
        &mut self,
        body: BodyArchetype,
        perspective: ListenerPerspective,
        lid: LidState,
        sample_rate: f32,
    ) -> SpaceTransitionAction {
        if body == self.current_body
            && perspective == self.current_perspective
            && lid == self.current_lid
            && !self.crossfading
        {
            return SpaceTransitionAction::None;
        }
        self.target_body = body;
        self.target_perspective = perspective;
        self.target_lid = lid;

        if self.recent_envelope <= SPACE_SILENCE_ENV_THRESHOLD {
            self.current_body = body;
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
    pub const fn current_body(&self) -> BodyArchetype {
        self.current_body
    }
    #[inline]
    #[must_use]
    pub const fn target_body(&self) -> BodyArchetype {
        self.target_body
    }
    #[inline]
    #[must_use]
    pub const fn current_perspective(&self) -> ListenerPerspective {
        self.current_perspective
    }
    #[inline]
    #[must_use]
    pub const fn current_lid(&self) -> LidState {
        self.current_lid
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
    pub const fn standby_radiator_idx(&self) -> usize {
        1 - self.active_radiator_idx
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

    /// Commits a pre-staged standby radiator bank without overwriting the active buffer.
    pub fn commit_standby(
        &mut self,
        perspective: ListenerPerspective,
        lid: LidState,
        sample_rate: f32,
    ) -> SpaceTransitionAction {
        self.commit_body_standby(self.current_body, perspective, lid, sample_rate)
    }

    /// Commits a pre-staged standby radiator bank for `(body, perspective, lid)`.
    pub fn commit_body_standby(
        &mut self,
        body: BodyArchetype,
        perspective: ListenerPerspective,
        lid: LidState,
        sample_rate: f32,
    ) -> SpaceTransitionAction {
        if self.crossfading {
            return SpaceTransitionAction::None;
        }
        if body == self.current_body
            && perspective == self.current_perspective
            && lid == self.current_lid
        {
            return SpaceTransitionAction::None;
        }
        self.target_body = body;
        self.target_perspective = perspective;
        self.target_lid = lid;
        if self.recent_envelope <= SPACE_SILENCE_ENV_THRESHOLD {
            self.active_radiator_idx = 1 - self.active_radiator_idx;
            self.current_body = body;
            self.current_perspective = perspective;
            self.current_lid = lid;
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

    pub fn step_block(&mut self, frames: usize) {
        if !self.crossfading {
            return;
        }
        self.elapsed_samples += frames;
        if self.elapsed_samples >= self.total_samples {
            self.active_radiator_idx = 1 - self.active_radiator_idx;
            self.current_body = self.target_body;
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
