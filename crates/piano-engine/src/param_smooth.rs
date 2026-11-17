//! Parameter smoothing and spatial crossfade transitions for real-time engine control.

use piano_spec::voicing::{
    A4_FREQ_DEFAULT, A4_FREQ_MAX, A4_FREQ_MIN, HAMMER_HARDNESS_DEFAULT, HAMMER_HARDNESS_MAX,
    HAMMER_HARDNESS_MIN, STRETCH_AMOUNT_DEFAULT, STRETCH_AMOUNT_MAX, STRETCH_AMOUNT_MIN,
    STRING_LENGTH_SCALE_DEFAULT, STRING_LENGTH_SCALE_MAX, STRING_LENGTH_SCALE_MIN,
    UNISON_DETUNE_SCALE_DEFAULT, UNISON_DETUNE_SCALE_MAX, UNISON_DETUNE_SCALE_MIN,
};
use piano_spec::VoicingSpec;

pub use crate::space_guard::{
    apply_raised_cosine_crossfade, raised_cosine_weights, IntoLidState, IntoListenerPerspective,
    SpaceCrossfadeGuard, SpaceTransitionAction, SPACE_CROSSFADE_S, SPACE_SILENCE_ENV_THRESHOLD,
};

/// Time constant for voicing parameter smoothing (`20 ms`).
pub const VOICING_SMOOTH_TAU_S: f32 = 0.020;
/// Convergence epsilon tolerance below which smoothed unit-scale parameters snap to target (`1e-4`).
pub const VOICING_SETTLE_TOL: f32 = 1.0e-4;

/// Voicing dirty flag: hammer hardness / action felt parameters changed.
pub const DIRTY_HAMMER: u8 = 1;
/// Voicing dirty flag: string tuning / unison / length scale / concert pitch changed.
pub const DIRTY_TUNING: u8 = 2;

const NUM_VOICING_PARAMS: usize = 5;

const DEFAULTS: [f32; NUM_VOICING_PARAMS] = [
    HAMMER_HARDNESS_DEFAULT,
    UNISON_DETUNE_SCALE_DEFAULT,
    STRETCH_AMOUNT_DEFAULT,
    STRING_LENGTH_SCALE_DEFAULT,
    A4_FREQ_DEFAULT,
];
const MINS: [f32; NUM_VOICING_PARAMS] = [
    HAMMER_HARDNESS_MIN,
    UNISON_DETUNE_SCALE_MIN,
    STRETCH_AMOUNT_MIN,
    STRING_LENGTH_SCALE_MIN,
    A4_FREQ_MIN,
];
const MAXS: [f32; NUM_VOICING_PARAMS] = [
    HAMMER_HARDNESS_MAX,
    UNISON_DETUNE_SCALE_MAX,
    STRETCH_AMOUNT_MAX,
    STRING_LENGTH_SCALE_MAX,
    A4_FREQ_MAX,
];

#[inline]
const fn settle_tol(idx: usize) -> f32 {
    let span = MAXS[idx] - MINS[idx];
    let scale = if span > 1.0 { span } else { 1.0 };
    VOICING_SETTLE_TOL * scale
}

/// Smoothing interpolation curve for parameter transitions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum SmoothCurve {
    #[default]
    Exponential,
    Linear,
}

/// Global 5-parameter voicing smoother (`hammer`, `unison`, `stretch`, `length`, `a4_freq_hz`)
/// with O(1) idle detection.
#[derive(Clone, Debug, PartialEq)]
pub struct VoicingSmoother {
    curve: SmoothCurve,
    active: bool,
    current: [f32; NUM_VOICING_PARAMS],
    target: [f32; NUM_VOICING_PARAMS],
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
        s.set_immediate_with_a4(
            spec.effective_global_hammer_hardness(),
            spec.effective_unison_detune_scale(),
            spec.effective_stretch_amount(),
            spec.effective_string_length_scale(),
            spec.effective_a4_freq_hz(),
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
        self.set_target_with_a4(hammer, unison, stretch, length, self.target[4]);
    }

    pub fn set_target_with_a4(
        &mut self,
        hammer: f32,
        unison: f32,
        stretch: f32,
        length: f32,
        a4_freq_hz: f32,
    ) {
        let vals = [hammer, unison, stretch, length, a4_freq_hz];
        let mut any_diff = false;
        for i in 0..NUM_VOICING_PARAMS {
            let t = if !vals[i].is_finite() {
                DEFAULTS[i]
            } else {
                vals[i].clamp(MINS[i], MAXS[i])
            };
            self.target[i] = t;
            if (self.current[i] - t).abs() > settle_tol(i) {
                any_diff = true;
            } else {
                self.current[i] = t;
            }
        }
        self.active = any_diff;
    }

    pub fn set_a4_target(&mut self, a4_freq_hz: f32) {
        self.set_target_with_a4(
            self.target[0],
            self.target[1],
            self.target[2],
            self.target[3],
            a4_freq_hz,
        );
    }

    pub fn set_immediate(&mut self, hammer: f32, unison: f32, stretch: f32, length: f32) {
        self.set_immediate_with_a4(hammer, unison, stretch, length, self.target[4]);
    }

    pub fn set_immediate_with_a4(
        &mut self,
        hammer: f32,
        unison: f32,
        stretch: f32,
        length: f32,
        a4_freq_hz: f32,
    ) {
        self.set_target_with_a4(hammer, unison, stretch, length, a4_freq_hz);
        self.current = self.target;
        self.active = false;
    }

    pub fn step_block(&mut self, frames: usize, sample_rate: f32) -> Option<VoicingSpec> {
        self.step_block_with_dirty(frames, sample_rate)
            .map(|(spec, _)| spec)
    }

    pub fn step_block_with_dirty(
        &mut self,
        frames: usize,
        sample_rate: f32,
    ) -> Option<(VoicingSpec, u8)> {
        if !self.active || frames == 0 || sample_rate <= 0.0 {
            return None;
        }
        let prev = self.current;
        let dt = (frames as f32) / sample_rate;
        match self.curve {
            SmoothCurve::Exponential => {
                let alpha = (-dt / VOICING_SMOOTH_TAU_S).exp();
                for i in 0..NUM_VOICING_PARAMS {
                    self.current[i] = self.target[i] + (self.current[i] - self.target[i]) * alpha;
                }
            }
            SmoothCurve::Linear => {
                let step_rate = dt / VOICING_SMOOTH_TAU_S;
                for i in 0..NUM_VOICING_PARAMS {
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
        for i in 0..NUM_VOICING_PARAMS {
            if (self.current[i] - self.target[i]).abs() <= settle_tol(i) {
                self.current[i] = self.target[i];
            }
        }
        if self.current == self.target {
            self.active = false;
        }
        let mut dirty = 0u8;
        if self.current[0] != prev[0] {
            dirty |= DIRTY_HAMMER;
        }
        if self.current[1] != prev[1]
            || self.current[2] != prev[2]
            || self.current[3] != prev[3]
            || self.current[4] != prev[4]
        {
            dirty |= DIRTY_TUNING;
        }
        if dirty == 0 {
            return None;
        }
        let mut spec = VoicingSpec::new();
        spec.hammer_hardness = self.current[0];
        spec.unison_detune_scale = self.current[1];
        spec.stretch_amount = self.current[2];
        spec.string_length_scale = self.current[3];
        spec.a4_freq_hz = self.current[4];
        Some((spec, dirty))
    }

    pub fn set_tuning_applied(&mut self, unison: f32, stretch: f32, length: f32) {
        self.set_tuning_applied_with_a4(unison, stretch, length, self.current[4]);
    }

    pub fn set_tuning_applied_with_a4(
        &mut self,
        unison: f32,
        stretch: f32,
        length: f32,
        a4_freq_hz: f32,
    ) {
        let vals = [unison, stretch, length, a4_freq_hz];
        for idx in 1..NUM_VOICING_PARAMS {
            let v = vals[idx - 1];
            let c = if v.is_finite() {
                v.clamp(MINS[idx], MAXS[idx])
            } else {
                DEFAULTS[idx]
            };
            self.current[idx] = c;
            self.target[idx] = c;
        }
        self.active = self.current[0] != self.target[0];
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
    #[inline]
    #[must_use]
    pub const fn current_a4_freq_hz(&self) -> f32 {
        self.current[4]
    }
}

#[cfg(test)]
#[path = "param_smooth_tests.rs"]
mod tests;
