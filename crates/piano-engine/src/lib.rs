//! Zero-allocation 4-stage real-time piano synthesizer engine.

pub(crate) mod diag;
pub(crate) mod drivers;
pub mod engine;
pub(crate) mod engine_core;
pub mod event_queue;
pub mod param_smooth;
pub(crate) mod perf_control;
pub mod space_control;
pub mod space_guard;
pub mod telemetry;

pub use engine::PianoEngine;
pub use event_queue::{EventQueue, MAX_EVENTS};
pub use param_smooth::{
    IntoLidState, IntoListenerPerspective, SpaceCrossfadeGuard, VoicingSmoother,
};
pub use piano_radiator::{
    embedded_body_bytes, space_blob_len_for_profile, space_blob_len_for_sample_rate, BodyArchetype,
    SpacePrepContext, SPACE_BLOB_HEADER_LEN, SPACE_BLOB_MAGIC_V2,
};
pub use piano_spec::{self, PerfPreset, PerfProfile};
pub use piano_strings::{voicing_blob_len, voicing_blob_len_with_profile, VoicingPrepContext};
pub use space_control::{IntoBodyArchetype, DEFAULT_REVERB_DECAY_S, DEFAULT_REVERB_MIX};
pub use telemetry::EngineTelemetry;

#[cfg(test)]
mod duplex_integration_tests;

#[cfg(test)]
mod engine_tests;

#[cfg(test)]
mod m3_tests;

#[cfg(test)]
mod m4_tests;

#[cfg(test)]
mod m5_tests;

#[cfg(test)]
mod m6_tests;

#[cfg(test)]
mod m6_acoustic_tests;

#[cfg(test)]
mod perf_profile_tests;

#[cfg(test)]
mod wave2_tests;

#[cfg(test)]
mod wave3_tests;

#[cfg(test)]
mod wave4_tests;

#[cfg(test)]
mod half_pedal_tests;
