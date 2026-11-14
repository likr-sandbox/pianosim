//! Shared piano specifications, sample-accurate events, and scaling constants.

#![cfg_attr(not(test), no_std)]

pub mod events;

pub use events::{u16_to_unit_f32, u7_to_u16, EventKind, PedalKind, SampleEvent};
