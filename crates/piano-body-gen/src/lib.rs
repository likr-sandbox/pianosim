//! Layer 1A: offline pure-Rust structural-acoustic modal solver and preset baker.

pub mod air_cavity;
pub mod bake;
pub mod linalg;
pub mod orthotropic_plate;
pub mod pole_cancel;
pub mod ribs_rim;
pub mod types;

#[cfg(test)]
mod tests;

pub use bake::{bake_concert_grand_preset, bake_concert_grand_with_diagnostics, BakeDiagnostics};
pub use orthotropic_plate::PlateSolveStats;
pub use pole_cancel::{
    eval_y_red_freq, eval_z_red_norm_freq, verify_pole_cancellation, InverseReducedBridge,
    PoleCancelReport, CALIBRATED_DIRECT_ZONE_GAIN,
};
pub use ribs_rim::{evaluate_plate_site, rim_signed_distance_m, CONCERT_ZONE_POS_M};
pub use types::{
    sitka_spruce_t60_s, t60_to_zeta, zeta_to_t60, ModeSource, SolvedBodyMode, MAX_BODY_MODE_HZ,
    MAX_MODAL_T60_S, MIN_BODY_MODE_HZ,
};
