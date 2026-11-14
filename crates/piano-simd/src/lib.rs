//! Portable 4-lane SIMD abstractions, 64-byte aligned buffers, FFT, and ADAA LUTs.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod aligned_buf;
pub mod f32x4;
pub mod fft_radix2;
pub mod lut_adaa;
mod math;
pub mod real_fft;

pub use aligned_buf::AlignedBuf;
pub use f32x4::F32x4;
pub use fft_radix2::{Complex, Radix2Fft};
pub use lut_adaa::{AdaaPowerLut, AdaaState, ADAA_DIFF_EPS};
pub use real_fft::RealFft;
