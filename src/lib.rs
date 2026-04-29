#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#[cfg(feature = "alloc")]
extern crate alloc;
pub mod barycentric;
pub mod decompose;
pub mod dither;
mod helpers;
#[cfg(feature = "image")]
pub mod image_adapter;
pub mod noise;

pub use decompose::Decomposer;
