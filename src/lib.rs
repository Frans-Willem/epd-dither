#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
extern crate alloc; // TODO: Can we put this behind a feature?
pub mod barycentric;
pub mod decompose;
pub mod dither;
mod helpers;
#[cfg(feature = "image")]
pub mod image_adapter;
pub mod noise;

pub use decompose::Decomposer;
