#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#[cfg(feature = "alloc")]
extern crate alloc;
pub mod barycentric;
pub mod decomposer_input;
pub mod decompose;
pub mod dither;
#[cfg(feature = "alloc")]
pub mod factory;
mod helpers;
#[cfg(feature = "image")]
pub mod image_adapter;
pub mod noise;
pub mod palette;
pub mod spectra6;

pub use decomposer_input::DecomposerInputColor;
pub use decompose::Decomposer;
pub use palette::Palette;
