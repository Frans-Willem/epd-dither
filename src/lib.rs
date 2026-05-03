#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#[cfg(feature = "alloc")]
extern crate alloc;
pub mod barycentric;
pub mod decompose;
pub mod dither;
#[cfg(feature = "alloc")]
pub mod registry;
mod helpers;
#[cfg(feature = "image")]
pub mod image;
pub mod noise;
pub mod palette;

pub use decompose::{Decomposer, DecomposerInputColor};
pub use palette::Palette;
