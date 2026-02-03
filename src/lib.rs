#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
extern crate alloc; // TODO: Can we put this behind a feature?
pub mod barycentric;
pub mod decompose;
mod helpers;
pub mod noise;
