#![no_std]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
pub mod barycentric;
pub mod decomposer6c;
pub mod noise;
mod helpers;
