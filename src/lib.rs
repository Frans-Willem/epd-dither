#![no_std]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
extern crate alloc; // TODO: Can we put this behind a feature?
pub mod barycentric;
pub mod decomposer6c; // TODO: Rename decomposer_octahedron
pub mod decomposer_bruteforce;
mod helpers;
pub mod noise;
