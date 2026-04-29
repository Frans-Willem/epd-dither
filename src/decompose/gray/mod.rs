//! 1-D / grayscale palette decomposers.
//!
//! Two decomposers, both `AsRef<[T]>`-storage-generic so they work with or
//! without an allocator:
//!
//!   * [`PureSpreadGrayDecomposer`]: asymmetric mean-preserving spread that
//!     peaks when the input coincides with a palette level.
//!   * [`OffsetBlendGrayDecomposer`]: linear blend of the bracket
//!     decompositions of `input ± offset`, with the blend factor chosen so
//!     the combined mean equals the input even when one side clamps.
//!
//! At parameter = 0 both reduce to the same plain bracket decomposition.

mod offset_blend;
mod pure_spread;

pub use offset_blend::OffsetBlendGrayDecomposer;
pub use pure_spread::PureSpreadGrayDecomposer;

/// Linearly-spaced 2-level grayscale palette (placeholder; replace with
/// measured panel reflectance values — see TODO.md).
pub const GRAYSCALE2: [u8; 2] = [0, 255];

/// Linearly-spaced 4-level grayscale palette (placeholder).
pub const GRAYSCALE4: [u8; 4] = [0, 85, 170, 255];

/// Linearly-spaced 16-level grayscale palette (placeholder).
pub const GRAYSCALE16: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
];
