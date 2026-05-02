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
