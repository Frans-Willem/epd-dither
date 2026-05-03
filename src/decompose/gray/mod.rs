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

/// `Ord::max` analogue for `T: PartialOrd`. Returns `a` on a tie or on
/// incomparable values (e.g. NaN), matching the convention `if a > b { a } else { b }`.
pub(super) fn partial_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// `Ord::clamp` analogue for `T: PartialOrd`. Incomparable inputs (e.g. NaN)
/// pass through unchanged.
pub(super) fn partial_clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}
