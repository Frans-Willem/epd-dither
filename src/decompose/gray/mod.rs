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

const fn replicate_rgb<const N: usize>(src: &[u8; N]) -> [[u8; 3]; N] {
    let mut out = [[0u8; 3]; N];
    let mut i = 0;
    while i < N {
        out[i] = [src[i]; 3];
        i += 1;
    }
    out
}

/// `GRAYSCALE2` widened to `[u8; 3]` per entry. Lets callers consume the
/// palette through the same `&[[u8; 3]]` interface as the chromatic
/// palettes without reaching for `alloc`.
pub const GRAYSCALE2_RGB: [[u8; 3]; 2] = replicate_rgb(&GRAYSCALE2);

/// `GRAYSCALE4` widened to `[u8; 3]` per entry.
pub const GRAYSCALE4_RGB: [[u8; 3]; 4] = replicate_rgb(&GRAYSCALE4);

/// `GRAYSCALE16` widened to `[u8; 3]` per entry.
pub const GRAYSCALE16_RGB: [[u8; 3]; 16] = replicate_rgb(&GRAYSCALE16);
