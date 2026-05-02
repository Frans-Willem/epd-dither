//! Pixel-color trait used by the dither factory: any type implementing
//! [`DecomposerInputColor`] can stand in as both a source pixel and a palette
//! entry. The factory picks the right method per strategy:
//!
//! * Octahedron / Naive use [`to_point`](DecomposerInputColor::to_point), giving
//!   a `Point3<f32>` in the decomposer's RGB-cube input space.
//! * Grayscale strategies use [`brightness`](DecomposerInputColor::brightness),
//!   plus [`is_grayscale`](DecomposerInputColor::is_grayscale) on each palette
//!   entry as a sanity check.
//!
//! Default impls for `image::Rgb<u8>` and `image::Rgb<f32>` live in
//! [`crate::image_adapter`] (gated on the `image` feature); other crates
//! supply their own.

use nalgebra::geometry::Point3;

pub trait DecomposerInputColor {
    /// Project into the decomposer's `Point3<f32>` input space (e.g. RGB
    /// cube for octahedron/naive). For 8-bit colour, divide by 255 to
    /// normalise into [0, 1].
    fn to_point(&self) -> Point3<f32>;

    /// Project into a 1-D brightness for grayscale strategies. The
    /// canonical choice is BT.709 luma applied directly in sRGB (no
    /// gamma round-trip) — see the README's Yule-Nielsen note.
    fn brightness(&self) -> f32;

    /// True iff the colour is achromatic (channels equal, or otherwise
    /// projects unambiguously to one brightness). Used to validate that
    /// a palette is suitable for a grayscale strategy.
    fn is_grayscale(&self) -> bool;
}

impl DecomposerInputColor for [u8; 3] {
    fn to_point(&self) -> Point3<f32> {
        let [r, g, b] = *self;
        Point3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
    }
    fn brightness(&self) -> f32 {
        let [r, g, b] = *self;
        let r = r as f32 / 255.0;
        let g = g as f32 / 255.0;
        let b = b as f32 / 255.0;
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }
    fn is_grayscale(&self) -> bool {
        let [r, g, b] = *self;
        r == g && g == b
    }
}
