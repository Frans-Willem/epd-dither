//! Library-grade enum equivalent of the binary's `--dither-palette` /
//! `--output-palette` arguments: names a built-in palette and hands back
//! its colour table as a `&'static [[u8; 3]]`.
//!
//! All built-in palette tables live here. Each is annotated with the
//! decomposers it's valid for; structural validity is enforced at
//! `Decomposer::new` time (e.g. `OctahedronDecomposer::new` errors out
//! on a non-convex-octahedron palette), so the annotation documents
//! intent rather than forming a hard contract.
//!
//! No-alloc throughout: the slice accessor borrows `'static` data, and
//! `FromStr`'s error type is a unit struct ([`InvalidPalette`]) with a
//! `Display` impl. Callers that want a formatted error string get one
//! automatically via the `ToString` blanket impl when they enable
//! `alloc`.

// ============================================================================
// 6-colour palettes
// ============================================================================

/// Naive 6-colour primaries-and-secondaries palette (black, white, yellow,
/// red, blue, green — order matches the reterminal e1002 driver).
///
/// Structurally a regular convex octahedron — three antipodal pairs
/// (K↔W, Y↔B, R↔G) sharing the centroid (0.5, 0.5, 0.5). Useful as a
/// sanity-check / untuned reference; not measured.
///
/// Valid decomposers: [`OctahedronDecomposer`](crate::decompose::octahedron::OctahedronDecomposer),
/// [`NaiveDecomposer`](crate::decompose::naive::NaiveDecomposer).
pub const NAIVE_RGB6: [[u8; 3]; 6] = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 255, 0],
    [255, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
];

/// 6-colour palette as used by the `epdoptimize` toolchain. The points do
/// **not** form a convex octahedron, so this palette is only valid with
/// [`NaiveDecomposer`](crate::decompose::naive::NaiveDecomposer) — feeding
/// it to [`OctahedronDecomposer`](crate::decompose::octahedron::OctahedronDecomposer)
/// produces either a constructor failure or incorrect barycentric
/// coordinates.
///
/// Mirrors the external `epdoptimize` library's palette for cross-library
/// benchmarking; **do not recalibrate**.
///
/// Valid decomposers: [`NaiveDecomposer`](crate::decompose::naive::NaiveDecomposer).
pub const EPDOPTIMIZE: [[u8; 3]; 6] = [
    [0x19, 0x1E, 0x21],
    [0xe8, 0xe8, 0xe8],
    [0xef, 0xde, 0x44],
    [0xb2, 0x13, 0x18],
    [0x21, 0x57, 0xba],
    [0x12, 0x5f, 0x20],
];

// ---- Spectra 6 family ------------------------------------------------------
//
// Measured Spectra 6 panel palettes from the 2026-04-30 reterminal-e1004
// calibration session. Each lays out the six entries in driver order —
// [K, W, Y, R, B, G] — matching the reterminal e1002/e1004 driver's
// nibble packing. The `calibration/convert.py` script emits its tables
// in capture-metadata order (K W B G R Y); the values here are reordered
// for the dither codebase.
//
// Five variants per illuminant, keyed off the `α` parameter of the BPC
// subtraction (see `calibration/convert.py` docstring for the full recipe):
//
// - `_<illuminant>` (no suffix): absolute — measured XYZ → sRGB with the
//   source-illuminant→D65 Bradford CAT only. Panel-realistic, narrow
//   lightness range.
// - `_<illuminant>_ADJUSTED`: α = α_max (largest BPC subtraction with no
//   negative-XYZ clipping required) + L*-symmetric Y scaling. The only
//   non-lossy mode: every patch's measured chromaticity is preserved
//   exactly.
// - `_<illuminant>_BPC{50,75,80,90,100}_ADJUSTED`: BPC at fixed α + L*-
//   symmetric scaling. Above α_max some warm-patch components clip to
//   zero (lossy), in exchange for darker blacks and a wider lightness
//   range.
//
// All 14 variants form valid convex octahedra.
//
// Valid decomposers (whole family):
// [`OctahedronDecomposer`](crate::decompose::octahedron::OctahedronDecomposer),
// [`NaiveDecomposer`](crate::decompose::naive::NaiveDecomposer).

/// Default Spectra 6 palette. Aliases [`SPECTRA6_D65_BPC80_ADJUSTED`].
///
/// Chosen as the default because its lightness spread (L* ≈ 11..89) most
/// closely matches the legacy phone-camera-derived placeholder, easing
/// regression-baseline continuity. For panel-realistic chromaticity prefer
/// [`SPECTRA6_D65_ADJUSTED`] (no clipping, narrower range).
pub const SPECTRA6: [[u8; 3]; 6] = SPECTRA6_D65_BPC80_ADJUSTED;

// ---------- D50 ---------------------------------------------------------

/// Absolute, D50 illuminant. Measured panel reflectance integrated under
/// D50 then converted to sRGB via Bradford CAT to D65.
#[rustfmt::skip]
pub const SPECTRA6_D50: [[u8; 3]; 6] = [
    [ 34,  42,  69], // K
    [138, 153, 164], // W
    [159, 143,   9], // Y
    [116,  34,  29], // R
    [  0,  72, 135], // B
    [ 61, 106,  89], // G
];

/// D50, BPC at α_max + L*-symmetric scaling. Lossless (no clipping).
#[rustfmt::skip]
pub const SPECTRA6_D50_ADJUSTED: [[u8; 3]; 6] = [
    [ 40,  49,  79], // K
    [180, 200, 211], // W
    [208, 186,   0], // Y
    [152,  36,   0], // R
    [  0,  92, 172], // B
    [ 78, 137, 108], // G
];

/// D50, BPC at α=0.50 + L*-symmetric scaling. Warm-patch Z may clip.
#[rustfmt::skip]
pub const SPECTRA6_D50_BPC50_ADJUSTED: [[u8; 3]; 6] = [
    [ 35,  43,  70], // K
    [187, 208, 218], // W
    [217, 194,   0], // Y
    [156,  25,   0], // R
    [  0,  92, 176], // B
    [ 79, 142, 105], // G
];

/// D50, BPC at α=0.75 + L*-symmetric scaling.
#[rustfmt::skip]
pub const SPECTRA6_D50_BPC75_ADJUSTED: [[u8; 3]; 6] = [
    [ 25,  32,  54], // K
    [202, 223, 231], // W
    [234, 208,   0], // Y
    [165,   0,   0], // R
    [  0,  95, 184], // B
    [ 82, 151, 102], // G
];

/// D50, BPC at α=0.80 + L*-symmetric scaling.
#[rustfmt::skip]
pub const SPECTRA6_D50_BPC80_ADJUSTED: [[u8; 3]; 6] = [
    [ 22,  29,  49], // K
    [206, 228, 236], // W
    [239, 212,   0], // Y
    [168,   0,   0], // R
    [  0,  96, 187], // B
    [ 83, 154, 102], // G
];

/// D50, BPC at α=0.90 + L*-symmetric scaling.
#[rustfmt::skip]
pub const SPECTRA6_D50_BPC90_ADJUSTED: [[u8; 3]; 6] = [
    [ 15,  20,  36], // K
    [218, 241, 248], // W
    [252, 224,   0], // Y
    [176,   0,   0], // R
    [  0, 100, 196], // B
    [ 86, 162, 102], // G
];

/// D50, full BPC (α=1.00) + L*-symmetric scaling. Panel-black → (0,0,0).
#[rustfmt::skip]
pub const SPECTRA6_D50_BPC100_ADJUSTED: [[u8; 3]; 6] = [
    [  0,   0,   0], // K
    [235, 255, 255], // W
    [255, 242,   0], // Y
    [188,   0,   0], // R
    [  0, 106, 209], // B
    [ 91, 174, 104], // G
];

// ---------- D65 ---------------------------------------------------------

/// Absolute, D65 illuminant.
#[rustfmt::skip]
pub const SPECTRA6_D65: [[u8; 3]; 6] = [
    [ 33,  43,  69], // K
    [138, 154, 164], // W
    [157, 144,   3], // Y
    [113,  35,  29], // R
    [  0,  72, 135], // B
    [ 60, 107,  88], // G
];

/// D65, BPC at α_max + L*-symmetric scaling. Lossless (no clipping).
#[rustfmt::skip]
pub const SPECTRA6_D65_ADJUSTED: [[u8; 3]; 6] = [
    [ 39,  49,  79], // K
    [179, 199, 210], // W
    [204, 186,   0], // Y
    [147,  36,   0], // R
    [  0,  92, 171], // B
    [ 76, 138, 106], // G
];

/// D65, BPC at α=0.50 + L*-symmetric scaling.
#[rustfmt::skip]
pub const SPECTRA6_D65_BPC50_ADJUSTED: [[u8; 3]; 6] = [
    [ 34,  43,  70], // K
    [187, 207, 216], // W
    [214, 193,   0], // Y
    [150,  26,   0], // R
    [  0,  92, 175], // B
    [ 77, 143, 103], // G
];

/// D65, BPC at α=0.75 + L*-symmetric scaling.
#[rustfmt::skip]
pub const SPECTRA6_D65_BPC75_ADJUSTED: [[u8; 3]; 6] = [
    [ 25,  32,  54], // K
    [201, 223, 230], // W
    [229, 208,   0], // Y
    [157,   0,   0], // R
    [  0,  95, 183], // B
    [ 80, 152,  99], // G
];

/// D65, BPC at α=0.80 + L*-symmetric scaling. Lightness spread closest to
/// the legacy phone-camera placeholder (≈77 L\*); aliased by [`SPECTRA6`].
#[rustfmt::skip]
pub const SPECTRA6_D65_BPC80_ADJUSTED: [[u8; 3]; 6] = [
    [ 22,  29,  49], // K
    [206, 227, 234], // W
    [234, 212,   0], // Y
    [159,   0,   0], // R
    [  0,  96, 186], // B
    [ 81, 155,  99], // G
];

/// D65, BPC at α=0.90 + L*-symmetric scaling.
#[rustfmt::skip]
pub const SPECTRA6_D65_BPC90_ADJUSTED: [[u8; 3]; 6] = [
    [ 14,  20,  36], // K
    [218, 240, 246], // W
    [247, 224,   0], // Y
    [167,   0,   0], // R
    [  0, 100, 195], // B
    [ 84, 163,  99], // G
];

/// D65, full BPC (α=1.00) + L*-symmetric scaling. Panel-black → (0,0,0).
#[rustfmt::skip]
pub const SPECTRA6_D65_BPC100_ADJUSTED: [[u8; 3]; 6] = [
    [  0,   0,   0], // K
    [235, 255, 255], // W
    [255, 242,   0], // Y
    [178,   0,   0], // R
    [  0, 106, 208], // B
    [ 90, 175, 100], // G
];

// ============================================================================
// 1-D grayscale palettes — RGB-widened so the same `&[[u8; 3]]` interface
// as the chromatic palettes works without reaching for `alloc`.
// ============================================================================

/// Linearly-spaced 2-level grayscale palette (placeholder; replace with
/// measured panel reflectance values — see TODO.md).
///
/// Valid decomposers:
/// [`OffsetBlendGrayDecomposer`](crate::decompose::gray::OffsetBlendGrayDecomposer),
/// [`PureSpreadGrayDecomposer`](crate::decompose::gray::PureSpreadGrayDecomposer).
pub const GRAYSCALE2_RGB: [[u8; 3]; 2] = [[0, 0, 0], [255, 255, 255]];

/// Linearly-spaced 4-level grayscale palette (placeholder).
///
/// Valid decomposers:
/// [`OffsetBlendGrayDecomposer`](crate::decompose::gray::OffsetBlendGrayDecomposer),
/// [`PureSpreadGrayDecomposer`](crate::decompose::gray::PureSpreadGrayDecomposer).
#[rustfmt::skip]
pub const GRAYSCALE4_RGB: [[u8; 3]; 4] = [
    [  0,   0,   0],
    [ 85,  85,  85],
    [170, 170, 170],
    [255, 255, 255],
];

/// Linearly-spaced 16-level grayscale palette (placeholder).
///
/// Valid decomposers:
/// [`OffsetBlendGrayDecomposer`](crate::decompose::gray::OffsetBlendGrayDecomposer),
/// [`PureSpreadGrayDecomposer`](crate::decompose::gray::PureSpreadGrayDecomposer).
#[rustfmt::skip]
pub const GRAYSCALE16_RGB: [[u8; 3]; 16] = [
    [0x00, 0x00, 0x00], [0x11, 0x11, 0x11], [0x22, 0x22, 0x22], [0x33, 0x33, 0x33],
    [0x44, 0x44, 0x44], [0x55, 0x55, 0x55], [0x66, 0x66, 0x66], [0x77, 0x77, 0x77],
    [0x88, 0x88, 0x88], [0x99, 0x99, 0x99], [0xAA, 0xAA, 0xAA], [0xBB, 0xBB, 0xBB],
    [0xCC, 0xCC, 0xCC], [0xDD, 0xDD, 0xDD], [0xEE, 0xEE, 0xEE], [0xFF, 0xFF, 0xFF],
];

// ============================================================================
// `Palette` enum + dispatch
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Palette {
    Naive,
    Spectra6,
    Spectra6D50,
    Spectra6D50Adjusted,
    Spectra6D50Bpc50Adjusted,
    Spectra6D50Bpc75Adjusted,
    Spectra6D50Bpc80Adjusted,
    Spectra6D50Bpc90Adjusted,
    Spectra6D50Bpc100Adjusted,
    Spectra6D65,
    Spectra6D65Adjusted,
    Spectra6D65Bpc50Adjusted,
    Spectra6D65Bpc75Adjusted,
    Spectra6D65Bpc80Adjusted,
    Spectra6D65Bpc90Adjusted,
    Spectra6D65Bpc100Adjusted,
    Epdoptimize,
    Grayscale2,
    Grayscale4,
    Grayscale16,
}

impl Palette {
    pub const LONG_HELP: &'static str = concat!(
        "Built-in palette to use.\n\n",
        "Accepted values:\n",
        "  naive\n",
        "  spectra6\n",
        "  spectra6-d50, spectra6-d50-adjusted\n",
        "  spectra6-d50-bpc{50,75,80,90,100}-adjusted\n",
        "  spectra6-d65, spectra6-d65-adjusted\n",
        "  spectra6-d65-bpc{50,75,80,90,100}-adjusted\n",
        "  epdoptimize\n",
        "  grayscale2, grayscale4, grayscale16\n",
    );

    pub fn as_rgb_slice(&self) -> &'static [[u8; 3]] {
        match self {
            Self::Naive => &NAIVE_RGB6,
            Self::Spectra6 => &SPECTRA6,
            Self::Spectra6D50 => &SPECTRA6_D50,
            Self::Spectra6D50Adjusted => &SPECTRA6_D50_ADJUSTED,
            Self::Spectra6D50Bpc50Adjusted => &SPECTRA6_D50_BPC50_ADJUSTED,
            Self::Spectra6D50Bpc75Adjusted => &SPECTRA6_D50_BPC75_ADJUSTED,
            Self::Spectra6D50Bpc80Adjusted => &SPECTRA6_D50_BPC80_ADJUSTED,
            Self::Spectra6D50Bpc90Adjusted => &SPECTRA6_D50_BPC90_ADJUSTED,
            Self::Spectra6D50Bpc100Adjusted => &SPECTRA6_D50_BPC100_ADJUSTED,
            Self::Spectra6D65 => &SPECTRA6_D65,
            Self::Spectra6D65Adjusted => &SPECTRA6_D65_ADJUSTED,
            Self::Spectra6D65Bpc50Adjusted => &SPECTRA6_D65_BPC50_ADJUSTED,
            Self::Spectra6D65Bpc75Adjusted => &SPECTRA6_D65_BPC75_ADJUSTED,
            Self::Spectra6D65Bpc80Adjusted => &SPECTRA6_D65_BPC80_ADJUSTED,
            Self::Spectra6D65Bpc90Adjusted => &SPECTRA6_D65_BPC90_ADJUSTED,
            Self::Spectra6D65Bpc100Adjusted => &SPECTRA6_D65_BPC100_ADJUSTED,
            Self::Epdoptimize => &EPDOPTIMIZE,
            Self::Grayscale2 => &GRAYSCALE2_RGB,
            Self::Grayscale4 => &GRAYSCALE4_RGB,
            Self::Grayscale16 => &GRAYSCALE16_RGB,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidPalette;

impl core::fmt::Display for InvalidPalette {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid palette name")
    }
}

impl core::error::Error for InvalidPalette {}

impl core::str::FromStr for Palette {
    type Err = InvalidPalette;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "naive" => Ok(Self::Naive),
            "spectra6" => Ok(Self::Spectra6),
            "spectra6-d50" => Ok(Self::Spectra6D50),
            "spectra6-d50-adjusted" => Ok(Self::Spectra6D50Adjusted),
            "spectra6-d50-bpc50-adjusted" => Ok(Self::Spectra6D50Bpc50Adjusted),
            "spectra6-d50-bpc75-adjusted" => Ok(Self::Spectra6D50Bpc75Adjusted),
            "spectra6-d50-bpc80-adjusted" => Ok(Self::Spectra6D50Bpc80Adjusted),
            "spectra6-d50-bpc90-adjusted" => Ok(Self::Spectra6D50Bpc90Adjusted),
            "spectra6-d50-bpc100-adjusted" => Ok(Self::Spectra6D50Bpc100Adjusted),
            "spectra6-d65" => Ok(Self::Spectra6D65),
            "spectra6-d65-adjusted" => Ok(Self::Spectra6D65Adjusted),
            "spectra6-d65-bpc50-adjusted" => Ok(Self::Spectra6D65Bpc50Adjusted),
            "spectra6-d65-bpc75-adjusted" => Ok(Self::Spectra6D65Bpc75Adjusted),
            "spectra6-d65-bpc80-adjusted" => Ok(Self::Spectra6D65Bpc80Adjusted),
            "spectra6-d65-bpc90-adjusted" => Ok(Self::Spectra6D65Bpc90Adjusted),
            "spectra6-d65-bpc100-adjusted" => Ok(Self::Spectra6D65Bpc100Adjusted),
            "epdoptimize" => Ok(Self::Epdoptimize),
            "grayscale2" => Ok(Self::Grayscale2),
            "grayscale4" => Ok(Self::Grayscale4),
            "grayscale16" => Ok(Self::Grayscale16),
            _ => Err(InvalidPalette),
        }
    }
}
