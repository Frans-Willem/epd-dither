//! Measured Spectra 6 panel palettes from the 2026-04-30 reterminal-e1004
//! calibration session.
//!
//! Each constant lays out the six palette entries in **driver order** —
//! `[K, W, Y, R, B, G]` — matching the reterminal e1002/e1004 driver's nibble
//! packing. The `calibration/convert.py` script emits its tables in
//! capture-metadata order (`K W B G R Y`); the values here are reordered for
//! the dither codebase.
//!
//! Five variants per illuminant, keyed off the `α` parameter of the
//! BPC subtraction (see `calibration/convert.py` docstring for the full
//! recipe):
//!
//! - `_<illuminant>` (no suffix): `absolute` — measured XYZ → sRGB with the
//!   source-illuminant→D65 Bradford CAT only. Panel-realistic, narrow
//!   lightness range.
//! - `_<illuminant>_ADJUSTED`: α = α_max (largest BPC subtraction with no
//!   negative-XYZ clipping required) + L*-symmetric Y scaling. The only
//!   non-lossy mode: every patch's measured chromaticity is preserved
//!   exactly.
//! - `_<illuminant>_BPC{50,75,80,90,100}_ADJUSTED`: BPC at fixed α + L*-
//!   symmetric scaling. Above α_max some warm-patch components clip to zero
//!   (lossy), in exchange for darker blacks and a wider lightness range.
//!
//! All 14 variants here form valid convex octahedra and are compatible
//! with [`crate::decompose::octahedron::OctahedronDecomposer`] (which
//! errors out on construction if its palette doesn't form one).

/// Default Spectra 6 palette. Aliases [`SPECTRA6_D65_BPC80_ADJUSTED`].
///
/// Chosen as the default because its lightness spread (L* ≈ 11..89) most
/// closely matches the legacy phone-camera-derived placeholder, easing
/// regression-master continuity. For panel-realistic chromaticity prefer
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
