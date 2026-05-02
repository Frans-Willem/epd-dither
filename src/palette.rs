//! Library-grade enum equivalent of the binary's `--dither-palette` /
//! `--output-palette` arguments: names a built-in RGB palette and hands
//! back its colour table as a `&'static [[u8; 3]]`.
//!
//! No-alloc throughout: the slice accessor borrows `'static` data, and
//! `FromStr`'s error type is a unit struct ([`InvalidPalette`]) with a
//! `Display` impl. Callers that want a formatted error string get one
//! automatically via the `ToString` blanket impl when they enable
//! `alloc`.

use crate::decompose::gray::{GRAYSCALE2_RGB, GRAYSCALE4_RGB, GRAYSCALE16_RGB};
use crate::decompose::naive::EPDOPTIMIZE;
use crate::decompose::octahedron::NAIVE_RGB6;
use crate::spectra6::{
    SPECTRA6, SPECTRA6_D50, SPECTRA6_D50_ADJUSTED, SPECTRA6_D50_BPC50_ADJUSTED,
    SPECTRA6_D50_BPC75_ADJUSTED, SPECTRA6_D50_BPC80_ADJUSTED, SPECTRA6_D50_BPC90_ADJUSTED,
    SPECTRA6_D50_BPC100_ADJUSTED, SPECTRA6_D65, SPECTRA6_D65_ADJUSTED, SPECTRA6_D65_BPC50_ADJUSTED,
    SPECTRA6_D65_BPC75_ADJUSTED, SPECTRA6_D65_BPC80_ADJUSTED, SPECTRA6_D65_BPC90_ADJUSTED,
    SPECTRA6_D65_BPC100_ADJUSTED,
};

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
