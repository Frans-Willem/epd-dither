//! Palette-indexed image buffer that doubles as an [`ImageWriter<usize>`]
//! sink and knows how to emit an indexed PNG. Picks the smallest legal
//! indexed-PNG bit depth (1/2/4/8) for the palette size and packs indices
//! MSB-first byte-aligned per scanline — exactly what the PNG spec wants.
//!
//! Available behind the `image` Cargo feature, which also pulls in `png`.

use crate::dither::{ImageReader, ImageSize, ImageWriter};
use alloc::vec;
use alloc::vec::Vec;
use image::Rgb;
use png::BitDepth;

/// Smallest indexed-PNG bit depth that holds `palette_size` distinct
/// indices. `None` for sizes outside `1..=256`.
fn bit_depth_for(palette_size: usize) -> Option<BitDepth> {
    match palette_size {
        1..=2 => Some(BitDepth::One),
        3..=4 => Some(BitDepth::Two),
        5..=16 => Some(BitDepth::Four),
        17..=256 => Some(BitDepth::Eight),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteSizeOutOfRange(pub usize);

impl core::fmt::Display for PaletteSizeOutOfRange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "palette size {} is out of range for indexed PNG (must be 1..=256)",
            self.0
        )
    }
}

impl core::error::Error for PaletteSizeOutOfRange {}

/// A palette whose size has been checked to fit indexed-PNG's 1/2/4/8-bit
/// layouts (1..=256 entries). Construct once via [`VerifiedPalette::new`];
/// [`PaletteImage::new`] then takes one infallibly.
#[derive(Clone)]
pub struct VerifiedPalette {
    /// Smallest indexed PNG bit depth that holds `palette.len()` indices.
    pub bit_depth: BitDepth,
    pub palette: Vec<Rgb<u8>>,
}

impl VerifiedPalette {
    pub fn new(palette: Vec<Rgb<u8>>) -> Result<Self, PaletteSizeOutOfRange> {
        let bit_depth = bit_depth_for(palette.len()).ok_or(PaletteSizeOutOfRange(palette.len()))?;
        Ok(Self { bit_depth, palette })
    }
}

/// A palette-indexed image: a packed `bit_depth`-bpp index array plus the
/// [`VerifiedPalette`] that interprets it. Implements [`ImageWriter<usize>`]
/// so it drops straight into the dither pipeline as the output sink, and
/// [`to_png`](Self::to_png) emits a ready-to-write indexed PNG.
pub struct PaletteImage {
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: usize,
    pub palette: VerifiedPalette,
    /// Packed indices.
    pub data: Vec<u8>,
}

impl PaletteImage {
    pub fn new(width: u32, height: u32, palette: VerifiedPalette) -> Self {
        let bpp = palette.bit_depth as usize;
        let px_per_byte = 8 / bpp;
        let bytes_per_row = (width as usize).div_ceil(px_per_byte);
        Self {
            width,
            height,
            bytes_per_row,
            palette,
            data: vec![0u8; bytes_per_row * height as usize],
        }
    }

    pub fn to_png(&self) -> Result<Vec<u8>, png::EncodingError> {
        let mut png_bytes: Vec<u8> = Vec::new();
        let mut encoder = png::Encoder::new(&mut png_bytes, self.width, self.height);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(self.palette.bit_depth);
        let palette_bytes: Vec<u8> = self.palette.palette.iter().flat_map(|rgb| rgb.0).collect();
        encoder.set_palette(palette_bytes);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.data)?;
        drop(writer);
        Ok(png_bytes)
    }
}

impl ImageSize for PaletteImage {
    fn width(&self) -> usize {
        self.width as usize
    }
    fn height(&self) -> usize {
        self.height as usize
    }
}

impl ImageReader<usize> for PaletteImage {
    fn get_pixel(&self, x: usize, y: usize) -> usize {
        let bpp = self.palette.bit_depth as usize;
        let px_per_byte = 8 / bpp;
        let byte_x = x / px_per_byte;
        let shift = (px_per_byte - 1 - (x % px_per_byte)) * bpp;
        let mask = ((1u32 << bpp) - 1) as u8;
        let byte = self.data[y * self.bytes_per_row + byte_x];
        ((byte >> shift) & mask) as usize
    }
}

impl ImageWriter<usize> for PaletteImage {
    fn put_pixel(&mut self, x: usize, y: usize, pixel: usize) {
        if x >= self.width as usize || y >= self.height as usize {
            return;
        }
        let bpp = self.palette.bit_depth as usize;
        let px_per_byte = 8 / bpp;
        let byte_x = x / px_per_byte;
        // MSB-first: the leftmost pixel occupies the highest bits.
        let shift = (px_per_byte - 1 - (x % px_per_byte)) * bpp;
        let mask = ((1u32 << bpp) - 1) as u8;
        let byte = &mut self.data[y * self.bytes_per_row + byte_x];
        *byte = (*byte & !(mask << shift)) | (((pixel as u8) & mask) << shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn writer(width: u32, height: u32, palette_size: usize) -> PaletteImage {
        let palette: Vec<Rgb<u8>> = (0..palette_size).map(|_| Rgb([0, 0, 0])).collect();
        PaletteImage::new(width, height, VerifiedPalette::new(palette).unwrap())
    }

    #[test]
    fn bit_depth_for_palette_sizes() {
        assert_eq!(bit_depth_for(0), None);
        assert_eq!(bit_depth_for(1), Some(BitDepth::One));
        assert_eq!(bit_depth_for(2), Some(BitDepth::One));
        assert_eq!(bit_depth_for(4), Some(BitDepth::Two));
        assert_eq!(bit_depth_for(6), Some(BitDepth::Four));
        assert_eq!(bit_depth_for(16), Some(BitDepth::Four));
        assert_eq!(bit_depth_for(17), Some(BitDepth::Eight));
        assert_eq!(bit_depth_for(256), Some(BitDepth::Eight));
        assert_eq!(bit_depth_for(257), None);
    }

    #[test]
    fn verified_palette_rejects_empty() {
        assert!(VerifiedPalette::new(vec![]).is_err());
    }

    #[test]
    fn verified_palette_rejects_oversized() {
        let palette: Vec<Rgb<u8>> = (0..257).map(|_| Rgb([0, 0, 0])).collect();
        assert!(VerifiedPalette::new(palette).is_err());
    }

    #[test]
    fn pack_8bit_writes_one_byte_per_pixel() {
        let mut w = writer(3, 2, 256);
        w.put_pixel(0, 0, 0xAA);
        w.put_pixel(2, 1, 0xBB);
        assert_eq!(w.data, vec![0xAA, 0, 0, 0, 0, 0xBB]);
    }

    #[test]
    fn pack_4bit_msb_first_and_rounds_rows() {
        let mut w = writer(3, 1, 16);
        w.put_pixel(0, 0, 0x0);
        w.put_pixel(1, 0, 0xA);
        w.put_pixel(2, 0, 0x5);
        // Row has 3 pixels at 4 bpp => 2 bytes (second byte's low nibble is padding).
        assert_eq!(w.data, vec![0x0A, 0x50]);
    }

    #[test]
    fn pack_2bit_packs_four_per_byte() {
        let mut w = writer(5, 1, 4);
        for (x, v) in [0usize, 1, 2, 3, 1].iter().enumerate() {
            w.put_pixel(x, 0, *v);
        }
        // 00 01 10 11 | 01 00 00 00  =>  0b00011011, 0b01000000
        assert_eq!(w.data, vec![0b00_01_10_11, 0b01_00_00_00]);
    }

    #[test]
    fn pack_1bit_msb_first() {
        let mut w = writer(9, 1, 2);
        for (x, v) in [1usize, 0, 1, 1, 0, 0, 1, 0, 1].iter().enumerate() {
            w.put_pixel(x, 0, *v);
        }
        // 10110010 | 10000000
        assert_eq!(w.data, vec![0b1011_0010, 0b1000_0000]);
    }

    #[test]
    fn put_pixel_overwrites_on_repeat() {
        let mut w = writer(2, 1, 16);
        w.put_pixel(0, 0, 0xF);
        w.put_pixel(0, 0, 0x3);
        assert_eq!(w.data, vec![0x30]);
    }

    #[test]
    fn put_pixel_out_of_bounds_is_ignored() {
        let mut w = writer(2, 2, 16);
        w.put_pixel(2, 0, 0xF);
        w.put_pixel(0, 2, 0xF);
        w.put_pixel(99, 99, 0xF);
        assert_eq!(w.data, vec![0x00, 0x00]);
    }

    #[test]
    fn get_pixel_round_trips_writes() {
        let mut w = writer(7, 3, 16);
        for y in 0..3 {
            for x in 0..7 {
                w.put_pixel(x, y, ((x * y + 1) & 0xF) as usize);
            }
        }
        for y in 0..3 {
            for x in 0..7 {
                assert_eq!(w.get_pixel(x, y), ((x * y + 1) & 0xF) as usize);
            }
        }
    }
}
