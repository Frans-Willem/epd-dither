//! `image`-crate adapter that plugs a `GenericImage` into the
//! [`crate::dither::diffuse`] pipeline. Available behind the `image`
//! Cargo feature.

use crate::dither::diffuse::{ImageReader, ImageSize, ImageWriter};

/// Combines an [`image::GenericImage`] source with a per-pixel noise function
/// and a caller-supplied paletted-output writer, exposing the trio as a
/// single value usable with [`diffuse_dither`](crate::dither::diffuse::diffuse_dither).
///
/// `image` itself has no paletted image type, so the writer half is left
/// fully generic: provide any [`ImageWriter`] (writing into a flat
/// `Vec<usize>`, a paletted PNG row buffer, a framebuffer, …). The reader
/// half pairs the input pixel with an optional noise sample so callers can
/// thread pre-baked noise (Bayer, IGN, white, file-driven) through the
/// pipeline.
pub struct PaletteDitheringWithNoise<I, F, W> {
    pub image: I,
    pub noise_fn: F,
    pub writer: W,
}

impl<I, F, W> ImageSize for PaletteDitheringWithNoise<I, F, W>
where
    I: image::GenericImage,
    F: Fn(usize, usize) -> Option<f32>,
    W: ImageWriter<usize>,
{
    fn width(&self) -> usize {
        self.image.width() as usize
    }
    fn height(&self) -> usize {
        self.image.height() as usize
    }
}

impl<I, F, W> ImageReader<(I::Pixel, Option<f32>)> for PaletteDitheringWithNoise<I, F, W>
where
    I: image::GenericImage,
    F: Fn(usize, usize) -> Option<f32>,
    W: ImageWriter<usize>,
{
    fn get_pixel(&self, x: usize, y: usize) -> (I::Pixel, Option<f32>) {
        (
            self.image.get_pixel(x as u32, y as u32),
            (self.noise_fn)(x, y),
        )
    }
}

impl<I, F, W> ImageWriter<usize> for PaletteDitheringWithNoise<I, F, W>
where
    I: image::GenericImage,
    F: Fn(usize, usize) -> Option<f32>,
    W: ImageWriter<usize>,
{
    fn put_pixel(&mut self, x: usize, y: usize, pixel: usize) {
        self.writer.put_pixel(x, y, pixel);
    }
}
