//! `image`-crate adapter that plugs a `GenericImage` into the
//! [`crate::dither::diffuse`] pipeline. Available behind the `image`
//! Cargo feature.
//!
//! These blanket impls let any [`image::GenericImageView`] act as an
//! [`ImageReader`] (and any [`image::GenericImage`] as both reader and
//! [`ImageWriter`]) directly — no wrapper required for the in-place case.
//! When the read side and write side need different concrete types, pair
//! them with [`crate::dither::ImageCombinedRW`].

use crate::decomposer_input::DecomposerInputColor;
use crate::dither::image_traits::{ImageReader, ImageSize, ImageWriter};
use nalgebra::geometry::Point3;

impl DecomposerInputColor for image::Rgb<f32> {
    fn to_point(&self) -> Point3<f32> {
        let [r, g, b] = self.0;
        Point3::new(r, g, b)
    }
    fn brightness(&self) -> f32 {
        let [r, g, b] = self.0;
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }
    fn is_grayscale(&self) -> bool {
        let [r, g, b] = self.0;
        r == g && g == b
    }
}

impl DecomposerInputColor for image::Rgb<u8> {
    fn to_point(&self) -> Point3<f32> {
        let [r, g, b] = self.0;
        Point3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
    }
    fn brightness(&self) -> f32 {
        let [r, g, b] = self.0;
        let r = r as f32 / 255.0;
        let g = g as f32 / 255.0;
        let b = b as f32 / 255.0;
        0.2126 * r + 0.7152 * g + 0.0722 * b
    }
    fn is_grayscale(&self) -> bool {
        let [r, g, b] = self.0;
        r == g && g == b
    }
}

impl<I> ImageSize for I
where
    I: image::GenericImageView,
{
    fn width(&self) -> usize {
        image::GenericImageView::width(self) as usize
    }
    fn height(&self) -> usize {
        image::GenericImageView::height(self) as usize
    }
}

impl<I> ImageReader<I::Pixel> for I
where
    I: image::GenericImageView,
{
    fn get_pixel(&self, x: usize, y: usize) -> I::Pixel {
        image::GenericImageView::get_pixel(self, x as u32, y as u32)
    }
}

impl<I> ImageWriter<I::Pixel> for I
where
    I: image::GenericImage,
{
    fn put_pixel(&mut self, x: usize, y: usize, pixel: I::Pixel) {
        image::GenericImage::put_pixel(self, x as u32, y as u32, pixel)
    }
}
