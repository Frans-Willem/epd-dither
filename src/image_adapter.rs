//! `image`-crate adapter that plugs a `GenericImage` into the
//! [`crate::dither::diffuse`] pipeline. Available behind the `image`
//! Cargo feature.
//!
//! These blanket impls let any [`image::GenericImageView`] act as an
//! [`ImageReader`] (and any [`image::GenericImage`] as both reader and
//! [`ImageWriter`]) directly — no wrapper required for the in-place case.
//! When the read side and write side need different concrete types, pair
//! them with [`crate::dither::ImageSplit`].

use crate::dither::image_traits::{ImageReader, ImageSize, ImageWriter};

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
