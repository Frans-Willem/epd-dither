//! Minimal image-shaped reader/writer traits used by the dither pipeline,
//! plus an [`ImageCombinedRW`] combiner. None of these depend on the `image`
//! Cargo feature; that crate's adapter lives in
//! [`crate::image::adapter`] and provides blanket impls of these traits
//! for [`image::GenericImage`] / [`image::GenericImageView`].

pub trait ImageSize {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
}

pub trait ImageReader<T> {
    fn get_pixel(&self, x: usize, y: usize) -> T;
}

pub trait ImageWriter<T> {
    fn put_pixel(&mut self, x: usize, y: usize, pixel: T);
}

/// Pairs a separate read side and write side into a single value usable
/// with [`diffuse_dither`](crate::dither::diffuse::diffuse_dither). The
/// pixel types on the two sides are independent, which is the typical
/// shape for paletted dithering: read `Rgb<f32>` (or whatever), write
/// `usize` palette indices.
///
/// Both sides must report matching dimensions; [`ImageCombinedRW::new`] checks
/// this and returns `None` on mismatch. Reported size on the combined
/// value comes from the reader.
pub struct ImageCombinedRW<R, W> {
    pub reader: R,
    pub writer: W,
}

impl<R, W> ImageCombinedRW<R, W>
where
    R: ImageSize,
    W: ImageSize,
{
    pub fn new(reader: R, writer: W) -> Option<Self> {
        if reader.width() == writer.width() && reader.height() == writer.height() {
            Some(Self { reader, writer })
        } else {
            None
        }
    }
}

impl<R, W> ImageSize for ImageCombinedRW<R, W>
where
    R: ImageSize,
{
    fn width(&self) -> usize {
        self.reader.width()
    }
    fn height(&self) -> usize {
        self.reader.height()
    }
}

impl<R, W, T> ImageReader<T> for ImageCombinedRW<R, W>
where
    R: ImageReader<T>,
{
    fn get_pixel(&self, x: usize, y: usize) -> T {
        self.reader.get_pixel(x, y)
    }
}

impl<R, W, T> ImageWriter<T> for ImageCombinedRW<R, W>
where
    W: ImageWriter<T>,
{
    fn put_pixel(&mut self, x: usize, y: usize, pixel: T) {
        self.writer.put_pixel(x, y, pixel)
    }
}
