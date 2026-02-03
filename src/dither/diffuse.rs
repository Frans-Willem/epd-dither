use crate::dither::diffusion_matrix::DiffusionMatrix;
use alloc::vec::Vec;
use core::ops::{AddAssign, Div, Mul};

pub trait PixelStrategy {
    type Source;
    type Target;
    type QuantizationError: Default
        + Clone
        + Mul<usize, Output = Self::QuantizationError>
        + Div<usize, Output = Self::QuantizationError>
        + AddAssign<Self::QuantizationError>;

    fn quantize(
        &self,
        source: Self::Source,
        error: Self::QuantizationError,
    ) -> (Self::Target, Self::QuantizationError);
}

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

fn add_usize_usize_clamped(a: usize, b: usize, limit: usize) -> Option<usize> {
    if a < limit && limit - a > b {
        Some(a + b)
    } else {
        None
    }
}

fn add_usize_isize_clamped(a: usize, b: isize, limit: usize) -> Option<usize> {
    if b > 0 {
        add_usize_usize_clamped(a, b as usize, limit)
    } else if b < 0 {
        let neg_b = (0 - b) as usize;
        if neg_b >= a { None } else { Some(a - neg_b) }
    } else {
        // b == 0
        if a < limit { Some(a) } else { None }
    }
}

struct RangeWithDir {
    min_inclusive: usize,
    max_exclusive: usize,
    dir: isize,
}

impl RangeWithDir {
    fn new(min_inclusive: usize, max_exclusive: usize, dir: isize) -> Self {
        Self {
            min_inclusive,
            max_exclusive,
            dir,
        }
    }
}

impl Iterator for RangeWithDir {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        if self.dir > 0 {
            if self.min_inclusive < self.max_exclusive {
                let ret = Some(self.min_inclusive);
                self.min_inclusive += self.dir as usize;
                ret
            } else {
                None
            }
        } else if self.dir < 0 {
            let neg_dir = (0 - self.dir) as usize;
            if self.max_exclusive - self.min_inclusive >= neg_dir {
                self.max_exclusive -= neg_dir;
                Some(self.max_exclusive)
            } else {
                None
            }
        } else {
            None
        }
    }
}

pub fn diffuse_dither<
    S: PixelStrategy,
    M: DiffusionMatrix,
    I: ImageSize + ImageReader<S::Source> + ImageWriter<S::Target>,
>(
    strategy: S,
    matrix: M,
    inout: &mut I,
    serpentine: bool,
) {
    // Store width and height once for easy access and to make sure it doesn't change out from under
    // us ;)
    let width = inout.width();
    let height = inout.height();
    // Get divisor & diffusion targets
    let error_divisor = matrix.divisor();
    let diffuse_targets = matrix.targets();
    // Find maximum y diffuse and height of error matrix (We're only ever working with a couple of
    // rows at a time, no need to allocate a full extra image)
    let max_y_diffuse = diffuse_targets
        .iter()
        .map(|(_, dy, _)| *dy)
        .min()
        .unwrap_or(0);
    let errors_height = max_y_diffuse + 1;
    let mut errors: Vec<S::QuantizationError> = Vec::new();
    errors.resize_with(width * errors_height, Default::default);
    for y in 0..height {
        let dir: isize = if serpentine && (y % 2) == 1 { -1 } else { 1 };
        for x in RangeWithDir::new(0, width, dir) {
            let source: S::Source = inout.get_pixel(x, y);
            // Use core::mem::take such that the error will be reset, as it will be re-used for the
            // next row.
            let error: S::QuantizationError =
                core::mem::take(&mut errors[x + ((y % errors_height) * width)]);
            let (target, error) = strategy.quantize(source, error / error_divisor);
            inout.put_pixel(x, y, target);
            // Diffuse the error
            for (dx, dy, mul) in diffuse_targets {
                if let (Some(tx), Some(ty)) = (
                    add_usize_isize_clamped(x, dx * dir, width),
                    add_usize_usize_clamped(y, *dy, height),
                ) {
                    errors[tx + ((ty % errors_height) * width)] += error.clone() * *mul;
                }
            }
        }
    }
}
