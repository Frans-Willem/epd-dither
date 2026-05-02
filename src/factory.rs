//! Convenience layer: build a `Box<dyn DynDitherer<T>>` from typed
//! configuration enums (via [`decompose_ditherer`]) or from strings (via
//! [`parse_decompose_ditherer`], which parses then delegates).
//!
//! Generic over the pixel type via [`DecomposerInputColor`]: the same colour value
//! is used both as a source pixel (read out of `T`) and as a palette
//! entry (in the slice passed in). Image-using callers get default
//! impls for `image::Rgb<u8>` / `image::Rgb<f32>` (see
//! [`crate::image_adapter`], gated on the `image` feature); the
//! [`[u8; 3]`](DecomposerInputColor) impl in [`crate::decomposer_input`] covers the
//! image-free case.
//!
//! [`NoiseSource::File`] and [`NoiseSource::Blue`] arms are gated on the
//! `image` feature — they decode an image — but the rest of the factory
//! works without it, including the all-strings entry
//! [`parse_decompose_ditherer`].

use crate::Decomposer;
use crate::decomposer_input::DecomposerInputColor;
use crate::decompose::gray::{OffsetBlendGrayDecomposer, PureSpreadGrayDecomposer};
use crate::decompose::naive::NaiveDecomposer;
use crate::decompose::octahedron::OctahedronDecomposer;
use crate::dither::diffusion_matrix::{DiffuseMethod, DiffusionMatrix, InvalidDiffuseMethod};
use crate::dither::{
    BundledDitherer, DecomposeStrategy, DecomposingDitherStrategy, DynDitherer, ImageReader,
    ImageSize, ImageWriter, InvalidDecomposeStrategy,
};
use crate::noise::{InvalidNoiseSource, NoiseSource};
use crate::palette::{InvalidPalette, Palette};
use alloc::boxed::Box;
use alloc::vec::Vec;
use nalgebra::geometry::Point3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactoryError {
    InvalidStrategy(InvalidDecomposeStrategy),
    InvalidNoise(InvalidNoiseSource),
    InvalidPalette(InvalidPalette),
    InvalidDiffuse(InvalidDiffuseMethod),
    /// Grayscale strategy chosen with a non-achromatic or non-ascending
    /// palette.
    NonGrayscalePalette,
    /// Decomposer construction returned `None` (e.g. the octahedron
    /// palette doesn't form a valid octahedron).
    DecomposerBuildFailed,
    /// Failed to load or decode an external noise image.
    #[cfg(feature = "image")]
    NoiseImageError,
}

impl core::fmt::Display for FactoryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidStrategy(e) => write!(f, "{e}"),
            Self::InvalidNoise(e) => write!(f, "{e}"),
            Self::InvalidPalette(e) => write!(f, "{e}"),
            Self::InvalidDiffuse(e) => write!(f, "{e}"),
            Self::NonGrayscalePalette => f.write_str(
                "grayscale strategy requires an achromatic, strictly-ascending palette",
            ),
            Self::DecomposerBuildFailed => f.write_str("decomposer construction failed"),
            #[cfg(feature = "image")]
            Self::NoiseImageError => f.write_str("failed to load or decode noise image"),
        }
    }
}

impl core::error::Error for FactoryError {}

/// True iff every entry is achromatic and the entries are strictly
/// ascending in brightness.
fn verify_grayscale_palette<Q: DecomposerInputColor>(p: &[Q]) -> bool {
    p.iter().all(|c| c.is_grayscale())
        && p.windows(2).all(|w| w[0].brightness() < w[1].brightness())
}

fn build_decomposing<D, F, Src, N, T>(
    decomposer: D,
    convert: F,
    noise_fn: Option<N>,
    matrix: impl DiffusionMatrix + Send + Sync + 'static,
) -> Box<dyn DynDitherer<T> + Send + Sync>
where
    D: Decomposer<f32> + Send + Sync + 'static,
    F: Fn(Src) -> D::Input + Send + Sync + 'static,
    Src: 'static,
    N: Fn(usize, usize) -> f32 + Send + Sync + 'static,
    T: ImageSize + ImageReader<Src> + ImageWriter<usize> + ?Sized + 'static,
{
    let strategy = DecomposingDitherStrategy::new(decomposer, convert);
    match noise_fn {
        Some(n) => Box::new(BundledDitherer::new(strategy.with_noise(n), matrix)),
        None => Box::new(BundledDitherer::new(strategy, matrix)),
    }
}

fn build_with_noise<P, Q, N, T>(
    strategy: DecomposeStrategy,
    palette: &[Q],
    noise_fn: Option<N>,
    matrix: impl DiffusionMatrix + Send + Sync + 'static,
) -> Result<Box<dyn DynDitherer<T> + Send + Sync>, FactoryError>
where
    P: DecomposerInputColor + 'static,
    Q: DecomposerInputColor,
    N: Fn(usize, usize) -> f32 + Send + Sync + 'static,
    T: ImageSize + ImageReader<P> + ImageWriter<usize> + ?Sized + 'static,
{
    match strategy {
        DecomposeStrategy::Octahedron(axis) => {
            let palette_points: Vec<Point3<f32>> = palette.iter().map(|q| q.to_point()).collect();
            let decomposer = OctahedronDecomposer::new(&palette_points)
                .ok_or(FactoryError::DecomposerBuildFailed)?
                .with_strategy(axis);
            Ok(build_decomposing(
                decomposer,
                |p: P| p.to_point(),
                noise_fn,
                matrix,
            ))
        }
        DecomposeStrategy::Naive(naive) => {
            let palette_points: Vec<Point3<f32>> = palette.iter().map(|q| q.to_point()).collect();
            let decomposer = NaiveDecomposer::new(&palette_points)
                .ok_or(FactoryError::DecomposerBuildFailed)?
                .with_strategy(naive);
            Ok(build_decomposing(
                decomposer,
                |p: P| p.to_point(),
                noise_fn,
                matrix,
            ))
        }
        DecomposeStrategy::GrayPureSpread(spread) => {
            if !verify_grayscale_palette(palette) {
                return Err(FactoryError::NonGrayscalePalette);
            }
            let levels: Vec<f32> = palette.iter().map(|q| q.brightness()).collect();
            let decomposer = PureSpreadGrayDecomposer::new(levels)
                .ok_or(FactoryError::DecomposerBuildFailed)?
                .with_spread_ratio(spread);
            Ok(build_decomposing(
                decomposer,
                |p: P| p.brightness(),
                noise_fn,
                matrix,
            ))
        }
        DecomposeStrategy::GrayOffsetBlend(distance) => {
            if !verify_grayscale_palette(palette) {
                return Err(FactoryError::NonGrayscalePalette);
            }
            let levels: Vec<f32> = palette.iter().map(|q| q.brightness()).collect();
            let decomposer = OffsetBlendGrayDecomposer::new(levels)
                .ok_or(FactoryError::DecomposerBuildFailed)?
                .with_distance(distance);
            Ok(build_decomposing(
                decomposer,
                |p: P| p.brightness(),
                noise_fn,
                matrix,
            ))
        }
    }
}

#[cfg(feature = "image")]
fn sample_luma_image(
    img: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>,
    x: usize,
    y: usize,
) -> f32 {
    img.get_pixel(x as u32 % img.width(), y as u32 % img.height())
        .0[0]
}

/// Build a `Box<dyn DynDitherer<T> + Send + Sync>` from already-parsed
/// configuration. Use [`parse_decompose_ditherer`] for the all-strings
/// entry point.
///
/// The returned trait object is `Send + Sync` so it can be moved into a
/// long-lived shared owner (e.g. an `Arc` shared across worker threads).
/// Every concrete piece the factory composes — built-in decomposers,
/// closures with captured constants, [`RefDiffusionMatrix`] — already
/// satisfies both auto traits.
pub fn decompose_ditherer<P, Q, T>(
    strategy: DecomposeStrategy,
    noise: NoiseSource,
    palette: &[Q],
    matrix: impl DiffusionMatrix + Send + Sync + 'static,
) -> Result<Box<dyn DynDitherer<T> + Send + Sync>, FactoryError>
where
    P: DecomposerInputColor + 'static,
    Q: DecomposerInputColor,
    T: ImageSize + ImageReader<P> + ImageWriter<usize> + ?Sized + 'static,
{
    match noise {
        NoiseSource::None => build_with_noise::<P, Q, fn(usize, usize) -> f32, T>(
            strategy, palette, None, matrix,
        ),
        NoiseSource::Bayer(Some(n)) => build_with_noise(
            strategy,
            palette,
            Some(move |x, y| crate::noise::bayer(x, y, n)),
            matrix,
        ),
        NoiseSource::Bayer(None) => build_with_noise(
            strategy,
            palette,
            Some(|x, y| crate::noise::bayer_inf(x, y)),
            matrix,
        ),
        NoiseSource::InterleavedGradient => build_with_noise(
            strategy,
            palette,
            Some(|x, y| crate::noise::interleaved_gradient_noise(x as f32, y as f32)),
            matrix,
        ),
        #[cfg(feature = "rand")]
        NoiseSource::White => {
            use rand::Rng;
            use rand::distr::StandardUniform;
            build_with_noise(
                strategy,
                palette,
                Some(|_x, _y| rand::rng().sample::<f32, _>(StandardUniform)),
                matrix,
            )
        }
        #[cfg(feature = "image")]
        NoiseSource::File(path) => {
            let img = image::ImageReader::open(&path)
                .map_err(|_| FactoryError::NoiseImageError)?
                .decode()
                .map_err(|_| FactoryError::NoiseImageError)?
                .to_luma32f();
            build_with_noise(
                strategy,
                palette,
                Some(move |x, y| sample_luma_image(&img, x, y)),
                matrix,
            )
        }
        #[cfg(feature = "image")]
        NoiseSource::Blue => {
            let img = image::load_from_memory(crate::noise::BLUE_NOISE_PNG)
                .map_err(|_| FactoryError::NoiseImageError)?
                .to_luma32f();
            build_with_noise(
                strategy,
                palette,
                Some(move |x, y| sample_luma_image(&img, x, y)),
                matrix,
            )
        }
    }
}

/// All-strings entry point: parse strategy/noise/palette/diffuse, then
/// hand off to [`decompose_ditherer`]. Generic over the source
/// pixel type `P`; the palette entry type is fixed to `[u8; 3]` since
/// that's what the [`Palette`] enum's slice accessor returns.
///
/// Returns `Box<dyn DynDitherer<T> + Send + Sync>` — see
/// [`decompose_ditherer`] for the rationale.
pub fn parse_decompose_ditherer<P, T>(
    strategy: &str,
    noise: &str,
    palette: &str,
    diffuse: &str,
) -> Result<Box<dyn DynDitherer<T> + Send + Sync>, FactoryError>
where
    P: DecomposerInputColor + 'static,
    T: ImageSize + ImageReader<P> + ImageWriter<usize> + ?Sized + 'static,
{
    let strategy: DecomposeStrategy = strategy.parse().map_err(FactoryError::InvalidStrategy)?;
    let noise: NoiseSource = noise.parse().map_err(FactoryError::InvalidNoise)?;
    let palette: Palette = palette.parse().map_err(FactoryError::InvalidPalette)?;
    let diffuse: DiffuseMethod = diffuse.parse().map_err(FactoryError::InvalidDiffuse)?;
    decompose_ditherer::<P, [u8; 3], T>(
        strategy,
        noise,
        palette.as_rgb_slice(),
        diffuse.to_matrix(),
    )
}
