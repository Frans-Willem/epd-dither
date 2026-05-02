#[cfg(feature = "alloc")]
use crate::Decomposer;
#[cfg(feature = "alloc")]
use crate::dither::diffuse::PixelStrategy;
#[cfg(feature = "alloc")]
use core::marker::PhantomData;
#[cfg(feature = "alloc")]
use core::ops::{AddAssign, Div, Mul};
#[cfg(feature = "alloc")]
use nalgebra::DVector;

/// Pixel strategy that decomposes a colour-space input into per-palette
/// weights via a [`Decomposer`], then picks one palette index per pixel —
/// either by sampling a positional noise value against the cumulative
/// weights, or (when no noise function is configured) by taking the
/// dominant component.
///
/// The source pixel type `Src` is converted into the decomposer's input by
/// `convert`. Positional noise is queried inside `quantize` from the
/// strategy's own `noise` field.
///
/// `noise` is `Option<N>` where `N: Fn(usize, usize) -> f32`: `None` means
/// no noise is applied (dominant-component selection); `Some(n)` means
/// `n(x, y)` is sampled per pixel. Default after [`new`](Self::new) is
/// `None`; use [`with_noise`](Self::with_noise) to plug one in.
///
/// The strategy emits a `usize` palette index as its target and a
/// per-component quantization error; whether and how that error is propagated
/// is the caller's choice via the [`DiffusionMatrix`](crate::dither::diffusion_matrix::DiffusionMatrix)
/// passed to [`diffuse_dither`](crate::dither::diffuse::diffuse_dither)
/// (use [`NO_DIFFUSE`](crate::dither::diffusion_matrix::NO_DIFFUSE) to
/// skip diffusion entirely).
#[cfg(feature = "alloc")]
pub struct DecomposingDitherStrategy<D, F, N, Src> {
    pub decomposer: D,
    pub convert: F,
    pub noise: Option<N>,
    _phantom: PhantomData<fn(Src)>,
}

/// Placeholder `N` for the noise-less default: the function pointer is
/// never called because `noise` is `None`, but `N` still needs a concrete
/// type. Bare `fn(usize, usize) -> f32` accepts any later [`with_noise`]
/// override that coerces to a fn pointer; capturing closures need an
/// explicit turbofish on `with_noise`.
#[cfg(feature = "alloc")]
type DefaultNoiseFn = fn(usize, usize) -> f32;

#[cfg(feature = "alloc")]
impl<D, F, Src> DecomposingDitherStrategy<D, F, DefaultNoiseFn, Src> {
    pub fn new(decomposer: D, convert: F) -> Self {
        Self {
            decomposer,
            convert,
            noise: None,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
impl<D, F, N, Src> DecomposingDitherStrategy<D, F, N, Src> {
    pub fn with_noise<N2>(self, noise: N2) -> DecomposingDitherStrategy<D, F, N2, Src>
    where
        N2: Fn(usize, usize) -> f32,
    {
        DecomposingDitherStrategy {
            decomposer: self.decomposer,
            convert: self.convert,
            noise: Some(noise),
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "alloc")]
#[derive(Clone, Default)]
pub struct DecomposedQuantizationError(Option<DVector<f32>>);

#[cfg(feature = "alloc")]
impl Mul<usize> for DecomposedQuantizationError {
    type Output = Self;
    fn mul(self, rhs: usize) -> Self {
        Self(self.0.map(|x| x * (rhs as f32)))
    }
}

#[cfg(feature = "alloc")]
impl Div<usize> for DecomposedQuantizationError {
    type Output = Self;
    fn div(self, rhs: usize) -> Self {
        Self(self.0.map(|x| x / (rhs as f32)))
    }
}

#[cfg(feature = "alloc")]
impl AddAssign<DecomposedQuantizationError> for DecomposedQuantizationError {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = match (core::mem::take(&mut self.0), rhs.0) {
            (a, None) => a,
            (None, b) => b,
            (Some(a), Some(b)) => Some(a + b),
        }
    }
}

#[cfg(feature = "alloc")]
impl<D, F, N, Src> PixelStrategy for DecomposingDitherStrategy<D, F, N, Src>
where
    D: Decomposer<f32>,
    F: Fn(Src) -> D::Input,
    N: Fn(usize, usize) -> f32,
{
    type Source = Src;
    type Target = usize;
    type QuantizationError = DecomposedQuantizationError;

    fn quantize(
        &self,
        source: Self::Source,
        x: usize,
        y: usize,
        error: Self::QuantizationError,
    ) -> (Self::Target, Self::QuantizationError) {
        let noise = self.noise.as_ref().map(|n| n(x, y));
        let mut decomposed = DVector::zeros(self.decomposer.palette_size());
        self.decomposer
            .decompose_into(&(self.convert)(source), decomposed.as_mut_slice());
        let decomposed = match error.0 {
            None => decomposed,
            Some(error) => decomposed + error,
        };
        let decomposed_clipped = decomposed.map(|x| if x < 0.0 { 0.0 } else { x });
        let decomposed_clipped_sum = decomposed_clipped.sum();
        let index = if let Some(noise) = noise
            && decomposed_clipped_sum > 0.0
        {
            let mut noise = noise * decomposed_clipped_sum;
            let mut index: usize = 0;
            while index + 1 < decomposed_clipped.nrows() && noise >= decomposed_clipped[index] {
                noise -= decomposed_clipped[index];
                index += 1;
            }
            index
        } else {
            decomposed.argmax().0
        };
        let mut error = decomposed;
        error[index] -= 1.0;
        (index, DecomposedQuantizationError(Some(error)))
    }
}

/// Library-grade enum equivalent of the binary's `--strategy` argument:
/// names a built-in decomposition strategy. The factory layer (see
/// [`crate::factory`], `image` feature) maps each variant to the
/// corresponding decomposer + convert function.
///
/// `Octahedron` and `Naive` carry the inner decomposer's strategy enum
/// directly, so they expose every axis/blend variant the inner enum
/// supports — not just the ones with a string spelling in `FromStr`.
///
/// No-spread 1-D grayscale is `GrayOffsetBlend(0.0)`, which the offset-blend
/// decomposer collapses to plain bracket decomposition via its `distance <= 0`
/// early-out. The string `"grayscale"` parses to that value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecomposeStrategy {
    Octahedron(crate::decompose::octahedron::OctahedronDecomposerAxisStrategy),
    Naive(crate::decompose::naive::NaiveDecomposerStrategy),
    /// `PureSpreadGrayDecomposer` with the given spread ratio in [0, 1].
    GrayPureSpread(f32),
    /// `OffsetBlendGrayDecomposer` with the given offset in input-space units.
    GrayOffsetBlend(f32),
}

impl DecomposeStrategy {
    pub const LONG_HELP: &'static str = concat!(
        "Decomposition strategy.\n\n",
        "Accepted values:\n",
        " octahedron-closest        Octahedron, pick closest axis (default)\n",
        " octahedron-furthest       Octahedron, pick furthest axis\n",
        " naive-mix                 Naive, favour mixed weights\n",
        " naive-dominant            Naive, favour dominant component\n",
        " naive-blend[:<p>]         Naive, smooth blend (default p=1)\n",
        " grayscale                 1-D grayscale, no spread\n",
        " gray-pure-spread:<r>      Pure-spread grayscale, r in [0, 1]\n",
        " gray-offset-blend:<r>     Offset-blend grayscale, r in [0, 1]\n\n",
        "Examples:\n",
        " --strategy octahedron-closest\n",
        " --strategy grayscale\n",
        " --strategy gray-pure-spread:0.25\n",
        " --strategy naive-blend:2\n",
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidDecomposeStrategy;

impl core::fmt::Display for InvalidDecomposeStrategy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid decompose-strategy name")
    }
}

impl core::error::Error for InvalidDecomposeStrategy {}

fn parse_unit_interval(s: &str) -> Result<f32, InvalidDecomposeStrategy> {
    let v = s.parse::<f32>().map_err(|_| InvalidDecomposeStrategy)?;
    if !(0.0..=1.0).contains(&v) {
        return Err(InvalidDecomposeStrategy);
    }
    Ok(v)
}

impl core::str::FromStr for DecomposeStrategy {
    type Err = InvalidDecomposeStrategy;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "grayscale" {
            return Ok(Self::GrayOffsetBlend(0.0));
        }
        if let Some(rest) = s.strip_prefix("octahedron-") {
            return Ok(Self::Octahedron(
                rest.parse().map_err(|_| InvalidDecomposeStrategy)?,
            ));
        }
        if let Some(rest) = s.strip_prefix("naive-") {
            return Ok(Self::Naive(
                rest.parse().map_err(|_| InvalidDecomposeStrategy)?,
            ));
        }
        if let Some(rest) = s.strip_prefix("gray-pure-spread:") {
            return Ok(Self::GrayPureSpread(parse_unit_interval(rest)?));
        }
        if let Some(rest) = s.strip_prefix("gray-offset-blend:") {
            return Ok(Self::GrayOffsetBlend(parse_unit_interval(rest)?));
        }
        Err(InvalidDecomposeStrategy)
    }
}
