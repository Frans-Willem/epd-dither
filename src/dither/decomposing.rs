use crate::Decomposer;
use crate::dither::diffuse::PixelStrategy;
use core::marker::PhantomData;
use core::ops::{AddAssign, Div, Mul};
use nalgebra::DVector;

/// Pixel strategy that decomposes a colour-space input into per-palette
/// weights via a [`Decomposer`], then picks one palette index per pixel —
/// either by sampling an external noise value against the cumulative weights,
/// or (when no noise is provided) by taking the dominant component.
///
/// The source pixel type `Src` is converted into the decomposer's input by
/// `convert`. The strategy emits a `usize` palette index as its target and a
/// per-component quantization error; whether and how that error is propagated
/// is the caller's choice via the [`DiffusionMatrix`](crate::dither::diffusion_matrix::DiffusionMatrix)
/// passed to [`diffuse_dither`](crate::dither::diffuse::diffuse_dither)
/// (use [`NoDiffuse`](crate::dither::diffusion_matrix::NoDiffuse) to skip
/// diffusion entirely).
pub struct DecomposingDitherStrategy<D, F, Src> {
    pub decomposer: D,
    pub convert: F,
    _phantom: PhantomData<fn(Src)>,
}

impl<D, F, Src> DecomposingDitherStrategy<D, F, Src> {
    pub fn new(decomposer: D, convert: F) -> Self {
        Self {
            decomposer,
            convert,
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone, Default)]
pub struct DecomposedQuantizationError(Option<DVector<f32>>);

impl Mul<usize> for DecomposedQuantizationError {
    type Output = Self;
    fn mul(self, rhs: usize) -> Self {
        Self(self.0.map(|x| x * (rhs as f32)))
    }
}

impl Div<usize> for DecomposedQuantizationError {
    type Output = Self;
    fn div(self, rhs: usize) -> Self {
        Self(self.0.map(|x| x / (rhs as f32)))
    }
}

impl AddAssign<DecomposedQuantizationError> for DecomposedQuantizationError {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = match (core::mem::take(&mut self.0), rhs.0) {
            (a, None) => a,
            (None, b) => b,
            (Some(a), Some(b)) => Some(a + b),
        }
    }
}

impl<D, F, Src> PixelStrategy for DecomposingDitherStrategy<D, F, Src>
where
    D: Decomposer<f32>,
    F: Fn(Src) -> D::Input,
{
    type Source = (Src, Option<f32>);
    type Target = usize;
    type QuantizationError = DecomposedQuantizationError;

    fn quantize(
        &self,
        source: Self::Source,
        error: Self::QuantizationError,
    ) -> (Self::Target, Self::QuantizationError) {
        let (source, noise) = source;
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
