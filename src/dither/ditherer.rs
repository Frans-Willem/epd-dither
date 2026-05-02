#[cfg(feature = "alloc")]
use crate::dither::diffuse::PixelStrategy;
#[cfg(feature = "alloc")]
use crate::dither::diffuse::diffuse_dither;
#[cfg(feature = "alloc")]
use crate::dither::diffusion_matrix::DiffusionMatrix;
use crate::dither::image_traits::{ImageReader, ImageSize, ImageWriter};

/// A dither pipeline with strategy, diffusion matrix, and (optional) input
/// preprocess pre-bundled, callable against any compatible reader/writer
/// pair.
///
/// Generic over the reader/writer at the method level: zero-cost dispatch,
/// not dyn-safe. For the type-erased path see [`DynDitherer`], which is
/// implemented for any `Ditherer` via a blanket impl.
pub trait Ditherer {
    type Input;
    type Output;

    fn dither_into<I>(&self, inout: &mut I)
    where
        I: ImageSize + ImageReader<Self::Input> + ImageWriter<Self::Output> + ?Sized;
}

/// Dyn-safe view of [`Ditherer`] with the reader/writer locked to a single
/// concrete `InOut`. Implementations of `Ditherer` get this for free via
/// the blanket impl below; bespoke implementors only have to write
/// `Ditherer`.
///
/// `Input` and `Output` are not associated types here — they're pinned by
/// `InOut`'s [`ImageReader`] / [`ImageWriter`] impls — so the dyn type
/// shortens to `dyn DynDitherer<MyInOut>`.
///
/// The method is named [`dyn_dither_into`](Self::dyn_dither_into) rather
/// than `dither_into` so that `Ditherer` and `DynDitherer` can both be in
/// scope on a concrete type without `bd.dither_into(...)` being
/// ambiguous.
pub trait DynDitherer<InOut: ?Sized> {
    fn dyn_dither_into(&self, inout: &mut InOut);
}

impl<T, InOut> DynDitherer<InOut> for T
where
    T: Ditherer,
    InOut: ImageSize + ImageReader<T::Input> + ImageWriter<T::Output> + ?Sized,
{
    fn dyn_dither_into(&self, inout: &mut InOut) {
        <T as Ditherer>::dither_into(self, inout);
    }
}

/// Default [`Ditherer`] implementation: a [`PixelStrategy`] paired with a
/// [`DiffusionMatrix`] and a `serpentine` flag. The strategy already owns
/// its own input-conversion function (see
/// [`DecomposingDitherStrategy`](crate::dither::DecomposingDitherStrategy)),
/// so the bundle adds no second preprocess layer.
///
/// `dither_into` is generic over the reader/writer at the call site, so a
/// single `BundledDitherer<S, M>` value works against any compatible
/// `InOut`. Type-erased use is via the `DynDitherer<InOut>` blanket impl
/// — `Box<dyn DynDitherer<MyInOut>>` works directly.
#[cfg(feature = "alloc")]
pub struct BundledDitherer<S, M> {
    pub strategy: S,
    pub matrix: M,
    pub serpentine: bool,
}

#[cfg(feature = "alloc")]
impl<S, M> BundledDitherer<S, M> {
    /// Defaults to serpentine traversal — matches every existing call site
    /// in the binary. Override with [`with_serpentine`](Self::with_serpentine).
    pub fn new(strategy: S, matrix: M) -> Self {
        Self {
            strategy,
            matrix,
            serpentine: true,
        }
    }

    pub fn with_serpentine(mut self, serpentine: bool) -> Self {
        self.serpentine = serpentine;
        self
    }
}

#[cfg(feature = "alloc")]
impl<S, M> Ditherer for BundledDitherer<S, M>
where
    S: PixelStrategy,
    M: DiffusionMatrix,
{
    type Input = S::Source;
    type Output = S::Target;

    fn dither_into<I>(&self, inout: &mut I)
    where
        I: ImageSize + ImageReader<S::Source> + ImageWriter<S::Target> + ?Sized,
    {
        diffuse_dither(&self.strategy, &self.matrix, inout, self.serpentine);
    }
}
