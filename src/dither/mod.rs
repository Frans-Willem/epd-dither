#[cfg(feature = "alloc")]
pub mod decomposing;
pub mod diffuse;
pub mod diffusion_matrix;
pub mod ditherer;
pub mod image_traits;

#[cfg(feature = "alloc")]
pub use decomposing::{
    DecomposeStrategy, DecomposedQuantizationError, DecomposingDitherStrategy,
    InvalidDecomposeStrategy,
};
#[cfg(feature = "alloc")]
pub use ditherer::BundledDitherer;
pub use ditherer::{Ditherer, DynDitherer};
pub use image_traits::{ImageReader, ImageSize, ImageSplit, ImageWriter};
