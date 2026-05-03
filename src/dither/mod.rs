pub mod diffuse;
pub mod diffusion_matrix;
pub mod ditherer;
#[cfg(feature = "alloc")]
pub mod with_decomposer;
pub mod image_traits;

#[cfg(feature = "alloc")]
pub use with_decomposer::{
    DecomposeStrategy, DecomposedQuantizationError, DecomposingDitherStrategy,
    InvalidDecomposeStrategy,
};
pub use ditherer::{BundledDitherer, Ditherer, DynDitherer};
pub use image_traits::{ImageCombinedRW, ImageReader, ImageSize, ImageWriter};
