pub mod decomposing;
pub mod diffuse;
pub mod diffusion_matrix;
pub mod ditherer;
pub mod image_traits;

pub use decomposing::{DecomposeStrategy, InvalidDecomposeStrategy};
#[cfg(feature = "alloc")]
pub use decomposing::{DecomposedQuantizationError, DecomposingDitherStrategy};
pub use ditherer::{BundledDitherer, Ditherer, DynDitherer};
pub use image_traits::{ImageCombinedRW, ImageReader, ImageSize, ImageWriter};
