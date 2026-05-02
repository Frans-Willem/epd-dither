#[cfg(feature = "alloc")]
pub mod decomposing;
pub mod diffuse;
pub mod diffusion_matrix;
pub mod image_traits;

#[cfg(feature = "alloc")]
pub use decomposing::{DecomposedQuantizationError, DecomposingDitherStrategy};
pub use image_traits::{ImageReader, ImageSize, ImageSplit, ImageWriter};
