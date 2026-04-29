#[cfg(feature = "alloc")]
pub mod decomposing;
pub mod diffuse;
pub mod diffusion_matrix;

#[cfg(feature = "alloc")]
pub use decomposing::{DecomposedQuantizationError, DecomposingDitherStrategy};
