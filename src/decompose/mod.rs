pub mod gray;
pub mod input;
pub mod naive;
pub mod octahedron;

pub use input::DecomposerInputColor;

/// Decomposes a colour-space point into per-palette weights.
///
/// `Input` is the colour-space type the decomposer operates on (e.g. [`nalgebra::Point3<T>`]
/// for an RGB decomposer, `T` for a grayscale decomposer). The weights are written into a
/// caller-provided buffer of length [`palette_size`](Self::palette_size), are non-negative,
/// and sum to one.
///
/// Concrete decomposers may expose ergonomic inherent methods returning their natural shape
/// (e.g. `Vector6<T>` for [`octahedron::OctahedronDecomposer`]); the trait itself stays
/// minimal and slice-based to suit hot-loop callers that pre-allocate once and reuse.
pub trait Decomposer<T> {
    type Input;

    /// Number of palette entries (required length of the `out` slice in
    /// [`decompose_into`](Self::decompose_into)).
    fn palette_size(&self) -> usize;

    /// Decompose `input` into `out`. `out.len()` must equal
    /// [`palette_size`](Self::palette_size).
    fn decompose_into(&self, input: &Self::Input, out: &mut [T]);
}
