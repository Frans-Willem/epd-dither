/// 6-colour palette as used by the `epdoptimize` toolchain. The points do
/// **not** form a convex octahedron, so this palette is only valid with
/// [`NaiveDecomposer`] — feeding it to `OctahedronDecomposer` produces
/// either a constructor failure or incorrect barycentric coordinates.
pub const EPDOPTIMIZE: [[u8; 3]; 6] = [
    [0x19, 0x1E, 0x21],
    [0xe8, 0xe8, 0xe8],
    [0xef, 0xde, 0x44],
    [0xb2, 0x13, 0x18],
    [0x21, 0x57, 0xba],
    [0x12, 0x5f, 0x20],
];

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum NaiveDecomposerStrategy {
    #[default]
    FavorMix,
    FavorDominant,
    /// Blend over all containing tetrahedra with weights
    /// `α_i ∝ (∏_j w_{i,j})^p`. `p = 0` averages equally; higher `p`
    /// concentrates on the most-interior tetrahedron. See
    /// `documentation/tetra-blend-research.md`.
    TetraBlend(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidNaiveDecomposerStrategy;

impl core::fmt::Display for InvalidNaiveDecomposerStrategy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid naive decomposer-strategy name")
    }
}

impl core::error::Error for InvalidNaiveDecomposerStrategy {}

impl core::str::FromStr for NaiveDecomposerStrategy {
    type Err = InvalidNaiveDecomposerStrategy;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mix" => Ok(Self::FavorMix),
            "dominant" => Ok(Self::FavorDominant),
            "blend" => Ok(Self::TetraBlend(1)),
            _ if s.starts_with("blend:") => {
                let p = s["blend:".len()..]
                    .parse::<u32>()
                    .map_err(|_| InvalidNaiveDecomposerStrategy)?;
                Ok(Self::TetraBlend(p))
            }
            _ => Err(InvalidNaiveDecomposerStrategy),
        }
    }
}

#[cfg(feature = "alloc")]
pub use alloc_impl::NaiveDecomposer;

#[cfg(feature = "alloc")]
mod alloc_impl {
    use super::NaiveDecomposerStrategy;
    use crate::barycentric::line::LineProjector;
    use crate::barycentric::tetrahedron::TetrahedronProjector;
    use crate::barycentric::triangle::TriangleProjector;
    use alloc::vec::Vec;
    use itertools::Itertools;
    use nalgebra::base::{OVector, Scalar, Vector4};
    use nalgebra::geometry::Point3;
    use nalgebra::{
        ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, ComplexField, Const,
    };
    use num_traits::identities::{One, Zero};
    use num_traits::zero;

    pub struct NaiveDecomposer<T: Scalar + ComplexField> {
        num_colors: usize,
        tetras: Vec<(TetrahedronProjector<T>, [usize; 4])>,
        faces: Vec<(TriangleProjector<T>, [usize; 3])>,
        edges: Vec<(LineProjector<T>, [usize; 2])>,
        // Strategy used by the [`Decomposer`](super::Decomposer) trait impl.
        strategy: NaiveDecomposerStrategy,
    }

    impl<T: Scalar> NaiveDecomposer<T>
    where
        T: ComplexField
            + ClosedSubAssign
            + ClosedMulAssign
            + ClosedAddAssign
            + ClosedDivAssign
            + Zero
            + One
            + PartialOrd,
    {
        pub fn new(colors: &[Point3<T>]) -> Option<Self> {
            let num_colors: usize = colors.len();
            let tetras: Vec<(TetrahedronProjector<T>, [usize; 4])> = (0..num_colors)
                .combinations(4)
                .filter_map(|vertex_indices| {
                    let vertex_indices: [usize; 4] = vertex_indices.try_into().ok()?;
                    let vertex_points = vertex_indices.map(|i| colors[i].clone());
                    let tetrahedron = TetrahedronProjector::new(vertex_points)?;
                    Some((tetrahedron, vertex_indices))
                })
                .collect();
            let faces: Vec<(TriangleProjector<T>, [usize; 3])> = (0..num_colors)
                .combinations(3)
                .filter_map(|vertex_indices| {
                    let vertex_indices: [usize; 3] = vertex_indices.try_into().ok()?;
                    let vertex_points = vertex_indices.map(|i| colors[i].clone());
                    let triangle = TriangleProjector::new(vertex_points)?;
                    Some((triangle, vertex_indices))
                })
                .collect();
            let edges: Vec<(LineProjector<T>, [usize; 2])> = (0..num_colors)
                .combinations(2)
                .filter_map(|vertex_indices| {
                    let vertex_indices: [usize; 2] = vertex_indices.try_into().ok()?;
                    let vertex_points = vertex_indices.map(|i| colors[i].clone());
                    let line = LineProjector::new(vertex_points)?;
                    Some((line, vertex_indices))
                })
                .collect();
            if !tetras.is_empty() || !faces.is_empty() || !edges.is_empty() {
                Some(Self {
                    num_colors,
                    tetras,
                    faces,
                    edges,
                    strategy: Default::default(),
                })
            } else {
                None
            }
        }

        /// Set the strategy used by [`Decomposer::decompose_into`](super::Decomposer::decompose_into).
        pub fn with_strategy(mut self, strategy: NaiveDecomposerStrategy) -> Self {
            self.strategy = strategy;
            self
        }

        /// Move local barycentric weights into the global-palette positions of
        /// `out`. `out` must already be zeroed. Local-vertex indices that exceed
        /// `num_colors` are silently dropped (their weight is discarded).
        fn write_global_barycentric<const N: usize>(
            &self,
            local_barycentric: OVector<T, Const<N>>,
            vertex_indices: &[usize; N],
            out: &mut [T],
        ) {
            let [local]: [[T; N]; 1] = local_barycentric.data.0;
            for (weight, &global_index) in local.into_iter().zip(vertex_indices.iter()) {
                if global_index < self.num_colors {
                    out[global_index] = weight;
                }
            }
        }

        fn compare_tetra_projection_favor_mix<'t>(
            a: (Vector4<T>, &'t [usize; 4]),
            b: (Vector4<T>, &'t [usize; 4]),
        ) -> (Vector4<T>, &'t [usize; 4]) {
            if b.0.max() < a.0.max() { b } else { a }
        }

        fn compare_tetra_projection_favor_dominant<'t>(
            a: (Vector4<T>, &'t [usize; 4]),
            b: (Vector4<T>, &'t [usize; 4]),
        ) -> (Vector4<T>, &'t [usize; 4]) {
            if b.0.max() > a.0.max() { b } else { a }
        }

        /// Blend all containing tetrahedra. Per tetrahedron the score is
        /// `α = (∏_j w_j)^power` (and `α = 1` if `power == 0`); the output is
        /// `Σ α·w / Σ α`. Returns `true` iff at least one tetrahedron
        /// contained `input` and the accumulated denominator was positive.
        ///
        /// `out` must already be zeroed.
        fn blend_tetras_into(&self, input: &Point3<T>, out: &mut [T], power: u32) -> bool {
            let mut total: T = zero();
            let mut found_any = false;
            for (tetra, vertex_indices) in self.tetras.iter() {
                let projected = tetra.project(input);
                if projected.min() < zero() {
                    continue;
                }
                found_any = true;
                // base = ∏_j w_j; alpha = base^power. Loop multiplications
                // because T only requires ComplexField, not num_traits::Pow.
                let mut base: T = T::one();
                for j in 0..4 {
                    base *= projected[j].clone();
                }
                let mut alpha: T = T::one();
                for _ in 0..power {
                    alpha *= base.clone();
                }
                for j in 0..4 {
                    let global = vertex_indices[j];
                    if global < self.num_colors {
                        out[global] += alpha.clone() * projected[j].clone();
                    }
                }
                total += alpha;
            }
            // Degenerate `total == 0` happens only when `input` lies on a
            // palette edge/vertex (every containing tetra has a zero weight)
            // and `power >= 1`. Fall through to the face/edge handler in
            // that case rather than producing NaN. Negated comparison
            // (rather than `total <= zero()`) so a NaN `total` also lands
            // on the fallback path.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !found_any || !(total > zero()) {
                for slot in out.iter_mut() {
                    *slot = zero();
                }
                return false;
            }
            for slot in out.iter_mut() {
                *slot = slot.clone() / total.clone();
            }
            true
        }
    }

    impl<T: Scalar> crate::decompose::Decomposer<T> for NaiveDecomposer<T>
    where
        T: ComplexField
            + ClosedSubAssign
            + ClosedMulAssign
            + ClosedAddAssign
            + ClosedDivAssign
            + Zero
            + One
            + PartialOrd,
    {
        type Input = Point3<T>;

        fn palette_size(&self) -> usize {
            self.num_colors
        }

        fn decompose_into(&self, input: &Point3<T>, out: &mut [T]) {
            for slot in out.iter_mut() {
                *slot = zero();
            }

            let handled = if let NaiveDecomposerStrategy::TetraBlend(power) = self.strategy {
                self.blend_tetras_into(input, out, power)
            } else {
                let in_tetras = self.tetras.iter().filter_map(|(tetra, vertex_indices)| {
                    let projected = tetra.project(input);
                    if projected.min() < zero() {
                        None
                    } else {
                        Some((projected, vertex_indices))
                    }
                });
                let in_tetras = in_tetras.reduce(match self.strategy {
                    NaiveDecomposerStrategy::FavorMix => Self::compare_tetra_projection_favor_mix,
                    NaiveDecomposerStrategy::FavorDominant => {
                        Self::compare_tetra_projection_favor_dominant
                    }
                    NaiveDecomposerStrategy::TetraBlend(_) => unreachable!(),
                });
                if let Some((local_barycentric, vertex_indices)) = in_tetras {
                    self.write_global_barycentric(local_barycentric, vertex_indices, out);
                    true
                } else {
                    false
                }
            };
            if handled {
                return;
            }
            let on_faces = self.faces.iter().filter_map(|(triangle, vertex_indices)| {
                let (projected, distance) = triangle.project(input);
                if projected.min() < zero() {
                    return None;
                }
                // Use distance^2, such that it is easier to compare (can ignore sign), and easier to
                // compare against edge distances.
                Some((
                    distance.clone() * distance.clone(),
                    projected,
                    vertex_indices,
                ))
            });
            let closest_face = on_faces.reduce(|a, b| if b.0 < a.0 { b } else { a });
            let on_edges = self.edges.iter().map(|(edge, vertex_indices)| {
                let (projected, _) = edge.clipping_project(input);
                let projected_pt = edge.bary_to_point(&projected);
                let distance_sq: T = T::from_real((projected_pt - input).norm_squared());
                (distance_sq, projected, vertex_indices)
            });
            let closest_edge = on_edges.reduce(|a, b| if b.0 < a.0 { b } else { a });
            match (closest_face, closest_edge) {
                (
                    Some((face_distance_sq, face_barycentric, face_vertex_indices)),
                    Some((edge_distance_sq, edge_barycentric, edge_vertex_indices)),
                ) => {
                    if edge_distance_sq < face_distance_sq {
                        self.write_global_barycentric(edge_barycentric, edge_vertex_indices, out);
                    } else {
                        self.write_global_barycentric(face_barycentric, face_vertex_indices, out);
                    }
                }
                (Some((_, local_barycentric, vertex_indices)), None) => {
                    self.write_global_barycentric(local_barycentric, vertex_indices, out);
                }
                (None, Some((_, local_barycentric, vertex_indices))) => {
                    self.write_global_barycentric(local_barycentric, vertex_indices, out);
                }
                _ => unreachable!(),
            }
        }
    }
}
