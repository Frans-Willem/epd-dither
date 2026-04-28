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

#[derive(Copy, Clone, Debug, Default)]
pub enum NaiveDecomposerStrategy {
    #[default]
    FavorMix,
    FavorDominant,
}

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
                let vertex_indices: [usize; 4] = vertex_indices.try_into().unwrap();
                let vertex_points = vertex_indices.map(|i| colors[i].clone());
                let tetrahedron = TetrahedronProjector::new(vertex_points)?;
                Some((tetrahedron, vertex_indices))
            })
            .collect();
        let faces: Vec<(TriangleProjector<T>, [usize; 3])> = (0..num_colors)
            .combinations(3)
            .filter_map(|vertex_indices| {
                let vertex_indices: [usize; 3] = vertex_indices.try_into().unwrap();
                let vertex_points = vertex_indices.map(|i| colors[i].clone());
                let triangle = TriangleProjector::new(vertex_points)?;
                Some((triangle, vertex_indices))
            })
            .collect();
        let edges: Vec<(LineProjector<T>, [usize; 2])> = (0..num_colors)
            .combinations(2)
            .filter_map(|vertex_indices| {
                let vertex_indices: [usize; 2] = vertex_indices.try_into().unwrap();
                let vertex_points = vertex_indices.map(|i| colors[i].clone());
                let line = LineProjector::new(vertex_points)?;
                Some((line, vertex_indices))
            })
            .collect();
        if tetras.len() > 0 || faces.len() > 0 || edges.len() > 0 {
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

}

impl<T: Scalar> super::Decomposer<T> for NaiveDecomposer<T>
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
            NaiveDecomposerStrategy::FavorDominant => Self::compare_tetra_projection_favor_dominant,
        });
        if let Some((local_barycentric, vertex_indices)) = in_tetras {
            self.write_global_barycentric(local_barycentric, vertex_indices, out);
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
            _ => panic!(),
        }
    }
}
