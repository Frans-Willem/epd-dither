use crate::barycentric::line::LineProjector;
use crate::barycentric::tetrahedron::TetrahedronProjector;
use crate::barycentric::triangle::TriangleProjector;
use alloc::vec::Vec;
use itertools::Itertools;
use nalgebra::base::{DVector, OVector, Scalar};
use nalgebra::geometry::Point3;
use nalgebra::{
    ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, ComplexField, Const,
};
use num_traits::identities::{One, Zero};
use num_traits::zero;

#[derive(Debug, Clone)]
pub enum DecomposerBruteforceStrategy {
    FavorMix,
    FavorDominant,
}

pub struct DecomposerBruteforce<T: Scalar + ComplexField> {
    num_colors: usize,
    tetras: Vec<(TetrahedronProjector<T>, [usize; 4])>,
    faces: Vec<(TriangleProjector<T>, [usize; 3])>,
    edges: Vec<(LineProjector<T>, [usize; 2])>,
}

impl<T: Scalar> DecomposerBruteforce<T>
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
            })
        } else {
            None
        }
    }

    fn to_global_barycentric_coordinates<const N: usize>(
        &self,
        local_barycentric: OVector<T, Const<N>>,
        vertex_indices: &[usize; N],
    ) -> DVector<T> {
        let mut global_barycentric = DVector::zeros(self.num_colors);
        for (local_index, global_index) in vertex_indices.iter().enumerate() {
            if *global_index < self.num_colors {
                global_barycentric[*global_index] = local_barycentric[local_index].clone();
            }
        }
        global_barycentric
    }

    pub fn decompose(&self, pt: &Point3<T>, strategy: DecomposerBruteforceStrategy) -> DVector<T> {
        let in_tetras = self.tetras.iter().filter_map(|(tetra, vertex_indices)| {
            let projected = tetra.project(pt);
            if projected.min() < zero() {
                None
            } else {
                Some((projected, vertex_indices))
            }
        });
        let in_tetras = match strategy {
            DecomposerBruteforceStrategy::FavorMix => {
                in_tetras.reduce(|a, b| if b.0.max() < a.0.max() { b } else { a })
            }
            DecomposerBruteforceStrategy::FavorDominant => {
                in_tetras.reduce(|a, b| if b.0.max() > a.0.max() { b } else { a })
            }
        };
        if let Some((local_barycentric, vertex_indices)) = in_tetras {
            return self.to_global_barycentric_coordinates(local_barycentric, vertex_indices);
        }
        let on_faces = self.faces.iter().filter_map(|(triangle, vertex_indices)| {
            let (projected, distance) = triangle.project(pt);
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
            let (projected, _) = edge.clipping_project(pt);
            let projected_pt = edge.bary_to_point(&projected);
            let distance_sq: T = T::from_real((projected_pt - pt).norm_squared());
            (distance_sq, projected, vertex_indices)
        });
        let closest_edge = on_edges.reduce(|a, b| if b.0 < a.0 { b } else { a });
        match (closest_face, closest_edge) {
            (
                Some((face_distance_sq, face_barycentric, face_vertex_indices)),
                Some((edge_distance_sq, edge_barycentric, edge_vertex_indices)),
            ) => {
                if edge_distance_sq < face_distance_sq {
                    self.to_global_barycentric_coordinates(edge_barycentric, edge_vertex_indices)
                } else {
                    self.to_global_barycentric_coordinates(face_barycentric, face_vertex_indices)
                }
            }
            (Some((_, local_barycentric, vertex_indices)), None) => {
                self.to_global_barycentric_coordinates(local_barycentric, vertex_indices)
            }
            (None, Some((_, local_barycentric, vertex_indices))) => {
                self.to_global_barycentric_coordinates(local_barycentric, vertex_indices)
            }
            _ => panic!(),
        }
    }
}
