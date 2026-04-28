use crate::barycentric::octahedron::OctahedronProjector;
use nalgebra::base::{Scalar, Vector3, Vector6};
use nalgebra::geometry::Point3;
use nalgebra::{ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, ComplexField};
use num_traits::identities::{One, Zero};
use num_traits::one;

struct LineDistanceCalculator<T: Scalar + ComplexField> {
    // P = origin + t * direction
    origin: Point3<T>,
    direction: Vector3<T>,
    direction_len_sq: T::RealField,
}

impl<T: Scalar> LineDistanceCalculator<T>
where
    T: ClosedSubAssign,
    T: ComplexField,
{
    fn new(vertices: [Point3<T>; 2]) -> Option<Self> {
        let [origin, target] = vertices;
        let direction = target - &origin;
        let direction_len_sq = direction.norm_squared();
        if direction_len_sq.is_zero() {
            None
        } else {
            Some(Self {
                origin,
                direction,
                direction_len_sq,
            })
        }
    }

    fn distance_squared(&self, pt: &Point3<T>) -> T::RealField {
        // See https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        self.direction.cross(&(&self.origin - pt)).norm_squared() / self.direction_len_sq.clone()
    }
}

struct OctahedronDecomposerAxis<T: Scalar + ComplexField> {
    // Which two poles are used as central axis
    poles: [usize; 2],
    // Calculator for distance to axis central line
    distance_calc: LineDistanceCalculator<T>,
    // Projector to find barycentric coordinates
    projector: OctahedronProjector<T>,
    // Ordering from projector local barycentric coordinates to global barycentric coordinates
    color_to_vertex_index: [usize; 6],
}

/// Decomposer for palettes whose points (colours) form a regular convex
/// octahedron (the structurally-symmetric Spectra 6 case). On a single core
/// of an ESP32-S3 it can decompose an 800×480 f32 image in under 5 seconds.
pub struct OctahedronDecomposer<T: Scalar + ComplexField> {
    // Possible axis to use in decomposition
    axis: [OctahedronDecomposerAxis<T>; 3],
    // Strategy used by the [`Decomposer`](super::Decomposer) trait impl.
    strategy: OctahedronDecomposerAxisStrategy,
}

#[derive(Copy, Clone, Debug, Default)]
pub enum OctahedronDecomposerAxisStrategy {
    Axis(usize),
    #[default]
    Closest,
    Furthest,
    Average,
}

impl<T: Scalar> OctahedronDecomposerAxis<T>
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
    fn new(vertex_index_to_color: [usize; 6], colors: &[Point3<T>; 6]) -> Option<Self> {
        let poles = [vertex_index_to_color[0], vertex_index_to_color[1]];
        let mut color_to_vertex_index: [usize; 6] = [0; 6];
        for (vertex_index, color_index) in vertex_index_to_color.iter().enumerate() {
            color_to_vertex_index[*color_index] = vertex_index;
        }
        let projector = OctahedronProjector::new(
            vertex_index_to_color.map(|color_index| colors[color_index].clone()),
        )?;
        let distance_calc =
            LineDistanceCalculator::new(poles.map(|color_index| colors[color_index].clone()))?;

        Some(Self {
            poles,
            projector,
            distance_calc,
            color_to_vertex_index,
        })
    }
    fn project(&self, color: &Point3<T>) -> (Vector6<T>, bool) {
        let (barycentric_local, is_inside) = self.projector.project(color);
        let barycentric_global =
            Vector6::from_fn(|row, _| barycentric_local[self.color_to_vertex_index[row]].clone());
        (barycentric_global, is_inside)
    }
}

impl<T: Scalar> OctahedronDecomposer<T>
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
        let colors: &[Point3<T>; 6] = colors.try_into().ok()?;
        let opposite_map = OctahedronProjector::find_opposites(colors)?;
        let axis: [OctahedronDecomposerAxis<T>; 3] =
            crate::helpers::opt_array_transpose(core::array::from_fn(|axis_index| {
                let vertex_index_to_color: [usize; 6] = [
                    opposite_map[axis_index % opposite_map.len()].0,
                    opposite_map[axis_index % opposite_map.len()].1,
                    opposite_map[(axis_index + 1) % opposite_map.len()].0,
                    opposite_map[(axis_index + 2) % opposite_map.len()].0,
                    opposite_map[(axis_index + 1) % opposite_map.len()].1,
                    opposite_map[(axis_index + 2) % opposite_map.len()].1,
                ];
                OctahedronDecomposerAxis::new(vertex_index_to_color, colors)
            }))?;
        Some(Self {
            axis,
            strategy: Default::default(),
        })
    }

    /// Set the axis-selection strategy used by [`Decomposer::decompose_into`](super::Decomposer::decompose_into).
    pub fn with_strategy(mut self, strategy: OctahedronDecomposerAxisStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn get_axis_from_color(&self, color_index: usize) -> Option<usize> {
        self.axis.iter().enumerate().find_map(|(axis_index, axis)| {
            if axis.poles[0] == color_index || axis.poles[1] == color_index {
                Some(axis_index)
            } else {
                None
            }
        })
    }
}

impl<T: Scalar> super::Decomposer<T> for OctahedronDecomposer<T>
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
        6
    }

    fn decompose_into(&self, input: &Point3<T>, out: &mut [T]) {
        let weights: Vector6<T> = match self.strategy {
            OctahedronDecomposerAxisStrategy::Axis(axis) => {
                let axis = &self.axis[axis % self.axis.len()];
                axis.project(input).0
            }
            OctahedronDecomposerAxisStrategy::Average => {
                let axis = &self.axis[0];
                let (mut barycentric_global, is_inside) = axis.project(input);
                if is_inside {
                    let mut divisor: T = one();
                    for axis_index in 1..self.axis.len() {
                        let (current, _) = self.axis[axis_index].project(input);
                        barycentric_global += current;
                        divisor += one();
                    }
                    barycentric_global /= divisor;
                }
                barycentric_global
            }
            OctahedronDecomposerAxisStrategy::Closest => {
                let (axis, _) = self
                    .axis
                    .iter()
                    .map(|axis| (axis, axis.distance_calc.distance_squared(input)))
                    .reduce(|a, b| if b.1 < a.1 { b } else { a })
                    .unwrap_or((&self.axis[0], num_traits::zero()));
                axis.project(input).0
            }
            OctahedronDecomposerAxisStrategy::Furthest => {
                let (axis, _) = self
                    .axis
                    .iter()
                    .map(|axis| (axis, axis.distance_calc.distance_squared(input)))
                    .reduce(|a, b| if b.1 > a.1 { b } else { a })
                    .unwrap_or((&self.axis[0], num_traits::zero()));
                axis.project(input).0
            }
        };
        // Owned `Matrix` doesn't implement `IntoIterator`; destructure the
        // single-column `ArrayStorage` to move each T out into `out`.
        let [weights]: [[T; 6]; 1] = weights.data.0;
        for (slot, weight) in out.iter_mut().zip(weights) {
            *slot = weight;
        }
    }
}
