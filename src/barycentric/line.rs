use nalgebra::base::{Scalar, Vector2, Vector3};
use nalgebra::geometry::Point3;
use nalgebra::{ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign, ComplexField};
use num_traits::identities::{One, Zero};
use num_traits::{one, zero};

pub struct LineProjector<T: Scalar + ComplexField> {
    pub origin: Point3<T>,
    pub direction: Vector3<T>,
    pub length_squared: T,
}

impl<
    T: Scalar
        + ComplexField
        + ClosedSubAssign
        + ClosedMulAssign
        + ClosedAddAssign
        + ClosedDivAssign
        + Zero
        + One
        + PartialOrd,
> LineProjector<T>
{
    pub fn new(vertices: [Point3<T>; 2]) -> Self {
        let [a, b] = vertices;
        let direction = b - &a;
        let origin = a;
        let length_squared = T::from_real(direction.norm_squared());
        LineProjector {
            origin,
            direction,
            length_squared,
        }
    }

    // Projects to barycentric coordinates
    pub fn project(&self, pt: &Point3<T>) -> Vector2<T> {
        if self.length_squared.is_zero() {
            // length of direction is zero, so line is a point
            return Vector2::new(one(), zero());
        }
        let origin_to_pt: Vector3<T> = pt - &self.origin;
        let origin_to_pt_dist_sq: T::RealField = origin_to_pt.norm_squared();
        if origin_to_pt_dist_sq.is_zero() {
            // A and P are the same, just return [1,0]
            return Vector2::new(one(), zero());
        }
        let t = origin_to_pt.dot(&self.direction) / self.length_squared.clone();
        Vector2::new(T::one() - t.clone(), t)
    }

    pub fn bary_to_point(&self, barycentric_coords: &Vector2<T>) -> Point3<T> {
        &self.origin + (&self.direction * barycentric_coords[1].clone())
    }

    // Returns closest barycentric coordinate on line (first retval)
    // If it was clipped (e.g. fell outside the two vertices) (second retval)
    // And optionally, if it has already been calculated
    // Second retval is None is the projection was on the line,
    // or Some(distance_squared) if it was clipped to the endpoints.
    // distance_squared is the squared euclidian distance from pt to the clipped point
    pub fn clipping_project(&self, pt: &Point3<T>) -> (Vector2<T>, bool) {
        let ret = self.project(pt);
        if ret[0] < zero() {
            (Vector2::new(zero(), one()), true)
        } else if ret[1] < zero() {
            (Vector2::new(one(), zero()), true)
        } else {
            (ret, false)
        }
    }
}
