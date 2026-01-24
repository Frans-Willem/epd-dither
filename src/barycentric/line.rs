use nalgebra::base::{Scalar, Vector3};
use nalgebra::geometry::Point3;
use nalgebra::{ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign};
use num_traits::identities::{One, Zero};
use num_traits::{one, zero};

pub struct LineProjector<T: Scalar> {
    pub origin: Point3<T>,
    pub direction: Vector3<T>,
    pub norm_squared: T,
}

impl<
    T: Scalar
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
        let norm_squared = direction.dot(&direction);
        LineProjector {
            origin,
            direction,
            norm_squared,
        }
    }

    // Projects to barycentric coordinates
    pub fn project(&self, pt: Point3<T>) -> [T; 2] {
        if self.norm_squared.is_zero() {
            // length of direction is zero, so line is a point
            return [one(), zero()];
        }
        let origin_to_pt: Vector3<T> = pt - &self.origin;
        let origin_to_pt_dist_sq: T = origin_to_pt.dot(&origin_to_pt);
        if origin_to_pt_dist_sq.is_zero() {
            // A and P are the same, just return [1,0]
            return [one(), zero()];
        }
        let t = origin_to_pt.dot(&self.direction) / self.norm_squared.clone();
        return [T::one() - t.clone(), t];
    }

    pub fn project_clipped(&self, pt: Point3<T>) -> ([T; 2], bool) {
        let ret = self.project(pt);
        if ret[0] < zero() {
            ([zero(), one()], true)
        } else if ret[1] < zero() {
            ([one(), zero()], true)
        } else {
            (ret, false)
        }
    }
}
