use super::prelude::*;

pub fn grid_index_from_point(p: &Point3<Real>, grid_size_inv: Real) -> Point3<i64> {
    Point3::new(
        (p.x * grid_size_inv + Real::EPSILON).floor() as i64,
        (p.y * grid_size_inv + Real::EPSILON).floor() as i64,
        (p.z * grid_size_inv + Real::EPSILON).floor() as i64,
    )
}
