use std::marker::PhantomData;

use crate::core::utils;
use crate::core::{index::GridIndex, prelude::*};

#[derive(Debug)]
pub struct SimpleRayCaster<T: GridIndex> {
    ray_length_steps: i64,
    current_step: i64,
    curr_index: Point3<i64>,
    t_to_next_boundary: Vector3<Real>,
    ray_step_signs: Vector3<Real>,
    t_step_size: Vector3<Real>,
    _marker: PhantomData<T>,
}

impl<T: GridIndex> SimpleRayCaster<T> {
    pub fn new(ray_start: &Point3<Real>, ray_end: &Point3<Real>, grid_size_inv: Real) -> Self {
        let start_scaled = ray_start * grid_size_inv;
        let end_scaled = ray_end * grid_size_inv;

        let curr_index = utils::grid_index_from_point(ray_start, grid_size_inv);
        let end_index = utils::grid_index_from_point(ray_end, grid_size_inv);

        let diff_index = end_index - curr_index;

        let ray_length_steps = diff_index.x.abs() + diff_index.y.abs() + diff_index.z.abs();
        let ray_scaled = end_scaled - start_scaled;

        let ray_step_signs = Vector3::new(
            ray_scaled.x.signum(),
            ray_scaled.y.signum(),
            ray_scaled.z.signum(),
        );
        let corrected_step = Vector3::new(
            ray_step_signs.x.max(0.0),
            ray_step_signs.y.max(0.0),
            ray_step_signs.z.max(0.0),
        );

        let start_scaled_shifted = start_scaled - curr_index.cast();

        let distance_to_boundaries = corrected_step - start_scaled_shifted;
        let t_to_next_boundary = Vector3::new(
            if ray_scaled.x.abs() < 0.0 {
                2.0
            } else {
                distance_to_boundaries.x / ray_scaled.x
            },
            if ray_scaled.y.abs() < 0.0 {
                2.0
            } else {
                distance_to_boundaries.y / ray_scaled.y
            },
            if ray_scaled.z.abs() < 0.0 {
                2.0
            } else {
                distance_to_boundaries.z / ray_scaled.z
            },
        );

        let t_step_size = Vector3::new(
            if ray_scaled.x.abs() < 0.0 {
                2.0
            } else {
                ray_step_signs.x / ray_scaled.x
            },
            if ray_scaled.y.abs() < 0.0 {
                2.0
            } else {
                ray_step_signs.y / ray_scaled.y
            },
            if ray_scaled.z.abs() < 0.0 {
                2.0
            } else {
                ray_step_signs.z / ray_scaled.z
            },
        );

        Self {
            ray_length_steps,
            current_step: 0,
            curr_index,
            t_to_next_boundary,
            ray_step_signs,
            t_step_size,
            _marker: PhantomData,
        }
    }

    pub fn steps_count(&self) -> usize {
        self.ray_length_steps as usize
    }

    pub fn next_index(&mut self) -> Option<T> {
        if self.current_step > self.ray_length_steps {
            return None;
        }

        let ret = Some(T::from(self.curr_index));
        self.current_step += 1;

        let t_min_index = self.t_to_next_boundary.imin();

        let rss = self.ray_step_signs[t_min_index] as i64;
        self.curr_index.coords[t_min_index] += rss;

        self.t_to_next_boundary[t_min_index] += self.t_step_size[t_min_index];

        ret
    }
}

impl<T: GridIndex> Iterator for SimpleRayCaster<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_index()
    }
}

#[cfg(test)]
mod test {
    use crate::core::index::GlobalIndex;

    use super::*;

    #[test]
    pub fn test_raycast1() {
        // forward
        let origin = Point3::new(0.0, 0.0, 0.0);
        let point_g = Point3::new(-1.0, 0.0, 0.0);

        let raycaster: SimpleRayCaster<GlobalIndex<4>> =
            SimpleRayCaster::new(&origin, &point_g, 1.0 / 0.5);

        let steps: Vec<_> = raycaster.collect();
        assert_eq!(
            &steps,
            &[
                GlobalIndex(Point3::new(0, 0, 0)),
                GlobalIndex(Point3::new(-1, 0, 0)),
                GlobalIndex(Point3::new(-2, 0, 0))
            ]
        );

        // forward, diagonal
        let origin = Point3::new(0.0, 0.0, 0.0);
        let point_g = Point3::new(1.0, 1.0, 1.0);

        let raycaster: SimpleRayCaster<GlobalIndex<4>> =
            SimpleRayCaster::new(&origin, &point_g, 1.0 / 0.5);
        let steps: Vec<_> = raycaster.collect();
        assert_eq!(
            &steps,
            &[
                GlobalIndex(Point3::new(0, 0, 0)),
                GlobalIndex(Point3::new(1, 0, 0)),
                GlobalIndex(Point3::new(1, 1, 0)),
                GlobalIndex(Point3::new(1, 1, 1)),
                GlobalIndex(Point3::new(2, 1, 1)),
                GlobalIndex(Point3::new(2, 2, 1)),
                GlobalIndex(Point3::new(2, 2, 2)),
            ]
        );
    }

    #[test]
    fn test_raycast2() {
        let origin = Point3::new(0.6, 0.0, 0.0);
        let point_g = Point3::new(0.8, 0.0, 1.0);

        let raycaster: SimpleRayCaster<GlobalIndex<4>> =
            SimpleRayCaster::new(&origin, &point_g, 1.0 / 0.5);
        let steps: Vec<_> = raycaster.collect();
        assert_eq!(
            &steps,
            &[
                GlobalIndex(Point3::new(1, 0, 0)),
                GlobalIndex(Point3::new(1, 0, 1)),
                GlobalIndex(Point3::new(1, 0, 2)),
            ]
        );
    }
}
