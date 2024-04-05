use std::fmt::Debug;

use super::{color::rainbow_map, prelude::*};

pub trait Voxel: Default + Clone + Copy + Debug {}

pub trait DrawableVoxel {
    /// (r,g,b,a)
    fn color(&self) -> Color;
}

/// Tsdf Voxel
#[derive(Debug, Default, Clone, Copy)]
pub struct Tsdf {
    pub distance: Real,
    pub weight: Real,
}

impl Voxel for Tsdf {}

impl DrawableVoxel for Tsdf {
    fn color(&self) -> Color {
        if self.weight.abs() > 1e-6 && self.distance < 0.2 {
            Color::new(0.7, 0.7, 0.7, 1.0)
        } else {
            Color::default()
        }
    }
}

/// Esdf Voxel
#[derive(Debug, Default, Clone, Copy)]
pub struct Esdf {
    pub distance: Real,
    pub observed: bool,
    pub hallucinated: bool,
    pub in_queue: bool,
    pub fixed: bool,
    pub parent: Option<Vector3<i32>>,
}

impl Voxel for Esdf {}

impl DrawableVoxel for Esdf {
    fn color(&self) -> Color {
        if self.distance != 0.0 && !self.fixed {
            rainbow_map(self.distance.abs() as f32 / 4.0)
        } else {
            Color::default()
        }
    }
}

/// Occupancy Voxel
#[derive(Debug, Default, Clone, Copy)]
pub struct Occupancy {
    pub probability: Real,
    pub observed: bool,
}

impl Voxel for Occupancy {}

#[derive(Debug, Default, Clone, Copy)]
pub struct IntensityVoxel {
    pub intensity: Real,
    pub weight: Real,
}

impl Voxel for IntensityVoxel {}
