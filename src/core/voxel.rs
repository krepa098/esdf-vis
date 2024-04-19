use std::fmt::Debug;

use bitflags::bitflags;
use nalgebra::Vector4;

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
// #[derive(Debug, Default, Clone, Copy)]
// pub struct Esdf {
//     pub distance: Real,
//     pub observed: bool,
//     pub hallucinated: bool,
//     pub in_queue: bool,
//     pub fixed: bool,
//     pub site_block_index: Option<Vector3<i32>>,
// }

impl Voxel for Esdf {}

impl DrawableVoxel for Esdf {
    fn color(&self) -> Color {
        if self.distance != 0.0 && !self.flags.contains(EsdfGpuFlags::Fixed) {
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

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    pub struct EsdfGpuFlags: u32 {
        const Observed = 1<<0;
        const Fixed = 1<<1;
        const HasSiteIndex = 1<<2;
        const Updated = 1<<3;
        const SpilledXPLus = 1<<4;
        const SpilledXMinus = 1<<5;
        const SpilledYPlus = 1<<6;
        const SpilledYMinus = 1<<7;
        const SpilledZPlus = 1<<8;
        const SpilledZMinus = 1<<9;
    }
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct Esdf {
    pub distance: Real,
    pub flags: EsdfGpuFlags,
    pub site_block_index: [i32; 3],
    pub _pad: [u8; 12],
}

// impl Voxel for Esdf {}

// impl From<Esdf> for EsdfGpu {
//     fn from(esdf: Esdf) -> Self {
//         let mut flags = EsdfGpuFlags::empty();

//         if esdf.fixed {
//             flags |= EsdfGpuFlags::Fixed;
//         }

//         if esdf.observed {
//             flags |= EsdfGpuFlags::Observed;
//         }

//         if esdf.site_block_index.is_some() {
//             flags |= EsdfGpuFlags::HasSiteIndex;
//         }

//         Self {
//             distance: esdf.distance,
//             flags,
//             site_block_index: esdf.site_block_index.unwrap_or_default().into(),
//             _pad: Default::default(),
//         }
//     }
// }
