use std::collections::BTreeSet;

use nalgebra::point;

use crate::core::index::{BlockIndex, GlobalIndex};
use crate::core::layer::Layer;
use crate::core::prelude::*;
use crate::core::voxel::Tsdf;

#[derive(Debug)]
pub struct TsdfIntegratorConfig {
    default_truncation_distance: Real,
}

impl Default for TsdfIntegratorConfig {
    fn default() -> Self {
        Self {
            default_truncation_distance: 0.2,
        }
    }
}

pub struct TsdfIntegrator {
    config: TsdfIntegratorConfig,
}

impl TsdfIntegrator {
    pub fn new(config: TsdfIntegratorConfig) -> Self {
        Self { config }
    }

    pub fn integrate_image<const VPS: usize>(
        &mut self,
        layer: &mut Layer<Tsdf, VPS>,
        image: &image::RgbImage,
        updated_block_indices: &mut BTreeSet<BlockIndex<VPS>>,
    ) {
        for y in 0..image.width() {
            for x in 0..image.height() {
                let global_index = GlobalIndex::from_point(
                    &point![x as Real, y as Real, 0.0],
                    layer.voxel_size_inv(),
                );
                let (block_index, voxel_index) = global_index.block_voxel_index();

                let mut lock = layer.allocate_block_by_index(&block_index).write();
                let voxel = lock.voxel_from_index_mut(&voxel_index);

                if image.get_pixel(x, y).0 == [0, 0, 0] {
                    if voxel.weight == 0.0 {
                        voxel.distance = self.config.default_truncation_distance;
                        voxel.weight = 1.0;
                        updated_block_indices.insert(block_index);
                    }
                } else if voxel.weight == 1.0 {
                    voxel.distance = 0.0;
                    voxel.weight = 0.0;

                    updated_block_indices.insert(block_index);
                }
            }
        }
    }
}
