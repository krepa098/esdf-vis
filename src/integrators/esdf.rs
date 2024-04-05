use nalgebra::point;

use crate::core::{
    index::{BlockIndex, VoxelIndex},
    layer::Layer,
    voxel::{Esdf, Tsdf},
};

use std::collections::BTreeSet;

#[derive(Default)]
pub struct EsdfIntegratorConfig {}

enum OpDir {
    XPlus,
    XMinus,
    YPlus,
    YMinus,
    ZPlus,
    ZMinus,
}

pub struct EsdfIntegrator {
    config: EsdfIntegratorConfig,
}

impl EsdfIntegrator {
    pub fn new(config: EsdfIntegratorConfig) -> Self {
        Self { config }
    }

    pub fn update_blocks<
        const VPS: usize,
        F: FnMut(&str, &Layer<Tsdf, VPS>, &Layer<Esdf, VPS>, &BlockIndex<VPS>),
    >(
        &mut self,
        tsdf_layer: &Layer<Tsdf, VPS>,
        esdf_layer: &mut Layer<Esdf, VPS>,
        mut callback: F,
    ) {
        let mut dirty_blocks = BTreeSet::new();
        let mut propagate_blocks = BTreeSet::new();

        for block_index in tsdf_layer.allocated_blocks_iter() {
            let tsdf_block = tsdf_layer.block_by_index(block_index).unwrap();
            let esdf_block = esdf_layer.allocate_block_by_index(block_index);
            let mut esdf_lock = esdf_block.write();

            for (i, tsdf_voxel) in tsdf_block.read().as_slice().iter().enumerate() {
                if tsdf_voxel.weight > 0.0 {
                    let esdf_voxel = esdf_lock.voxel_from_lin_index_mut(i);
                    esdf_voxel.distance = tsdf_voxel.distance;
                    esdf_voxel.fixed = true;
                    esdf_voxel.observed = true;

                    dirty_blocks.insert(*block_index);
                }
            }
        }

        while !dirty_blocks.is_empty() {
            while let Some(block_index) = dirty_blocks.pop_first() {
                Self::sweep_block(OpDir::XPlus, &block_index, esdf_layer);
                callback("sweep: x+", tsdf_layer, esdf_layer, &block_index);
                Self::sweep_block(OpDir::XMinus, &block_index, esdf_layer);
                callback("sweep: x-", tsdf_layer, esdf_layer, &block_index);
                Self::sweep_block(OpDir::YPlus, &block_index, esdf_layer);
                callback("sweep: y+", tsdf_layer, esdf_layer, &block_index);
                Self::sweep_block(OpDir::YMinus, &block_index, esdf_layer);
                callback("sweep: y-", tsdf_layer, esdf_layer, &block_index);

                propagate_blocks.insert(block_index);
            }

            while let Some(block_index) = propagate_blocks.pop_first() {
                if let Some(dirty_block_index) =
                    Self::propagate_to_neighbour(OpDir::XPlus, &block_index, esdf_layer)
                {
                    dirty_blocks.insert(dirty_block_index);
                    callback("prop.: x+", tsdf_layer, esdf_layer, &dirty_block_index);
                }

                if let Some(dirty_block_index) =
                    Self::propagate_to_neighbour(OpDir::XMinus, &block_index, esdf_layer)
                {
                    dirty_blocks.insert(dirty_block_index);
                    callback("prop.: x-", tsdf_layer, esdf_layer, &dirty_block_index);
                }

                if let Some(dirty_block_index) =
                    Self::propagate_to_neighbour(OpDir::YPlus, &block_index, esdf_layer)
                {
                    dirty_blocks.insert(dirty_block_index);
                    callback("prop.: y+", tsdf_layer, esdf_layer, &dirty_block_index);
                }

                if let Some(dirty_block_index) =
                    Self::propagate_to_neighbour(OpDir::YMinus, &block_index, esdf_layer)
                {
                    dirty_blocks.insert(dirty_block_index);
                    callback("prop.: y-", tsdf_layer, esdf_layer, &dirty_block_index);
                }
            }
        }
    }

    fn sweep_block<const VPS: usize>(
        dir: OpDir,
        index: &BlockIndex<VPS>,
        esdf_layer: &mut Layer<Esdf, VPS>,
    ) {
        let (step, order) = match dir {
            OpDir::XPlus => (1i32, [2, 1, 0]),
            OpDir::XMinus => (-1, [2, 1, 0]),
            OpDir::YPlus => (1, [2, 0, 1]),
            OpDir::YMinus => (-1, [2, 0, 1]),
            OpDir::ZPlus => (1, [1, 0, 2]),
            OpDir::ZMinus => (-1, [1, 0, 2]),
        };

        let voxel_size = esdf_layer.voxel_size();
        let mut lock = esdf_layer.block_by_index(index).unwrap().write();

        for u in 0..VPS {
            for v in 0..VPS {
                let w_range = if step > 0 {
                    create_range_chain(1, VPS - 1)
                } else {
                    create_range_chain(VPS - 2, 0)
                };

                for w in w_range {
                    let mut p = point![0, 0, 0];
                    p[order[0]] = u;
                    p[order[1]] = v;
                    p[order[2]] = w;

                    let voxel_index = VoxelIndex(p);

                    p[order[2]] = (w as i32 - step) as usize;
                    let parent_voxel_index = VoxelIndex(p);

                    let parent_voxel = lock.voxel_from_index(&parent_voxel_index);
                    let parent_fixed = parent_voxel.fixed;
                    let parent_dist = parent_voxel.distance;

                    let voxel = lock.voxel_from_index_mut(&voxel_index);

                    if parent_fixed && !voxel.observed {
                        if !voxel.fixed {
                            voxel.distance = parent_dist + voxel_size;
                            voxel.fixed = true;
                        } else {
                            voxel.distance = voxel.distance.min(parent_dist + voxel_size);
                        }
                    }
                }
            }
        }
    }

    fn propagate_to_neighbour<const VPS: usize>(
        dir: OpDir,
        pivot_index: &BlockIndex<VPS>,
        esdf_layer: &mut Layer<Esdf, VPS>,
    ) -> Option<BlockIndex<VPS>> {
        let voxel_size = esdf_layer.voxel_size();

        let nblock_index = match dir {
            OpDir::XPlus => BlockIndex::new(pivot_index.x + 1, pivot_index.y, pivot_index.z),
            OpDir::XMinus => BlockIndex::new(pivot_index.x - 1, pivot_index.y, pivot_index.z),
            OpDir::YPlus => BlockIndex::new(pivot_index.x, pivot_index.y + 1, pivot_index.z),
            OpDir::YMinus => BlockIndex::new(pivot_index.x, pivot_index.y - 1, pivot_index.z),
            OpDir::ZPlus => BlockIndex::new(pivot_index.x, pivot_index.y, pivot_index.z + 1),
            OpDir::ZMinus => BlockIndex::new(pivot_index.x, pivot_index.y, pivot_index.z - 1),
        };

        let (n_index, p_index, order) = match dir {
            OpDir::XPlus => (0usize, VPS - 1, [2, 1, 0]),
            OpDir::XMinus => (VPS - 1, 0, [2, 1, 0]),
            OpDir::YPlus => (0, VPS - 1, [2, 0, 1]),
            OpDir::YMinus => (VPS - 1, 0, [2, 0, 1]),
            OpDir::ZPlus => (0, VPS - 1, [1, 0, 2]),
            OpDir::ZMinus => (VPS - 1, 0, [1, 0, 2]),
        };

        let mut dirty = false;

        if let Some(neighbour_block) = esdf_layer.block_by_index(&nblock_index) {
            let pivot_block = esdf_layer.block_by_index(pivot_index).unwrap().read();

            let mut nlock = neighbour_block.write();

            for u in 0..VPS {
                for v in 0..VPS {
                    let mut p = point![0, 0, 0];
                    p[order[0]] = u;
                    p[order[1]] = v;
                    p[order[2]] = p_index;

                    let p_voxel_index = VoxelIndex(p);

                    p[order[2]] = n_index;
                    let n_voxel_index = VoxelIndex(p);

                    // propagate through the sides
                    let pivot_voxel = pivot_block.voxel_from_index(&p_voxel_index);
                    let pivot_fixed = pivot_voxel.fixed;
                    let pivot_dist = pivot_voxel.distance;

                    let neighbour_voxel = nlock.voxel_from_index_mut(&n_voxel_index);
                    let neighbour_fixed = neighbour_voxel.fixed;
                    let neighbour_observed = neighbour_voxel.observed;

                    if pivot_fixed && !neighbour_observed {
                        if neighbour_fixed {
                            // found a shorter distance?
                            if neighbour_voxel.distance > pivot_dist + voxel_size {
                                neighbour_voxel.distance = pivot_dist + voxel_size;
                                dirty = true;
                            }
                        } else {
                            neighbour_voxel.distance = pivot_dist + voxel_size;
                            neighbour_voxel.fixed = true;
                            dirty = true;
                        }
                    }
                }
            }
        }

        dirty.then_some(nblock_index)
    }
}

fn create_range_chain(a: usize, b: usize) -> impl Iterator<Item = usize> {
    #[allow(clippy::reversed_empty_ranges)]
    let (part1, part2) = if a <= b {
        (a..=b, 1..=0)
    } else {
        (1..=0, b..=a)
    };

    part1.chain(part2.rev())
}
