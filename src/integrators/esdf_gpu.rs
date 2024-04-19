use nalgebra::point;

use crate::{
    core::{
        index::{BlockIndex, VoxelIndex},
        layer::Layer,
        voxel::{Esdf, EsdfGpuFlags, Tsdf},
    },
    wgpu_utils,
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

    pub async fn update_blocks<
        const VPS: usize,
        F: FnMut(&str, &Layer<Tsdf, VPS>, &Layer<Esdf, VPS>, &BlockIndex<VPS>),
    >(
        &mut self,
        tsdf_layer: &Layer<Tsdf, VPS>,
        esdf_layer: &mut Layer<Esdf, VPS>,
        updated_blocks: &BTreeSet<BlockIndex<VPS>>,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        mut callback: F,
    ) {
        let mut dirty_blocks = BTreeSet::new();
        let mut propagate_blocks = BTreeSet::new();
        let mut sites_indices_to_clear = BTreeSet::new();
        let mut blocks_to_clear = updated_blocks.clone();

        // allocate all blocks from the tsdf layer
        for block_index in tsdf_layer.allocated_blocks_iter() {
            esdf_layer.allocate_block_by_index(block_index);
        }

        // create a list of sites to clear
        for block_index in updated_blocks {
            let esdf_block = esdf_layer.allocate_block_by_index(block_index);
            let mut esdf_lock = esdf_block.write();

            for voxel in esdf_lock.as_mut_slice() {
                if voxel.flags.contains(EsdfGpuFlags::HasSiteIndex) {
                    sites_indices_to_clear.insert(BlockIndex::<VPS>::new(
                        voxel.site_block_index[0],
                        voxel.site_block_index[1],
                        voxel.site_block_index[2],
                    ));
                }
            }
        }

        // create a list of all blocks containing sites to clear
        let mut open_list = updated_blocks.clone();
        let mut closed_list = BTreeSet::new();
        while let Some(index) = open_list.pop_first() {
            closed_list.insert(index);

            if let Some(esdf_block) = esdf_layer.block_by_index_mut(&index) {
                {
                    let mut esdf_lock = esdf_block.write();

                    for voxel in esdf_lock.as_mut_slice() {
                        if voxel.flags.contains(EsdfGpuFlags::HasSiteIndex)
                            && sites_indices_to_clear.contains(&BlockIndex::<VPS>::new(
                                voxel.site_block_index[0],
                                voxel.site_block_index[1],
                                voxel.site_block_index[2],
                            ))
                        {
                            blocks_to_clear.insert(index);
                        }
                    }
                }

                // also explore its neighbours
                for neighbour in index.neighbours() {
                    if !closed_list.contains(&neighbour.index)
                        && esdf_layer.block_by_index(&neighbour.index).is_some()
                    {
                        open_list.insert(neighbour.index);
                    }
                }
            }

            // blocks_to_check.clear();
        }

        // reset blocks
        for block_index in &blocks_to_clear {
            {
                let esdf_block = esdf_layer.allocate_block_by_index(block_index);
                let mut esdf_lock = esdf_block.write();
                esdf_lock.reset_voxels();
            }

            callback("clear site", tsdf_layer, esdf_layer, block_index);
        }

        // transfer tsdf to esdf
        for block_index in &blocks_to_clear {
            let tsdf_block = tsdf_layer.block_by_index(block_index).unwrap();
            let esdf_block = esdf_layer.allocate_block_by_index(block_index);
            let mut esdf_lock = esdf_block.write();

            for (i, tsdf_voxel) in tsdf_block.read().as_slice().iter().enumerate() {
                let esdf_voxel = esdf_lock.voxel_from_lin_index_mut(i);

                if tsdf_voxel.weight > 0.0 {
                    esdf_voxel.distance = tsdf_voxel.distance;
                    esdf_voxel.flags.insert(EsdfGpuFlags::Fixed);
                    esdf_voxel.flags.insert(EsdfGpuFlags::Observed);
                    esdf_voxel.site_block_index = block_index.coords.into();
                    dirty_blocks.insert(*block_index);
                } else {
                    esdf_voxel.distance = 0.0;
                    esdf_voxel.flags.remove(EsdfGpuFlags::Fixed);
                    esdf_voxel.flags.remove(EsdfGpuFlags::Observed);
                    esdf_voxel.flags.remove(EsdfGpuFlags::HasSiteIndex);
                }
            }
        }

        while !dirty_blocks.is_empty() {
            // while let Some(block_index) = dirty_blocks.pop_first() {
            //     // Self::sweep_block(OpDir::XPlus, &block_index, esdf_layer);
            //     // callback("sweep: x+", tsdf_layer, esdf_layer, &block_index);
            //     // Self::sweep_block(OpDir::XMinus, &block_index, esdf_layer);
            //     // callback("sweep: x-", tsdf_layer, esdf_layer, &block_index);
            //     // Self::sweep_block(OpDir::YPlus, &block_index, esdf_layer);
            //     // callback("sweep: y+", tsdf_layer, esdf_layer, &block_index);
            //     // Self::sweep_block(OpDir::YMinus, &block_index, esdf_layer);
            //     // callback("sweep: y-", tsdf_layer, esdf_layer, &block_index);

            //     propagate_blocks.insert(block_index);
            // }
            Self::sweep_gpu(esdf_layer, device, queue, &dirty_blocks).await;
            propagate_blocks = dirty_blocks.clone();
            callback(
                "sweep: gpu",
                tsdf_layer,
                esdf_layer,
                &BlockIndex(point![0, 0, 0]),
            );
            dirty_blocks.clear();

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
                    let parent_fixed = parent_voxel.flags.contains(EsdfGpuFlags::Fixed);
                    let parent_dist = parent_voxel.distance;
                    let parent_site_block_index = parent_voxel.site_block_index;

                    let voxel = lock.voxel_from_index_mut(&voxel_index);

                    if parent_fixed && !voxel.flags.contains(EsdfGpuFlags::Observed) {
                        if !voxel.flags.contains(EsdfGpuFlags::Fixed) {
                            voxel.distance = parent_dist + voxel_size;
                            voxel.flags.insert(EsdfGpuFlags::Fixed);
                            voxel.site_block_index = parent_site_block_index;
                        } else if voxel.distance > parent_dist + voxel_size {
                            voxel.distance = voxel.distance.min(parent_dist + voxel_size);
                            voxel.site_block_index = parent_site_block_index;
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
                    let pivot_fixed = pivot_voxel.flags.contains(EsdfGpuFlags::Fixed);
                    let pivot_dist = pivot_voxel.distance;
                    let pivot_site_block_index = pivot_voxel.site_block_index;

                    let neighbour_voxel = nlock.voxel_from_index_mut(&n_voxel_index);
                    let neighbour_fixed = neighbour_voxel.flags.contains(EsdfGpuFlags::Fixed);
                    let neighbour_observed = neighbour_voxel.flags.contains(EsdfGpuFlags::Observed);

                    if pivot_fixed && !neighbour_observed {
                        if neighbour_fixed {
                            // found a shorter distance?
                            if neighbour_voxel.distance > pivot_dist + voxel_size {
                                neighbour_voxel.distance = pivot_dist + voxel_size;
                                neighbour_voxel.site_block_index = pivot_site_block_index;
                                dirty = true;
                            }
                        } else {
                            neighbour_voxel.distance = pivot_dist + voxel_size;
                            neighbour_voxel.flags |= EsdfGpuFlags::Fixed;
                            neighbour_voxel.site_block_index = pivot_site_block_index;
                            dirty = true;
                        }
                    }
                }
            }
        }

        dirty.then_some(nblock_index)
    }

    async fn sweep_gpu<const VPS: usize>(
        esdf_layer: &mut Layer<Esdf, VPS>,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        dirty_blocks: &BTreeSet<BlockIndex<VPS>>,
    ) {
        let blocks: Vec<_> = dirty_blocks
            .iter()
            .map(|index| esdf_layer.block_by_index(index).unwrap())
            .collect();

        wgpu_utils::sweep_blocks(device, queue, blocks.as_slice()).await;
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
