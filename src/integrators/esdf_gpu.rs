use crate::{
    core::{
        index::BlockIndex,
        layer::Layer,
        voxel::{Esdf, EsdfFlags, Tsdf},
    },
    wgpu_utils::{GpuPropagate, GpuSweep},
};

use std::{collections::BTreeSet, time::Duration};

#[derive(Default)]
pub struct EsdfIntegratorConfig {}

pub struct EsdfIntegrator {
    config: EsdfIntegratorConfig,
    sweep_cache: GpuSweep,
    propgate_cache: GpuPropagate,
}

impl EsdfIntegrator {
    pub fn new(
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        config: EsdfIntegratorConfig,
    ) -> Self {
        Self {
            config,
            sweep_cache: GpuSweep::new(device, queue),
            propgate_cache: GpuPropagate::new(device, queue),
        }
    }

    pub fn update_blocks<
        const VPS: usize,
        F: FnMut(&str, &Layer<Tsdf, VPS>, &Layer<Esdf, VPS>, &[BlockIndex<VPS>], Duration),
    >(
        &mut self,
        tsdf_layer: &Layer<Tsdf, VPS>,
        esdf_layer: &mut Layer<Esdf, VPS>,
        updated_blocks: &BTreeSet<BlockIndex<VPS>>,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        mut callback: F,
    ) {
        let start = std::time::Instant::now();

        let mut dirty_blocks = BTreeSet::new();
        let mut sites_indices_to_clear = BTreeSet::new();
        let mut blocks_to_clear = updated_blocks.clone();

        callback(
            "tsdf updated",
            tsdf_layer,
            esdf_layer,
            &updated_blocks.iter().copied().collect::<Vec<_>>(),
            Duration::from_millis(500),
        );

        // allocate all blocks from the tsdf layer
        for block_index in tsdf_layer.allocated_blocks_iter() {
            esdf_layer.allocate_block_by_index(block_index);
        }

        // create a list of site *indices* to clear
        for block_index in updated_blocks {
            let esdf_block = esdf_layer.allocate_block_by_index(block_index);
            let esdf_lock = esdf_block.read();

            for voxel in esdf_lock.as_slice() {
                if voxel.flags.contains(EsdfFlags::HasSiteIndex) {
                    sites_indices_to_clear.insert(BlockIndex::new(
                        voxel.site_block_index[0],
                        voxel.site_block_index[1],
                        voxel.site_block_index[2],
                    ));
                }
            }
        }

        // create a list of all blocks containing sites to clear
        let mut open_list = updated_blocks.clone(); // blocks to visit
        let mut closed_list = BTreeSet::new(); // blocks already visited
        while let Some(index) = open_list.pop_first() {
            closed_list.insert(index);

            if let Some(esdf_block) = esdf_layer.block_by_index(&index) {
                {
                    let esdf_lock = esdf_block.read();
                    let mut flagged_clear = false;

                    for voxel in esdf_lock.as_slice() {
                        if voxel.flags.contains(EsdfFlags::HasSiteIndex)
                            && sites_indices_to_clear.contains(&BlockIndex::<VPS>::new(
                                voxel.site_block_index[0],
                                voxel.site_block_index[1],
                                voxel.site_block_index[2],
                            ))
                        {
                            blocks_to_clear.insert(index);
                            flagged_clear = true;
                            break;
                        }
                    }

                    if flagged_clear {
                        // also explore its neighbors
                        for neighbour in index.neighbors6() {
                            if !closed_list.contains(&neighbour.index)
                                && esdf_layer.contains(&neighbour.index)
                            {
                                open_list.insert(neighbour.index);
                            }
                        }
                    } else {
                        // propagate from these blocks as
                        // they are still valid
                        dirty_blocks.insert(index);
                    }
                }
            }
        }

        // reset blocks
        for block_index in &blocks_to_clear {
            {
                let esdf_block = esdf_layer.allocate_block_by_index(block_index);
                let mut esdf_lock = esdf_block.write();
                esdf_lock.reset_voxels();
            }

            callback(
                "clear site",
                tsdf_layer,
                esdf_layer,
                &[*block_index],
                Duration::from_millis(50),
            );
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
                    esdf_voxel
                        .flags
                        .insert(EsdfFlags::Fixed | EsdfFlags::Observed | EsdfFlags::HasSiteIndex);
                    esdf_voxel.site_block_index = block_index.coords.into();
                    dirty_blocks.insert(*block_index);
                } else {
                    esdf_voxel.distance = 0.0;
                    esdf_voxel.flags.remove(EsdfFlags::all());
                }
            }
        }

        while !dirty_blocks.is_empty() {
            let start = std::time::Instant::now();
            // sweep
            self.sweep_gpu(esdf_layer, device, queue, &dirty_blocks);
            let indices: Vec<_> = dirty_blocks.iter().copied().collect();
            callback(
                "sweep: xy (GPU)",
                tsdf_layer,
                esdf_layer,
                &indices,
                Duration::from_millis(1000),
            );

            // propagate
            dirty_blocks = self.propagate_gpu(esdf_layer, device, queue, &dirty_blocks);
            let indices: Vec<_> = dirty_blocks.iter().copied().collect();
            callback(
                "prop.: xy (GPU)",
                tsdf_layer,
                esdf_layer,
                &indices,
                Duration::from_millis(1000),
            );
            println!(
                "ESDF pass finished in {:?}",
                std::time::Instant::now().duration_since(start)
            );
        }

        println!(
            "=> ESDF update finished in {:?}",
            std::time::Instant::now().duration_since(start)
        );
    }

    fn sweep_gpu<const VPS: usize>(
        &mut self,
        esdf_layer: &mut Layer<Esdf, VPS>,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        dirty_blocks: &BTreeSet<BlockIndex<VPS>>,
    ) {
        firestorm::profile_method!("sweep_gpu");

        let blocks: Vec<_> = dirty_blocks
            .iter()
            .map(|index| esdf_layer.block_by_index(index).unwrap())
            .collect();

        self.sweep_cache.submit(device, queue, &blocks);
        // wgpu_utils::sweep_blocks(device, queue, blocks.as_slice()).await;
    }

    fn propagate_gpu<const VPS: usize>(
        &mut self,
        esdf_layer: &mut Layer<Esdf, VPS>,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        dirty_blocks: &BTreeSet<BlockIndex<VPS>>,
    ) -> BTreeSet<BlockIndex<VPS>> {
        firestorm::profile_method!("propagate_gpu");

        // padded list of blocks (blocks themselves + direct neighbors)
        let block_indices_of_interest = BTreeSet::from_iter(
            dirty_blocks
                .iter()
                .flat_map(|index| index.neighbors6_include_self().map(|p| p.index))
                .filter(|p| esdf_layer.contains(p)),
        );

        // BlockIndex -> u32
        let block_index_map = std::collections::BTreeMap::from_iter(
            block_indices_of_interest
                .iter()
                .enumerate()
                .map(|p| (p.1, p.0)),
        );

        // u32 -> BlockIndex
        let block_index_map_inv = std::collections::BTreeMap::from_iter(
            block_indices_of_interest
                .iter()
                .enumerate()
                .map(|p| (p.0, p.1)),
        );

        // a subset of all blocks to upload to the GPU
        let blocks: Vec<_> = block_indices_of_interest
            .iter()
            .filter_map(|p| esdf_layer.block_by_index(p))
            .collect();

        // work indices for the shader
        let workgroup_block_indices: Vec<u32> = dirty_blocks
            .iter()
            .flat_map(|p| {
                p.neighbors6_include_self().map(|p| {
                    block_index_map
                        .get(&p.index)
                        .copied()
                        .map(|p| p as u32)
                        .unwrap_or(u32::MAX) // MAX indicates an invalid index
                })
            })
            .collect();

        let block_info =
            self.propgate_cache
                .submit(device, queue, &workgroup_block_indices, &blocks);
        // wgpu_utils::propagate_blocks(device, queue, &workgroup_block_indices, &blocks).await;

        // blocks updated by the shader are considered dirty and have to be swept again
        BTreeSet::from_iter(
            block_info
                .iter()
                .enumerate()
                .filter(|p| p.1.flags.contains(EsdfFlags::Updated))
                .map(|p| *block_index_map_inv.get(&p.0).unwrap())
                .copied(),
        )
    }
}
