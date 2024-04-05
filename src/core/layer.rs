use std::collections::HashMap;

use super::prelude::*;

use super::{block::Block, index::BlockIndex, voxel::Voxel};

pub struct Layer<VoxelType: Voxel, const VPS: usize> {
    block_size: Real,
    block_size_inv: Real,
    voxel_size: Real,
    voxel_size_inv: Real,
    blocks: HashMap<BlockIndex<VPS>, Block<VoxelType, VPS>>,
}

impl<VoxelType: Voxel + Copy, const VPS: usize> Layer<VoxelType, VPS> {
    pub fn new(voxel_size: Real) -> Self {
        let block_size = voxel_size * (VPS as Real);
        let block_size_inv = block_size.recip();
        let voxel_size_inv = voxel_size.recip();

        Self {
            block_size,
            block_size_inv,
            voxel_size,
            voxel_size_inv,
            blocks: HashMap::new(),
        }
    }

    #[inline]
    pub fn voxel_size(&self) -> Real {
        self.voxel_size
    }

    #[inline]
    pub fn voxel_size_inv(&self) -> Real {
        self.voxel_size_inv
    }

    #[inline]
    pub fn voxels_per_side(&self) -> usize {
        VPS
    }

    #[inline]
    pub fn block_size(&self) -> Real {
        self.block_size
    }

    #[inline]
    pub fn block_size_inv(&self) -> Real {
        self.block_size_inv
    }

    pub fn block_by_index(&self, index: &BlockIndex<VPS>) -> Option<&Block<VoxelType, VPS>> {
        self.blocks.get(index)
    }

    pub fn block_by_index_mut(
        &mut self,
        index: &BlockIndex<VPS>,
    ) -> Option<&mut Block<VoxelType, VPS>> {
        self.blocks.get_mut(index)
    }

    pub fn allocate_block_by_index(
        &mut self,
        index: &BlockIndex<VPS>,
    ) -> &mut Block<VoxelType, VPS> {
        if !self.blocks.contains_key(index) {
            self.blocks.insert(
                *index,
                Block::new(self.voxel_size, self.origin_from_index(index)),
            );
        }

        self.blocks.get_mut(index).unwrap()
    }

    pub fn origin_from_index(&self, index: &BlockIndex<VPS>) -> Point3<Real> {
        Point3::new(
            (index.x as Real) * self.block_size,
            (index.y as Real) * self.block_size,
            (index.z as Real) * self.block_size,
        )
    }

    pub fn center_point_from_index(&self, index: &BlockIndex<VPS>) -> Point3<Real> {
        Point3::new(
            ((index.x as Real) + 0.5) * self.block_size,
            ((index.y as Real) + 0.5) * self.block_size,
            ((index.z as Real) + 0.5) * self.block_size,
        )
    }

    pub fn allocated_blocks_iter(&self) -> impl Iterator<Item = &BlockIndex<VPS>> {
        self.blocks.keys()
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
    }

    pub fn min_max_pred<R: PartialOrd + Copy, F: Fn(&VoxelType) -> R>(
        &self,
        pred: F,
    ) -> (Option<R>, Option<R>) {
        let mut curr_max: Option<R> = None;
        let mut curr_min: Option<R> = None;

        for block_index in self.allocated_blocks_iter() {
            let block = self.block_by_index(block_index).unwrap();

            for voxel in block.read().as_slice() {
                let p = pred(voxel);

                if *curr_max.get_or_insert(p) < p {
                    curr_max = Some(p);
                };
                if *curr_min.get_or_insert(p) > p {
                    curr_min = Some(p);
                };
            }
        }

        (curr_min, curr_max)
    }
}
