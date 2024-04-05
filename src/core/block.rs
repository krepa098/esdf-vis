use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::index::VoxelIndex;
use super::prelude::*;
use super::storage::VoxelStorage;
use super::voxel::Voxel;

#[derive(Debug)]
pub struct Block<VoxelType: Voxel, const VPS: usize> {
    voxel_count: usize,
    voxel_size: Real,
    voxel_inv_size: Real,
    block_size: Real,
    block_size_inv: Real,
    origin: Point3<Real>,
    voxels: RwLock<VoxelStorage<VoxelType, VPS, VPS, VPS>>,
}

impl<VoxelType: Voxel, const VPS: usize> Block<VoxelType, VPS> {
    pub fn new(voxel_size: Real, origin: Point3<Real>) -> Self {
        let vps = VPS as Real;
        let voxel_count = VPS * VPS * VPS;
        let block_size = vps * voxel_size;

        Self {
            voxel_count,
            voxel_size,
            voxel_inv_size: voxel_size.recip(),
            block_size,
            block_size_inv: block_size.recip(),
            origin,
            voxels: RwLock::new(VoxelStorage::default()),
        }
    }

    pub fn origin(&self) -> &Point3<Real> {
        &self.origin
    }

    pub fn lin_index_from_voxel_index(index: &VoxelIndex<VPS>) -> usize {
        assert!(index.x < VPS);
        assert!(index.y < VPS);
        assert!(index.z < VPS);

        let x = index.x;
        let y = index.y;
        let z = index.z;

        x + VPS * (y + z * VPS)
    }

    pub fn voxel_index_from_lin_index(index: usize) -> VoxelIndex<VPS> {
        let (q, rem) = num_integer::div_rem(index, VPS * VPS);
        let z = q;
        let (q, rem) = num_integer::div_rem(rem, VPS);
        let y = q;
        let x = rem;

        VoxelIndex(Point3::new(x, y, z))
    }

    pub fn voxel_position_from_lin_index(&self, index: usize) -> Point3<Real> {
        let voxel_index = Self::voxel_index_from_lin_index(index);
        Point3::new(
            self.origin.x + (voxel_index.x as Real + 0.5) * self.voxel_size,
            self.origin.y + (voxel_index.y as Real + 0.5) * self.voxel_size,
            self.origin.z + (voxel_index.z as Real + 0.5) * self.voxel_size,
        )
    }

    pub fn read(&self) -> BlockReadLock<VoxelType, VPS> {
        BlockReadLock {
            voxels: self.voxels.read(),
        }
    }

    pub fn write(&self) -> BlockWriteLock<VoxelType, VPS> {
        BlockWriteLock {
            voxels: self.voxels.write(),
        }
    }
}

pub struct BlockWriteLock<'a, VoxelType: Voxel, const VPS: usize> {
    voxels: RwLockWriteGuard<'a, VoxelStorage<VoxelType, VPS, VPS, VPS>>,
}

impl<'a, VoxelType: Voxel, const VPS: usize> BlockWriteLock<'a, VoxelType, VPS> {
    #[inline]
    pub fn voxel_iter(&self) -> impl Iterator<Item = &VoxelType> {
        self.voxels.as_slice().iter()
    }

    #[inline]
    pub fn as_slice(&self) -> &[VoxelType] {
        self.voxels.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [VoxelType] {
        self.voxels.as_mut_slice()
    }

    #[inline]
    pub fn voxel_from_lin_index(&self, index: usize) -> &VoxelType {
        self.as_slice().get(index).unwrap()
    }

    #[inline]
    pub fn voxel_from_lin_index_mut(&mut self, index: usize) -> &mut VoxelType {
        self.as_mut_slice().get_mut(index).unwrap()
    }

    #[inline]
    pub fn voxel_from_index(&self, index: &VoxelIndex<VPS>) -> &VoxelType {
        let lin_index = Block::<VoxelType, VPS>::lin_index_from_voxel_index(index);
        self.as_slice().get(lin_index).unwrap()
    }

    #[inline]
    pub fn voxel_from_index_mut(&mut self, index: &VoxelIndex<VPS>) -> &mut VoxelType {
        let lin_index = Block::<VoxelType, VPS>::lin_index_from_voxel_index(index);
        self.as_mut_slice().get_mut(lin_index).unwrap()
    }
}

pub struct BlockReadLock<'a, VoxelType: Voxel, const VPS: usize> {
    voxels: RwLockReadGuard<'a, VoxelStorage<VoxelType, VPS, VPS, VPS>>,
}

impl<'a, VoxelType: Voxel, const VPS: usize> BlockReadLock<'a, VoxelType, VPS> {
    #[inline]
    pub fn voxel_iter(&self) -> impl Iterator<Item = &VoxelType> {
        self.voxels.as_slice().iter()
    }

    #[inline]
    pub fn as_slice(&self) -> &[VoxelType] {
        self.voxels.as_slice()
    }

    #[inline]
    pub fn voxel_from_lin_index(&self, index: usize) -> &VoxelType {
        self.as_slice().get(index).unwrap()
    }

    #[inline]
    pub fn voxel_from_index(&self, index: &VoxelIndex<VPS>) -> &VoxelType {
        let lin_index = Block::<VoxelType, VPS>::lin_index_from_voxel_index(index);
        self.as_slice().get(lin_index).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Default, Clone, Copy)]
    struct TestVoxel {
        distance: f32,
    }

    impl Voxel for TestVoxel {}

    #[test]
    fn test_block_index() {
        assert_eq!(
            Block::<TestVoxel, 3>::lin_index_from_voxel_index(&VoxelIndex(Point3::new(2, 2, 2))),
            26
        );
        assert_eq!(
            Block::<TestVoxel, 3>::voxel_index_from_lin_index(26),
            VoxelIndex(Point3::new(2, 2, 2))
        );

        assert_eq!(
            Block::<TestVoxel, 3>::voxel_index_from_lin_index(0),
            VoxelIndex(Point3::new(0, 0, 0))
        );
    }
}
