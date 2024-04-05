use std::ops::{Add, Deref, Sub};

use super::{prelude::*, utils::grid_index_from_point};

pub trait GridIndex: From<Point3<i64>> {}

impl<const VPS: usize> GridIndex for GlobalIndex<VPS> {}
impl<const VPS: usize> GridIndex for BlockIndex<VPS> {}

/// Global Index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalIndex<const VPS: usize>(pub Point3<i64>);

impl<const VPS: usize> Deref for GlobalIndex<VPS> {
    type Target = Point3<i64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const VPS: usize> Sub for GlobalIndex<VPS> {
    type Output = GlobalIndex<VPS>;

    fn sub(self, rhs: Self) -> Self::Output {
        GlobalIndex((self.0 - rhs.0).into())
    }
}

impl<const VPS: usize> Add for GlobalIndex<VPS> {
    type Output = GlobalIndex<VPS>;

    fn add(self, rhs: Self) -> Self::Output {
        GlobalIndex((self.0.coords + rhs.0.coords).into())
    }
}

impl<const VPS: usize> GlobalIndex<VPS> {
    pub fn from_point(p: &Point3<Real>, grid_size_inv: Real) -> Self {
        Self::from(grid_index_from_point(p, grid_size_inv))
    }

    pub fn from_block_and_local_lin_index(
        block_index: &BlockIndex<VPS>,
        local_lin_index: usize,
    ) -> Self {
        assert!(local_lin_index < VPS * VPS * VPS);

        let mut p = block_index.0 * VPS as i32;

        let (q, rem) = num_integer::div_rem(local_lin_index, VPS * VPS);
        p.z += q as i32;
        let (q, rem) = num_integer::div_rem(rem, VPS);
        p.y += q as i32;
        p.x += rem as i32;

        GlobalIndex(p.cast())
    }

    pub fn from_block_and_voxel_index(
        block_index: &BlockIndex<VPS>,
        voxel_index: &VoxelIndex<VPS>,
    ) -> Self {
        Self(
            (block_index.0.cast::<i64>().coords * VPS as i64 + voxel_index.0.cast::<i64>().coords)
                .into(),
        )
    }

    pub fn block_voxel_index(&self) -> (BlockIndex<VPS>, VoxelIndex<VPS>) {
        (self.block_index(), self.local_voxel_index())
    }

    pub fn block_index(&self) -> BlockIndex<VPS> {
        BlockIndex(Point3::new(
            self.x.div_euclid(VPS as i64) as i32,
            self.y.div_euclid(VPS as i64) as i32,
            self.z.div_euclid(VPS as i64) as i32,
        ))
    }

    pub fn local_voxel_index(&self) -> VoxelIndex<VPS> {
        VoxelIndex(Point3::new(
            self.x.rem_euclid(VPS as i64) as usize,
            self.y.rem_euclid(VPS as i64) as usize,
            self.z.rem_euclid(VPS as i64) as usize,
        ))
    }

    pub fn center(&self, voxel_size: Real) -> Point3<Real> {
        Point3::new(
            ((self.x as Real) + 0.5) * voxel_size,
            ((self.y as Real) + 0.5) * voxel_size,
            ((self.z as Real) + 0.5) * voxel_size,
        )
    }

    pub fn neighbours(&self) -> GlobalIndexNeighbourIter<VPS> {
        GlobalIndexNeighbourIter { pivot: self, n: 0 }
    }
}

impl<const VPS: usize> From<Point3<i64>> for GlobalIndex<VPS> {
    fn from(value: Point3<i64>) -> Self {
        Self(value.cast())
    }
}

pub struct GlobalIndexNeighbourIter<'a, const VPS: usize> {
    pivot: &'a GlobalIndex<VPS>,
    n: usize,
}

pub struct Neighbour<const VPS: usize> {
    pub index: GlobalIndex<VPS>,
    pub dir: Vector3<i64>,
    pub grid_dist: Real,
}

impl<'a, const VPS: usize> Iterator for GlobalIndexNeighbourIter<'a, VPS> {
    type Item = Neighbour<VPS>;

    fn next(&mut self) -> Option<Self::Item> {
        #[allow(clippy::approx_constant)]
        const SQRT_2: Real = 1.414213562;
        const SQRT_3: Real = 1.732050808;

        const NEIGHBOUR_OFFSETS: [(Vector3<i64>, Real); 26] = [
            (Vector3::new(-1, 0, 0), 1.0),
            (Vector3::new(1, 0, 0), 1.0),
            (Vector3::new(0, -1, 0), 1.0),
            (Vector3::new(0, 1, 0), 1.0),
            (Vector3::new(0, 0, -1), 1.0),
            (Vector3::new(0, 0, 1), 1.0),
            (Vector3::new(-1, -1, 0), SQRT_2),
            (Vector3::new(-1, 1, 0), SQRT_2),
            (Vector3::new(1, -1, 0), SQRT_2),
            (Vector3::new(1, 1, 0), SQRT_2),
            (Vector3::new(-1, 0, -1), SQRT_2),
            (Vector3::new(-1, 0, 1), SQRT_2),
            (Vector3::new(1, 0, -1), SQRT_2),
            (Vector3::new(1, 0, 1), SQRT_2),
            (Vector3::new(0, -1, -1), SQRT_2),
            (Vector3::new(0, -1, 1), SQRT_2),
            (Vector3::new(0, 1, -1), SQRT_2),
            (Vector3::new(0, 1, 1), SQRT_2),
            (Vector3::new(-1, -1, -1), SQRT_3),
            (Vector3::new(-1, -1, 1), SQRT_3),
            (Vector3::new(-1, 1, -1), SQRT_3),
            (Vector3::new(-1, 1, 1), SQRT_3),
            (Vector3::new(1, -1, -1), SQRT_3),
            (Vector3::new(1, -1, 1), SQRT_3),
            (Vector3::new(1, 1, -1), SQRT_3),
            (Vector3::new(1, 1, 1), SQRT_3),
        ];

        let offset = NEIGHBOUR_OFFSETS.get(self.n);
        let next = offset.map(|(offset, dist)| Neighbour {
            index: GlobalIndex(self.pivot.0 + offset),
            dir: offset.to_owned(),
            grid_dist: *dist,
        });

        self.n += 1;
        next
    }
}

/// Voxel Index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelIndex<const VPS: usize>(pub Point3<usize>);

impl<const VPS: usize> Deref for VoxelIndex<VPS> {
    type Target = Point3<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const VPS: usize> VoxelIndex<VPS> {
    pub fn linear_index(&self) -> usize {
        self.x + VPS * (self.y + self.z * VPS)
    }
}

/// Block Index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockIndex<const VPS: usize>(pub Point3<i32>);

impl<const VPS: usize> Deref for BlockIndex<VPS> {
    type Target = Point3<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const VPS: usize> BlockIndex<VPS> {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self(Point3::new(x, y, z))
    }
}

impl<const VPS: usize> From<Point3<i64>> for BlockIndex<VPS> {
    fn from(value: Point3<i64>) -> Self {
        Self(value.cast())
    }
}

impl<const VPS: usize> BlockIndex<VPS> {
    fn hash(&self) -> i32 {
        (self.x * 18397) + (self.y * 20483) + (self.z * 29303)
    }
}

impl<const VPS: usize> Ord for BlockIndex<VPS> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash().cmp(&other.hash())
    }
}

impl<const VPS: usize> PartialOrd for BlockIndex<VPS> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_global_index() {
        let global_index: GlobalIndex<3> =
            GlobalIndex::from_point(&Point3::new(1.0, 2.0, 3.0), 1.0 / 0.5);
        assert_eq!(global_index, GlobalIndex(Point3::new(2, 4, 6)));

        let global_index: GlobalIndex<3> = GlobalIndex(Point3::new(5, 5, 5));
        let block_index = global_index.block_index();
        assert_eq!(block_index, BlockIndex(Point3::new(1, 1, 1)));

        let global_index: GlobalIndex<3> = GlobalIndex(Point3::new(-5, -5, -5));
        let block_index = global_index.block_index();
        assert_eq!(block_index, BlockIndex(Point3::new(-2, -2, -2)));

        // vps = 3
        // |global| -> |block, local voxel|
        // 0 -> b:0, v:0
        // -1 -> b:-1, v:2
        // -2 -> b:-1, v:1
        // -3 -> b:-1, v:0
        // -4 -> b:-2, v:2
        let global_index: GlobalIndex<3> = GlobalIndex(Point3::new(-1, -4, -5));
        let voxel_index = global_index.local_voxel_index();
        assert_eq!(voxel_index, VoxelIndex(Point3::new(2, 2, 1)));

        let global_index: GlobalIndex<32> = GlobalIndex(Point3::new(-4, 19, 0));
        let voxel_index = global_index.local_voxel_index();
        assert_eq!(voxel_index, VoxelIndex(Point3::new(28, 19, 0)));

        let global_index: GlobalIndex<4> = GlobalIndex(Point3::new(0, -4, 0));
        let voxel_index = global_index.local_voxel_index();
        assert_eq!(voxel_index, VoxelIndex(Point3::new(0, 0, 0)));
    }

    #[test]
    fn test_block_index_from_global_index() {
        let global_index: GlobalIndex<3> = GlobalIndex(Point3::new(-1, -2, -3));

        assert_eq!(
            global_index.block_index(),
            BlockIndex(Point3::new(-1, -1, -1))
        );

        let global_index: GlobalIndex<3> = GlobalIndex(Point3::new(1, 2, 3));

        assert_eq!(global_index.block_index(), BlockIndex(Point3::new(0, 0, 1)));
    }

    #[test]
    fn test_block_from_lin_index() {
        let block_index = BlockIndex(Point3::new(0, 0, 0));
        let global_index: GlobalIndex<3> =
            GlobalIndex::from_block_and_local_lin_index(&block_index, (3 * 3 * 3) - 1);

        assert_eq!(global_index, GlobalIndex(Point3::new(2, 2, 2)));

        let global_index: GlobalIndex<3> =
            GlobalIndex::from_block_and_local_lin_index(&block_index, (1 * 2 * 3) - 1);

        assert_eq!(global_index, GlobalIndex(Point3::new(2, 1, 0)));
    }
}