#[derive(Debug)]
pub struct VoxelStorage<T, const H: usize, const W: usize, const D: usize>(pub [[[T; H]; W]; D]);

impl<T, const H: usize, const W: usize, const D: usize> VoxelStorage<T, H, W, D> {
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: this is OK because ArrayStorage is contiguous.
        unsafe { self.as_slice_unchecked() }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: this is OK because ArrayStorage is contiguous.
        unsafe { self.as_mut_slice_unchecked() }
    }

    #[inline]
    fn ptr(&self) -> *const T {
        self.0.as_ptr() as *const T
    }

    #[inline]
    fn ptr_mut(&mut self) -> *mut T {
        self.0.as_mut_ptr() as *mut T
    }

    #[inline]
    unsafe fn as_slice_unchecked(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr(), H * W * D)
    }

    #[inline]
    unsafe fn as_mut_slice_unchecked(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr_mut(), H * W * D)
    }
}

impl<T: Default + Copy, const H: usize, const W: usize, const D: usize> Default
    for VoxelStorage<T, H, W, D>
{
    #[inline]
    fn default() -> Self {
        Self([[[T::default(); H]; W]; D])
    }
}
