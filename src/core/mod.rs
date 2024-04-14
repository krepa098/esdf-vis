pub mod block;
pub mod color;
pub mod index;
pub mod layer;
pub mod storage;
pub mod utils;
pub mod voxel;

pub mod prelude {
    pub type Real = f32;

    pub use nalgebra::Point3;

    pub use nalgebra::Vector3;

    pub type Color = nalgebra::Vector4<f32>;
}
