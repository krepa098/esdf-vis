use std::time::Duration;

use ab_glyph::{FontArc, PxScale};
use image::{buffer::ConvertBuffer, Delay, RgbImage};
use imageproc::drawing::draw_text_mut;
use nalgebra::point;

use crate::core::{
    color::rainbow_map,
    index::{BlockIndex, GlobalIndex, VoxelIndex},
    layer::Layer,
    voxel::{Esdf, Tsdf},
};

static COLOR_OF_INTEREST: [u8; 3] = [255, 0, 255];
static COLOR_GRID: [u8; 3] = [0, 0, 0];
static COLOR_TSDF: [u8; 3] = [150, 150, 150];

pub struct Renderer {
    frames: Vec<RgbImage>,
    font: FontArc,
}

impl Renderer {
    pub fn new() -> Self {
        let font =
            FontArc::try_from_slice(include_bytes!("../fonts/DejaVuSansCondensed.ttf")).unwrap();

        Self {
            frames: vec![],
            font,
        }
    }

    pub fn render_tsdf_layer<const VPS: usize>(
        &mut self,
        tsdf_layer: &Layer<Tsdf, VPS>,
        esdf_layer: &Layer<Esdf, VPS>,
        block_of_interest: &BlockIndex<VPS>,
        op: &str,
    ) {
        let mut blocks_in_x = 0;
        let mut blocks_in_y = 0;

        for block in tsdf_layer.allocated_blocks_iter() {
            blocks_in_x = blocks_in_x.max(block.coords.x);
            blocks_in_y = blocks_in_y.max(block.coords.y);
        }

        // make space for all blocks + block boundaries
        let bottom_padding = 12;
        let img_width = (blocks_in_x + 1) * VPS as i32 + blocks_in_x + 2;
        let img_height = (blocks_in_y + 1) * VPS as i32 + blocks_in_y + 2 + bottom_padding;

        let mut img = image::RgbImage::new(img_width as u32, img_height as u32);
        img.fill(255);

        // render block boundaries (block grid)
        for bx in 0..=blocks_in_x + 1 {
            for y in 0..img.height() - bottom_padding as u32 {
                img.get_pixel_mut((bx * VPS as i32 + bx) as u32, y).0 = COLOR_GRID;
            }
        }
        for by in 0..=blocks_in_y + 1 {
            for x in 0..img.width() {
                img.get_pixel_mut(x, (by * VPS as i32 + by) as u32).0 = COLOR_GRID;
            }
        }

        // render esdf voxels
        if let (Some(d_min), Some(d_max)) = esdf_layer.min_max_pred(|v| v.distance) {
            let d_range = d_max - d_min;

            for block_index in esdf_layer.allocated_blocks_iter() {
                if let Some(block) = esdf_layer.block_by_index(block_index) {
                    for vx in 0..VPS {
                        for vy in 0..VPS {
                            let voxel_index = VoxelIndex(point![vx, vy, 0]);
                            let lock = block.read();
                            let voxel = lock.voxel_from_index(&voxel_index);

                            if voxel.fixed {
                                let index = GlobalIndex::from_block_and_voxel_index(
                                    block_index,
                                    &voxel_index,
                                );

                                let color =
                                    rainbow_map(((voxel.distance - d_min) / d_range) as f32);

                                img.get_pixel_mut(
                                    (index.x + block_index.x as i64) as u32 + 1,
                                    (index.y + block_index.y as i64) as u32 + 1,
                                )
                                .0 = [
                                    (color.x * 255.0) as u8,
                                    (color.y * 255.0) as u8,
                                    (color.z * 255.0) as u8,
                                ];
                            }
                        }
                    }
                }
            }
        }

        // render tsdf voxels
        for block_index in tsdf_layer.allocated_blocks_iter() {
            if let Some(block) = tsdf_layer.block_by_index(block_index) {
                for vx in 0..VPS {
                    for vy in 0..VPS {
                        let voxel_index = VoxelIndex(point![vx, vy, 0]);
                        let blockr = block.read();
                        let voxel = blockr.voxel_from_index(&voxel_index);

                        if voxel.weight > 0.0 && voxel.distance <= 0.4 {
                            let index =
                                GlobalIndex::from_block_and_voxel_index(block_index, &voxel_index);

                            img.get_pixel_mut(
                                (index.x + block_index.x as i64 + 1) as u32,
                                (index.y + block_index.y as i64 + 1) as u32,
                            )
                            .0 = COLOR_TSDF;
                        }
                    }
                }
            }
        }

        // render frame around block of interest
        for vx in 0..=VPS + 1 {
            img.get_pixel_mut(
                (vx + block_of_interest.x as usize * VPS + block_of_interest.x as usize) as u32,
                (block_of_interest.y as usize * VPS + block_of_interest.y as usize) as u32,
            )
            .0 = COLOR_OF_INTEREST;
            img.get_pixel_mut(
                (vx + block_of_interest.x as usize * VPS + block_of_interest.x as usize) as u32,
                (block_of_interest.y as usize * VPS + block_of_interest.y as usize + VPS + 1)
                    as u32,
            )
            .0 = COLOR_OF_INTEREST;
        }
        for vy in 0..=VPS + 1 {
            img.get_pixel_mut(
                (block_of_interest.x as usize * VPS + block_of_interest.x as usize) as u32,
                (vy + block_of_interest.y as usize * VPS + block_of_interest.y as usize) as u32,
            )
            .0 = COLOR_OF_INTEREST;
            img.get_pixel_mut(
                ((block_of_interest.x as usize + 1) * VPS + block_of_interest.x as usize + 1)
                    as u32,
                (vy + block_of_interest.y as usize * VPS + block_of_interest.y as usize) as u32,
            )
            .0 = COLOR_OF_INTEREST;
        }

        // render op text
        let height = 8.0;
        let scale = PxScale {
            x: height * 2.0,
            y: height,
        };
        let y_pos = img.height() as i32 - bottom_padding;
        draw_text_mut(
            &mut img,
            image::Rgb([0, 0, 255]),
            0,
            y_pos,
            scale,
            &self.font,
            op,
        );

        self.frames.push(img);
    }

    pub fn render_gif(&mut self, path: &str) {
        use image::codecs::gif::GifEncoder;

        let file = std::fs::File::create(path).unwrap();
        let mut encoder = GifEncoder::new(file);
        encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .unwrap();
        encoder
            .encode_frames(self.frames.iter().map(|f| {
                image::Frame::from_parts(
                    f.convert(),
                    0,
                    0,
                    Delay::from_saturating_duration(Duration::from_millis(100)),
                )
            }))
            .unwrap();
    }
}
