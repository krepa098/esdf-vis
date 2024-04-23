use wgpu::{include_wgsl, util::DeviceExt, Device, PushConstantRange, Queue, RequestDeviceError};

use crate::core::{
    block::Block,
    voxel::{Esdf, EsdfFlags},
};

pub async fn create_adapter() -> Result<(Device, Queue), RequestDeviceError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("cannot get adapter");

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: (2048 << 20) - 1, // 2GB,
                    max_buffer_size: (2048 << 20) - 1,                 // 2GB
                    max_push_constant_size: 128,
                    ..Default::default()
                },
            },
            None,
        )
        .await
}

#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
pub struct TestData {
    pos: [i32; 4],
}

pub struct GpuSweep {
    shader_module: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    compute_pipeline: wgpu::ComputePipeline,
    voxel_storage_buffer: wgpu::Buffer,
    block_info_storage_buffer: wgpu::Buffer,
    timestamp_query_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    voxel_readback_buffer: wgpu::Buffer,
    block_info_readback_buffer: wgpu::Buffer,
    timestamp_readback_buffer: wgpu::Buffer,
    timestamp_query_set: wgpu::QuerySet,
}

impl GpuSweep {
    pub fn new(device: &Device, queue: &mut Queue) -> Self {
        let shader_module = device.create_shader_module(include_wgsl!("shaders/sweep.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        let voxel_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 256,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
                | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
        });

        let block_info_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 128,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
                | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
        });

        let timestamp_query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8 * 2,
            usage: wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
                | wgpu::BufferUsages::COPY_SRC // allow as source buffer for copy_buffer
                | wgpu::BufferUsages::QUERY_RESOLVE, // allow for query use
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_info_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let voxel_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 256, //voxel_storage_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_info_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 256, //block_info_storage_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: timestamp_query_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        Self {
            shader_module,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
            voxel_storage_buffer,
            block_info_storage_buffer,
            timestamp_query_buffer,
            bind_group,
            voxel_readback_buffer,
            block_info_readback_buffer,
            timestamp_readback_buffer,
            timestamp_query_set,
        }
    }

    pub fn submit<const VPS: usize>(
        &mut self,
        device: &Device,
        queue: &mut Queue,
        blocks: &[&Block<Esdf, VPS>],
    ) {
        firestorm::profile_method!("submit");
        firestorm::profile_section!(prepare);

        // prepare data
        let mut voxels = Vec::with_capacity(blocks.len() * VPS * VPS * VPS);
        for block in blocks {
            let lock = block.read();
            for voxel in lock.as_slice() {
                voxels.push(*voxel);
            }
        }
        let voxel_data: &[u8] = bytemuck::cast_slice(&voxels);

        let block_info = vec![BlockInfo::default(); blocks.len()];
        let block_info_data: &[u8] = bytemuck::cast_slice(&block_info);

        queue.write_buffer(&self.voxel_storage_buffer, 0, voxel_data);
        queue.write_buffer(&self.block_info_storage_buffer, 0, block_info_data);

        // record command
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // initial time
        encoder.write_timestamp(&self.timestamp_query_set, 0);

        {
            let mut comp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            comp_pass.set_bind_group(0, &self.bind_group, &[]);
            comp_pass.set_pipeline(&self.compute_pipeline);
            comp_pass.dispatch_workgroups(blocks.len() as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.voxel_storage_buffer,
            0,
            &self.voxel_readback_buffer,
            0,
            voxel_data.len() as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.block_info_storage_buffer,
            0,
            &self.block_info_readback_buffer,
            0,
            block_info_data.len() as u64,
        );

        // end time
        encoder.write_timestamp(&self.timestamp_query_set, 1);
        encoder.resolve_query_set(
            &self.timestamp_query_set,
            0..2,
            &self.timestamp_query_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.timestamp_query_buffer,
            0,
            &self.timestamp_readback_buffer,
            0,
            self.timestamp_readback_buffer.size(),
        );

        drop(prepare);

        // submit
        firestorm::profile_section!(submit_poll);
        queue.submit(Some(encoder.finish()));

        // readback
        self.timestamp_readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.block_info_readback_buffer
            .slice(..block_info_data.len() as u64)
            .map_async(wgpu::MapMode::Read, |_| {});

        self.voxel_readback_buffer
            .slice(..voxel_data.len() as u64)
            .map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::Maintain::Wait);

        drop(submit_poll);

        {
            firestorm::profile_section!(readback);

            let bytes: &[u8] = &self
                .voxel_readback_buffer
                .slice(..voxel_data.len() as u64)
                .get_mapped_range();
            let data: &[Esdf] = bytemuck::cast_slice(bytes);

            let voxel_blocks: Vec<_> = data.chunks(VPS * VPS * VPS).collect();

            let bytes: &[u8] = &self.timestamp_readback_buffer.slice(..).get_mapped_range();
            let counts: &[u64; 2] = bytemuck::from_bytes(bytes);

            println!(
                "GPU sweep:\t{} blocks\t{:?}",
                blocks.len(),
                TimestampGpu::new(counts, queue).duration()
            );

            let bytes: &[u8] = &self
                .block_info_readback_buffer
                .slice(..block_info_data.len() as u64)
                .get_mapped_range();
            let block_info: &[BlockInfo] = bytemuck::cast_slice(bytes);

            // writeback
            for (block, voxel_data) in blocks.iter().zip(voxel_blocks) {
                block.write().as_mut_slice().copy_from_slice(voxel_data);
            }
        }

        self.voxel_readback_buffer.unmap();
        self.block_info_readback_buffer.unmap();
        self.timestamp_readback_buffer.unmap();
    }
}

pub struct GpuPropagate {
    shader_module: wgpu::ShaderModule,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    compute_pipeline: wgpu::ComputePipeline,
    voxel_storage_buffer: wgpu::Buffer,
    block_info_storage_buffer: wgpu::Buffer,
    timestamp_query_buffer: wgpu::Buffer,
    workgroup_block_indices_storage_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    voxel_readback_buffer: wgpu::Buffer,
    block_info_readback_buffer: wgpu::Buffer,
    timestamp_readback_buffer: wgpu::Buffer,
    timestamp_query_set: wgpu::QuerySet,
}

impl GpuPropagate {
    pub fn new(device: &Device, queue: &mut Queue) -> Self {
        let shader_module = device.create_shader_module(include_wgsl!("shaders/propagate.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        let voxel_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 256,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
                | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
        });

        let block_info_storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 128,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
                | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
        });

        let workgroup_block_indices_storage_buffer =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 1024 * 1024 * 128,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
            | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
            });

        let timestamp_query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8 * 2,
            usage: wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
                | wgpu::BufferUsages::COPY_SRC // allow as source buffer for copy_buffer
                | wgpu::BufferUsages::QUERY_RESOLVE, // allow for query use
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: workgroup_block_indices_storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_info_storage_buffer.as_entire_binding(),
                },
            ],
        });

        let voxel_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let block_info_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024 * 1024 * 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: timestamp_query_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });

        Self {
            shader_module,
            bind_group_layout,
            pipeline_layout,
            compute_pipeline,
            voxel_storage_buffer,
            workgroup_block_indices_storage_buffer,
            block_info_storage_buffer,
            timestamp_query_buffer,
            bind_group,
            voxel_readback_buffer,
            block_info_readback_buffer,
            timestamp_readback_buffer,
            timestamp_query_set,
        }
    }

    pub fn submit<const VPS: usize>(
        &mut self,
        device: &Device,
        queue: &mut Queue,
        workgroup_block_indices: &[u32],
        blocks: &[&Block<Esdf, VPS>],
    ) -> Vec<BlockInfo> {
        firestorm::profile_method!("submit");
        firestorm::profile_section!(prepare);

        // prepare data
        let mut voxels = Vec::with_capacity(blocks.len() * VPS * VPS * VPS);

        {
            firestorm::profile_section!(prep_voxels);
            for block in blocks {
                let lock = block.read();
                for voxel in lock.as_slice() {
                    voxels.push(*voxel);
                }
            }
        }

        let voxel_data: &[u8] = bytemuck::cast_slice(&voxels);

        let block_info = vec![BlockInfo::default(); blocks.len()];
        let block_info_data: &[u8] = bytemuck::cast_slice(&block_info);

        let workgroup_block_indices_data: &[u8] = bytemuck::cast_slice(workgroup_block_indices);

        queue.write_buffer(&self.voxel_storage_buffer, 0, voxel_data);
        queue.write_buffer(&self.block_info_storage_buffer, 0, block_info_data);
        queue.write_buffer(
            &self.workgroup_block_indices_storage_buffer,
            0,
            workgroup_block_indices_data,
        );

        // record command
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // initial time
        encoder.write_timestamp(&self.timestamp_query_set, 0);

        {
            let mut comp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            comp_pass.set_bind_group(0, &self.bind_group, &[]);
            comp_pass.set_pipeline(&self.compute_pipeline);
            let settings = PropagateSettings {
                dir: PropagateSettings::X,
            };
            comp_pass.set_push_constants(0, bytemuck::cast_slice(&[settings]));
            comp_pass.dispatch_workgroups(workgroup_block_indices.len() as u32 / 7, 1, 1);

            let settings = PropagateSettings {
                dir: PropagateSettings::Y,
            };
            comp_pass.set_push_constants(0, bytemuck::cast_slice(&[settings]));
            comp_pass.dispatch_workgroups(workgroup_block_indices.len() as u32 / 7, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.voxel_storage_buffer,
            0,
            &self.voxel_readback_buffer,
            0,
            voxel_data.len() as u64,
        );

        encoder.copy_buffer_to_buffer(
            &self.block_info_storage_buffer,
            0,
            &self.block_info_readback_buffer,
            0,
            block_info_data.len() as u64,
        );

        // end time
        encoder.write_timestamp(&self.timestamp_query_set, 1);
        encoder.resolve_query_set(
            &self.timestamp_query_set,
            0..2,
            &self.timestamp_query_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.timestamp_query_buffer,
            0,
            &self.timestamp_readback_buffer,
            0,
            self.timestamp_readback_buffer.size(),
        );

        drop(prepare);

        // submit
        firestorm::profile_section!(submit_poll);
        queue.submit(Some(encoder.finish()));

        // readback
        self.timestamp_readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});

        self.block_info_readback_buffer
            .slice(..block_info_data.len() as u64)
            .map_async(wgpu::MapMode::Read, |_| {});

        self.voxel_readback_buffer
            .slice(..voxel_data.len() as u64)
            .map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::Maintain::Wait);
        drop(submit_poll);

        firestorm::profile_section!(readback);

        let block_info = {
            let bytes: &[u8] = &self
                .voxel_readback_buffer
                .slice(..voxel_data.len() as u64)
                .get_mapped_range();
            let data: &[Esdf] = bytemuck::cast_slice(bytes);

            let voxel_blocks: Vec<_> = data.chunks(VPS * VPS * VPS).collect();

            let bytes: &[u8] = &self.timestamp_readback_buffer.slice(..).get_mapped_range();
            let counts: &[u64; 2] = bytemuck::from_bytes(bytes);

            println!(
                "GPU propgate:\t{} blocks\t{:?}",
                blocks.len(),
                TimestampGpu::new(counts, queue).duration()
            );

            let bytes: &[u8] = &self
                .block_info_readback_buffer
                .slice(..block_info_data.len() as u64)
                .get_mapped_range();
            let block_info: &[BlockInfo] = bytemuck::cast_slice(bytes);

            // writeback
            for (block, voxel_data) in blocks.iter().zip(voxel_blocks) {
                block.write().as_mut_slice().copy_from_slice(voxel_data);
            }

            block_info.to_owned()
        };

        self.voxel_readback_buffer.unmap();
        self.block_info_readback_buffer.unmap();
        self.timestamp_readback_buffer.unmap();

        drop(readback);

        block_info
    }
}

#[derive(Debug, Clone, Copy)]
struct TimestampGpu {
    ns: [f32; 2],
    delta_ns: f32,
}

impl TimestampGpu {
    pub fn new(counts: &[u64; 2], queue: &Queue) -> Self {
        let period = queue.get_timestamp_period();

        Self {
            ns: [counts[0] as f32 * period, counts[1] as f32 * period],
            delta_ns: (counts[1] - counts[0]) as f32 * period,
        }
    }

    pub fn duration(&self) -> std::time::Duration {
        std::time::Duration::from_nanos(self.delta_ns as u64)
    }
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct BlockInfo {
    pub flags: EsdfFlags,
    pub updated_voxels: u32,
}

#[derive(Debug, Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
struct PropagateSettings {
    dir: u32,
}

impl PropagateSettings {
    const X: u32 = 0 + 1;
    const Y: u32 = 2 + 1;
    const Z: u32 = 4 + 1;
}
