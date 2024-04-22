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

pub async fn shader(device: &Device, queue: &mut Queue) {
    let shader_module = device.create_shader_module(include_wgsl!("shaders/test.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: wgpu::BindingType::Buffer {
                has_dynamic_offset: false,
                min_binding_size: None,
                ty: wgpu::BufferBindingType::Storage { read_only: false },
            },
        }],
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

    let data = vec![TestData::default(); 8];

    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::try_cast_slice(&data).unwrap(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
            | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: storage_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut comp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        comp_pass.set_bind_group(0, &bind_group, &[]);
        comp_pass.set_pipeline(&compute_pipeline);
        comp_pass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &readback_buffer,
        0,
        readback_buffer.size(),
    );

    queue.submit(Some(encoder.finish()));

    let buffer_slice = readback_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let m: &[u8] = &readback_buffer.slice(..).get_mapped_range();

    let data: &[TestData] = bytemuck::cast_slice(m);
    dbg!(data);
}

pub async fn sweep_blocks<const VPS: usize>(
    device: &Device,
    queue: &mut Queue,
    blocks: &[&Block<Esdf, VPS>],
) {
    if blocks.is_empty() {
        println!("sweep_blocks: blocks is empty");
        return;
    }

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

    let voxel_data: Vec<u8> = blocks
        .iter()
        .flat_map(|p| bytemuck::cast_slice(p.read().as_slice()).to_owned())
        .collect();

    let voxel_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &voxel_data,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
            | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
    });

    let block_info_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vec![BlockInfo::default(); blocks.len()]),
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
        size: voxel_storage_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let block_info_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: block_info_storage_buffer.size(),
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

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // initial time
    encoder.write_timestamp(&timestamp_query_set, 0);

    {
        let mut comp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        comp_pass.set_bind_group(0, &bind_group, &[]);
        comp_pass.set_pipeline(&compute_pipeline);
        comp_pass.dispatch_workgroups(blocks.len() as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &voxel_storage_buffer,
        0,
        &voxel_readback_buffer,
        0,
        voxel_readback_buffer.size(),
    );

    encoder.copy_buffer_to_buffer(
        &block_info_storage_buffer,
        0,
        &block_info_readback_buffer,
        0,
        block_info_readback_buffer.size(),
    );

    // end time
    encoder.write_timestamp(&timestamp_query_set, 1);
    encoder.resolve_query_set(&timestamp_query_set, 0..2, &timestamp_query_buffer, 0);
    encoder.copy_buffer_to_buffer(
        &timestamp_query_buffer,
        0,
        &timestamp_readback_buffer,
        0,
        timestamp_readback_buffer.size(),
    );

    // submit
    queue.submit(Some(encoder.finish()));

    // readback
    timestamp_readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});
    block_info_readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});

    voxel_readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::Maintain::Wait);

    let bytes: &[u8] = &voxel_readback_buffer.slice(..).get_mapped_range();
    let data: &[Esdf] = bytemuck::cast_slice(bytes);

    let voxel_blocks: Vec<_> = data.chunks(VPS * VPS * VPS).collect();

    println!("GPU sweep: Updated {} blocks", blocks.len());

    let bytes: &[u8] = &timestamp_readback_buffer.slice(..).get_mapped_range();
    let counts: &[u64; 2] = bytemuck::from_bytes(bytes);
    dbg!(TimestampGpu::new(counts, queue).duration());

    let bytes: &[u8] = &block_info_readback_buffer.slice(..).get_mapped_range();
    let block_info: &[BlockInfo] = bytemuck::cast_slice(bytes);
    // dbg!(block_info);

    // writeback
    for (block, voxel_data) in blocks.iter().zip(voxel_blocks) {
        block.write().as_mut_slice().copy_from_slice(voxel_data);
    }
}

pub async fn propagate_blocks<const VPS: usize>(
    device: &Device,
    queue: &mut Queue,
    workgroup_block_indices: &[u32],
    blocks: &[&Block<Esdf, VPS>],
) -> Vec<BlockInfo> {
    if blocks.is_empty() {
        println!("sweep_blocks: blocks is empty");
        return vec![];
    }

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
            range: 0..8,
        }],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });

    let voxel_data: Vec<u8> = blocks
        .iter()
        .flat_map(|p| bytemuck::cast_slice(p.read().as_slice()).to_owned())
        .collect();

    let workgroup_block_indices_storage_buffer =
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(workgroup_block_indices),
            usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
            | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
        });

    let voxel_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &voxel_data,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST // allow as destination buffer for copy_buffer
            | wgpu::BufferUsages::COPY_SRC, // allow as source buffer for copy_buffer
    });

    let block_info_storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vec![BlockInfo::default(); blocks.len()]),
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
        size: voxel_storage_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let block_info_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: block_info_storage_buffer.size(),
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

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // initial time
    encoder.write_timestamp(&timestamp_query_set, 0);

    {
        let mut comp_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        comp_pass.set_bind_group(0, &bind_group, &[]);
        comp_pass.set_pipeline(&compute_pipeline);
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
        &voxel_storage_buffer,
        0,
        &voxel_readback_buffer,
        0,
        voxel_readback_buffer.size(),
    );

    encoder.copy_buffer_to_buffer(
        &block_info_storage_buffer,
        0,
        &block_info_readback_buffer,
        0,
        block_info_readback_buffer.size(),
    );

    // end time
    encoder.write_timestamp(&timestamp_query_set, 1);
    encoder.resolve_query_set(&timestamp_query_set, 0..2, &timestamp_query_buffer, 0);
    encoder.copy_buffer_to_buffer(
        &timestamp_query_buffer,
        0,
        &timestamp_readback_buffer,
        0,
        timestamp_readback_buffer.size(),
    );

    // submit
    queue.submit(Some(encoder.finish()));

    // readback / map buffers
    timestamp_readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});

    voxel_readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {
            dbg!("mapped");
        });

    block_info_readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::Maintain::Wait);

    let bytes: &[u8] = &voxel_readback_buffer.slice(..).get_mapped_range();
    let voxel_data: &[Esdf] = bytemuck::cast_slice(bytes);

    println!("GPU propgate: Updated {} blocks", blocks.len());

    let bytes: &[u8] = &timestamp_readback_buffer.slice(..).get_mapped_range();
    let counts: &[u64; 2] = bytemuck::from_bytes(bytes);
    dbg!(TimestampGpu::new(counts, queue).duration());

    let bytes: &[u8] = &block_info_readback_buffer.slice(..).get_mapped_range();
    let block_info: &[BlockInfo] = bytemuck::cast_slice(bytes);

    // writeback
    let voxel_blocks: Vec<_> = voxel_data.chunks(VPS * VPS * VPS).collect();
    for (block, voxel_data) in blocks.iter().zip(voxel_blocks) {
        block.write().as_mut_slice().copy_from_slice(voxel_data);
    }

    block_info.to_owned()
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
