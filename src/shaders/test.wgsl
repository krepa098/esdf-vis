

struct TestData {
    pos: vec4<i32>,
};

struct EsdfGpuVoxel {
    distance: f32,
    flags: u32,
    site_parent_index: vec3<i32>,
};

struct TestDataArray {
    data: array<TestData>,
};

@group(0) 
@binding(0) 
var<storage, read_write> buffer: TestDataArray;

fn index_to_hash(index: vec3<i32>) -> i32 {
    return  (index.x * 18397) + (index.y * 20483) + (index.z * 29303);
}

@compute 
@workgroup_size(8,1,1) // multiple of 32
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    buffer.data[local_index].pos = vec4(i32(local_index));
}

