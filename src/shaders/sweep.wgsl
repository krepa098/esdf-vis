const VPS: u32 = 16u;
const VoxelSize: f32 = 1.0;

// flags
const Observed: u32         = 1u << 0;
const Fixed: u32            = 1u << 1;
const HasSiteIndex: u32     = 1u << 2;
const Updated: u32          = 1u << 3;
const SpilledXPLus : u32    = 1u << 4;
const SpilledXMinus: u32    = 1u << 5;
const SpilledYPlus: u32     = 1u << 6;
const SpilledYMinus: u32    = 1u << 7;
const SpilledZPlus: u32     = 1u << 8;
const SpilledZMinus: u32    = 1u << 9;

struct EsdfVoxel {
    distance: f32,
    flags: u32,
    site_block_index: vec3<i32>,
};

struct Block {
    voxels: array<EsdfVoxel, (VPS*VPS*VPS)>,
};

struct BlockInfo {
    flags: atomic<u32>,
    updated_voxels: atomic<u32>,
    // _pad: array<u32, 2>,
};

// vars
@group(0) 
@binding(0) 
var<storage, read_write> block_voxels: array<Block>;

@group(0) 
@binding(1) 
var<storage, read_write> block_info: array<BlockInfo>;

var<workgroup> voxel_wg_data: array<EsdfVoxel, (VPS*VPS*VPS)>;

// helpers
fn index_to_lin(index: vec3<u32>) -> u32 {
    return index.x + VPS * (index.y + index.z * VPS);
}

fn is_fixed(voxel: ptr<function, EsdfVoxel>) -> bool {
    return ((*voxel).flags & Fixed) > 0;
}

fn is_observed(voxel: ptr<function, EsdfVoxel>) -> bool {
    return ((*voxel).flags & Observed) > 0;
}

fn update_voxel(voxel: ptr<function, EsdfVoxel>, parent_voxel: ptr<function, EsdfVoxel>) -> bool {  
    if (is_fixed(parent_voxel) && !is_observed(voxel)) {
        if (!is_fixed(voxel)) {
            (*voxel).distance = (*parent_voxel).distance + VoxelSize;
            (*voxel).flags |= Fixed;
            (*voxel).site_block_index = (*parent_voxel).site_block_index;

            return true;

        } else if ((*voxel).distance > (*parent_voxel).distance + VoxelSize) {
            (*voxel).distance = min((*voxel).distance, (*parent_voxel).distance + VoxelSize);
            (*voxel).site_block_index = (*parent_voxel).site_block_index;

            return true;
        }
    }

    return false;
}

// main
@compute 
@workgroup_size(VPS,VPS,1) // multiple of 32
fn main(
    // @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    // @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let block_id = workgroup_id.x;

    // fetch (global to shared memory)
    for (var w: u32 = 0; w < VPS; w++) {
        let i = index_to_lin(vec3(local_id.x,local_id.y,w));

        voxel_wg_data[i] = block_voxels[block_id].voxels[i];
    }

    workgroupBarrier();

    // x+
    for (var w: u32 = 1; w < VPS; w++) {
        let i = index_to_lin(vec3(w, local_id.x,local_id.y));
        let i_p = index_to_lin(vec3(w-1, local_id.x,local_id.y));

        var voxel = voxel_wg_data[i];
        var parent_voxel = voxel_wg_data[i_p];

        if (update_voxel(&voxel, &parent_voxel)) {
            atomicAdd(&block_info[block_id].updated_voxels, 1u);
            atomicOr(&block_info[block_id].flags, SpilledXPLus);
        }

        voxel_wg_data[i] = voxel;
    }

    workgroupBarrier();

    // x-
    for (var w: u32 = VPS - 1; w > 0; w--) {
        let i = index_to_lin(vec3(w-1, local_id.x,local_id.y));
        let i_p = index_to_lin(vec3(w, local_id.x,local_id.y));

        var voxel = voxel_wg_data[i];
        var parent_voxel = voxel_wg_data[i_p];

        if (update_voxel(&voxel, &parent_voxel)) {
            atomicAdd(&block_info[block_id].updated_voxels, 1u);
            atomicOr(&block_info[block_id].flags, SpilledXMinus);
        }

        voxel_wg_data[i] = voxel;
    }

    workgroupBarrier();

    // y+
    for (var w: u32 = 1; w < VPS; w++) {
        let i = index_to_lin(vec3(local_id.x,w,local_id.y));
        let i_p = index_to_lin(vec3(local_id.x,w-1,local_id.y));

        var voxel = voxel_wg_data[i];
        var parent_voxel = voxel_wg_data[i_p];

        if (update_voxel(&voxel, &parent_voxel)) {
            atomicAdd(&block_info[block_id].updated_voxels, 1u);
            atomicOr(&block_info[block_id].flags, SpilledYPlus);
        }

        voxel_wg_data[i] = voxel;
    }

    workgroupBarrier();

    // y- and writeback
    for (var w: u32 = VPS - 1; w > 0; w--) {
        let i = index_to_lin(vec3(local_id.x,w-1,local_id.y));
        let i_p = index_to_lin(vec3(local_id.x,w,local_id.y));

        var voxel = voxel_wg_data[i];
        var parent_voxel = voxel_wg_data[i_p];

        if (update_voxel(&voxel, &parent_voxel)) {
            atomicAdd(&block_info[block_id].updated_voxels, 1u);
            atomicOr(&block_info[block_id].flags, SpilledYMinus);
        }

        voxel_wg_data[i] = voxel;
    }

    workgroupBarrier();
    
    // writeback (shared to global memory)
    for (var w: u32 = 0; w < VPS; w++) {
        let i = index_to_lin(vec3(local_id.x,local_id.y,w));

        block_voxels[block_id].voxels[i] = voxel_wg_data[i];
    }
}

