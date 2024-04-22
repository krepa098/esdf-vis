// VPS=4  | the fastest, many blocks
// VPS=8  | still fast, fewer blocks
// VPS=16 | slowest, does not work with shared memory (requires 80k, available 64k?)
//
const VPS: u32 = 8u;
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

const Invalid: u32          = 0xFFFFFFFF;

const Stride: u32           = 7; // [self, x+, x-, y+, y-, z+, z-]

struct EsdfVoxel {
    site_block_index: vec3<i32>,
    distance: f32,
    flags: u32,
};

struct Block {
    voxels: array<EsdfVoxel, (VPS*VPS*VPS)>,
};

struct BlockInfo {
    flags: atomic<u32>,
    updated_voxels: atomic<u32>,
};

struct Settings {
    dir_block_index_offset: u32,
};

// bindings
@group(0) 
@binding(0) 
var<storage, read_write> block_voxels: array<Block>;

@group(0) 
@binding(1) 
var<storage, read_write> workgroup_block_indices: array<u32>;

@group(0) 
@binding(2) 
var<storage, read_write> block_info: array<BlockInfo>;

// push constants
var<push_constant> settings: Settings;

// helpers
fn voxel_index_to_lin(index: vec3<u32>) -> u32 {
    return index.x + VPS * (index.y + index.z * VPS);
}

fn is_fixed(flags: u32) -> bool {
    return (flags & Fixed) > 0;
}

fn is_observed(flags: u32) -> bool {
    return (flags & Observed) > 0;
}

fn update_voxel(block_index: u32, voxel_index: u32, parent_block_index: u32, parent_voxel_index: u32) -> bool {  
    let voxel = &(block_voxels[block_index].voxels[voxel_index]);
    let parent_voxel = &(block_voxels[parent_block_index].voxels[parent_voxel_index]);

    let is_parent_fixed = is_fixed((*parent_voxel).flags);
    let is_voxel_observed = is_observed((*voxel).flags);
    let is_voxel_fixed = is_fixed((*voxel).flags);

    if (is_parent_fixed && !is_voxel_observed) {
        if (!is_voxel_fixed) {
            (*voxel).distance = (*parent_voxel).distance + VoxelSize;
            (*voxel).flags |= Fixed | HasSiteIndex;
            (*voxel).site_block_index = (*parent_voxel).site_block_index;

            return true;

        } else if ((*voxel).distance > (*parent_voxel).distance + VoxelSize) {
            (*voxel).distance = (*parent_voxel).distance + VoxelSize;
            (*voxel).site_block_index = (*parent_voxel).site_block_index;

            return true;
        }
    }

    return false;
}

// main
@compute 
@workgroup_size(VPS,VPS,1) // multiple of 32 (NV) or 64 (AMD)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let parent_block_index = workgroup_block_indices[workgroup_id.x * Stride];
    let prop_p_block_index = workgroup_block_indices[workgroup_id.x * Stride + settings.dir_block_index_offset];
    let prop_m_block_index = workgroup_block_indices[workgroup_id.x * Stride + settings.dir_block_index_offset +1];

    // voxel indexing
    // x
    var index_p = vec3(0, local_id.x, local_id.y);
    var index_m = vec3(VPS-1, local_id.x, local_id.y);

    // y?
    if (settings.dir_block_index_offset == 3) {
        index_p = vec3(local_id.x, 0, local_id.y);
        index_m = vec3(local_id.x, VPS-1, local_id.y);
        
    // z?
    } else if (settings.dir_block_index_offset == 5) {
        index_p = vec3(local_id.x, local_id.y, 0);
        index_m = vec3(local_id.x, local_id.y, VPS-1);
    }

    var i_p = voxel_index_to_lin(index_p);
    var i_m = voxel_index_to_lin(index_m);

    if (prop_p_block_index != Invalid) {
        if (update_voxel(prop_p_block_index, i_p, parent_block_index, i_m)) {
            atomicOr(&block_info[prop_p_block_index].flags, Updated);
        }
    }

    
    if (prop_m_block_index != Invalid) {
        if (update_voxel(prop_m_block_index, i_m, parent_block_index, i_p)) {
            atomicOr(&block_info[prop_m_block_index].flags, Updated);
        }
    }
}

