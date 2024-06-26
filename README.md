# ESDF Generation Visualized
Euclidean signed distance fields (ESDF) contain the distance from any point to the nearest surface and are, e.g., used for path planning and collision checking in robotics.

This is a basic implementation of the ```nvblox``` ESDF generation algorithm outlined in [1].

## Preview
![](.media/out.gif)

The *sweep and propagate* algorithm runs on the CPU and operates on a per-block basis.
The *sweeps* in x, y, and z directions are followed by a *propagation* phase, where voxels are transferred across block boundaries. 
This iterative process continues until reaching convergence.

The *sites* are block indices referring to the origin, i.e., the block with the closest surface to the voxel.
If the surfaces change, the *sites* are used to identify the blocks that need to be cleared and recalculated.

The original algorithm is a bit smarter and executes those operations in parallel (primarily on the GPU).

## References
[1] Millane, Alexander, et al. "nvblox: GPU-Accelerated Incremental Signed Distance Field Mapping." arXiv preprint arXiv:2311.00626 (2023).
