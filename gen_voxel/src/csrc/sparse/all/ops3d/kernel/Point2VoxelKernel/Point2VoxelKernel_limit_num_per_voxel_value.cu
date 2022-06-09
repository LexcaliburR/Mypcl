#include <csrc/sparse/all/ops3d/kernel/Point2VoxelKernel.h>
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace kernel {

using TensorView = cumm::common::TensorView;
using TensorViewHashKernel = cumm::common::TensorViewHashKernel;
using Layout = csrc::sparse::all::ops3d::layout_ns::TensorGeneric;

__global__ void limit_num_per_voxel_value(int* num_per_voxel, int num_voxels,
                                          int num_points_per_voxel)
{
    for (int i : tv::KernelLoopX<int>(num_voxels)) {
        int count = min(num_points_per_voxel, num_per_voxel[i]);
        num_per_voxel[i] = count;
    }
}

}  // namespace kernel
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc