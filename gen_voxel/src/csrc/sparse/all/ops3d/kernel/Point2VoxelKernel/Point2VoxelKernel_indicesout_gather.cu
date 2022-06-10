#include <csrc/sparse/all/ops3d/kernel/Point2VoxelKernel.h>
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace kernel {

using TensorView = cumm::common::TensorView;
using TensorViewHashKernel = cumm::common::TensorViewHashKernel;
using Layout = csrc::sparse::all::ops3d::layout_ns::TensorGeneric;

__global__ void indicesout_gather(int* indices, int* indices_out,
                                  int num_voxels, int num_coords,
                                  int num_coords_out)
{
    for (int i : tv::KernelLoopX<int>(num_voxels)) {
        for (int j = 0; j < num_coords; ++j) {
            indices_out[1 + i * num_coords_out + j] =
                indices[i * num_coords + j];
        }
    }
}

}  // namespace kernel
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc