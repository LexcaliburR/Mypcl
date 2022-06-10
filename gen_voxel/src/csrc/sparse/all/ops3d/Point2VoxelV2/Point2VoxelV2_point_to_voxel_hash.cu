#include <csrc/sparse/all/ops3d/Point2VoxelV2.h>
#include <csrc/sparse/all/ops3d/kernel/Point2VoxelKernel.h>

namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {

using TensorView = cumm::common::TensorView;
using Point2VoxelCommon = csrc::sparse::all::ops3d::p2v_c::Point2VoxelCommon;
using Layout = csrc::sparse::all::ops3d::layout_ns::TensorGeneric;

std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>
Point2VoxelV2::point_to_voxel_hash(tv::Tensor points, bool clear_voxels,
                                   bool padding, std::uintptr_t stream_int)
{
    tv::Tensor points_voxel_id = tv::empty({points.dim(0)}, tv::int64, 0);
    int64_t expected_hash_data_num = points.dim(0) * 2;
    if (hashdata.dim(0) < expected_hash_data_num) {
        hashdata = tv::zeros({expected_hash_data_num}, tv::custom128, 0);
    }
    if (point_indice_data.dim(0) < points.dim(0)) {
        point_indice_data = tv::zeros({points.dim(0)}, tv::int64, 0);
    }
    return point_to_voxel_hash_static(
        points,
        voxels,
        voxels_out,
        indices,
        indices_out,
        num_per_voxel,
        hashdata,
        point_indice_data,
        points_voxel_id,
        Point2VoxelCommon::tvarray2array(vsize),
        Point2VoxelCommon::tvarray2array(grid_size),
        Point2VoxelCommon::tvarray2array(grid_stride),
        Point2VoxelCommon::tvarray2array(coors_range),
        clear_voxels,
        padding,
        stream_int);
}

}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc