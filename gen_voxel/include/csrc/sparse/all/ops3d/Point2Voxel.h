#pragma once

#include <cumm/common/TensorView.h>
#include <csrc/sparse/all/ops3d/p2v_c/Point2VoxelCommon.h>
#include <csrc/sparse/all/ops3d/layout_ns/TensorGeneric.h>

namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {

using TensorView = cumm::common::TensorView;
using Point2VoxelCommon = csrc::sparse::all::ops3d::p2v_c::Point2VoxelCommon;
using Layout = csrc::sparse::all::ops3d::layout_ns::TensorGeneric;

struct Point2Voxel
{
    tv::Tensor hashdata;
    tv::Tensor point_indice_data;
    tv::Tensor voxels;
    tv::Tensor indices;
    tv::Tensor num_per_voxel;
    tv::array<float, 3> vsize;
    tv::array<float, 6> coors_range;
    tv::array<int, 3> grid_size;
    tv::array<int, 3> grid_stride;

    std::array<int, 3> get_grid_size();
    /**
     * @param vsize_xyz
     * @param coors_range_xyz
     * @param num_point_features
     * @param max_num_voxels
     * @param max_num_points_per_voxel
     */
    Point2Voxel(std::array<float, 3> vsize_xyz,
                std::array<float, 6> coors_range_xyz, int num_point_features,
                int max_num_voxels, int max_num_points_per_voxel);
    /**
     * @param points
     * @param clear_voxels
     * @param empty_mean
     * @param stream_int
     */
    std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point_to_voxel_hash(
        tv::Tensor points, bool clear_voxels = true, bool empty_mean = false,
        std::uintptr_t stream_int = 0);
    /**
     * @param points
     * @param voxels
     * @param indices
     * @param num_per_voxel
     * @param hashdata
     * @param point_indice_data
     * @param points_voxel_id
     * @param vsize
     * @param grid_size
     * @param grid_stride
     * @param coors_range
     * @param clear_voxels whether to clear voxel buffer
     * @param empty_mean true: fill the empty point feature with the mean val of
     * valid points in this voxel, false: fill the empty point featuer with zero
     * @param stream_int
     */
    static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor>
    point_to_voxel_hash_static(
        tv::Tensor points, tv::Tensor voxels, tv::Tensor indices,
        tv::Tensor num_per_voxel, tv::Tensor hashdata,
        tv::Tensor point_indice_data, tv::Tensor points_voxel_id,
        std::array<float, 3> vsize, std::array<int, 3> grid_size,
        std::array<int, 3> grid_stride, std::array<float, 6> coors_range,
        bool clear_voxels = true, bool empty_mean = false,
        std::uintptr_t stream_int = 0);
};
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc