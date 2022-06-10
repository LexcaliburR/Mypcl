#include <csrc/sparse/all/ops3d/Point2VoxelV2.h>

namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {

using TensorView = cumm::common::TensorView;
using Point2VoxelCommon = csrc::sparse::all::ops3d::p2v_c::Point2VoxelCommon;
using Layout = csrc::sparse::all::ops3d::layout_ns::TensorGeneric;

std::array<int, 3> Point2VoxelV2::get_grid_size()
{
    std::array<int, 3> res;
    for (int i = 0; i < 3; ++i) {
        res[i] = grid_size[i];
    }
    return res;
}

}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc