#pragma once
#include <cumm/common/TensorView.h>
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace p2v_c {

using TensorView = cumm::common::TensorView;

struct Point2VoxelCommon
{
    /**
     * @param vsize_xyz
     * @param coors_range_xyz
     */
    static std::tuple<std::array<float, 3>, std::array<int, 3>,
                      std::array<int, 3>, std::array<float, 6>>
    calc_meta_data(std::array<float, 3> vsize_xyz,
                   std::array<float, 6> coors_range_xyz);
    template <typename T, size_t N>
    static tv::array<T, N> array2tvarray(std::array<T, N> arr)
    {
        tv::array<T, N> tarr;
        for (int i = 0; i < N; ++i) {
            tarr[i] = arr[i];
        }
        return tarr;
    }
    template <typename T, size_t N>
    static std::array<T, N> tvarray2array(tv::array<T, N> arr)
    {
        std::array<T, N> tarr;
        for (int i = 0; i < N; ++i) {
            tarr[i] = arr[i];
        }
        return tarr;
    }
};
}  // namespace p2v_c
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc