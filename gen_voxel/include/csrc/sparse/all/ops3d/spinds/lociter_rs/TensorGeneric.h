#pragma once
#include <cumm/common/TensorView.h>
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace spinds {
namespace lociter_rs {

using TensorView = cumm::common::TensorView;

struct TensorGeneric
{
    tv::array<int, 2> strides;
    TV_HOST_DEVICE_INLINE TensorGeneric(tv::array<int, 2> const& strides)
        : strides(strides)
    {}
    TV_HOST_DEVICE_INLINE static TensorGeneric from_shape(
        const tv::array<int, 3>& shape)
    {
        return TensorGeneric({shape[1] * shape[2], shape[2]});
    }
    TV_HOST_DEVICE_INLINE int64_t
    operator()(const tv::array<int, 3>& indexes) const
    {
        return indexes[2] + int64_t(strides[0] * indexes[0]) +
               int64_t(strides[1] * indexes[1]);
    }
    TV_HOST_DEVICE_INLINE int64_t operator()(const int* indexes) const
    {
        return indexes[2] + int64_t(strides[0] * indexes[0]) +
               int64_t(strides[1] * indexes[1]);
    }
    TV_HOST_DEVICE_INLINE tv::array<int, 3> inverse(int64_t index) const
    {
        tv::array<int, 3> out;
        int64_t residual = index;
        out[0] = int(residual / strides[0]);
        residual = residual % strides[0];
        out[1] = int(residual / strides[1]);
        out[2] = int(residual % strides[1]);
        return out;
    }
    TV_HOST_DEVICE_INLINE void inverse(int64_t index,
                                       tv::array<int, 3>& out) const
    {
        int64_t residual = index;
        out[0] = int(residual / strides[0]);
        residual = residual % strides[0];
        out[1] = int(residual / strides[1]);
        out[2] = int(residual % strides[1]);
    }
    TV_HOST_DEVICE_INLINE void inverse(int64_t index, int& idx_0, int& idx_1,
                                       int& idx_2) const
    {
        int64_t residual = index;
        idx_0 = int(residual / strides[0]);
        residual = residual % strides[0];
        idx_1 = int(residual / strides[1]);
        idx_2 = int(residual % strides[1]);
    }
    TV_HOST_DEVICE_INLINE void inverse(int64_t index, int* out) const
    {
        int64_t residual = index;
        out[0] = int(residual / strides[0]);
        residual = residual % strides[0];
        out[1] = int(residual / strides[1]);
        out[2] = int(residual % strides[1]);
    }
};
}  // namespace lociter_rs
}  // namespace spinds
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc