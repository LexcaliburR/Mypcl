#pragma once
#include <tensorview/math/fastmath.h>
#include <cumm/common/TensorView.h>
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
namespace layout_ns {

using TensorView = cumm::common::TensorView;

struct TensorGeneric
{
    tv::array<int, 2> strides;
    tv::array<unsigned int, 2> multipliers;
    tv::array<unsigned int, 2> shift_rights;
    TV_HOST_DEVICE_INLINE TensorGeneric(tv::array<int, 2> const& strides)
        : strides(strides)
    {
        tv::math::find_divisor(multipliers[0], shift_rights[0], strides[0]);
        tv::math::find_divisor(multipliers[1], shift_rights[1], strides[1]);
    }
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
        int residual;
        tv::math::fast_divmod(out[0],
                              residual,
                              index,
                              strides[0],
                              multipliers[0],
                              shift_rights[0]);
        tv::math::fast_divmod(out[1],
                              out[2],
                              residual,
                              strides[1],
                              multipliers[1],
                              shift_rights[1]);
        return out;
    }
    TV_HOST_DEVICE_INLINE void inverse(int64_t index,
                                       tv::array<int, 3>& out) const
    {
        int residual;
        tv::math::fast_divmod(out[0],
                              residual,
                              index,
                              strides[0],
                              multipliers[0],
                              shift_rights[0]);
        tv::math::fast_divmod(out[1],
                              out[2],
                              residual,
                              strides[1],
                              multipliers[1],
                              shift_rights[1]);
    }
    TV_HOST_DEVICE_INLINE void inverse(int64_t index, int& idx_0, int& idx_1,
                                       int& idx_2) const
    {
        int residual;
        tv::math::fast_divmod(idx_0,
                              residual,
                              index,
                              strides[0],
                              multipliers[0],
                              shift_rights[0]);
        tv::math::fast_divmod(idx_1,
                              idx_2,
                              residual,
                              strides[1],
                              multipliers[1],
                              shift_rights[1]);
    }
    TV_HOST_DEVICE_INLINE void inverse(int64_t index, int* out) const
    {
        int residual;
        tv::math::fast_divmod(out[0],
                              residual,
                              index,
                              strides[0],
                              multipliers[0],
                              shift_rights[0]);
        tv::math::fast_divmod(out[1],
                              out[2],
                              residual,
                              strides[1],
                              multipliers[1],
                              shift_rights[1]);
    }
};
}  // namespace layout_ns
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc