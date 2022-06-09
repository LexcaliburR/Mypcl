#pragma once

#include <cumm/conv/bases/ConvEnum.h>
#include <cumm/conv/params/ConvProblemCommon.h>
#include <cumm/common/TensorView.h>

namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu3d {
namespace spinds {
using ConvEnum = cumm::conv::bases::ConvEnum;
using ConvProblemCommon = cumm::conv::params::ConvProblemCommon;
using TensorView = cumm::common::TensorView;
struct ConvProblem
{
    int N;
    int C;
    int K;
    tv::array<int, 3> input_dims;
    tv::array<int, 3> output_dims;
    tv::array<int, 3> ksize;
    tv::array<int, 3> padding;
    tv::array<int, 3> stride;
    tv::array<int, 3> dilation;
    ConvEnum::Mode mode;
    int split_k_slices;
    int groups;
    TV_HOST_DEVICE_INLINE ConvProblem(
        int N, int C, int K, tv::array<int, 3> input_dims,
        tv::array<int, 3> output_dims, tv::array<int, 3> ksize,
        tv::array<int, 3> padding, tv::array<int, 3> stride,
        tv::array<int, 3> dilation,
        ConvEnum::Mode mode = ConvEnum::Mode::kCrossCorrelation,
        int split_k_slices = 1, int groups = 1)
        : N(N),
          C(C),
          K(K),
          input_dims(input_dims),
          output_dims(output_dims),
          ksize(ksize),
          padding(padding),
          stride(stride),
          dilation(dilation),
          mode(mode),
          split_k_slices(split_k_slices),
          groups(groups)
    {}
    TV_HOST_DEVICE_INLINE static tv::array<int, 3> calc_output_dims(
        tv::array<int, 3> input_dims, tv::array<int, 3> ksize,
        tv::array<int, 3> padding, tv::array<int, 3> stride,
        tv::array<int, 3> dilation)
    {
        tv::array<int, 3> out;
        for (int i = 0; i < 3; ++i) {
            out[i] =
                ((input_dims[i] + padding[i] * 2 - ksize[i] * dilation[i]) /
                 stride[i]) +
                1;
        }
        return out;
    }
    TV_HOST_DEVICE_INLINE tv::array<int, 3> implicit_gemm_mnk(
        ConvEnum::OpType op_type)
    {
        int ksize_prod = tv::arrayops::prod(ksize);
        int in_prod = tv::arrayops::prod(input_dims);
        int out_prod = tv::arrayops::prod(output_dims);
        return ConvProblemCommon::implicit_gemm_mnk(
            op_type, N, C, K, ksize_prod, in_prod, out_prod, false);
    }
    TV_HOST_DEVICE_INLINE int implicit_gemm_k_iterations(
        ConvEnum::OpType op_type, int tile_shape_k)
    {
        int ksize_prod = tv::arrayops::prod(ksize);
        int in_prod = tv::arrayops::prod(input_dims);
        int out_prod = tv::arrayops::prod(output_dims);
        switch (op_type) {
            case ConvEnum::OpType::kForward:
                return ksize_prod *
                       tv::div_up(tv::div_up(C, split_k_slices), tile_shape_k);
            case ConvEnum::OpType::kBackwardInput:
                return ksize_prod *
                       tv::div_up(tv::div_up(K, split_k_slices), tile_shape_k);
            case ConvEnum::OpType::kBackwardWeight:
                return tv::div_up(tv::div_up(N * out_prod, split_k_slices),
                                  tile_shape_k);
            default:
                return 0;
        }
        return 0;
    }
    TV_HOST_DEVICE_INLINE tv::array<int, 5> get_input_shape()
    {
        return {N, input_dims[0], input_dims[1], input_dims[2], C};
    }
    TV_HOST_DEVICE_INLINE tv::array<int, 5> get_weight_shape()
    {
        return {K, ksize[0], ksize[1], ksize[2], C};
    }
    TV_HOST_DEVICE_INLINE tv::array<int, 5> get_output_shape()
    {
        return {N, output_dims[0], output_dims[1], output_dims[2], K};
    }
};
}  // namespace spinds
}  // namespace ops_cpu3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc