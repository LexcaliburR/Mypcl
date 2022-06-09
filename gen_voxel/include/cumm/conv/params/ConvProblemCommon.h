#pragma once
#include <cumm/conv/bases/ConvEnum.h>
#include <cumm/common/TensorView.h>
namespace cumm {
namespace conv {
namespace params {
using ConvEnum = cumm::conv::bases::ConvEnum;
using TensorView = cumm::common::TensorView;
struct ConvProblemCommon {
  TV_HOST_DEVICE_INLINE static tv::array<int, 3> implicit_gemm_mnk(ConvEnum::OpType op_type, int N, int C, int K, int kernel_volume, int in_prod, int out_prod, bool mask_sparse)   {
    
    if (mask_sparse){
        switch (op_type) {
            case ConvEnum::OpType::kForward:
                return {N, K, C * kernel_volume};
            case ConvEnum::OpType::kBackwardInput:
                return {N, C, K * kernel_volume};
            case ConvEnum::OpType::kBackwardWeight:
                return {K, C * kernel_volume, N};
            default:
                return {};
        }
        return {};
    }else{
        switch (op_type) {
            case ConvEnum::OpType::kForward:
                return {N * out_prod, K, C * kernel_volume};
            case ConvEnum::OpType::kBackwardInput:
                return {N * in_prod, C, K * kernel_volume};
            case ConvEnum::OpType::kBackwardWeight:
                return {K, C * kernel_volume, N * out_prod};
            default:
                return {};
        }
        return {};
    }
  }
};
} // namespace params
} // namespace conv
} // namespace cumm