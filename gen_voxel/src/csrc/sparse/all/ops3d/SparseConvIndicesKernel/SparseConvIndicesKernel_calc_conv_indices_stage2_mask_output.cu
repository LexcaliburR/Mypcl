#include <csrc/sparse/all/ops3d/SparseConvIndicesKernel.h>

namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {

using TensorView = cumm::common::TensorView;
using TensorViewKernel = cumm::common::TensorViewKernel;
using TensorViewHashKernel = cumm::common::TensorViewHashKernel;
using ThrustLib = cumm::common::ThrustLib;
using ConvLocIter = csrc::sparse::all::ops3d::spinds::ConvOutLocIter;
using ConvProblem = csrc::sparse::all::ops_cpu3d::spinds::ConvProblem;

__global__ void calc_conv_indices_stage2_mask_output(int* indice_pairs_bwd,
                                                     uint32_t* mask_bwd,
                                                     int num_indices_in, int kv)
{
    for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
        uint32_t mask = 0;
        for (int filter_offset = 0; filter_offset < kv; ++filter_offset) {
            auto val =
                indice_pairs_bwd[filter_offset * num_indices_in + input_index];
            mask |= (val != -1) << filter_offset;
        }
        mask_bwd[input_index] = mask;
    }
}

}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc