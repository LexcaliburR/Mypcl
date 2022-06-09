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

__global__ void calc_conv_indices_stage1(
    ConvLocIter loc_iter, const int* indices_in, int32_t* indice_pairs,
    int32_t* indice_pairs_for_uniq, int* indice_num_per_loc, int num_indices_in,
    int indices_pair_size, int RS, bool transposed)
{
    int filter_offset = blockIdx.y;
    loc_iter.set_filter_offset(filter_offset);
    int indices_pair_size_mul_RS = indices_pair_size * RS;
    int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
    for (int i : tv::KernelLoopX<int>(num_indices_in)) {
        tv::array<int, 4> npq_offset;
        bool valid;
        if (transposed) {
            valid = loc_iter.query_nhw_out(indices_in + i * 4, npq_offset);
        } else {
            valid = loc_iter.query_npq(indices_in + i * 4, npq_offset);
        }
        if (valid) {
            int old_num =
                tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
            int32_t offset = loc_iter.layout_npq(npq_offset);
            if (old_num < indices_pair_size) {
                indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                indice_pairs[indices_pair_size_mul_RS +
                             filter_offset_mul_indices_pair_size + old_num] =
                    offset;
                indice_pairs_for_uniq[filter_offset_mul_indices_pair_size +
                                      old_num] = offset;
            }
        }
    }
}

}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc