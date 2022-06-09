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
int SparseConvIndicesKernel::generate_conv_inds_stage1_5(tv::Tensor indice_pairs_uniq, int64_t uniq_size, std::uintptr_t stream_int)   {
  
  thrust::device_ptr<int32_t> ptr_tr(indice_pairs_uniq.data_ptr<int32_t>());
  auto thrust_ctx = thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(stream_int));
  thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
  auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
  auto num_out_act = new_end - ptr_tr - 1;
  return num_out_act;
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc