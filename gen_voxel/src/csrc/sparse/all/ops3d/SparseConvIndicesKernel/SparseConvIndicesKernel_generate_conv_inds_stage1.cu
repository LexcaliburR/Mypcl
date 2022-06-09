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
void SparseConvIndicesKernel::generate_conv_inds_stage1(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  // TODO stream
  // TODO handle num input == 0
  int kv = tv::arrayops::prod(ksize);
  TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
  TV_ASSERT_RT_ERR(tv::arrayops::prod(input_dims) <= std::numeric_limits<int32_t>::max(), 
      "kernel volume must smaller than max value of int32_t");
  // indice_pairs: [2, kv, indices.dim(0)]
  // indice_pairs_uniq: [indice_pairs.size() / 2 + 1]
  tv::check_shape(indice_pairs, {2, kv, indices.dim(0)});
  tv::check_shape(indice_num_per_loc, {kv});
  int64_t uniq_size = indice_pairs.size() / 2 + 1;
  TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= uniq_size, "error");
  TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
  int64_t expected_out_size = indices.dim(0) * kv;
  tv::cuda::Launch launcher_num_act_in(indices.dim(0), reinterpret_cast<cudaStream_t>(stream_int));
  // tv::cuda::Launch launcher_num_act_in_2(indices.dim(0));
  launcher_num_act_in.blocks.y = kv;
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  ConvLocIter loc_iter(problem);
  tv::cuda::Launch launcher_clean_uniq(uniq_size, reinterpret_cast<cudaStream_t>(stream_int));
  launcher_clean_uniq(clean_indices_uniq, indice_pairs_uniq.data_ptr<int32_t>(), uniq_size);
  launcher_num_act_in(calc_conv_indices_stage1, loc_iter, indices.data_ptr<const int>(), 
      indice_pairs.data_ptr<int32_t>(), 
      indice_pairs_uniq.data_ptr<int32_t>(), indice_num_per_loc.data_ptr<int>(), indices.dim(0),
      indice_pairs.dim(2), kv, transposed);
  // thrust::device_ptr<int32_t> ptr_tr(indice_pairs_uniq.data_ptr<int32_t>());
  // auto thrust_ctx = thrust::cuda::par.on(reinterpret_cast<cudaStream_t>(stream_int));
  // thrust::sort(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
  // auto new_end = thrust::unique(thrust_ctx, ptr_tr, ptr_tr + uniq_size);
  // auto num_out_act = new_end - ptr_tr - 1;
  // return num_out_act;
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc