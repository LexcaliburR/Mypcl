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
int SparseConvIndicesKernel::generate_conv_inds_stage2_mask(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  // TODO stream
  // TODO handle num input == 0
  int kv = tv::arrayops::prod(ksize);
  // indice_pairs_bwd: [kv, indices.dim(0)]
  // indice_pairs_fwd: [kv, out_inds.dim(0)]
  auto ctx = tv::Context();
  ctx.set_cuda_stream(custream);
  // out_inds: [MaxSize, 4]
  // auto timer = tv::CudaContextTimer<>();
  tv::check_shape(indice_pairs_bwd, {kv, indices.dim(0)});
  tv::check_shape(indice_pairs_fwd, {kv, num_out_act});
  tv::check_shape(out_inds, {num_out_act, 4});
  tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
  launcher_num_act_in.blocks.y = kv;
  tv::cuda::Launch launcher_num_act_in_no_y(indices.dim(0), custream);
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  ConvLocIter loc_iter(problem);
  // TODO handle invalid num_out_act
  indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_out_act);
  tv::cuda::Launch lanucher_build_hash(num_out_act, custream);
  using V = int32_t;
  using KeyType = int32_t;
  constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();
  using table_t =
      tv::hash::LinearHashTable<KeyType, V, tv::hash::Murmur3Hash<KeyType>,
                                  kEmptyKey, false>;
  using pair_t = typename table_t::value_type;
  TV_ASSERT_RT_ERR(hashdata.dim(0) >= num_out_act, "hash size not enough");
  table_t hash = table_t(hashdata.data_ptr<pair_t>(), hashdata.dim(0));
  hash.clear(custream);
  lanucher_build_hash(build_conv_hash_table<table_t>, hash, 
      out_inds.data_ptr<int>(), indice_pairs_uniq.data_ptr<const int32_t>(), 
      loc_iter.layout_npq, num_out_act);
  if (!mask_bwd.empty()){
      // auto timer = tv::CudaContextTimer<>();
      launcher_num_act_in(calc_conv_indices_stage2_mask<table_t>, hash, 
          indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
          mask_fwd.data_ptr<uint32_t>(), mask_bwd.data_ptr<uint32_t>(),
          indice_pairs_bwd.dim(1), indice_pairs_fwd.dim(1));
      // tv::ssprint("calc_conv_indices_stage2_mask", timer.report() / 1000.0);
      launcher_num_act_in_no_y(calc_conv_indices_stage2_mask_output, indice_pairs_bwd.data_ptr<int>(), 
          mask_bwd.data_ptr<uint32_t>(),
          indice_pairs_bwd.dim(1), kv);
      // tv::ssprint("calc_conv_indices_stage2_mask_output", timer.report() / 1000.0);
      if (mask_fwd.dim(0) == 2){
          mask_fwd[1].copy_(mask_fwd[0], ctx);
      }
      if (mask_bwd.dim(0) == 2){
          mask_bwd[1].copy_(mask_bwd[0], ctx);
      }
  }else{
      launcher_num_act_in(calc_conv_indices_stage2_inference_mask<table_t>, hash, 
          indice_pairs_fwd.data_ptr<int>(), indice_pairs_bwd.data_ptr<int>(), 
          mask_fwd.data_ptr<uint32_t>(),
          indice_pairs_bwd.dim(1), indice_pairs_fwd.dim(1));
      if (mask_fwd.dim(0) == 2){
          mask_fwd[1].copy_(mask_fwd[0], ctx);
      }
  }
  return num_out_act;
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc