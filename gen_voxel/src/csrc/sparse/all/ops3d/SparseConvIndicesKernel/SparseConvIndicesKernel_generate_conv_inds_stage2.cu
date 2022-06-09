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
int SparseConvIndicesKernel::generate_conv_inds_stage2(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor out_inds, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed, std::uintptr_t stream_int)   {
  
  auto custream = reinterpret_cast<cudaStream_t>(stream_int);
  // TODO stream
  // TODO handle num input == 0
  int kv = tv::arrayops::prod(ksize);
  TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
  // indice_pairs: [2, kv, indices.dim(0)]
  // indice_pairs_uniq: [indice_pairs.size() / 2 + 1]
  // out_inds: [MaxSize, 4]
  // auto timer = tv::CudaContextTimer<>();
  int64_t uniq_size = indice_pairs.size() / 2 + 1;
  TV_ASSERT_RT_ERR(indice_pairs_uniq.dim(0) >= num_out_act, "error");
  TV_ASSERT_RT_ERR(out_inds.dim(0) >= num_out_act && out_inds.dim(1) == 4, "error");
  tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
  launcher_num_act_in.blocks.y = kv;
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
  launcher_num_act_in(calc_conv_indices_stage2<table_t>, hash, 
      indice_pairs[1].data_ptr<int>(), indices.dim(0), 
      indice_pairs.dim(2));
  return num_out_act;
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc