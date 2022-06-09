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

int SparseConvIndicesKernel::generate_subm_conv_inds(
    tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs,
    tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size,
    tv::array<int, 3> input_dims, tv::array<int, 3> ksize,
    tv::array<int, 3> dilation, tv::Tensor indice_pair_mask, bool backward,
    std::uintptr_t stream_int)
{
    auto custream = reinterpret_cast<cudaStream_t>(stream_int);
    auto ctx = tv::Context();
    ctx.set_cuda_stream(custream);
    if (!indice_pair_mask.empty()) {
        TV_ASSERT_INVALID_ARG(tv::arrayops::prod(ksize) < 32,
                              "for now only support 32bit mask");
    }
    // TODO stream
    // TODO handle num input == 0
    tv::array<int, 3> stride, padding;
    for (int i = 0; i < 3; ++i) {
        TV_ASSERT_RT_ERR(ksize[i] % 2 == 1, "subm only support odd ksize");
        stride[i] = 1;
        padding[i] = (ksize[i] / 2) * dilation[i];
    }
    int kv = tv::arrayops::prod(ksize);
    TV_ASSERT_RT_ERR(kv == indice_pairs.dim(1), "error");
    // indice_pairs: [2, kv, indices.dim(0)]
    // out_inds: [MaxSize, 4]
    // auto timer = tv::CudaContextTimer<>();
    TV_ASSERT_RT_ERR(indice_num_per_loc.dim(0) == kv, "error");
    tv::cuda::Launch launcher_num_act_in(indices.dim(0), custream);
    launcher_num_act_in.blocks.y = (kv / 2) + 1;
    // launcher_num_act_in.blocks.y = kv;
    TV_ASSERT_RT_ERR(
        tv::arrayops::prod(input_dims) <= std::numeric_limits<int32_t>::max(),
        "kernel volume must smaller than max value of int32_t");
    ConvProblem problem(batch_size,
                        1,
                        1,
                        input_dims,
                        input_dims,
                        ksize,
                        padding,
                        stride,
                        dilation);
    ConvLocIter loc_iter(problem);
    tv::cuda::Launch lanucher_build_hash(indices.dim(0), custream);
    using V = int32_t;
    using KeyType = int32_t;
    constexpr KeyType kEmptyKey = std::numeric_limits<KeyType>::max();
    using table_t = tv::hash::LinearHashTable<KeyType,
                                              V,
                                              tv::hash::Murmur3Hash<KeyType>,
                                              kEmptyKey,
                                              false>;
    using pair_t = typename table_t::value_type;
    TV_ASSERT_RT_ERR(hashdata.dim(0) >= indices.dim(0), "hash size not enough");
    table_t hash = table_t(hashdata.data_ptr<pair_t>(), hashdata.dim(0));
    hash.clear(custream);
    // tv::ssprint("clear hash time", hashdata.dim(0), timer.report() / 1000.0);
    lanucher_build_hash(build_subm_conv_hash_table<table_t>,
                        hash,
                        indices.data_ptr<const int>(),
                        loc_iter.layout_npq,
                        indices.dim(0));
    // tv::ssprint("build_hash time", timer.report() / 1000.0);
    if (!indice_pair_mask.empty()) {
        TV_ASSERT_INVALID_ARG(indice_pair_mask.ndim() == 2, "error");
        if (indice_pair_mask.dim(0) == 2) {
            auto mask_0 = indice_pair_mask[0];
            tv::cuda::Launch lanucher_fill(mask_0.size(), custream);
            lanucher_fill(cudakers::fill_kernel<uint32_t>,
                          mask_0.data_ptr<uint32_t>(),
                          (1 << (kv / 2)),
                          mask_0.size());
            indice_pair_mask[1].zero_(ctx);
            auto kernel = &calc_subm_conv_indices_split_mask<table_t>;
            launcher_num_act_in(kernel,
                                loc_iter,
                                hash,
                                indices.data_ptr<int>(),
                                indice_pairs.data_ptr<int>(),
                                indice_pair_mask[0].data_ptr<uint32_t>(),
                                indice_pair_mask[1].data_ptr<uint32_t>(),
                                indices.dim(0),
                                indice_pairs.dim(2),
                                kv);
        } else {
            tv::cuda::Launch lanucher_fill(indice_pair_mask.size(), custream);
            lanucher_fill(cudakers::fill_kernel<uint32_t>,
                          indice_pair_mask.data_ptr<uint32_t>(),
                          (1 << (kv / 2)),
                          indice_pair_mask.size());
            TV_ASSERT_RT_ERR(indice_pair_mask.dim(0) == 1, "error");
            launcher_num_act_in(calc_subm_conv_indices_mask<table_t>,
                                loc_iter,
                                hash,
                                indices.data_ptr<int>(),
                                indice_pairs.data_ptr<int>(),
                                indice_pair_mask.data_ptr<uint32_t>(),
                                indices.dim(0),
                                indice_pairs.dim(2),
                                kv);
        }
    } else {
        launcher_num_act_in(calc_subm_conv_indices<table_t>,
                            loc_iter,
                            hash,
                            indices.data_ptr<int>(),
                            indice_pairs.data_ptr<int>(),
                            indice_num_per_loc.data_ptr<int>(),
                            indices.dim(0),
                            indice_pairs.dim(2),
                            kv);
    }
    // tv::ssprint("gem subm conv inds time", timer.report() / 1000.0);
    return indices.dim(0);
}
}  // namespace ops3d
}  // namespace all
}  // namespace sparse
}  // namespace csrc