#pragma once
#include <cumm/common/TensorView.h>
#include <cumm/common/TensorViewKernel.h>
#include <cumm/common/TensorViewHashKernel.h>
#include <cumm/common/ThrustLib.h>
#include <csrc/sparse/all/ops3d/spinds/ConvOutLocIter.h>
#include <csrc/sparse/all/ops_cpu3d/spinds/ConvProblem.h>
#include <csrc/sparse/all/ops3d/cudakers/CudaCommonKernel.h>

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

/**
 * @param loc_iter 
 * @param indices_in 
 * @param indice_pairs 
 * @param indice_pairs_for_uniq 
 * @param indice_num_per_loc 
 * @param num_indices_in 
 * @param indices_pair_size 
 * @param RS 
 * @param transposed 
 */
__global__ void calc_conv_indices_stage1(ConvLocIter loc_iter, const int* indices_in, int32_t* indice_pairs, int32_t* indice_pairs_for_uniq, int* indice_num_per_loc, int num_indices_in, int indices_pair_size, int RS, bool transposed);
template <typename TTable>
__global__ void build_conv_hash_table(TTable table, int* indices_out, const int32_t* indice_pairs_for_uniq, spinds::LayoutNPQ layout_npq, int num_indices)   {
  
  for (int output_index : tv::KernelLoopX<int>(num_indices)) {
      int32_t output_coord_offset = indice_pairs_for_uniq[output_index];
      layout_npq.inverse(output_coord_offset, indices_out + 4 * output_index);
      table.insert(output_coord_offset, output_index);
  }
}
template <typename TTable>
__global__ void calc_conv_indices_stage2(TTable table, int* indice_pairs_out_part, int num_indices_in, int indices_pair_size)   {
  
  int filter_offset = blockIdx.y;
  auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
  for (int i : tv::KernelLoopX<int>(num_indices_in)) {
      int32_t output_coord_offset = indice_pairs_out_part_filter[i];
      if (output_coord_offset > -1){
          auto ptr = table.lookup_ptr(output_coord_offset);
          if (ptr){
              indice_pairs_out_part_filter[i] = ptr->second;
          }
      }
  }
}
/**
 * @param loc_iter 
 * @param indices_in 
 * @param indice_pairs_bwd 
 * @param indice_pairs_for_uniq 
 * @param indice_num_per_loc 
 * @param num_indices_in 
 * @param RS 
 * @param transposed 
 */
__global__ void calc_conv_indices_stage1_mask(ConvLocIter loc_iter, const int* indices_in, int32_t* indice_pairs_bwd, int32_t* indice_pairs_for_uniq, int* indice_num_per_loc, int num_indices_in, int RS, bool transposed);
template <typename TTable>
__global__ void calc_conv_indices_stage2_mask(TTable table, int* indice_pairs_fwd, int* indice_pairs_bwd, uint32_t* mask_fwd, uint32_t* mask_bwd, int num_indices_in, int num_indices_out)   {
  
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_fwd = (1u << (filter_offset));
  // TODO following rule for even kernel size is wrong. 
  // uint32_t filter_mask_bwd = (1u << (gridDim.y - 1 - filter_offset));
  auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
  auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
      int32_t output_coord_offset = indice_pairs_bwd_filter[input_index];
      if (output_coord_offset > -1){
          auto ptr = table.lookup_ptr(output_coord_offset);
          if (ptr){
              auto output_index = ptr->second;
              atomicOr(mask_fwd + output_index, filter_mask_fwd);
              // atomicOr(mask_bwd + input_index, filter_mask_bwd);
              indice_pairs_fwd_filter[output_index] = input_index;
              indice_pairs_bwd_filter[input_index] = output_index;
          }
      }
  }
}
/**
 * @param indice_pairs_bwd 
 * @param mask_bwd 
 * @param num_indices_in 
 * @param kv 
 */
__global__ void calc_conv_indices_stage2_mask_output(int* indice_pairs_bwd, uint32_t* mask_bwd, int num_indices_in, int kv);
template <typename TTable>
__global__ void calc_conv_indices_stage2_inference_mask(TTable table, int* indice_pairs_fwd, int* indice_pairs_bwd, uint32_t* mask_fwd, int num_indices_in, int num_indices_out)   {
  
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_fwd = (1u << (filter_offset));
  auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
  auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
      int32_t output_coord_offset = indice_pairs_bwd_filter[input_index];
      if (output_coord_offset > -1){
          auto ptr = table.lookup_ptr(output_coord_offset);
          if (ptr){
              auto output_index = ptr->second;
              atomicOr(mask_fwd + output_index, filter_mask_fwd);
              indice_pairs_fwd_filter[output_index] = input_index;
          }
      }
  }
}
template <typename TTable>
__global__ void build_subm_conv_hash_table(TTable table, const int* indices_in, spinds::LayoutNPQ layout_npq, int num_indices)   {
  
  for (int i : tv::KernelLoopX<int>(num_indices)) {
      int32_t index = layout_npq(indices_in + i * 4);
      table.insert(index, i);
  }
}
/**
 * @param indice_pairs_for_uniq 
 * @param size 
 */
__global__ void clean_indices_uniq(int32_t* indice_pairs_for_uniq, int32_t size);
template <typename TTable>
__global__ void calc_subm_conv_indices(ConvLocIter loc_iter, TTable table, const int* indices_in, int32_t* indice_pairs, int* indice_num_per_loc, int num_indices_in, int indices_pair_size, int RS)   {
  
  int filter_offset = blockIdx.y;
  loc_iter.set_filter_offset(filter_offset);
  int indices_pair_size_mul_RS = indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
  if (filter_offset == (RS / 2)){
      for (int i : tv::KernelLoopX<int>(num_indices_in)) {
          indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
          indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
      }
  } else {
      for (int i : tv::KernelLoopX<int>(num_indices_in)) {
          tv::array<int, 4> npq_offset;
          if (loc_iter.query_npq_no_stride(indices_in + i * 4, npq_offset)){
              int32_t offset = loc_iter.layout_npq(npq_offset);
              auto item = table.lookup(offset); // performance bound
              if (!item.empty()){
                  int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                  indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                  indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = item.second;
                  indice_pairs[filter_offset_mul_indices_pair_size_1 + old_num] = item.second;
                  indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
              }
          }
      }
  }
}
template <typename TTable>
__global__ void calc_subm_conv_indices_mask(ConvLocIter loc_iter, TTable table, const int* indices_in, int32_t* indice_pairs, uint32_t* mask, int num_indices, int indices_pair_size, int RS)   {
  
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_out = (1u << (filter_offset));
  uint32_t filter_mask_in = (1u << (RS - 1 - filter_offset));
  // uint32_t filter_mask_center = (1u << (RS / 2));
  loc_iter.set_filter_offset(filter_offset);
  int indices_pair_size_mul_RS = indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
  if (filter_offset == (RS / 2)){
      for (int i : tv::KernelLoopX<int>(num_indices)) {
          // atomicOr(mask + i, filter_mask_center);
          indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
          indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
      }
  } else {
      for (int output_index : tv::KernelLoopX<int>(num_indices)) {
          // find input offset from output offset
          tv::array<int, 4> nhw_offset;
          // table: input indice coord to output index (or output indice coord to input index)
          if (loc_iter.query_nhw(indices_in + output_index * 4, nhw_offset)){
              int32_t offset = loc_iter.layout_npq(nhw_offset);
              auto item = table.lookup(offset);
              if (!item.empty()) {
                  auto input_index = item.second; // we find a input indice idx.
                  atomicOr(mask + output_index, filter_mask_out);
                  atomicOr(mask + input_index, filter_mask_in);
                  // for this output, we set correct input idx.
                  indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                  indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + input_index] = output_index;
                  // the output in "input location" connect this output idx in another location.
                  indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                  indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
              }
          }
      }
  }
}
template <typename TTable>
__global__ void calc_subm_conv_indices_split_mask(ConvLocIter loc_iter, TTable table, const int* indices_in, int32_t* indice_pairs, uint32_t* mask1, uint32_t* mask2, int num_indices, int indices_pair_size, int RS)   {
  
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_out = (1u << (filter_offset));
  uint32_t filter_mask_in = (1u << (RS - 1 - filter_offset));
  // uint32_t filter_mask_center = (1u << (RS / 2));
  loc_iter.set_filter_offset(filter_offset);
  auto indice_ptr_inv = indice_pairs + indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
  if (filter_offset == (RS / 2)){
      for (int i : tv::KernelLoopX<int>(num_indices)) {
          indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
          indice_ptr_inv[filter_offset_mul_indices_pair_size + i] = i;
      }
  } else {
      for (int output_index : tv::KernelLoopX<int>(num_indices)) {
          // find input offset from output offset
          tv::array<int, 4> nhw_offset;
          // table: input indice coord to output index (or output indice coord to input index)
          if (loc_iter.query_nhw(indices_in + output_index * 4, nhw_offset)){
              int32_t offset = loc_iter.layout_npq(nhw_offset);
              auto item = table.lookup(offset);
              if (!item.empty()) {
                  auto input_index = item.second; // we find a input indice idx.
                  atomicOr(mask1 + output_index, filter_mask_out);
                  atomicOr(mask2 + input_index, filter_mask_in);
                  // for this output, we set correct input idx.
                  indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                  // the output in "input location" connect this output idx in another location.
                  indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                  indice_ptr_inv[filter_offset_mul_indices_pair_size + input_index] = output_index;
                  indice_ptr_inv[filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
              }
          }
      }
  }
}
struct SparseConvIndicesKernel {
  /**
   * @param indices 
   * @param indice_pairs 
   * @param indice_pairs_uniq 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static void generate_conv_inds_stage1(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indice_pairs_uniq 
   * @param uniq_size 
   * @param stream_int 
   */
  static int generate_conv_inds_stage1_5(tv::Tensor indice_pairs_uniq, int64_t uniq_size, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata 
   * @param indice_pairs 
   * @param indice_pairs_uniq 
   * @param out_inds 
   * @param num_out_act 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static int generate_conv_inds_stage2(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor out_inds, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param indice_pairs_bwd 
   * @param indice_pairs_uniq 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static void generate_conv_inds_mask_stage1(tv::Tensor indices, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata 
   * @param indice_pairs_fwd 
   * @param indice_pairs_bwd 
   * @param indice_pairs_uniq 
   * @param out_inds 
   * @param mask_fwd 
   * @param mask_bwd 
   * @param num_out_act 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static int generate_conv_inds_stage2_mask(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param input_dims 
   * @param ksize 
   * @param dilation 
   * @param indice_pair_mask 
   * @param backward 
   * @param stream_int 
   */
  static int generate_subm_conv_inds(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> dilation, tv::Tensor indice_pair_mask = tv::Tensor(), bool backward = false, std::uintptr_t stream_int = 0);
};
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc