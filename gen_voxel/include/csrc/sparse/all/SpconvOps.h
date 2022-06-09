#pragma once
#include <csrc/sparse/all/ThrustCustomAllocatorV2.h>
#include <csrc/sparse/all/ops_cpu1d/Point2VoxelCPU.h>
#include <csrc/sparse/all/ops_cpu1d/SparseConvIndicesCPU.h>
#include <csrc/sparse/all/ops1d/Point2Voxel.h>
#include <csrc/sparse/all/ops_cpu2d/Point2VoxelCPU.h>
#include <csrc/sparse/all/ops_cpu2d/SparseConvIndicesCPU.h>
#include <csrc/sparse/all/ops2d/Point2Voxel.h>
#include <csrc/sparse/all/ops_cpu3d/Point2VoxelCPU.h>
#include <csrc/sparse/all/ops_cpu3d/SparseConvIndicesCPU.h>
#include <csrc/sparse/all/ops3d/Point2Voxel.h>
#include <csrc/sparse/all/ops_cpu4d/Point2VoxelCPU.h>
#include <csrc/sparse/all/ops_cpu4d/SparseConvIndicesCPU.h>
#include <csrc/sparse/all/ops4d/Point2Voxel.h>
namespace csrc {
namespace sparse {
namespace all {
using ThrustCustomAllocatorV2 = csrc::sparse::all::ThrustCustomAllocatorV2;
using Point2Voxel1DCPU = csrc::sparse::all::ops_cpu1d::Point2VoxelCPU;
using SpconvIndicesCPU1D = csrc::sparse::all::ops_cpu1d::SparseConvIndicesCPU;
using Point2Voxel1D = csrc::sparse::all::ops1d::Point2Voxel;
using Point2Voxel2DCPU = csrc::sparse::all::ops_cpu2d::Point2VoxelCPU;
using SpconvIndicesCPU2D = csrc::sparse::all::ops_cpu2d::SparseConvIndicesCPU;
using Point2Voxel2D = csrc::sparse::all::ops2d::Point2Voxel;
using Point2Voxel3DCPU = csrc::sparse::all::ops_cpu3d::Point2VoxelCPU;
using SpconvIndicesCPU3D = csrc::sparse::all::ops_cpu3d::SparseConvIndicesCPU;
using Point2Voxel3D = csrc::sparse::all::ops3d::Point2Voxel;
using Point2Voxel4DCPU = csrc::sparse::all::ops_cpu4d::Point2VoxelCPU;
using SpconvIndicesCPU4D = csrc::sparse::all::ops_cpu4d::SparseConvIndicesCPU;
using Point2Voxel4D = csrc::sparse::all::ops4d::Point2Voxel;
struct SpconvOps {
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
  static void generate_conv_inds_stage1(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indice_pairs_uniq 
   * @param ndim 
   * @param uniq_size 
   * @param stream_int 
   */
  static int generate_conv_inds_stage1_5(tv::Tensor indice_pairs_uniq, int ndim, int64_t uniq_size, std::uintptr_t stream_int = 0);
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
  static int generate_conv_inds_stage2(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor out_inds, int num_out_act, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
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
  static void generate_conv_inds_mask_stage1(tv::Tensor indices, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
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
  static int generate_conv_inds_mask_stage2(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
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
  static int generate_subm_conv_inds(tv::Tensor indices, tv::Tensor hashdata, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> dilation, tv::Tensor indice_pair_mask = tv::Tensor(), bool backward = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   */
  static int generate_conv_inds_cpu(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false);
  /**
   * @param indices 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param input_dims 
   * @param ksize 
   * @param dilation 
   */
  static int generate_subm_conv_inds_cpu(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> dilation);
  /**
   * @param out 
   * @param inp 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void maxpool_forward(tv::Tensor out, tv::Tensor inp, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param dout 
   * @param dinp 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void maxpool_backward(tv::Tensor out, tv::Tensor inp, tv::Tensor dout, tv::Tensor dinp, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   * @param stream 
   */
  static void maxpool_implicit_gemm_forward(tv::Tensor out, tv::Tensor inp, tv::Tensor inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param dout 
   * @param dinp 
   * @param inds 
   * @param stream 
   */
  static void maxpool_implicit_gemm_backward(tv::Tensor out, tv::Tensor inp, tv::Tensor dout, tv::Tensor dinp, tv::Tensor inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param out_inds 
   * @param in_inds 
   */
  static void maxpool_forward_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor out_inds, tv::Tensor in_inds);
  /**
   * @param out 
   * @param inp 
   * @param dout 
   * @param dinp 
   * @param out_inds 
   * @param in_inds 
   */
  static void maxpool_backward_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor dout, tv::Tensor dinp, tv::Tensor out_inds, tv::Tensor in_inds);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   */
  static void gather_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor inds);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   */
  static void scatter_add_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor inds);
  /**
   * @param data 
   * @param indices 
   * @param stream 
   */
  static tv::Tensor sort_1d_by_key(tv::Tensor data, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0);
  /**
   * @param data 
   * @param alloc_func 
   * @param indices 
   * @param stream 
   */
  static tv::Tensor sort_1d_by_key_allocator(tv::Tensor data, std::function<std::uintptr_t(std::size_t)> alloc_func, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0);
  /**
   * @param data 
   * @param mask 
   * @param indices 
   * @param stream 
   * @param mask_output 
   */
  static tv::Tensor sort_1d_by_key_split(tv::Tensor data, tv::Tensor mask, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, bool mask_output = false);
  /**
   * @param data 
   * @param alloc_func 
   * @param mask 
   * @param indices 
   * @param stream 
   * @param mask_output 
   */
  static tv::Tensor sort_1d_by_key_split_allocator(tv::Tensor data, std::function<std::uintptr_t(std::size_t)> alloc_func, tv::Tensor mask, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, bool mask_output = false);
  /**
   * @param a 
   */
  static tv::Tensor count_bits(tv::Tensor a);
  /**
   * @param vsize_xyz 
   * @param coors_range_xyz 
   */
  static std::tuple<std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>> calc_point2voxel_meta_data(std::vector<float> vsize_xyz, std::vector<float> coors_range_xyz);
  /**
   * @param points 
   * @param voxels 
   * @param indices 
   * @param num_per_voxel 
   * @param densehashdata 
   * @param pc_voxel_id 
   * @param vsize 
   * @param grid_size 
   * @param grid_stride 
   * @param coors_range 
   * @param empty_mean 
   * @param clear_voxels 
   */
  static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point2voxel_cpu(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor densehashdata, tv::Tensor pc_voxel_id, std::vector<float> vsize, std::vector<int> grid_size, std::vector<int> grid_stride, std::vector<float> coors_range, bool empty_mean = false, bool clear_voxels = true);
  /**
   * @param points 
   * @param voxels 
   * @param indices 
   * @param num_per_voxel 
   * @param hashdata 
   * @param point_indice_data 
   * @param pc_voxel_id 
   * @param vsize 
   * @param grid_size 
   * @param grid_stride 
   * @param coors_range 
   * @param empty_mean 
   * @param clear_voxels 
   * @param stream_int 
   */
  static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point2voxel_cuda(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor hashdata, tv::Tensor point_indice_data, tv::Tensor pc_voxel_id, std::vector<float> vsize, std::vector<int> grid_size, std::vector<int> grid_stride, std::vector<float> coors_range, bool empty_mean = false, bool clear_voxels = true, std::uintptr_t stream_int = 0);
};
} // namespace all
} // namespace sparse
} // namespace csrc