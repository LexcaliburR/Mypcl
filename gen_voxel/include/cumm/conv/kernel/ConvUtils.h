#pragma once
#include <cumm/common/TensorView.h>
namespace cumm {
namespace conv {
namespace kernel {
using TensorView = cumm::common::TensorView;
struct ConvUtils {
  /**
   * @param m 
   * @param n 
   * @param k 
   * @param tile_m 
   * @param tile_n 
   * @param split_k_slice 
   * @param kv 
   * @param op_type 
   */
  static tv::array<int, 3> get_spconv_logical_tile_count(int m, int n, int k, int tile_m, int tile_n, int split_k_slice, int kv, int op_type);
};
} // namespace kernel
} // namespace conv
} // namespace cumm