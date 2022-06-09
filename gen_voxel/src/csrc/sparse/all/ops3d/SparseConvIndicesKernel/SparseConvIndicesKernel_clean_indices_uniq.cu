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
__global__ void clean_indices_uniq(int32_t* indice_pairs_for_uniq, int32_t size)   {
  
  for (int32_t i : tv::KernelLoopX<int32_t>(size)) {
      indice_pairs_for_uniq[i] = std::numeric_limits<int32_t>::max();
  }
}
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc