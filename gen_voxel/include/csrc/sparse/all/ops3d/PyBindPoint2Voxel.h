#pragma once

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
struct PyBindPoint2Voxel {
  /**
   * @param module 
   */
  static void bind_Point2Voxel(const pybind11::module_& module);
};
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc