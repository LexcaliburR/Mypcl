#pragma once
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace csrc {
namespace sparse {
namespace all {
struct PyBindSpconvOps {
  /**
   * @param module 
   */
  static void bind_SpconvOps(const pybind11::module_& module);
};
} // namespace all
} // namespace sparse
} // namespace csrc