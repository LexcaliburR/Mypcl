#pragma once
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace cumm {
namespace common {
struct PyBindCompileInfo {
  /**
   * @param module 
   */
  static void bind_CompileInfo(const pybind11::module_& module);
};
} // namespace common
} // namespace cumm