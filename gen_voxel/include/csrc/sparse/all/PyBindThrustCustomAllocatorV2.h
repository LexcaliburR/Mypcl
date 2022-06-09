#pragma once
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace csrc {
namespace sparse {
namespace all {
struct PyBindThrustCustomAllocatorV2 {
  /**
   * @param module 
   */
  static void bind_ThrustCustomAllocatorV2(const pybind11::module_& module);
};
} // namespace all
} // namespace sparse
} // namespace csrc