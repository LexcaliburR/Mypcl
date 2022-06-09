#pragma once
#include <vector>
#include <tuple>
#include <string>
namespace cumm {
namespace common {
struct CompileInfo {
  
  static std::vector<std::tuple<int, int>> get_compiled_cuda_arch();
};
} // namespace common
} // namespace cumm