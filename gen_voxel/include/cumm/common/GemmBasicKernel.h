#pragma once
#include <tensorview/gemm/arch/memory.h>
#include <tensorview/gemm/arch/transpose.h>
#include <tensorview/gemm/arch/semaphore.h>
#include <cumm/common/GemmBasic.h>
namespace cumm {
namespace common {
using GemmBasic = cumm::common::GemmBasic;
struct GemmBasicKernel {
};
} // namespace common
} // namespace cumm