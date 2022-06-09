#pragma once
namespace csrc {
namespace sparse {
namespace all {
namespace cudakers {

template <typename T>
__global__ void arange_kernel(T* data, int size)
{
    for (int i : tv::KernelLoopX<int>(size)) {
        data[i] = T(i);
    }
}
template <typename T>
__global__ void fill_kernel(T* data, T val, int size)
{
    for (int i : tv::KernelLoopX<int>(size)) {
        data[i] = T(val);
    }
}

struct CudaCommonKernel
{};

}  // namespace cudakers
}  // namespace all
}  // namespace sparse
}  // namespace csrc