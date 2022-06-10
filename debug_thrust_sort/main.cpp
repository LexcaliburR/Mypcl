#include <iostream>
#include <vector>

#include <torch/all.h>
#include <thrust/sort.h>

std::vector<float> get_arr()
{
    std::vector<float> ret(20, 0.0f);
    ret[0] = 19.0f;
    ret[2] = 13.0f;
    ret[3] = 2.0f;
    ret[1] = 9.0f;
    ret[4] = 30.0f;
    ret[5] = 2.0f;
    ret[6] = 4.0f;
    ret[7] = 5.0f;

    return ret;
}

int main(int argc, char* argv[])
{
    std::cout << "--------- CPU ---------------" << std::endl;
    auto data1 = get_arr();

    thrust::sort(data1.begin(), data1.end());
    torch::Tensor t_tensor = torch::from_blob((void*)data1.data(), {20});
    // std::cout << t_tensor << std::endl;

    torch::Tensor t_tensor_sorted1;
    torch::Tensor t_tensor_sorted2;
    std::tie(t_tensor_sorted1, t_tensor_sorted2) =
        at::native::_unique_cpu(t_tensor);

    std::cout << t_tensor << std::endl;
    std::cout << t_tensor_sorted1 << std::endl;
    std::cout << t_tensor_sorted2 << std::endl;

    std::cout << "--------- GPU ---------------" << std::endl;

    auto data2 = get_arr();
    // void* data_cuda;
    // cudaMalloc(&data_cuda, sizeof(float) * 20);
    // cudaMemcpy(
    //     data_cuda, data2.data(), sizeof(float) * 20, cudaMemcpyHostToDevice);

    torch::TensorOptions opt = torch::TensorOptions();
    torch::Tensor t_tensor_cuda =
        torch::from_blob((void*)data2.data(), {20}).cuda();

    torch::Tensor t_tensor_cuda_uniqued1;
    torch::Tensor t_tensor_cuda_uniqued2;
    // std::tie(t_tensor_cuda_uniqued1, t_tensor_cuda_uniqued2) =
    //     at::native::_unique_cuda(t_tensor_cuda);

    thrust::sort(t_tensor_cuda.data_ptr<float>(),
                 t_tensor_cuda.data_ptr<float>() + 20);

    std::cout << t_tensor_cuda.device() << std::endl;
    std::cout << t_tensor_cuda << std::endl;
}