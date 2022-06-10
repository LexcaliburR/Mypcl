
#include <string>
#include <tuple>

#include <glog/logging.h>

#include <tensorview/tensor.h>
#include <tensorview/cuda/driverops.h>
#include <csrc/sparse/all/ops3d/Point2Voxel.h>
#include <csrc/sparse/all/ops3d/Point2VoxelV2.h>

#include "common/tic_toc.h"
#include "common/binfile_tools.h"

using namespace lidar::common::utils;
using namespace csrc::sparse::all::ops3d;
using namespace tv::detail;

int main(int argc, char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_log_prefix = true;

    BinReader reader(5);

    float* points = nullptr;
    std::string file_name =
        "/home/lishiqi/DATA/testdata/H01_1653012509.450013.bin";
    int point_num = 0;

    reader.read_from_file_rootpath(file_name, points, &point_num);

    Point2VoxelV2 gen(
        {0.1f, 0.1f, 0.1f}, {-80, -80, -6, 80, 80, 6}, 5, 200000, 5);

    PERF_BLOCK_START(true);
    for (int i = 0; i < 1; ++i) {
        LOG(INFO) << "+++++++++++++++++++++++++++++++++++++++++";

        auto points_tv =
            tv::from_blob((void*)points, {point_num, 5}, tv::DType::float32);
        auto points_tv_cuda = points_tv.cuda();
        PERF_BLOCK_END("cpu to gpu");

        tv::Tensor voxels_th_cuda;
        tv::Tensor indices_th_cuda;
        tv::Tensor num_p_in_vx_th_cuda;

        std::tie(voxels_th_cuda, indices_th_cuda, num_p_in_vx_th_cuda) =
            gen.point_to_voxel_hash(points_tv_cuda, true, true);
        auto voxels_th = voxels_th_cuda.cpu();
        auto indices_th = indices_th_cuda.cpu();

        // auto num_p_in_vx_th = num_p_in_vx_th_cuda.cpu();
        LOG(INFO) << "voxels shape : " << voxels_th_cuda.shape();
        LOG(INFO) << "indices shape : " << indices_th_cuda.shape();
        LOG(INFO) << "num per voxel shape : " << num_p_in_vx_th_cuda.shape();
        LOG(INFO) << voxels_th[0];
        LOG(INFO) << indices_th[0];
        LOG(INFO) << voxels_th[1];
        LOG(INFO) << indices_th[1];

        PERF_BLOCK_END("cuda voxel");
        // bool clear_voxels, bool empty_mean
        std::tie(voxels_th_cuda, indices_th_cuda, num_p_in_vx_th_cuda) =
            gen.point_to_voxel_hash(points_tv_cuda, true, false);

        voxels_th = voxels_th_cuda.cpu();
        indices_th = indices_th_cuda.cpu();
        // voxels_th = voxels_th_cuda.cpu();
        // LOG(INFO) << voxels_th_cuda.shape();
        // LOG(INFO) << voxels_th[0];
        // PERF_BLOCK_END("cuda voxel");
        // auto voxels_th = voxels_th_cuda.cpu();
        LOG(INFO) << "voxels shape : " << voxels_th_cuda.shape();
        LOG(INFO) << "indices shape : " << indices_th_cuda.shape();
        LOG(INFO) << "num per voxel shape : " << num_p_in_vx_th_cuda.shape();
        // cpu padding
        // tv::Tensor new_vox = tv::zeros({200000, 5, 5}, tv::DType::float32);
        // std::copy(voxels_th.data<float>(),
        //           voxels_th.data<float>() + voxels_th.size(),
        //           new_vox.data<float>());

        // gpu padding
        // tv::Tensor new_vox_gpu = tv::zeros(
        //     {200000, 5, 5}, tv::DType::float32, voxels_th_cuda.device());
        // tv::dev2dev(new_vox_gpu.raw_data(),
        //             voxels_th_cuda.raw_data(),
        //             voxels_th_cuda.size() *
        //                 tv::detail::sizeof_dtype(voxels_th_cuda.dtype()));

        // auto voxels_th = voxels_th_cuda.cpu();
        // auto voxels_new = new_vox_gpu.cpu();

        PERF_BLOCK_END("padding");
    }

    google::ShutdownGoogleLogging();
}