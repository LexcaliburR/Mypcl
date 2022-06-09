
#include <string>

#include <glog/logging.h>

#include <tensorview/tensor.h>

#include "common/tic_toc.h"
#include "common/binfile_tools.h"
#include "csrc/sparse/all/ops3d/Point2Voxel.h"

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

    Point2Voxel gen(
        {0.1f, 0.1f, 0.1f}, {-80, -80, -6, 80, 80, 6}, 5, 200000, 5);

    auto points_tv =
        tv::from_blob((void*)points, {point_num, 5}, tv::DType::float32);

    auto points_tv_cuda = points_tv.cuda();
    LOG(INFO) << points_tv.raw_size();
    LOG(INFO) << points_tv.size();
    LOG(INFO) << points_tv.storage_size();
    LOG(INFO) << points_tv.itemsize();
    LOG(INFO) << points_tv.byte_offset();
    LOG(INFO) << points_tv.device();
    LOG(INFO) << points_tv.empty();

    LOG(INFO) << "-----------------------------------------";
    LOG(INFO) << points_tv_cuda.raw_size();
    LOG(INFO) << points_tv_cuda.size();
    LOG(INFO) << points_tv_cuda.storage_size();
    LOG(INFO) << points_tv_cuda.itemsize();
    LOG(INFO) << points_tv_cuda.byte_offset();
    LOG(INFO) << points_tv_cuda.device();
    LOG(INFO) << points_tv_cuda.empty();

    PERF_BLOCK_START(true);

    for (int i = 0; i < 1000; ++i) {
        auto ret = gen.point_to_voxel_hash(points_tv_cuda);
        PERF_BLOCK_END("cuda voxel");
    }

    google::ShutdownGoogleLogging();
}