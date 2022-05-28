#include <string>

#include "filter/voxelDSampler.h"
#include "utils/readbin.h"
#include "utils/pcdconvertion.h"
#include "utils/tic_toc.h"

using namespace mypcl;

int main(int argc, char* argv[])
{
    std::string file_path =
        "/home/develop/Mypcl/sampledata/H01_1634009186.127198.bin";
    float* points = new float[0];
    int point_num = utils::readPointBin(file_path, points, 5, 4);

    auto input = utils::arrToVecPoint(points, point_num);

    filter::VoxelDSampleSort downsample({0.5, 0.5, 0.5},
                                        filter::SampleType::CENTER);

    std::vector<base::PointXYZI> new_pts;

    TIME_START(false);
    downsample.downSample(input, &new_pts);
    TIME_END("down sample");

    LOG(INFO) << new_pts.size() << std::endl;
    return 0;
}
