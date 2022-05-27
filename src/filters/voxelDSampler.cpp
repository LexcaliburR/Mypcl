/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-05-28 00:11:25
 * @FilePath: /Mypcl/src/filters/voxelDSampler.cpp
 * @Description:
 */
#include "voxelDSampler.h"

#include <algorithm>

#include "utils/colorfullogging.h"
namespace mypcl {
namespace filter {

VoxelDSampleSort::VoxelDSampleSort(const std::vector<float>& grid_size,
                                   SampleType type = SampleType::RANDOM)
    : grid_size_(grid_size), type_(type)
{}

void VoxelDSampleSort::downSample(const std::vector<PointXYZI>& s_points,
                                  std::vector<PointXYZI>* t_points)
{
    // 1.get min/max
    std::vector<float> limit = getMinMax(s_points);

    // 2.get dim
    std::vector<size_t> dims = getDims(s_points);

    // 3.compute index
    const auto& dim_x = dims[0];
    const auto& dim_y = dims[1];
    const auto& dim_z = dims[2];

    // std::vector<size_t> indexs(s_points.size(), 0);
    std::vector<std::pair<size_t, size_t>> indexs;
    for (size_t i = 0; i < s_points.size(); ++i) {
        size_t idx_x = s_points[i].x / dim_x;
        size_t idx_y = s_points[i].y / dim_y;
        size_t idx_z = s_points[i].z / dim_z;

        size_t coord = idx_x + idx_y * dim_x + idx_z * dim_y * dim_z;
        auto index = std::make_pair(coord, i);
        indexs.push_back(index);
    }

    // 4.sort
    std::sort(indexs.begin(),
              indexs.end(),
              [](const std::pair<size_t, size_t>& before,
                 const std::pair<size_t, size_t>& after) {
                  before.first < after.first;
              });

    // 5.downsample centerwise or randomwise
    if (type_ == SampleType::CENTER) {
        centerSample(s_points, indexs, t_points);
    } else if (type_ == SampleType::RANDOM) {
        randomSample(s_points, indexs, t_points);
    } else {
        LOG(ERROR) << BOLDRED << "do not supported sample type! " << RESET;
    }
    return;
}

std::vector<float> VoxelDSampleSort::getMinMax(
    const std::vector<PointXYZI>& grid_size)
{}
std::vector<size_t> VoxelDSampleSort::getDims(
    const std::vector<PointXYZI>& grid_size)
{}
void VoxelDSampleSort::centerSample(
    const std::vector<PointXYZI>& s_points,
    std::vector<std::pair<size_t, size_t>> indexes,
    std::vector<PointXYZI>* t_points)
{}
void VoxelDSampleSort::randomSample(
    const std::vector<PointXYZI>& s_points,
    std::vector<std::pair<size_t, size_t>> indexes,
    std::vector<PointXYZI>* t_points)
{}

}  // namespace filter
}  // namespace mypcl