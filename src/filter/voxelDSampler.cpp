/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: lishiqi
 * @LastEditTime: 2022-05-28 04:56:35
 * @FilePath: /Mypcl/src/filter/voxelDSampler.cpp
 * @Description:
 */
#include "voxelDSampler.h"

#include <algorithm>
#include <math.h>

#include "utils/colorfullogging.h"
namespace mypcl {
namespace filter {

VoxelDSampleSort::VoxelDSampleSort(const std::vector<float>& grid_size,
                                   SampleType type = SampleType::RANDOM)
    : grid_size_(grid_size), type_(type)
{}

void VoxelDSampleSort::downSample(const std::vector<base::PointXYZI>& s_points,
                                  std::vector<base::PointXYZI>* t_points)
{
    // 1.get min/max
    std::vector<float> limit = getMinMax(s_points);

    // 2.get dim
    std::vector<size_t> dims = getDims(limit);

    // 3.compute index
    // std::vector<size_t> indexs(s_points.size(), 0);
    std::vector<std::pair<size_t, size_t>> indexs;
    for (size_t i = 0; i < s_points.size(); ++i) {
        size_t idx_x = s_points[i].x / grid_size_[0];
        size_t idx_y = s_points[i].y / grid_size_[1];
        size_t idx_z = s_points[i].z / grid_size_[2];

        size_t coord = idx_x + idx_y * dims[0] + idx_z * dims[0] * dims[1];
        auto index = std::make_pair(coord, i);
        indexs.push_back(index);
    }

    // 4.sort
    std::sort(indexs.begin(),
              indexs.end(),
              [](const std::pair<size_t, size_t>& before,
                 const std::pair<size_t, size_t>& after) {
                  return before.first < after.first;
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
    const std::vector<base::PointXYZI>& s_points)
{
    std::vector<float> minmax = {
        1000.0f, 1000.0f, 1000.0f, -1000.0f, -1000.0f, -1000.0f};
    auto& min_x = minmax[0];
    auto& min_y = minmax[1];
    auto& min_z = minmax[2];
    auto& max_x = minmax[3];
    auto& max_y = minmax[4];
    auto& max_z = minmax[5];

    for (const auto& pt : s_points) {
        if (min_x > pt.x) min_x = pt.x;
        if (min_y > pt.y) min_y = pt.y;
        if (min_y > pt.z) min_z = pt.z;
        if (max_x < pt.x) max_x = pt.x;
        if (max_y < pt.y) max_y = pt.y;
        if (max_z < pt.z) max_z = pt.z;
    }

    return minmax;
}

std::vector<size_t> VoxelDSampleSort::getDims(const std::vector<float>& minmax)
{
    std::vector<size_t> dims(3, 0);
    dims[0] = (minmax[3] - minmax[0]) / grid_size_[0];
    dims[1] = (minmax[4] - minmax[1]) / grid_size_[1];
    dims[2] = (minmax[5] - minmax[2]) / grid_size_[2];
    return dims;
}

void VoxelDSampleSort::centerSample(
    const std::vector<base::PointXYZI>& s_points,
    const std::vector<std::pair<size_t, size_t>>& indexes,
    std::vector<base::PointXYZI>* t_points)
{
    if (!t_points) {
        *t_points = std::vector<base::PointXYZI>();
    } else if (!t_points->empty()) {
        t_points->clear();
    }

    size_t pre = 0;
    float cx = 0;
    float cy = 0;
    float cz = 0;
    float ci = 0;
    size_t num_pt_in_voxel = 1;
    for (size_t i = 1; i < indexes.size(); ++i) {
        if (indexes[i].first == indexes[pre].first) {
            cx += s_points[indexes[i].second].x;
            cy += s_points[indexes[i].second].y;
            cz += s_points[indexes[i].second].z;
            ci += s_points[indexes[i].second].i;
            ++num_pt_in_voxel;
        } else {
            t_points->emplace_back(base::PointXYZI{cx / num_pt_in_voxel,
                                                   cy / num_pt_in_voxel,
                                                   cz / num_pt_in_voxel,
                                                   ci / num_pt_in_voxel});
            cx = 0;
            cy = 0;
            cz = 0;
            ci = 0;
            num_pt_in_voxel = 1;
        }
        pre = i;
    }
}

void VoxelDSampleSort::randomSample(
    const std::vector<base::PointXYZI>& s_points,
    std::vector<std::pair<size_t, size_t>> indexes,
    std::vector<base::PointXYZI>* t_points)
{
    if (!t_points) {
        *t_points = std::vector<base::PointXYZI>();
    } else if (!t_points->empty()) {
        t_points->clear();
    }

    size_t pre = 0;
    for (size_t i = 1; i < indexes.size(); ++i) {
        if (indexes[i].first == indexes[pre].first) {
            continue;
        } else {
            t_points->emplace_back(s_points[indexes[i].second]);
        }
        pre = i;
    }
}

}  // namespace filter
}  // namespace mypcl