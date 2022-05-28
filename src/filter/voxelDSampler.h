/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-05-28 12:35:13
 * @FilePath: /Mypcl/src/filter/voxelDSampler.h
 * @Description:
 */
#pragma once
#include <vector>

#include "base/points.h"

namespace mypcl {
namespace filter {

enum class SampleType {
    CENTER = 0,
    RANDOM = 1,
};

class VoxelDSampleSort
{
public:
    VoxelDSampleSort(const std::vector<float>& grid_size, SampleType type);
    void downSample(const std::vector<base::PointXYZI>& s_points,
                    std::vector<base::PointXYZI>* t_points);
    void setVoxelSize(float x, float y, float z);
    void setSampleType(SampleType type);

private:
    std::vector<float> getMinMax(const std::vector<base::PointXYZI>& s_points);
    std::vector<size_t> getDims(const std::vector<float>& minmax);

    /**
     * @description:
     * @param indexes: coords, point index in source points list
     */
    void centerSample(const std::vector<base::PointXYZI>& s_points,
                      const std::vector<std::pair<size_t, size_t>>& indexes,
                      std::vector<base::PointXYZI>* t_points);

    void randomSample(const std::vector<base::PointXYZI>& s_points,
                      std::vector<std::pair<size_t, size_t>> indexes,
                      std::vector<base::PointXYZI>* t_points);

private:
    SampleType type_;
    std::vector<float> grid_size_;
};

inline void VoxelDSampleSort::setVoxelSize(float x, float y, float z)
{
    grid_size_[0] = x;
    grid_size_[1] = y;
    grid_size_[2] = z;
}

inline void VoxelDSampleSort::setSampleType(SampleType type) { type_ = type; }

}  // namespace filter
}  // namespace mypcl
