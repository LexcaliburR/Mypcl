/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-05-28 00:10:41
 * @FilePath: /Mypcl/src/filters/voxelDSampler.h
 * @Description:
 */
#include <vector>

#include "base/points.h"
namespace mypcl {
namespace filter {

using namespace mypcl::base;

enum class SampleType {
    CENTER = 0,
    RANDOM = 1,
};

class VoxelDSampleSort
{
public:
    VoxelDSampleSort(const std::vector<float>& grid_size, SampleType type);
    void downSample(const std::vector<PointXYZI>& s_points,
                    std::vector<PointXYZI>* t_points);
    void setVoxelSize(float x, float y, float z);
    void setSampleType(SampleType type);

private:
    std::vector<float> getMinMax(const std::vector<PointXYZI>& grid_size);
    std::vector<size_t> getDims(const std::vector<PointXYZI>& grid_size);
    void centerSample(const std::vector<PointXYZI>& s_points,
                      std::vector<std::pair<size_t, size_t>> indexes,
                      std::vector<PointXYZI>* t_points);
    void randomSample(const std::vector<PointXYZI>& s_points,
                      std::vector<std::pair<size_t, size_t>> indexes,
                      std::vector<PointXYZI>* t_points);

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
