/*
 * @Author: L.
 * @Date: 2022-05-18 11:58:11
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-06-09 11:28:30
 * @FilePath: /Mypcl/gen_voxel/example/common/binfile_tools.h
 * @Description:
 */
#pragma once

#include <string>
#include <memory>
// #include "base/rspointcloud.h"

namespace lidar {
namespace common {
namespace utils {

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum,
                 int point_feature_num);

int readBinFile(std::string& filename, void*& bufPtr, int point_feature_num);

class BinReader
{
public:
    BinReader(int num_feature) : num_feature_(num_feature){};
    ~BinReader(){};
    void read_from_file_rootpath(const std::string file_rootpath,
                                 float*& pc_output, int* point_num);

    // /**
    //  * @description: load bin-format pointcloud and convert to
    //  pcl::PointCloud,
    //  * do not filter points by nan
    //  * @param file_path[In], bin file path
    //  * @param pc_output[Out], the output
    //  */
    // void read_from_file_rootpath(const std::string file_rootpath,
    //                              pcl::PointCloud<RsPointXYZI>::Ptr
    //                              pc_output);
    void set_num_feature(int num_feature);

private:
    int read_bin_file(std::string file_path, int num_feature, void*& buffer_ptr,
                      int* point_num);

    int num_feature_;
};  // class BinReader

// class IReader
// {
// public:
//     IReader() = default;
//     virtual ~IReader() = 0;

//     virtual void read_from_file_rootpath(const std::string file_rootpath,
//                                          float*& pc_output, int* point_num) =
//                                          0;
//     virtual void read_from_file_rootpath(
//         const std::string file_rootpath,
//         pcl::PointCloud<RsPointXYZI>::Ptr pc_output) = 0;
// };

// inline IReader::~IReader(){};

// class BinReader : public IReader
// {
// public:
//     BinReader();
//     ~BinReader() override;
//     void read_from_file_rootpath(const std::string file_rootpath,
//                                  float*& pc_output, int* point_num) override;

//     /**
//      * @description: load bin-format pointcloud and convert to
//      pcl::PointCloud,
//      * do not filter points by nan
//      * @param file_path[In], bin file path
//      * @param pc_output[Out], the output
//      */
//     void read_from_file_rootpath(
//         const std::string file_rootpath,
//         pcl::PointCloud<RsPointXYZI>::Ptr pc_output) override;
//     void set_num_feature(int num_feature);

// private:
//     int read_bin_file(std::string file_path, int num_feature, void*&
//     buffer_ptr,
//                       int* point_num);

//     int num_feature_;
// };  // class BinReader

// enum class InputFileType {
//     BIN = 0,
//     // BAG = 1,
// };

// class ReaderFactory
// {
// public:
//     static std::shared_ptr<IReader> make(InputFileType file_type);
// };  // class ReaderFactory

}  // namespace utils
}  // namespace common
}  // namespace lidar