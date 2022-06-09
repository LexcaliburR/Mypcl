/*
 * @Author: L.
 * @Date: 2022-05-18 12:00:02
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-06-09 11:35:49
 * @FilePath: /Mypcl/gen_voxel/example/common/binfile_tools.cpp
 * @Description:
 */
#include <fstream>

#include <glog/logging.h>

// #include "colorfullogging.h"
#include "log_defs.h"
#include "binfile_tools.h"

namespace lidar {
namespace common {
namespace utils {

bool readBinFile(std::string& filename, void*& bufPtr, int& pointNum,
                 int point_feature_num)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        LOG(ERROR) << BOLDRED << "[Error] Open file " << filename << " failed"
                   << RESET;
        return false;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    bufPtr = malloc(fileSize);
    if (bufPtr == nullptr) {
        LOG(ERROR) << BOLDRED
                   << "[Error] Malloc Memory Failed! Size: " << fileSize
                   << RESET;
        return false;
    }
    // read the data:
    file.read((char*)bufPtr, fileSize);
    file.close();

    pointNum = fileSize / sizeof(float) / point_feature_num;
    if (fileSize / sizeof(float) % point_feature_num != 0) {
        LOG(ERROR) << BOLDRED << "[Error] File Size Error! " << fileSize
                   << RESET;
    }
    LOG(INFO) << "[INFO] pointNum : " << pointNum << std::endl;
    return true;
}

int readBinFile(std::string& filename, void*& bufPtr, int point_feature_num)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        LOG(ERROR) << BOLDRED << "[Error] Open file " << filename << " failed"
                   << RESET;
        return 0;
    }
    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    bufPtr = malloc(fileSize);
    if (bufPtr == nullptr) {
        LOG(ERROR) << BOLDRED
                   << "[Error] Malloc Memory Failed! Size: " << fileSize
                   << RESET;
        return 0;
    }
    // read the data:
    file.read((char*)bufPtr, fileSize);
    file.close();

    int pointNum = fileSize / sizeof(float) / point_feature_num;
    if (fileSize / sizeof(float) % point_feature_num != 0) {
        LOG(ERROR) << BOLDRED << "[Error] File Size Error! " << fileSize
                   << RESET;
    }
    LOG(INFO) << "[INFO] pointNum : " << pointNum << std::endl;
    return pointNum;
}

void BinReader::set_num_feature(int num_feature) { num_feature_ = num_feature; }

void BinReader::read_from_file_rootpath(const std::string file_path,
                                        float*& pc_output, int* point_num)
{
    void* buffer_ptr = nullptr;
    int result = read_bin_file(file_path, num_feature_, buffer_ptr, point_num);
    pc_output = static_cast<float*>(buffer_ptr);
    return;
}

// void BinReader::read_from_file_rootpath(
//     const std::string file_path, pcl::PointCloud<RsPointXYZI>::Ptr pc_output)
// {
//     float* points_array = nullptr;
//     int point_num = 0;
//     read_from_file_rootpath(file_path, points_array, &point_num);

//     for (int i = 0; i < point_num; i++) {
//         RsPointXYZI tmp = RsPointXYZI();
//         tmp.x = points_array[i * num_feature_];
//         tmp.y = points_array[i * num_feature_ + 1];
//         tmp.z = points_array[i * num_feature_ + 2];
//         // tmp.intensity = points_array[i * num_feature_ + 3];
//         tmp.intensity = (int)(points_array[i * num_feature_ + 3] * 255);
//         // std::cout << tmp.intensity << std::endl;
//         pc_output->push_back(tmp);
//     }
// }

int BinReader::read_bin_file(std::string file_path, int num_feature,
                             void*& buffer_ptr, int* point_num)
{
    std::streampos fileSize;
    std::ifstream file(file_path, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    buffer_ptr = malloc(fileSize);
    if (buffer_ptr == nullptr) {
        LOG(ERROR) << BOLDRED
                   << "[Error] Malloc Memory Failed! Size: " << fileSize
                   << RESET;
        return false;
    }

    // read the data:
    file.read((char*)buffer_ptr, fileSize);
    file.close();

    *point_num = fileSize / sizeof(float) / num_feature;
    if (fileSize / sizeof(float) % num_feature != 0) {
        LOG(ERROR) << BOLDRED << "[Error] File Size Error! " << fileSize
                   << RESET;
    }

    // for (int i = 0; i < 10; ++i) {
    //     std::cout << ((float*)buffer_ptr)[i * 5 + 0] << " "
    //               << ((float*)buffer_ptr)[i * 5 + 1] << " "
    //               << ((float*)buffer_ptr)[i * 5 + 2] << " "
    //               << ((float*)buffer_ptr)[i * 5 + 3] << " "
    //               << ((float*)buffer_ptr)[i * 5 + 3] << " "
    //               << ((int*)buffer_ptr)[i * 5 + 4] << std::endl;
    // }
    LOG(INFO) << "[INFO] point_num : " << *point_num;
    return true;
}

// std::shared_ptr<IReader> ReaderFactory::make(InputFileType file_type)
// {
//     if (file_type == InputFileType::BIN) {
//         auto reader = std::make_shared<BinReader>();
//         return reader;
//     } else {
//         std::cout << "unsupport file type " << std::endl;
//     }

//     return nullptr;
// }

}  // namespace utils
}  // namespace common
}  // namespace lidar