/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: Lexcaliburr lishiqi0111@gmail.com
 * @LastEditTime: 2022-05-27 22:59:38
 * @FilePath: /Mypcl/src/utils/readbin.h
 * @Description:
 */
#pragma once
#include <fstream>
#include <string>
#include "colorfullogging.h"

namespace mypcl {
namespace utils {

template <class T>
int readPointBin(std::string path, T* buffer, int feature_dim, int use_dim)
{
    *buffer = 0;

    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    std::streampos file_sizes = file.tellg();
    file.seekg(0, std::ios::beg);

    buffer_ptr = malloc(fileSize);

    if (buffer_ptr == nullptr) {
        LOG(WARNING) << YELLOW << "empty file !" << RESET;
    }

    file.read((char*))buffer, file_sizes);
    file.close();

    if (file_sizes / sizeof(T) % feature_dim != 0) {
        LOG(ERROR) << RED << "file size error!" << RESET;
        return 0
    }
    int point_num = file_sizes / sizeof(T) / feature_dim;
    return point_num;
}

}  // namespace utils
}  // namespace mypcl