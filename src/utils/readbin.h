/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-05-28 12:20:55
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
int readPointBin(std::string path, T*& buffer, int feature_dim, int use_dim)
{
    char* buffer_ptr = 0;

    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    std::streampos file_sizes = file.tellg();
    file.seekg(0, std::ios::beg);

    buffer_ptr = (char*)malloc(file_sizes);

    if (buffer_ptr == nullptr) {
        LOG(WARNING) << YELLOW << "empty file !" << RESET;
    }

    file.read(buffer_ptr, file_sizes);
    file.close();

    if (file_sizes / sizeof(T) % feature_dim != 0) {
        LOG(ERROR) << RED << "file size error!" << RESET;
        return 0;
    }

    int point_num = file_sizes / sizeof(T) / feature_dim;

    buffer = new T[point_num * use_dim];
    T* data = (T*)buffer_ptr;
    for (int i = 0; i < point_num; ++i) {
        for (int j = 0; j < use_dim; ++j) {
            buffer[i * use_dim + j] = data[i * feature_dim + j];
        }
    }

    delete buffer_ptr;
    return point_num;
};

}  // namespace utils
}  // namespace mypcl