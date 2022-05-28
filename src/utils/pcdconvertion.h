/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:20:06
 * @LastEditors: lishiqi
 * @LastEditTime: 2022-05-28 04:52:03
 * @FilePath: /Mypcl/src/utils/pcdconvertion.h
 * @Description:
 */
#pragma once
#include <fstream>
#include <string>
#include <vector>
#include "colorfullogging.h"
#include "readbin.h"
#include "base/points.h"

namespace mypcl {
namespace utils {

std::vector<base::PointXYZI> arrToVecPoint(float* pc_in, const int point_num)
{
    std::vector<base::PointXYZI> ret(point_num);
    for (int i = 0; i < point_num; ++i) {
        ret[i].x = pc_in[i * 4 + 0];
        ret[i].y = pc_in[i * 4 + 1];
        ret[i].z = pc_in[i * 4 + 2];
        ret[i].i = pc_in[i * 4 + 3];
    }
    return ret;
}

}  // namespace utils
}  // namespace mypcl