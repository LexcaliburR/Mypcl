/*
 * @Author: Lexcaliburr lishiqi0111@gmail.com
 * @Date: 2022-05-27 22:47:21
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-05-28 12:33:19
 * @FilePath: /Mypcl/src/base/points.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置:
 */
#pragma once

#include <iostream>
namespace mypcl {
namespace base {

struct PointXYZI
{
    float x;
    float y;
    float z;
    float i;

private:
    friend std::ostream& operator<<(std::ostream& os, const PointXYZI& pt);
};

inline std::ostream& operator<<(std::ostream& os, const PointXYZI& pt)
{
    os << "x: " << pt.x << "  y: " << pt.y << " z: " << pt.z << " i: " << pt.i;
    return os;
}

}  // namespace base
}  // namespace mypcl