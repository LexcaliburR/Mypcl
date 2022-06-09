/*
 * @Author: lishiqi
 * @Date: 2021-11-22 11:28:30
 * @LastEditors: Lexcaliburr
 * @LastEditTime: 2022-06-09 11:45:35
 */
#pragma once

#include <iostream>
#include <string>

// #include "colorfullogging.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_ROS
#include <ros/time.h>
#else
#include <chrono>
#endif

namespace lidar {
namespace common {
namespace utils {
class TicToc
{
public:
    explicit TicToc(bool sync_cuda = false) : sync_cuda_(sync_cuda)
    {
#ifdef USE_ROS
        ros::Time::init();
#else

#endif
    }

    void Tic()
    {
#ifdef USE_CUDA
        if (sync_cuda_) cudaDeviceSynchronize();
#endif

#ifdef USE_ROS
        t_ = ros::Time::now().toSec();
#else
        t_ = std::chrono::steady_clock::now();
#endif
    }

    double Toc(const std::string& remark = "")
    {
#ifdef USE_CUDA
        if (sync_cuda_) cudaDeviceSynchronize();
#endif

#ifdef USE_ROS
        double ms = (ros::Time::now().toSec() - t_) * 1000;
#else
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_);
        double ms = double(duration.count()) *
                    std::chrono::microseconds::period::num /
                    std::chrono::microseconds::period::den * 1000;
#endif

        if (!remark.empty()) {
            // std::cout << remark << ":" << ms << "ms" << std::endl;
            LOG(INFO) << remark << ":" << ms << "ms";
        }

        return ms;
    }

private:
#ifdef USE_ROS
    double t_;
#else
    std::chrono::steady_clock::time_point t_;
#endif
    bool sync_cuda_;
};

}  // namespace utils
}  // namespace common
}  // namespace lidar

#define PERF_BLOCK_START(sync_cuda)                   \
    lidar::common::utils::TicToc _tictoc_(sync_cuda); \
    _tictoc_.Tic();

#define PERF_BLOCK_END(indicator) \
    _tictoc_.Toc(indicator);      \
    _tictoc_.Tic();
