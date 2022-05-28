#pragma once

#include <chrono>
#include <string>

#include "colorfullogging.h"

namespace mypcl {
namespace utils {

class TicToc
{
public:
    explicit TicToc(bool sync_cuda = false) : sync_cuda_(sync_cuda) {}
    void tic() { t_ = std::chrono::steady_clock::now(); };
    void toc(const std::string& remark = "")
    {
        // if(sync_cuda_) cudaDevicesSynchronize();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t_);
        double ms = double(duration.count()) *
                    std::chrono::milliseconds::period::num /
                    std::chrono::microseconds::period::den * 1000;

        if (!remark.empty()) {
            LOG(INFO) << remark << ":" << ms << "ms";
        }
    };

private:
    bool sync_cuda_;
    std::chrono::steady_clock::time_point t_;
};

}  // namespace utils

#define TIME_START(sync_cuda)          \
    utils::TicToc _tictoc_(sync_cuda); \
    _tictoc_.tic();

#define TIME_END(remark)  \
    _tictoc_.toc(remark); \
    _tictoc_.tic();

}  // namespace mypcl