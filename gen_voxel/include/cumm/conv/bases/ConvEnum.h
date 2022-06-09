#pragma once
namespace cumm {
namespace conv {
namespace bases {
struct ConvEnum {
  enum class Mode {
    kConvolution = 0,
    kCrossCorrelation = 1,
  };
  enum class OpType {
    kForward = 0,
    kBackwardInput = 1,
    kBackwardWeight = 2,
  };
  enum class IterAlgo {
    kAnalytic = 0,
    kOptimized = 1,
  };
  enum class LayoutType {
    kChannelFirst = 0,
    kChannelLast = 1,
  };
};
} // namespace bases
} // namespace conv
} // namespace cumm