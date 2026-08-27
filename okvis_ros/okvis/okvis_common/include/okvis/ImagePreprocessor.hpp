/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *********************************************************************************/

#ifndef INCLUDE_OKVIS_IMAGEPREPROCESSOR_HPP_
#define INCLUDE_OKVIS_IMAGEPREPROCESSOR_HPP_

#include <mutex>
#include <vector>

#include <okvis/Parameters.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace okvis {

struct ImageRegionStatistics {
  double mean = 0.0;
  double standardDeviation = 0.0;
  double darkFraction = 0.0;
  double saturatedFraction = 0.0;
};

/// Compute row-major image statistics for a regular grid.
std::vector<ImageRegionStatistics> computeImageGridStatistics(const cv::Mat& image,
                                                              int rows = 4,
                                                              int cols = 4);

/// Thread-safe production image contrast preprocessor.
class ImagePreprocessor {
 public:
  explicit ImagePreprocessor(const HistogramParams& parameters);

  /// Apply the configured operation. Input and output are 8-bit single-channel images.
  void apply(const cv::Mat& input, cv::Mat& output);

  const HistogramParams& parameters() const { return parameters_; }

 private:
  HistogramParams parameters_;
  cv::Ptr<cv::CLAHE> clahe_;
  std::mutex mutex_;
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_IMAGEPREPROCESSOR_HPP_
