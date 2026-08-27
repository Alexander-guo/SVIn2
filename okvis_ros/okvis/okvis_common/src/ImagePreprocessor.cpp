/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *********************************************************************************/

#include <okvis/ImagePreprocessor.hpp>

#include <algorithm>
#include <stdexcept>

namespace okvis {

namespace {
cv::Mat asGray(const cv::Mat& image) {
  if (image.channels() == 1) return image;
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  return gray;
}
}  // namespace

std::vector<ImageRegionStatistics> computeImageGridStatistics(const cv::Mat& image,
                                                              int rows,
                                                              int cols) {
  if (image.empty()) throw std::invalid_argument("Cannot compute statistics for an empty image");
  if (rows <= 0 || cols <= 0) throw std::invalid_argument("Image statistics grid must be positive");

  const cv::Mat gray = asGray(image);
  std::vector<ImageRegionStatistics> result;
  result.reserve(static_cast<size_t>(rows * cols));
  for (int row = 0; row < rows; ++row) {
    const int y0 = row * gray.rows / rows;
    const int y1 = (row + 1) * gray.rows / rows;
    for (int col = 0; col < cols; ++col) {
      const int x0 = col * gray.cols / cols;
      const int x1 = (col + 1) * gray.cols / cols;
      const cv::Mat region = gray(cv::Range(y0, y1), cv::Range(x0, x1));
      cv::Scalar mean;
      cv::Scalar deviation;
      cv::meanStdDev(region, mean, deviation);
      const double pixels = static_cast<double>(region.total());
      ImageRegionStatistics statistics;
      statistics.mean = mean[0];
      statistics.standardDeviation = deviation[0];
      statistics.darkFraction = pixels > 0.0 ? cv::countNonZero(region < 20) / pixels : 0.0;
      statistics.saturatedFraction = pixels > 0.0 ? cv::countNonZero(region > 245) / pixels : 0.0;
      result.push_back(statistics);
    }
  }
  return result;
}

ImagePreprocessor::ImagePreprocessor(const HistogramParams& parameters) : parameters_(parameters) {
  if (parameters_.claheClipLimit <= 0.0) throw std::invalid_argument("CLAHE clip limit must be positive");
  if (parameters_.claheTilesGridSize <= 0) throw std::invalid_argument("CLAHE tile grid size must be positive");
  if (parameters_.histogramMethod == CLAHE) {
    clahe_ = cv::createCLAHE(parameters_.claheClipLimit,
                             cv::Size(parameters_.claheTilesGridSize, parameters_.claheTilesGridSize));
  }
}

void ImagePreprocessor::apply(const cv::Mat& input, cv::Mat& output) {
  if (input.empty()) throw std::invalid_argument("Cannot preprocess an empty image");
  if (input.type() != CV_8UC1) throw std::invalid_argument("ImagePreprocessor requires CV_8UC1 input");

  std::lock_guard<std::mutex> lock(mutex_);
  switch (parameters_.histogramMethod) {
    case CLAHE:
      clahe_->apply(input, output);
      break;
    case HISTOGRAM:
      cv::equalizeHist(input, output);
      break;
    case NONE:
    default:
      input.copyTo(output);
      break;
  }
}

}  // namespace okvis
