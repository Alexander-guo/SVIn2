#include <gtest/gtest.h>

#include <future>
#include <vector>

#include <okvis/ImagePreprocessor.hpp>

namespace {

cv::Mat unequalIlluminationImage() {
  cv::Mat image(128, 128, CV_8UC1);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      const int base = x < image.cols / 2 ? 20 : 150;
      image.at<unsigned char>(y, x) = static_cast<unsigned char>(std::min(255, base + (x + y) % 50));
    }
  }
  return image;
}

okvis::HistogramParams histogramParameters(okvis::HistogramMethod method) {
  okvis::HistogramParams parameters;
  parameters.histogramMethod = method;
  parameters.claheClipLimit = 5.0;
  parameters.claheTilesGridSize = 8;
  return parameters;
}

}  // namespace

TEST(ImagePreprocessor, PreservesNoneAndHistogramModes) {
  const cv::Mat input = unequalIlluminationImage();
  okvis::HistogramParams parameters = histogramParameters(okvis::NONE);
  okvis::ImagePreprocessor none(parameters);
  cv::Mat unchanged;
  none.apply(input, unchanged);
  EXPECT_EQ(cv::countNonZero(input != unchanged), 0);

  parameters.histogramMethod = okvis::HISTOGRAM;
  okvis::ImagePreprocessor histogram(parameters);
  cv::Mat equalized;
  histogram.apply(input, equalized);
  EXPECT_EQ(equalized.type(), input.type());
  EXPECT_EQ(equalized.size(), input.size());
  EXPECT_GT(cv::countNonZero(input != equalized), 0);
}

TEST(ImagePreprocessor, ClaheIsDeterministicAndDoesNotCreateHardTileSeams) {
  cv::Mat input(128, 128, CV_8UC1);
  for (int y = 0; y < input.rows; ++y) {
    for (int x = 0; x < input.cols; ++x) input.at<unsigned char>(y, x) = static_cast<unsigned char>(x);
  }
  okvis::HistogramParams parameters = histogramParameters(okvis::CLAHE);
  parameters.claheClipLimit = 2.0;
  parameters.claheTilesGridSize = 4;
  okvis::ImagePreprocessor preprocessor(parameters);
  cv::Mat first;
  cv::Mat second;
  preprocessor.apply(input, first);
  preprocessor.apply(input, second);
  EXPECT_EQ(cv::countNonZero(first != second), 0);

  double maximumBoundaryJump = 0.0;
  double maximumInteriorJump = 0.0;
  for (int x = 1; x < first.cols; ++x) {
    const double jump = cv::mean(cv::abs(first.col(x) - first.col(x - 1)))[0];
    if (x % 32 == 0) maximumBoundaryJump = std::max(maximumBoundaryJump, jump);
    else maximumInteriorJump = std::max(maximumInteriorJump, jump);
  }
  EXPECT_LE(maximumBoundaryJump, maximumInteriorJump + 2.0);
}

TEST(ImagePreprocessor, SupportsConcurrentCallsWithoutChangingResults) {
  const cv::Mat input = unequalIlluminationImage();
  okvis::HistogramParams parameters = histogramParameters(okvis::CLAHE);
  okvis::ImagePreprocessor preprocessor(parameters);
  cv::Mat reference;
  preprocessor.apply(input, reference);

  std::vector<std::future<cv::Mat>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.emplace_back(std::async(std::launch::async, [&preprocessor, &input]() {
      cv::Mat output;
      preprocessor.apply(input, output);
      return output;
    }));
  }
  for (auto& future : futures) EXPECT_EQ(cv::countNonZero(reference != future.get()), 0);
}

TEST(ImagePreprocessor, HandlesUniformAndNoisyImagesAcrossCameraInstances) {
  okvis::HistogramParams parameters = histogramParameters(okvis::CLAHE);
  parameters.claheClipLimit = 5.0;
  parameters.claheTilesGridSize = 8;
  okvis::ImagePreprocessor cameraZero(parameters);
  okvis::ImagePreprocessor cameraOne(parameters);

  const cv::Mat uniform(128, 128, CV_8UC1, cv::Scalar(24));
  cv::Mat noise(128, 128, CV_8UC1);
  cv::RNG random(42);
  random.fill(noise, cv::RNG::NORMAL, 24, 5);

  cv::Mat uniformOutput;
  cv::Mat noisyOutput;
  auto uniformFuture = std::async(std::launch::async, [&]() { cameraZero.apply(uniform, uniformOutput); });
  auto noisyFuture = std::async(std::launch::async, [&]() { cameraOne.apply(noise, noisyOutput); });
  uniformFuture.get();
  noisyFuture.get();

  EXPECT_EQ(uniformOutput.type(), CV_8UC1);
  EXPECT_EQ(uniformOutput.size(), uniform.size());
  EXPECT_EQ(noisyOutput.type(), CV_8UC1);
  EXPECT_EQ(noisyOutput.size(), noise.size());
  EXPECT_EQ(cv::countNonZero(uniformOutput != uniformOutput.at<unsigned char>(0, 0)), 0);
  EXPECT_GT(cv::countNonZero(noisyOutput != noisyOutput.at<unsigned char>(0, 0)), 0);
}

TEST(ImagePreprocessor, ReportsFourByFourRegionStatistics) {
  const auto statistics = okvis::computeImageGridStatistics(unequalIlluminationImage());
  ASSERT_EQ(statistics.size(), 16u);
  EXPECT_LT(statistics.front().mean, statistics.back().mean);
  EXPECT_GE(statistics.front().darkFraction, 0.0);
  EXPECT_LE(statistics.front().darkFraction, 1.0);
}
