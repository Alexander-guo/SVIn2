#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <brisk/brisk.h>
#include <okvis/SpatialKeypointBalancer.hpp>
#include <okvis/cameras/DoubleSphereCamera.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <opencv2/imgproc.hpp>

namespace {

std::vector<cv::KeyPoint> imbalancedCandidates() {
  std::vector<cv::KeyPoint> candidates;
  for (int i = 0; i < 100; ++i) {
    candidates.emplace_back(cv::Point2f(10.0f + i % 30, 10.0f + i / 30), 8.0f, 0.0f, 1000.0f - i);
  }
  for (int cell = 1; cell < 4; ++cell) {
    const int col = cell % 2;
    const int row = cell / 2;
    for (int i = 0; i < 25; ++i) {
      candidates.emplace_back(cv::Point2f(60.0f * col + 10.0f + i % 10,
                                          60.0f * row + 10.0f + i / 10),
                              8.0f, 0.0f, 100.0f - i);
    }
  }
  return candidates;
}

cv::Mat darkAndBrightCheckerboard() {
  cv::Mat image(256, 256, CV_8UC1);
  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 2; ++col) {
      const bool dark = row == 0 && col == 0;
      const int low = dark ? 15 : 120;
      const int high = dark ? 55 : 230;
      for (int y = row * 128; y < (row + 1) * 128; ++y) {
        for (int x = col * 128; x < (col + 1) * 128; ++x) {
          image.at<unsigned char>(y, x) = ((x / 12 + y / 12) % 2 == 0) ? low : high;
        }
      }
    }
  }
  return image;
}

}  // namespace

TEST(SpatialKeypointBalancer, ReservesEachCellAndKeepsFinalBudget) {
  okvis::SpatialBalancingParams parameters;
  parameters.enable = true;
  parameters.rows = 2;
  parameters.cols = 2;
  parameters.minimumPerValidCell = 10;
  const auto result = okvis::balanceSpatialKeypoints(
      imbalancedCandidates(), nullptr, cv::Size(120, 120), 80, parameters);
  ASSERT_EQ(result.keypoints.size(), 80u);
  ASSERT_EQ(result.selectedPerCell.size(), 4u);
  for (size_t count : result.selectedPerCell) EXPECT_GE(count, 10u);
}

TEST(SpatialKeypointBalancer, IsDeterministicForReversedInput) {
  okvis::SpatialBalancingParams parameters;
  parameters.enable = true;
  parameters.rows = 2;
  parameters.cols = 2;
  parameters.minimumPerValidCell = 10;
  std::vector<cv::KeyPoint> forward = imbalancedCandidates();
  std::vector<cv::KeyPoint> reverse = forward;
  std::reverse(reverse.begin(), reverse.end());
  const auto first = okvis::balanceSpatialKeypoints(forward, nullptr, cv::Size(120, 120), 80, parameters);
  const auto second = okvis::balanceSpatialKeypoints(reverse, nullptr, cv::Size(120, 120), 80, parameters);
  ASSERT_EQ(first.keypoints.size(), second.keypoints.size());
  for (size_t index = 0; index < first.keypoints.size(); ++index) {
    EXPECT_EQ(first.keypoints[index].pt, second.keypoints[index].pt);
    EXPECT_EQ(first.keypoints[index].response, second.keypoints[index].response);
  }
}

TEST(SpatialKeypointBalancer, SharesAnOvercommittedReserveAcrossCells) {
  okvis::SpatialBalancingParams parameters;
  parameters.enable = true;
  parameters.rows = 2;
  parameters.cols = 2;
  parameters.minimumPerValidCell = 10;
  const auto result = okvis::balanceSpatialKeypoints(
      imbalancedCandidates(), nullptr, cv::Size(120, 120), 6, parameters);
  ASSERT_EQ(result.keypoints.size(), 6u);
  ASSERT_EQ(result.selectedPerCell.size(), 4u);
  EXPECT_EQ(*std::max_element(result.selectedPerCell.begin(), result.selectedPerCell.end()) -
                *std::min_element(result.selectedPerCell.begin(), result.selectedPerCell.end()),
            1u);
}

TEST(SpatialKeypointBalancer, RejectsInvalidDoubleSphereCandidates) {
  using Camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;
  std::shared_ptr<const okvis::cameras::CameraBase> camera = std::make_shared<Camera>(
      624, 624, 230.0, 230.0, 312.0, 312.0, 0.0, 0.8, okvis::cameras::NoDistortion::testObject());
  std::vector<cv::KeyPoint> candidates;
  candidates.emplace_back(cv::Point2f(312.0f, 312.0f), 8.0f, 0.0f, 10.0f);
  candidates.emplace_back(cv::Point2f(0.0f, 0.0f), 8.0f, 0.0f, 20.0f);
  okvis::SpatialBalancingParams parameters;
  const auto result = okvis::balanceSpatialKeypoints(candidates, camera, cv::Size(624, 624), 10, parameters);
  EXPECT_EQ(result.keypoints.size(), 1u);
  EXPECT_EQ(result.invalidCandidates, 1u);
}

TEST(SpatialKeypointBalancer, LocalDetectionCoversDarkAndBrightCellsWithinBudget) {
  const cv::Mat image = darkAndBrightCheckerboard();
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> detector(5.0, 0, 1.0, 1000);
  okvis::SpatialBalancingParams parameters;
  parameters.enable = true;
  parameters.rows = 2;
  parameters.cols = 2;
  parameters.candidateMultiplier = 3;
  parameters.minimumPerValidCell = 4;
  parameters.localPaddingPixels = 24;

  const auto result =
      okvis::detectAndBalanceSpatialKeypoints(image, detector, nullptr, 40, parameters);
  EXPECT_LE(result.keypoints.size(), 40u);
  ASSERT_EQ(result.rawDetectionsPerCell.size(), 4u);
  ASSERT_EQ(result.candidatesPerCell.size(), 4u);
  ASSERT_EQ(result.selectedPerCell.size(), 4u);
  for (size_t cell = 0; cell < 4; ++cell) {
    EXPECT_GT(result.rawDetectionsPerCell[cell], 0u) << "cell " << cell;
    EXPECT_GT(result.candidatesPerCell[cell], 0u) << "cell " << cell;
    EXPECT_GE(result.selectedPerCell[cell], 4u) << "cell " << cell;
  }
}

TEST(SpatialKeypointBalancer, LocalDetectionIsDeterministicAndOwnsEachPointOnce) {
  const cv::Mat image = darkAndBrightCheckerboard();
  okvis::SpatialBalancingParams parameters;
  parameters.enable = true;
  parameters.rows = 2;
  parameters.cols = 2;
  parameters.minimumPerValidCell = 4;
  parameters.localPaddingPixels = 32;
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> firstDetector(5.0, 0, 1.0, 1000);
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> secondDetector(5.0, 0, 1.0, 1000);
  const auto first =
      okvis::detectAndBalanceSpatialKeypoints(image, firstDetector, nullptr, 50, parameters);
  const auto second =
      okvis::detectAndBalanceSpatialKeypoints(image, secondDetector, nullptr, 50, parameters);
  ASSERT_EQ(first.keypoints.size(), second.keypoints.size());
  for (size_t index = 0; index < first.keypoints.size(); ++index) {
    EXPECT_FLOAT_EQ(first.keypoints[index].pt.x, second.keypoints[index].pt.x);
    EXPECT_FLOAT_EQ(first.keypoints[index].pt.y, second.keypoints[index].pt.y);
    EXPECT_FLOAT_EQ(first.keypoints[index].response, second.keypoints[index].response);
  }
  const size_t selectedTotal =
      std::accumulate(first.selectedPerCell.begin(), first.selectedPerCell.end(), size_t{0});
  EXPECT_EQ(selectedTotal, first.keypoints.size());
  for (size_t lhs = 0; lhs < first.keypoints.size(); ++lhs) {
    for (size_t rhs = lhs + 1; rhs < first.keypoints.size(); ++rhs) {
      EXPECT_TRUE(first.keypoints[lhs].pt != first.keypoints[rhs].pt);
    }
  }
}

TEST(SpatialKeypointBalancer, LocalDetectionRejectsInvalidCameraAndMaskedRegions) {
  using Camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;
  auto camera = std::make_shared<Camera>(
      256, 256, 100.0, 100.0, 128.0, 128.0, 0.0, 0.8, okvis::cameras::NoDistortion::testObject());
  cv::Mat cameraMask = cv::Mat::zeros(256, 256, CV_8UC1);
  cv::rectangle(cameraMask, cv::Rect(64, 64, 64, 64), cv::Scalar(255), cv::FILLED);
  ASSERT_TRUE(camera->setMask(cameraMask));

  const cv::Mat image = darkAndBrightCheckerboard();
  brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator> detector(5.0, 0, 1.0, 1000);
  okvis::SpatialBalancingParams parameters;
  parameters.enable = true;
  parameters.rows = 2;
  parameters.cols = 2;
  parameters.minimumPerValidCell = 4;
  parameters.localPaddingPixels = 24;
  const auto result =
      okvis::detectAndBalanceSpatialKeypoints(image, detector, camera, 80, parameters);

  EXPECT_GT(result.invalidCandidates, 0u);
  EXPECT_LE(result.keypoints.size(), 80u);
  for (const cv::KeyPoint& keypoint : result.keypoints) {
    Eigen::Vector3d direction;
    EXPECT_TRUE(camera->backProject(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), &direction));
    EXPECT_EQ(cameraMask.at<unsigned char>(static_cast<int>(keypoint.pt.y),
                                           static_cast<int>(keypoint.pt.x)),
              0);
  }
}

TEST(SpatialKeypointBalancer, MeasuresBoundaryDensity) {
  std::vector<cv::KeyPoint> boundary;
  for (int y = 10; y < 90; y += 10) boundary.emplace_back(cv::Point2f(50.0f, static_cast<float>(y)), 8.0f);
  EXPECT_GT(okvis::keypointBoundaryDensityRatio(boundary, cv::Size(100, 100), 2), 1.0);
}
