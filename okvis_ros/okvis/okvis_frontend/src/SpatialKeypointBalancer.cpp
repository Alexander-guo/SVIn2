/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *********************************************************************************/

#include <okvis/SpatialKeypointBalancer.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

#include <opencv2/imgproc.hpp>

namespace okvis {

namespace {
bool strongerKeypoint(const cv::KeyPoint& lhs, const cv::KeyPoint& rhs) {
  if (lhs.response != rhs.response) return lhs.response > rhs.response;
  return std::tie(lhs.pt.x, lhs.pt.y, lhs.size, lhs.angle, lhs.octave, lhs.class_id) <
         std::tie(rhs.pt.x, rhs.pt.y, rhs.size, rhs.angle, rhs.octave, rhs.class_id);
}

int cellIndex(const cv::KeyPoint& keypoint, const cv::Size& size, int rows, int cols) {
  if (!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y) || keypoint.pt.x < 0.0f ||
      keypoint.pt.y < 0.0f || keypoint.pt.x >= size.width || keypoint.pt.y >= size.height) {
    return -1;
  }
  const int col = std::min(cols - 1, static_cast<int>(keypoint.pt.x * cols / size.width));
  const int row = std::min(rows - 1, static_cast<int>(keypoint.pt.y * rows / size.height));
  return row * cols + col;
}

bool cameraValid(const cv::KeyPoint& keypoint,
                 const std::shared_ptr<const cameras::CameraBase>& geometry) {
  if (!geometry) return true;
  Eigen::Vector3d direction;
  if (!geometry->backProject(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), &direction) || !direction.allFinite()) {
    return false;
  }
  if (geometry->hasMask()) {
    const int x = static_cast<int>(keypoint.pt.x);
    const int y = static_cast<int>(keypoint.pt.y);
    if (x < 0 || y < 0 || x >= geometry->mask().cols || y >= geometry->mask().rows ||
        geometry->mask().at<unsigned char>(y, x) != 0) {
      return false;
    }
  }
  return true;
}

cv::Rect cellRectangle(const cv::Size& imageSize, int row, int col, int rows, int cols) {
  const int x0 = col * imageSize.width / cols;
  const int x1 = (col + 1) * imageSize.width / cols;
  const int y0 = row * imageSize.height / rows;
  const int y1 = (row + 1) * imageSize.height / rows;
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

cv::Rect paddedRectangle(const cv::Rect& cell, const cv::Size& imageSize, int padding) {
  const int x0 = std::max(0, cell.x - padding);
  const int y0 = std::max(0, cell.y - padding);
  const int x1 = std::min(imageSize.width, cell.x + cell.width + padding);
  const int y1 = std::min(imageSize.height, cell.y + cell.height + padding);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}
}  // namespace

std::vector<size_t> countSpatialKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                          const cv::Size& imageSize,
                                          int rows,
                                          int cols) {
  std::vector<size_t> counts(static_cast<size_t>(std::max(0, rows * cols)), 0);
  if (rows <= 0 || cols <= 0 || imageSize.width <= 0 || imageSize.height <= 0) return counts;
  for (const cv::KeyPoint& keypoint : keypoints) {
    const int cell = cellIndex(keypoint, imageSize, rows, cols);
    if (cell >= 0) ++counts[static_cast<size_t>(cell)];
  }
  return counts;
}

SpatialKeypointBalanceResult balanceSpatialKeypoints(
    const std::vector<cv::KeyPoint>& candidates,
    const std::shared_ptr<const cameras::CameraBase>& geometry,
    const cv::Size& imageSize,
    size_t maximumKeypoints,
    const SpatialBalancingParams& parameters) {
  SpatialKeypointBalanceResult result;
  const int cellCount = parameters.rows * parameters.cols;
  result.candidatesPerCell.assign(static_cast<size_t>(cellCount), 0);
  result.rawDetectionsPerCell = countSpatialKeypoints(candidates, imageSize, parameters.rows, parameters.cols);
  result.selectedPerCell.assign(static_cast<size_t>(cellCount), 0);
  if (cellCount <= 0 || imageSize.width <= 0 || imageSize.height <= 0 || maximumKeypoints == 0) return result;

  std::vector<std::vector<cv::KeyPoint>> cells(static_cast<size_t>(cellCount));
  for (const cv::KeyPoint& keypoint : candidates) {
    const int cell = cellIndex(keypoint, imageSize, parameters.rows, parameters.cols);
    if (cell < 0 || !std::isfinite(keypoint.response) || !cameraValid(keypoint, geometry)) {
      ++result.invalidCandidates;
      continue;
    }
    cells[static_cast<size_t>(cell)].push_back(keypoint);
  }
  for (size_t cell = 0; cell < cells.size(); ++cell) {
    std::sort(cells[cell].begin(), cells[cell].end(), strongerKeypoint);
    result.candidatesPerCell[cell] = cells[cell].size();
  }

  result.keypoints.reserve(std::min(maximumKeypoints, candidates.size()));
  std::vector<size_t> consumed(cells.size(), 0);
  for (size_t level = 0;
       level < static_cast<size_t>(parameters.minimumPerValidCell) && result.keypoints.size() < maximumKeypoints;
       ++level) {
    for (size_t cell = 0; cell < cells.size() && result.keypoints.size() < maximumKeypoints; ++cell) {
      if (level < cells[cell].size()) {
        result.keypoints.push_back(cells[cell][level]);
        ++result.selectedPerCell[cell];
        ++consumed[cell];
      }
    }
  }

  struct RemainingCandidate {
    cv::KeyPoint keypoint;
    size_t cell = 0;
  };
  std::vector<RemainingCandidate> remaining;
  for (size_t cell = 0; cell < cells.size(); ++cell) {
    for (size_t index = consumed[cell]; index < cells[cell].size(); ++index) {
      remaining.push_back({cells[cell][index], cell});
    }
  }
  std::sort(remaining.begin(), remaining.end(), [](const auto& lhs, const auto& rhs) {
    if (strongerKeypoint(lhs.keypoint, rhs.keypoint)) return true;
    if (strongerKeypoint(rhs.keypoint, lhs.keypoint)) return false;
    return lhs.cell < rhs.cell;
  });
  for (const RemainingCandidate& candidate : remaining) {
    if (result.keypoints.size() >= maximumKeypoints) break;
    result.keypoints.push_back(candidate.keypoint);
    ++result.selectedPerCell[candidate.cell];
  }
  return result;
}

SpatialKeypointBalanceResult detectAndBalanceSpatialKeypoints(
    const cv::Mat& image,
    cv::FeatureDetector& detector,
    const std::shared_ptr<const cameras::CameraBase>& geometry,
    size_t maximumKeypoints,
    const SpatialBalancingParams& parameters) {
  SpatialKeypointBalanceResult empty;
  const int cellCount = parameters.rows * parameters.cols;
  empty.rawDetectionsPerCell.assign(static_cast<size_t>(std::max(0, cellCount)), 0);
  empty.candidatesPerCell.assign(static_cast<size_t>(std::max(0, cellCount)), 0);
  empty.selectedPerCell.assign(static_cast<size_t>(std::max(0, cellCount)), 0);
  if (image.empty() || cellCount <= 0 || maximumKeypoints == 0) return empty;

  const size_t multiplier = static_cast<size_t>(std::max(1, parameters.candidateMultiplier));
  const size_t targetCandidateCount =
      maximumKeypoints > std::numeric_limits<size_t>::max() / multiplier
          ? std::numeric_limits<size_t>::max()
          : maximumKeypoints * multiplier;
  const size_t perCellCandidateCap = std::max(
      static_cast<size_t>(std::max(0, parameters.minimumPerValidCell)),
      (targetCandidateCount + static_cast<size_t>(cellCount) - 1) / static_cast<size_t>(cellCount));

  std::vector<cv::KeyPoint> candidates;
  candidates.reserve(std::min(targetCandidateCount,
                              static_cast<size_t>(cellCount) * perCellCandidateCap));
  for (int row = 0; row < parameters.rows; ++row) {
    for (int col = 0; col < parameters.cols; ++col) {
      const size_t cell = static_cast<size_t>(row * parameters.cols + col);
      const cv::Rect ownership = cellRectangle(image.size(), row, col, parameters.rows, parameters.cols);
      if (ownership.empty()) continue;
      const cv::Rect padded = paddedRectangle(ownership, image.size(), parameters.localPaddingPixels);
      // BRISK requires continuous input. Padding supplies detector context;
      // ownership filtering below prevents duplicates and artificial ROI-edge features.
      const cv::Mat localImage = image(padded).clone();
      std::vector<cv::KeyPoint> localKeypoints;
      detector.detect(localImage, localKeypoints);
      std::vector<cv::KeyPoint> validOwned;
      validOwned.reserve(localKeypoints.size());
      for (cv::KeyPoint keypoint : localKeypoints) {
        keypoint.pt.x += static_cast<float>(padded.x);
        keypoint.pt.y += static_cast<float>(padded.y);
        if (!ownership.contains(cv::Point(cvFloor(keypoint.pt.x), cvFloor(keypoint.pt.y)))) continue;
        ++empty.rawDetectionsPerCell[cell];
        if (!std::isfinite(keypoint.response) || !cameraValid(keypoint, geometry)) {
          ++empty.invalidCandidates;
          continue;
        }
        validOwned.push_back(keypoint);
      }
      std::sort(validOwned.begin(), validOwned.end(), strongerKeypoint);
      if (validOwned.size() > perCellCandidateCap) validOwned.resize(perCellCandidateCap);
      empty.candidatesPerCell[cell] = validOwned.size();
      candidates.insert(candidates.end(), validOwned.begin(), validOwned.end());
    }
  }

  SpatialKeypointBalanceResult balanced = balanceSpatialKeypoints(
      candidates, geometry, image.size(), maximumKeypoints, parameters);
  balanced.rawDetectionsPerCell = std::move(empty.rawDetectionsPerCell);
  balanced.candidatesPerCell = std::move(empty.candidatesPerCell);
  balanced.invalidCandidates += empty.invalidCandidates;
  return balanced;
}

double keypointBoundaryDensityRatio(const std::vector<cv::KeyPoint>& keypoints,
                                    const cv::Size& imageSize,
                                    int tileGridSize,
                                    int halfWidthPixels) {
  if (tileGridSize <= 1 || halfWidthPixels <= 0 || imageSize.area() <= 0) return 0.0;
  cv::Mat boundaryMask = cv::Mat::zeros(imageSize, CV_8UC1);
  for (int tile = 1; tile < tileGridSize; ++tile) {
    const int x = tile * imageSize.width / tileGridSize;
    const int y = tile * imageSize.height / tileGridSize;
    cv::rectangle(boundaryMask,
                  cv::Rect(std::max(0, x - halfWidthPixels), 0,
                           std::min(imageSize.width, x + halfWidthPixels + 1) - std::max(0, x - halfWidthPixels),
                           imageSize.height),
                  cv::Scalar(255), cv::FILLED);
    cv::rectangle(boundaryMask,
                  cv::Rect(0, std::max(0, y - halfWidthPixels), imageSize.width,
                           std::min(imageSize.height, y + halfWidthPixels + 1) - std::max(0, y - halfWidthPixels)),
                  cv::Scalar(255), cv::FILLED);
  }
  size_t boundaryCount = 0;
  size_t interiorCount = 0;
  for (const cv::KeyPoint& keypoint : keypoints) {
    const int x = static_cast<int>(keypoint.pt.x);
    const int y = static_cast<int>(keypoint.pt.y);
    if (x < 0 || y < 0 || x >= imageSize.width || y >= imageSize.height) continue;
    if (boundaryMask.at<unsigned char>(y, x) != 0) ++boundaryCount;
    else ++interiorCount;
  }
  const double boundaryArea = cv::countNonZero(boundaryMask);
  const double interiorArea = imageSize.area() - boundaryArea;
  if (boundaryArea <= 0.0 || interiorArea <= 0.0) return 0.0;
  const double boundaryDensity = boundaryCount / boundaryArea;
  const double interiorDensity = interiorCount / interiorArea;
  if (interiorDensity <= 0.0) return boundaryDensity > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  return boundaryDensity / interiorDensity;
}

}  // namespace okvis
