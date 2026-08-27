/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *********************************************************************************/

#ifndef INCLUDE_OKVIS_SPATIALKEYPOINTBALANCER_HPP_
#define INCLUDE_OKVIS_SPATIALKEYPOINTBALANCER_HPP_

#include <memory>
#include <vector>

#include <okvis/Parameters.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace okvis {

struct SpatialKeypointBalanceResult {
  std::vector<cv::KeyPoint> keypoints;
  std::vector<size_t> rawDetectionsPerCell;
  std::vector<size_t> candidatesPerCell;
  std::vector<size_t> selectedPerCell;
  size_t invalidCandidates = 0;
};

/// Detect independently in padded cell-local images before imposing any
/// image-wide budget, then balance the camera-valid candidates.
SpatialKeypointBalanceResult detectAndBalanceSpatialKeypoints(
    const cv::Mat& image,
    cv::FeatureDetector& detector,
    const std::shared_ptr<const cameras::CameraBase>& geometry,
    size_t maximumKeypoints,
    const SpatialBalancingParams& parameters);

/// Deterministically reserve candidates in each valid image cell, then fill globally by response.
SpatialKeypointBalanceResult balanceSpatialKeypoints(
    const std::vector<cv::KeyPoint>& candidates,
    const std::shared_ptr<const cameras::CameraBase>& geometry,
    const cv::Size& imageSize,
    size_t maximumKeypoints,
    const SpatialBalancingParams& parameters);

/// Row-major regular-grid counts without changing feature membership.
std::vector<size_t> countSpatialKeypoints(const std::vector<cv::KeyPoint>& keypoints,
                                          const cv::Size& imageSize,
                                          int rows = 4,
                                          int cols = 4);

/// Ratio of keypoint density near internal tile boundaries to density in tile interiors.
double keypointBoundaryDensityRatio(const std::vector<cv::KeyPoint>& keypoints,
                                    const cv::Size& imageSize,
                                    int tileGridSize,
                                    int halfWidthPixels = 2);

}  // namespace okvis

#endif  // INCLUDE_OKVIS_SPATIALKEYPOINTBALANCER_HPP_
