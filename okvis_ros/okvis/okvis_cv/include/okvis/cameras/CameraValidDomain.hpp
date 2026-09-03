/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *********************************************************************************/

#ifndef INCLUDE_OKVIS_CAMERAS_CAMERAVALIDDOMAIN_HPP_
#define INCLUDE_OKVIS_CAMERAS_CAMERAVALIDDOMAIN_HPP_

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <okvis/cameras/CameraBase.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace okvis {
namespace cameras {

/// Build an OpenCV-style validity mask: nonzero pixels are valid for feature
/// detection. CameraBase's optional mask uses the opposite convention
/// (nonzero means masked/invalid), and is folded into the result here.
inline cv::Mat cameraValidDomainMask(const std::shared_ptr<const CameraBase>& geometry,
                                     const cv::Size& imageSize) {
  cv::Mat validMask(imageSize, CV_8UC1, cv::Scalar(255));
  if (!geometry || imageSize.width <= 0 || imageSize.height <= 0) return validMask;

  // A pinhole camera has no model-domain exclusion inside its configured
  // rectangular image. Only an explicit CameraBase mask can invalidate it.
  const bool rectangularPinholeDomain = geometry->type().rfind("PinholeCamera<", 0) == 0;

  for (int y = 0; y < imageSize.height; ++y) {
    unsigned char* row = validMask.ptr<unsigned char>(y);
    for (int x = 0; x < imageSize.width; ++x) {
      bool valid = rectangularPinholeDomain;
      if (!rectangularPinholeDomain) {
        Eigen::Vector3d direction;
        const Eigen::Vector2d pixelCenter(static_cast<double>(x) + 0.5,
                                          static_cast<double>(y) + 0.5);
        valid = geometry->backProject(pixelCenter, &direction) && direction.allFinite();
      }
      if (valid && geometry->hasMask()) {
        valid = x < geometry->mask().cols && y < geometry->mask().rows &&
                geometry->mask().at<unsigned char>(y, x) == 0;
      }
      row[x] = valid ? 255 : 0;
    }
  }
  return validMask;
}

inline bool isValidCameraKeypoint(const cv::KeyPoint& keypoint,
                                  const cv::Mat& validMask,
                                  const std::shared_ptr<const CameraBase>& geometry) {
  if (!std::isfinite(keypoint.pt.x) || !std::isfinite(keypoint.pt.y)) return false;
  const int x = cvFloor(keypoint.pt.x);
  const int y = cvFloor(keypoint.pt.y);
  if (x < 0 || y < 0 || x >= validMask.cols || y >= validMask.rows ||
      validMask.at<unsigned char>(y, x) == 0) {
    return false;
  }
  if (!geometry) return true;
  Eigen::Vector3d direction;
  return geometry->backProject(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), &direction) &&
         direction.allFinite();
}

inline void retainValidCameraKeypoints(std::vector<cv::KeyPoint>* keypoints,
                                       const cv::Mat& validMask,
                                       const std::shared_ptr<const CameraBase>& geometry) {
  if (!keypoints) return;
  keypoints->erase(std::remove_if(keypoints->begin(), keypoints->end(),
                                  [&](const cv::KeyPoint& keypoint) {
                                    return !isValidCameraKeypoint(keypoint, validMask, geometry);
                                  }),
                   keypoints->end());
}

/// Clone an image and remove invalid-domain content without modifying the
/// image stored in Frame. The custom BRISK detector cannot accept a mask.
inline cv::Mat cameraDomainMaskedDetectionImage(const cv::Mat& image,
                                                const cv::Mat& validMask) {
  cv::Mat workingImage = image.clone();
  if (!workingImage.empty() && !validMask.empty()) {
    workingImage.setTo(cv::Scalar::all(0), validMask == 0);
  }
  return workingImage;
}

/// Overlay invalid pixels in BGR red at 50 percent opacity. Pixels selected by
/// validMask remain unchanged.
inline void overlayInvalidCameraDomain50Percent(cv::Mat* bgrImage,
                                                const cv::Mat& validMask) {
  if (!bgrImage || bgrImage->empty() || bgrImage->type() != CV_8UC3 ||
      validMask.empty() || validMask.size() != bgrImage->size()) {
    return;
  }
  const cv::Mat invalidMask = validMask == 0;
  if (cv::countNonZero(invalidMask) == 0) return;
  cv::Mat redImage(bgrImage->size(), bgrImage->type(), cv::Scalar(0, 0, 255));
  cv::Mat blended;
  cv::addWeighted(*bgrImage, 0.5, redImage, 0.5, 0.0, blended);
  blended.copyTo(*bgrImage, invalidMask);
}

/// Overlay invalid pixels in BGR blue at 30 percent opacity. This is used for
/// user-configured exclusion masks, independently of the camera model domain.
inline void overlayConfiguredCameraMask30PercentBlue(cv::Mat* bgrImage,
                                                     const cv::Mat& validMask) {
  if (!bgrImage || bgrImage->empty() || bgrImage->type() != CV_8UC3 ||
      validMask.empty() || validMask.size() != bgrImage->size()) {
    return;
  }
  const cv::Mat invalidMask = validMask == 0;
  if (cv::countNonZero(invalidMask) == 0) return;
  cv::Mat blueImage(bgrImage->size(), bgrImage->type(), cv::Scalar(150, 100, 0));
  cv::Mat blended;
  cv::addWeighted(*bgrImage, 0.7, blueImage, 0.3, 0.0, blended);
  blended.copyTo(*bgrImage, invalidMask);
}

}  // namespace cameras
}  // namespace okvis

#endif  // INCLUDE_OKVIS_CAMERAS_CAMERAVALIDDOMAIN_HPP_
