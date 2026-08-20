/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Mar 31, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file implementation/Frame.hpp
 * @brief Header implementation file for the Frame class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "opencv2/highgui/highgui.hpp"  // Sharmin
#include "opencv2/imgproc/imgproc.hpp"  // Sharmin
using namespace cv;                     // NOLINT

/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {

uint32_t frameFloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::array<uint32_t, 7> frameKeypointOrderKey(const cv::KeyPoint& keypoint) {
  return {frameFloatBits(keypoint.pt.x),
          frameFloatBits(keypoint.pt.y),
          frameFloatBits(keypoint.size),
          frameFloatBits(keypoint.response),
          static_cast<uint32_t>(static_cast<int32_t>(keypoint.octave)),
          static_cast<uint32_t>(static_cast<int32_t>(keypoint.class_id)),
          frameFloatBits(keypoint.angle)};
}

}  // namespace

// a constructor that uses the specified geometry,
/// detector and extractor
Frame::Frame(const cv::Mat& image,
             std::shared_ptr<cameras::CameraBase>& cameraGeometry,
             std::shared_ptr<cv::FeatureDetector>& detector,
             std::shared_ptr<cv::DescriptorExtractor>& extractor)
    : image_(image), cameraGeometry_(cameraGeometry), detector_(detector), extractor_(extractor) {}

// set the frame image;
void Frame::setImage(const cv::Mat& image) { image_ = image; }

// set the geometry
void Frame::setGeometry(std::shared_ptr<const cameras::CameraBase> cameraGeometry) { cameraGeometry_ = cameraGeometry; }

// set the detector
void Frame::setDetector(std::shared_ptr<cv::FeatureDetector> detector) { detector_ = detector; }

// set the extractor
void Frame::setExtractor(std::shared_ptr<cv::DescriptorExtractor> extractor) { extractor_ = extractor; }

// obtain the image
const cv::Mat& Frame::image() const { return image_; }

// get the base class geometry (will be slow to use)
std::shared_ptr<const cameras::CameraBase> Frame::geometry() const { return cameraGeometry_; }

// get the specific geometry (will be fast to use)
template <class GEOMETRY_T>
std::shared_ptr<const GEOMETRY_T> Frame::geometryAs() const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    std::dynamic_pointer_cast<const GEOMETRY_T>(cameraGeometry_),
                    "incorrect pointer cast requested. " << cameraGeometry_->distortionType());
  return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#else
  return std::static_pointer_cast<const GEOMETRY_T>(cameraGeometry_);
#endif
}

// detect keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        returns the number of detected points.
int Frame::detect() {
  // make sure things are set to zero for safety
  keypoints_.clear();
  descriptors_.resize(0);
  landmarkIds_.clear();  // resizing and filling in zeros in Frame::describe() as some keypoints are removed there.

  // run the detector
  OKVIS_ASSERT_TRUE_DBG(Exception, detector_ != NULL, "Detector not initialised!");
  detector_->detect(image_, keypoints_);
  return keypoints_.size();
}

// describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        \param extractionDirection the extraction direction in camera frame
///        returns the number of detected points.
int Frame::describe(const Eigen::Vector3d& extractionDirection) {
  // check initialisation
  OKVIS_ASSERT_TRUE_DBG(Exception, extractor_ != NULL, "Detector not initialised!");

  // orient the keypoints according to the extraction direction:
  Eigen::Vector3d ep;
  Eigen::Vector2d reprojection;
  Eigen::Matrix<double, 2, 3> Jacobian;
  Eigen::Vector2d eg_projected;
  for (size_t k = 0; k < keypoints_.size(); ++k) {
    cv::KeyPoint& ckp = keypoints_[k];
    // project ray
    cameraGeometry_->backProject(Eigen::Vector2d(ckp.pt.x, ckp.pt.y), &ep);
    // obtain image Jacobian
    cameraGeometry_->project(ep, &reprojection, &Jacobian);
    // multiply with gravity direction
    eg_projected = Jacobian * extractionDirection;
    double angle = atan2(eg_projected[1], eg_projected[0]);
    // set
    ckp.angle = angle / M_PI * 180.0;
  }

  // extraction
  extractor_->compute(image_, keypoints_, descriptors_);
  landmarkIds_ = std::vector<uint64_t>(keypoints_.size(), 0);
  canonicalizeFeatureOrder();
  return keypoints_.size();
}
// describe keypoints. This uses virtual function calls.
///        That's a negligibly small overhead for many detections.
///        \param extractionDirection the extraction direction in camera frame
///        returns the number of detected points.
template <class GEOMETRY_T>
int Frame::describeAs(const Eigen::Vector3d& extractionDirection) {
  // check initialisation
  OKVIS_ASSERT_TRUE_DBG(Exception, extractor_ != NULL, "Detector not initialised!");

  // orient the keypoints according to the extraction direction:
  Eigen::Vector3d ep;
  Eigen::Vector2d reprojection;
  Eigen::Matrix<double, 2, 3> Jacobian;
  Eigen::Vector2d eg_projected;
  for (size_t k = 0; k < keypoints_.size(); ++k) {
    cv::KeyPoint& ckp = keypoints_[k];
    // project ray
    geometryAs<GEOMETRY_T>()->backProject(Eigen::Vector2d(ckp.pt.x, ckp.pt.y), &ep);
    // obtain image Jacobian
    geometryAs<GEOMETRY_T>()->project(ep, &reprojection, &Jacobian);
    // multiply with gravity direction
    eg_projected = Jacobian * extractionDirection;
    double angle = atan2(eg_projected[1], eg_projected[0]);
    // set
    ckp.angle = angle / M_PI * 180.0;
  }

  // extraction
  extractor_->compute(image_, keypoints_, descriptors_);
  canonicalizeFeatureOrder();
  return keypoints_.size();
}

// access a specific keypoint in OpenCV format
bool Frame::getCvKeypoint(size_t keypointIdx, cv::KeyPoint& keypoint) const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    keypointIdx < keypoints_.size(),
                    "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size());
  keypoint = keypoints_[keypointIdx];
  return keypointIdx < keypoints_.size();
#else
  keypoint = keypoints_[keypointIdx];
  return true;
#endif
}

// get a specific keypoint
bool Frame::getKeypoint(size_t keypointIdx, Eigen::Vector2d& keypoint) const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    keypointIdx < keypoints_.size(),
                    "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size());
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x, keypoints_[keypointIdx].pt.y);
  return keypointIdx < keypoints_.size();
#else
  keypoint = Eigen::Vector2d(keypoints_[keypointIdx].pt.x, keypoints_[keypointIdx].pt.y);
  return true;
#endif
}

// get the size of a specific keypoint
bool Frame::getKeypointSize(size_t keypointIdx, double& keypointSize) const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    keypointIdx < keypoints_.size(),
                    "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size());
  keypointSize = keypoints_[keypointIdx].size;
  return keypointIdx < keypoints_.size();
#else
  keypointSize = keypoints_[keypointIdx].size;
  return true;
#endif
}

// access the descriptor -- CAUTION: high-speed version.
///        returns NULL if out of bounds.
const unsigned char* Frame::keypointDescriptor(size_t keypointIdx) {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    keypointIdx < keypoints_.size(),
                    "keypointIdx " << keypointIdx << "out of range: keypoints has size " << keypoints_.size());
  return descriptors_.data + descriptors_.cols * keypointIdx;
#else
  return descriptors_.data + descriptors_.cols * keypointIdx;
#endif
}

// Set the landmark ID
bool Frame::setLandmarkId(size_t keypointIdx, uint64_t landmarkId) {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    keypointIdx < landmarkIds_.size(),
                    "keypointIdx " << keypointIdx << "out of range: landmarkIds_ has size " << landmarkIds_.size());
  landmarkIds_[keypointIdx] = landmarkId;
  return keypointIdx < keypoints_.size();
#else
  landmarkIds_[keypointIdx] = landmarkId;
  return true;
#endif
}

// Sharmin
// Set the landmark ID
/*bool Frame::setContourLandmarkId(size_t keypointIdx, uint64_t landmarkId)
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < contour_landmarkIds_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarkIds_ has size "
          << contour_landmarkIds_.size());
  contour_landmarkIds_[keypointIdx] = landmarkId;
  return keypointIdx < contour_keypoints_.size();
#else
  contour_landmarkIds_[keypointIdx] = landmarkId;
  return true;
#endif
}*/

// Access the landmark ID
uint64_t Frame::landmarkId(size_t keypointIdx) const {
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(Exception,
                    keypointIdx < landmarkIds_.size(),
                    "keypointIdx " << keypointIdx << "out of range: landmarkIds has size " << landmarkIds_.size());
  return landmarkIds_[keypointIdx];
#else
  return landmarkIds_[keypointIdx];
#endif
}

// @Sharmin
// Access the landmark ID
/*uint64_t Frame::contour_landmarkId(size_t keypointIdx) const
{
#ifndef NDEBUG
  OKVIS_ASSERT_TRUE(
      Exception,
      keypointIdx < contour_landmarkIds_.size(),
      "keypointIdx " << keypointIdx << "out of range: landmarkIds has size "
          << contour_landmarkIds_.size());
  return contour_landmarkIds_[keypointIdx];
#else
  return contour_landmarkIds_[keypointIdx];
#endif
}*/

// provide keypoints externally
inline bool Frame::resetKeypoints(const std::vector<cv::KeyPoint>& keypoints) {
  keypoints_ = keypoints;
  landmarkIds_ = std::vector<uint64_t>(keypoints_.size(), 0);
  return true;
}

// provide descriptors externally
inline bool Frame::resetDescriptors(const cv::Mat& descriptors) {
  descriptors_ = descriptors;
  return true;
}

void Frame::canonicalizeFeatureOrder() {
  if (keypoints_.size() < 2) return;

  const bool descriptorsAligned =
      descriptors_.rows == static_cast<int>(keypoints_.size()) && descriptors_.cols > 0;
  if (descriptors_.rows > 0 && !descriptorsAligned) return;

  std::vector<size_t> order(keypoints_.size());
  for (size_t index = 0; index < order.size(); ++index) order[index] = index;

  std::sort(order.begin(), order.end(), [this, descriptorsAligned](size_t lhs, size_t rhs) {
    const std::array<uint32_t, 7> lhsKey = frameKeypointOrderKey(keypoints_[lhs]);
    const std::array<uint32_t, 7> rhsKey = frameKeypointOrderKey(keypoints_[rhs]);
    if (lhsKey != rhsKey) {
      return std::lexicographical_compare(lhsKey.begin(), lhsKey.end(), rhsKey.begin(), rhsKey.end());
    }
    if (!descriptorsAligned) return false;

    const unsigned char* lhsDescriptor = descriptors_.ptr<unsigned char>(static_cast<int>(lhs));
    const unsigned char* rhsDescriptor = descriptors_.ptr<unsigned char>(static_cast<int>(rhs));
    const size_t rowBytes = static_cast<size_t>(descriptors_.cols) * descriptors_.elemSize();
    return std::lexicographical_compare(lhsDescriptor,
                                        lhsDescriptor + rowBytes,
                                        rhsDescriptor,
                                        rhsDescriptor + rowBytes);
  });

  std::vector<cv::KeyPoint> sortedKeypoints;
  sortedKeypoints.reserve(keypoints_.size());
  std::vector<uint64_t> sortedLandmarkIds;
  if (landmarkIds_.size() == keypoints_.size()) sortedLandmarkIds.reserve(landmarkIds_.size());

  cv::Mat sortedDescriptors;
  if (descriptorsAligned) {
    sortedDescriptors.create(descriptors_.rows, descriptors_.cols, descriptors_.type());
  }

  for (size_t sortedIndex = 0; sortedIndex < order.size(); ++sortedIndex) {
    const size_t originalIndex = order[sortedIndex];
    sortedKeypoints.push_back(keypoints_[originalIndex]);
    if (!sortedLandmarkIds.empty() || landmarkIds_.size() == keypoints_.size()) {
      sortedLandmarkIds.push_back(landmarkIds_[originalIndex]);
    }
    if (descriptorsAligned) {
      descriptors_.row(static_cast<int>(originalIndex)).copyTo(sortedDescriptors.row(static_cast<int>(sortedIndex)));
    }
  }

  keypoints_.swap(sortedKeypoints);
  if (landmarkIds_.size() == keypoints_.size()) landmarkIds_.swap(sortedLandmarkIds);
  if (descriptorsAligned) descriptors_ = sortedDescriptors;
}

size_t Frame::numKeypoints() const { return keypoints_.size(); }

// @Sharmin
size_t Frame::contour_numKeypoints() const { return contour_keypoints_.size(); }

// @Sharmin
std::vector<cv::KeyPoint> Frame::contourKeypoints() const { return contour_keypoints_; }

// @Sharmin
cv::Mat Frame::contourDescriptors() const { return contour_descriptors_; }

}  // namespace okvis
