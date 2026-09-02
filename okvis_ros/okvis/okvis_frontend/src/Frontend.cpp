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
 *  Created on: Mar 27, 2015
 *      Author: Andreas Forster (an.forster@gmail.com)
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file Frontend.cpp
 * @brief Source file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 * @Modified by Sharmin Rahman
 * @Last Modified: 09/19/2018
 */

#include <brisk/brisk.h>
#include <glog/logging.h>

#include <okvis/Frontend.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// cameras and distortions
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/DoubleSphereCamera.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/RansacPoseDiagnostics.hpp>
#include <sstream>
#include <vector>
// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {
bool featureDistributionDiagnosticsOptIn() {
  const char* value = std::getenv("SVIN2_ENABLE_DRIFT_DIAGNOSTICS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

std::string spatialCounts(const std::vector<size_t>& values) {
  std::ostringstream stream;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) stream << ',';
    stream << values[index];
  }
  return stream.str();
}

double safeRatio(size_t numerator, size_t denominator) {
  return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
}

size_t finiteSupportInCurrentFrame(const Estimator& estimator,
                                   const std::shared_ptr<MultiFrame>& frame) {
  if (!frame) return 0;
  size_t count = 0;
  for (size_t camera = 0; camera < frame->numFrames(); ++camera) {
    for (size_t keypoint = 0; keypoint < frame->numKeypoints(camera); ++keypoint) {
      const uint64_t id = frame->landmarkId(camera, keypoint);
      if (id != 0 && estimator.isLandmarkAdded(id) && estimator.isLandmarkInitialized(id)) ++count;
    }
  }
  return count;
}

struct PendingIntensityCounts {
  size_t total = 0;
  size_t originalBright = 0;
  size_t originalDark = 0;
  size_t originalUnknown = 0;
};

void incrementPendingIntensity(uint8_t flags, PendingIntensityCounts* counts) {
  ++counts->total;
  if ((flags & Frame::PreHistogramIntensityValid) == 0) {
    ++counts->originalUnknown;
  } else if ((flags & Frame::PreHistogramDark) != 0) {
    ++counts->originalDark;
  } else {
    ++counts->originalBright;
  }
}

}  // namespace

// Constructor.
Frontend::Frontend(size_t numCameras)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      lastFeatureDiagnosticSecond_(numCameras, std::numeric_limits<int64_t>::min()),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))), //TODO(Guo): consider make matcher thread num configurable
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2) {
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
  }
  initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint>* keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_, "Camera index exceeds number of cameras.");
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);

  bool diagnosticsEnabled = false;
  if (featureDistributionDiagnosticsOptIn()) {
    const int64_t diagnosticSecond = static_cast<int64_t>(std::floor(frameOut->timestamp().toSec()));
    diagnosticsEnabled = lastFeatureDiagnosticSecond_[cameraIndex] != diagnosticSecond;
    if (diagnosticsEnabled) lastFeatureDiagnosticSecond_[cameraIndex] = diagnosticSecond;
  }
  SpatialKeypointBalanceResult spatialResult;
  if (spatialBalancingParameters_.enable) {
    spatialResult = detectAndBalanceSpatialKeypoints(frameOut->image(cameraIndex),
                                                     *featureDetectors_[cameraIndex],
                                                     frameOut->geometry(cameraIndex),
                                                     briskDetectionMaximumKeypoints_,
                                                     spatialBalancingParameters_);
    frameOut->resetKeypoints(cameraIndex, spatialResult.keypoints);
  } else {
    frameOut->detect(cameraIndex);
  }

  std::vector<cv::KeyPoint> selectedKeypoints;
  if (diagnosticsEnabled) {
    selectedKeypoints.reserve(frameOut->numKeypoints(cameraIndex));
    for (size_t index = 0; index < frameOut->numKeypoints(cameraIndex); ++index) {
      cv::KeyPoint keypoint;
      frameOut->getCvKeypoint(cameraIndex, index, keypoint);
      selectedKeypoints.push_back(keypoint);
    }
  }

  // ExtractionDirection == gravity direction in camera frame
  Eigen::Vector3d g_in_W(0, 0, -1);
  Eigen::Vector3d extractionDirection = T_WC.inverse().C() * g_in_W;
  frameOut->describe(cameraIndex, extractionDirection);

  if (diagnosticsEnabled) {
    std::vector<cv::KeyPoint> describedKeypoints;
    describedKeypoints.reserve(frameOut->numKeypoints(cameraIndex));
    for (size_t index = 0; index < frameOut->numKeypoints(cameraIndex); ++index) {
      cv::KeyPoint keypoint;
      frameOut->getCvKeypoint(cameraIndex, index, keypoint);
      describedKeypoints.push_back(keypoint);
    }
    const cv::Size imageSize = frameOut->image(cameraIndex).size();
    const std::vector<size_t> selectedCellCounts = countSpatialKeypoints(selectedKeypoints, imageSize);
    const std::vector<size_t> describedCellCounts = countSpatialKeypoints(describedKeypoints, imageSize);
    const std::vector<size_t> rawCellCounts = spatialBalancingParameters_.enable
                                                  ? spatialResult.rawDetectionsPerCell
                                                  : selectedCellCounts;
    const std::vector<size_t> candidateCellCounts = spatialBalancingParameters_.enable
                                                        ? spatialResult.candidatesPerCell
                                                        : selectedCellCounts;
    const size_t rawCount = std::accumulate(rawCellCounts.begin(), rawCellCounts.end(), size_t{0});
    const size_t candidateCount =
        std::accumulate(candidateCellCounts.begin(), candidateCellCounts.end(), size_t{0});
    LOG(INFO) << std::setprecision(17) << "[FEATURE_DISTRIBUTION_DIAGNOSTIC]"
              << " timestamp=" << frameOut->timestamp().toSec() << " frame=" << frameOut->id()
              << " camera=" << cameraIndex << " balancing=" << spatialBalancingParameters_.enable
              << " raw_detection_count=" << rawCount << " valid_candidate_count=" << candidateCount
              << " selected_count=" << selectedKeypoints.size()
              << " described_count=" << describedKeypoints.size()
              << " invalid_count=" << spatialResult.invalidCandidates
              << " raw_detection_cells=" << spatialCounts(rawCellCounts)
              << " valid_candidate_cells=" << spatialCounts(candidateCellCounts)
              << " selected_cells=" << spatialCounts(selectedCellCounts)
              << " described_cells=" << spatialCounts(describedCellCounts)
              << " boundary_density_ratio="
              << keypointBoundaryDensityRatio(describedKeypoints,
                                               imageSize,
                                               imagePreprocessingTileGridSize_);
  }

  // set detector/extractor to nullpointer? TODO(later) or not?
  return true;
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator,
    okvis::kinematics::Transformation& /*T_WS_propagated*/,  // TODO(sleutenegger): why is this not used here?
    const okvis::VioParameters& params,
    const std::shared_ptr<okvis::MapPointVector> /*map*/,  // TODO(sleutenegger): why is this not used here?
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool* asKeyframe) {
  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(
        Exception, distortionType == params.nCameraSystem.distortionType(i), "mixed frame types are not supported yet");
  }
  int numMatchesToKeyframes = 0;

  // find camera model (Currently only supports all cameras having the same model. TODO: add support for mixed camera models)
  const std::string cameraModel0 = params.nCameraSystem.cameraGeometry(0)->type();
  const bool isDoubleSphereCameraModel = cameraModel0.rfind("DoubleSphereCamera<", 0) == 0;
  const bool isPinholeCameraModel = cameraModel0.rfind("PinholeCamera<", 0) == 0;
  OKVIS_ASSERT_TRUE(Exception,
                    isPinholeCameraModel || isDoubleSphereCameraModel,
                    "Unsupported camera model: " + cameraModel0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    const std::string cameraModelI = params.nCameraSystem.cameraGeometry(i)->type();
    const bool sameModelClass =
        (isDoubleSphereCameraModel && cameraModelI.rfind("DoubleSphereCamera<", 0) == 0) ||
        (isPinholeCameraModel && cameraModelI.rfind("PinholeCamera<", 0) == 0);
    OKVIS_ASSERT_TRUE(
        Exception, sameModelClass, "mixed camera model classes are not supported yet");
  }

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() > 1) {
    int requiredMatches = 5;

    double uncertainMatchFraction = 0;
    bool rotationOnly = false;

    // match to last keyframe
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        numMatchesToKeyframes = matchToKeyframes<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false, &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        numMatchesToKeyframes = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false, &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        numMatchesToKeyframes = matchToKeyframes<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), rotationOnly, false, &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        OKVIS_ASSERT_TRUE(Exception,
                          isDoubleSphereCameraModel,
                          "NoDistortion is only supported with DoubleSphereCamera in this frontend path.");
        numMatchesToKeyframes = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, false, &uncertainMatchFraction);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchKeyframesTimer.stop();
    if (!isInitialized_) {
      if (!rotationOnly) {
        isInitialized_ = true;
        LOG(INFO) << "Initialized!";
      }
    }

    if (numMatchesToKeyframes <= requiredMatches) {
      LOG(WARNING) << "Tracking failure (matching to keyframes). Number of total matches (3D-2D + 2D-2D): "
                   << numMatchesToKeyframes;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);

    // match to last frame
    int numMatchesToLastFrame = 0;
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        numMatchesToLastFrame = matchToLastFrame<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(), false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        numMatchesToLastFrame = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        numMatchesToLastFrame = matchToLastFrame<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), false);

        break;
      }
      case okvis::cameras::NCameraSystem::NoDistortion: {
        OKVIS_ASSERT_TRUE(Exception,
                          isDoubleSphereCameraModel,
                          "NoDistortion is only supported with DoubleSphereCamera in this frontend path.");
        numMatchesToLastFrame = matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion> > >(
            estimator, params, framesInOut->id(), false);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }

    if (numMatchesToLastFrame <= requiredMatches && estimator.numFrames() >= 2) {
        if (estimator.isKeyframe(estimator.frameIdByAge(1))) {
          // LOG(WARNING) << "Skip to match to last frame. Last frame is a keyframe.";
        } else {
          LOG(WARNING) << "Tracking failure (matching to last frame). Number of total matches (3D-2D + 2D-2D): "
                       << numMatchesToLastFrame;
        }
    }
    matchToLastFrameTimer.stop();
  } else {
    *asKeyframe = true;  // first frame needs to be keyframe
  }
  // do stereo match to get new landmarks
  TimerSwitchable matchStereoTimer("2.4.3 matchStereo");
  switch (distortionType) {
    case okvis::cameras::NCameraSystem::RadialTangential: {
      matchStereo<VioKeyframeWindowMatchingAlgorithm<
          okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(estimator, params, framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > >(
          estimator, params, framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8: {
      matchStereo<VioKeyframeWindowMatchingAlgorithm<
          okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > >(
          estimator, params, framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::NoDistortion: {
      OKVIS_ASSERT_TRUE(Exception,
                        isDoubleSphereCameraModel,
                        "NoDistortion is only supported with DoubleSphereCamera in this frontend path.");
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion> > >(
          estimator, params, framesInOut);
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
      break;
  }
  matchStereoTimer.stop();

  return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(const okvis::ImuMeasurementDeque& imuMeasurements,
                           const okvis::ImuParameters& imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias& speedAndBiases,
                           const okvis::Time& t_start,
                           const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const {
  if (imuMeasurements.size() < 2) {
    LOG(WARNING) << "- Skipping propagation as only one IMU measurement has been given to frontend."
                 << " Normal when starting up.";
    return 0;
  }
  int measurements_propagated = okvis::ceres::ImuError::propagation(
      imuMeasurements, imuParams, T_WS_propagated, speedAndBiases, t_start, t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::doWeNeedANewKeyframe(const okvis::Estimator& estimator,
                                    std::shared_ptr<okvis::MultiFrame> currentFrame) {
  // Sharmin: Modified for Scale refinement
  // if (estimator.numFrames() < 2) {
  if (estimator.numFrames() < 2 || estimator.stateCount_ < 6) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_) return false;

  double overlap = 0.0;
  double ratio = 0.0;

  // go through all the frames and try to match the initialized keypoints
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) {
    // get the hull of all keypoints in current frame
    std::vector<cv::Point2f> frameBPoints, frameBHull;
    std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > frameBLandmarks;

    const size_t numB = currentFrame->numKeypoints(im);
    frameBPoints.reserve(numB);
    frameBLandmarks.reserve(numB);
    Eigen::Vector2d keypoint;
    for (size_t k = 0; k < numB; ++k) {
      currentFrame->getKeypoint(im, k, keypoint);
      // insert it
      frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      // also remember matches
      if (currentFrame->landmarkId(im, k) != 0) {
        frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      }
    }

    if (frameBPoints.size() < 3) continue;
    cv::convexHull(frameBPoints, frameBHull);
    if (frameBMatches.size() < 3) continue;
    cv::convexHull(frameBMatches, frameBMatchesHull);

    // areas
    double frameBArea = cv::contourArea(frameBHull);
    double frameBMatchesArea = cv::contourArea(frameBMatchesHull);

    // overlap area
    double overlapArea = frameBMatchesArea / frameBArea;
    // matching ratio inside overlap area: count
    int pointsInFrameBMatchesArea = 0;
    if (frameBMatchesHull.size() > 2) {
      for (size_t k = 0; k < frameBPoints.size(); ++k) {
        if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false) > 0) {
          pointsInFrameBMatchesArea++;
        }
      }
    }
    double matchingRatio = static_cast<double>(frameBMatches.size()) / static_cast<double>(pointsInFrameBMatchesArea);

    // calculate overlap score
    overlap = std::max(overlapArea, overlap);
    ratio = std::max(matchingRatio, ratio);
  }

  // take a decision
  if (overlap > keyframeInsertionOverlapThreshold_ && ratio > keyframeInsertionMatchingRatioThreshold_)
    return false;
  else
    return true;
}

// Match a new multiframe to existing keyframes
template <class MATCHING_ALGORITHM>
int Frontend::matchToKeyframes(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers) {
  rotationOnly = true;
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  int retCtr = 0;
  int numUncertainMatches = 0;

  // go through all the frames and try to match the initialized keypoints
  // 3D2D matching with keyframes
  size_t kfcounter = 0;
  for (size_t age = 1; age < estimator.numFrames(); ++age) {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId)) continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      MATCHING_ALGORITHM matchingAlgorithm(
          estimator, MATCHING_ALGORITHM::Match3D2D, briskMatchingThreshold_, usePoseUncertainty,
          false, params.diagnostics.landmarkPromotion);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 3D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
    }
    kfcounter++;
    if (kfcounter > 2) break;
  }

  kfcounter = 0;
  bool firstFrame = true;
  // Note Sharmin: age = 0 is the current frame. age is a reverse iterator over StateMap
  for (size_t age = 1; age < estimator.numFrames(); ++age) {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId)) continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      MATCHING_ALGORITHM matchingAlgorithm(
          estimator, MATCHING_ALGORITHM::Match2D2D, briskMatchingThreshold_, usePoseUncertainty,
          false, params.diagnostics.landmarkPromotion);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 2D-2D for initialization of new (mono-)correspondences
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
      if (params.diagnostics.bearingTracking) {
        LOG_EVERY_N(INFO, 10000) << "[BEARING_TRACK_DIAGNOSTIC] context=keyframe"
                                  << " frame_a=" << olderFrameId << " frame_b=" << currentFrameId
                                  << " camera=" << im << " matches=" << matchingAlgorithm.numMatches()
                                  << " bearing_only=" << matchingAlgorithm.numBearingOnlyMatches()
                                  << " finite=" << matchingAlgorithm.numFiniteMatches()
                                  << " promotions=" << matchingAlgorithm.numPromotions()
                                  << " promotion_attempts=" << matchingAlgorithm.numPromotionAttempts()
                                  << " deferred_observations="
                                  << matchingAlgorithm.numPromotionDeferredObservations()
                                  << " deferred_frames=" << matchingAlgorithm.numPromotionDeferredFrames()
                                  << " deferred_parallax=" << matchingAlgorithm.numPromotionDeferredParallax()
                                  << " rejected_candidates=" << matchingAlgorithm.numRejectedCandidates()
                                  << " active_pending=" << matchingAlgorithm.numActivePendingTracks();
      }
    }

    // remove outliers
    // only do RANSAC 3D2D with most recent KF
    if (kfcounter == 0 && isInitialized_)
      runRansac3d2d(estimator, params.nCameraSystem, estimator.multiFrame(currentFrameId), removeOutliers);

    bool rotationOnly_tmp = false;
    // Pending bearing tracks remain useful for rotation/relative-pose outlier
    // rejection after initialization, but only initialize the pose at startup.
    const int bearingRansacInliers = runRansac2d2d(
        estimator, params, currentFrameId, olderFrameId, !isInitialized_, removeOutliers, rotationOnly_tmp);
    if (params.diagnostics.bearingTracking) {
      LOG_EVERY_N(INFO, 10000) << "[BEARING_RANSAC_DIAGNOSTIC] context=keyframe"
                                << " frame_a=" << olderFrameId << " frame_b=" << currentFrameId
                                << " inliers=" << bearingRansacInliers
                                << " rotation_only=" << rotationOnly_tmp;
    }
    // Sharmin: commented for scale
    if (firstFrame) {
      rotationOnly = rotationOnly_tmp;
      firstFrame = false;
    }

    kfcounter++;
    if (kfcounter > 1) break;
  }

  // calculate fraction of safe matches
  if (uncertainMatchFraction) {
    *uncertainMatchFraction =
        retCtr > 0 ? static_cast<double>(numUncertainMatches) / static_cast<double>(retCtr) : 0.0;
  }

  return retCtr;
}

// Match a new multiframe to the last frame.
template <class MATCHING_ALGORITHM>
int Frontend::matchToLastFrame(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    LOG(WARNING) << "Not enough frames to match to last frame. Estimator has " << estimator.numFrames()
                 << " frames.";
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);

  if (estimator.isKeyframe(lastFrameId)) {
    // already done
    // LOG(WARNING) << "Last frame is a keyframe. Not matching to last frame.";
    return 0;
  }

  int retCtr = 0;

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(
        estimator, MATCHING_ALGORITHM::Match3D2D, briskMatchingThreshold_, usePoseUncertainty,
        false, params.diagnostics.landmarkPromotion);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 3D-2D
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
    // LOG(INFO) << "Number of matches to last frame (3D-2D): " << matchingAlgorithm.numMatches();
  }

  runRansac3d2d(estimator, params.nCameraSystem, estimator.multiFrame(currentFrameId), removeOutliers, true);

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(
        estimator, MATCHING_ALGORITHM::Match2D2D, briskMatchingThreshold_, usePoseUncertainty,
        false, params.diagnostics.landmarkPromotion);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 2D-2D for initialization of new (mono-)correspondences
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
    if (params.diagnostics.bearingTracking) {
      LOG_EVERY_N(INFO, 10000) << "[BEARING_TRACK_DIAGNOSTIC] context=last_frame"
                                << " frame_a=" << lastFrameId << " frame_b=" << currentFrameId
                                << " camera=" << im << " matches=" << matchingAlgorithm.numMatches()
                                << " bearing_only=" << matchingAlgorithm.numBearingOnlyMatches()
                                << " finite=" << matchingAlgorithm.numFiniteMatches()
                                << " promotions=" << matchingAlgorithm.numPromotions()
                                << " promotion_attempts=" << matchingAlgorithm.numPromotionAttempts()
                                << " deferred_observations="
                                << matchingAlgorithm.numPromotionDeferredObservations()
                                << " deferred_frames=" << matchingAlgorithm.numPromotionDeferredFrames()
                                << " deferred_parallax=" << matchingAlgorithm.numPromotionDeferredParallax()
                                << " rejected_candidates=" << matchingAlgorithm.numRejectedCandidates()
                                << " active_pending=" << matchingAlgorithm.numActivePendingTracks();
    }
    // LOG(INFO) << "Number of matches to last frame (2D-2D): " << matchingAlgorithm.numMatches();
  }

  // remove outliers
  bool rotationOnly = false;
  const int bearingRansacInliers =
      runRansac2d2d(estimator, params, currentFrameId, lastFrameId, false, removeOutliers,
                    rotationOnly, true);
  if (params.diagnostics.bearingTracking) {
    LOG_EVERY_N(INFO, 10000) << "[BEARING_RANSAC_DIAGNOSTIC] context=last_frame"
                              << " frame_a=" << lastFrameId << " frame_b=" << currentFrameId
                              << " inliers=" << bearingRansacInliers << " rotation_only=" << rotationOnly;
  }

  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
// Sharmin: modified to add scale-refinement
template <class MATCHING_ALGORITHM>
void Frontend::matchStereo(okvis::Estimator& estimator,
                           const okvis::VioParameters& params,
                           std::shared_ptr<okvis::MultiFrame> multiFrame) {
  bool useSCM = false;  // Sharmin
  const size_t camNumber = multiFrame->numFrames();
  const uint64_t mfId = multiFrame->id();

  for (size_t im0 = 0; im0 < camNumber; im0++) {
    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) {
      // first, check the possibility for overlap
      // TODO(test): implement this in the Multiframe.

      // Keep the policy separate from calibrated visibility: an opposing pair
      // may have a tiny formal overlap that is unsuitable for feature matching.
      const bool hasOverlap = multiFrame->hasOverlap(im0, im1);
      const bool crossCameraDiagnostics = params.diagnostics.crossCameraMatching;
      size_t overlapPixels = 0;
      size_t overlapTotalPixels = 0;
      double overlapFraction = 0.0;
      if (crossCameraDiagnostics && hasOverlap) {
        const cv::Mat overlapMask = multiFrame->overlap(im0, im1);
        overlapPixels = overlapMask.empty() ? 0 : static_cast<size_t>(cv::countNonZero(overlapMask));
        overlapTotalPixels = overlapMask.empty() ? 0 : overlapMask.total();
        overlapFraction = overlapTotalPixels == 0
                              ? 0.0
                              : static_cast<double>(overlapPixels) / overlapTotalPixels;
      }

      if (!params.sensors_information.enableCrossCameraMatching || !hasOverlap) {
        if (crossCameraDiagnostics) {
          LOG(INFO) << std::setprecision(17) << "[CROSS_CAMERA_MATCH_DIAGNOSTIC]"
                    << " timestamp=" << multiFrame->timestamp().toSec() << " frame=" << mfId
                    << " camera_a=" << im0 << " camera_b=" << im1
                    << " enabled=" << params.sensors_information.enableCrossCameraMatching
                    << " has_overlap=" << hasOverlap << " overlap_pixels=" << overlapPixels
                    << " overlap_total_pixels=" << overlapTotalPixels
                    << " overlap_fraction=" << overlapFraction
                    << " keypoints_a=" << multiFrame->numKeypoints(im0)
                    << " keypoints_b=" << multiFrame->numKeypoints(im1)
                    << " matches_2d2d=0 bearing_only=0 finite=0 promotions=0 rejected_2d2d=0"
                    << " matches_3d2d_forward=0 matches_3d2d_reverse=0 runtime_ms=0";
        }
        continue;
      }

      std::chrono::steady_clock::time_point pairStart;
      if (crossCameraDiagnostics) pairStart = std::chrono::steady_clock::now();

      MATCHING_ALGORITHM matchingAlgorithm(
          estimator,
          MATCHING_ALGORITHM::Match2D2D,
          briskMatchingThreshold_,
          false,
          false,
          params.diagnostics.landmarkPromotion);  // TODO(test): verify when uncertainty-based matching is restored
      matchingAlgorithm.setFrames(mfId, mfId, im0, im1);  // newest frame

      // match 2D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      const size_t matches2d2d = crossCameraDiagnostics ? matchingAlgorithm.numMatches() : 0;
      const size_t bearingOnly = crossCameraDiagnostics ? matchingAlgorithm.numBearingOnlyMatches() : 0;
      const size_t finite = crossCameraDiagnostics ? matchingAlgorithm.numFiniteMatches() : 0;
      const size_t promotions = crossCameraDiagnostics ? matchingAlgorithm.numPromotions() : 0;
      const size_t rejected2d2d = crossCameraDiagnostics ? matchingAlgorithm.numRejectedCandidates() : 0;

      // match 3D-2D
      matchingAlgorithm.setMatchingType(MATCHING_ALGORITHM::Match3D2D);
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      const size_t matches3d2dForward = crossCameraDiagnostics ? matchingAlgorithm.numMatches() : 0;

      // match 2D-3D
      matchingAlgorithm.setFrames(mfId, mfId, im1, im0);  // newest frame
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      const size_t matches3d2dReverse = crossCameraDiagnostics ? matchingAlgorithm.numMatches() : 0;

      if (crossCameraDiagnostics) {
        const double runtimeMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pairStart).count();
        LOG(INFO) << std::setprecision(17) << "[CROSS_CAMERA_MATCH_DIAGNOSTIC]"
                  << " timestamp=" << multiFrame->timestamp().toSec() << " frame=" << mfId
                  << " camera_a=" << im0 << " camera_b=" << im1 << " enabled=1 has_overlap=1"
                  << " overlap_pixels=" << overlapPixels << " overlap_total_pixels=" << overlapTotalPixels
                  << " overlap_fraction=" << overlapFraction
                  << " keypoints_a=" << multiFrame->numKeypoints(im0)
                  << " keypoints_b=" << multiFrame->numKeypoints(im1)
                  << " matches_2d2d=" << matches2d2d << " bearing_only=" << bearingOnly
                  << " finite=" << finite << " promotions=" << promotions
                  << " rejected_2d2d=" << rejected2d2d
                  << " matches_3d2d_forward=" << matches3d2dForward
                  << " matches_3d2d_reverse=" << matches3d2dReverse
                  << " runtime_ms=" << runtimeMs;
      }
    }
  }

  /***********   Scale Refinement: Added by Sharmin ****************/

  bool rotationOnly_tmp = false;
  bool removeOutliers = true;
  if (camNumber < 2) {
    // Scale refinement needs at least a stereo pair; skip safely for mono configurations.
    isScaleRefined_ = true;
  } else {
    // do RANSAC 2D2D for initialization only
    if (!isScaleRefined_ && numStatesToRefineScale_ <= 5) {
      std::cout << "Performing Ransac2d2d to refine scale." << std::endl;
      int numInliers =
          runRansac2d2dToRefineScale(estimator, params, mfId, mfId, true, removeOutliers, rotationOnly_tmp);
      std::cout << "ransac2d2d num_inliers: " << numInliers << std::endl;
      if (numInliers > 15) {
        std::cout << "To refine scale: num_state " << numStatesToRefineScale_ << " num_inliers: " << numInliers
                  << std::endl;
        numStatesToRefineScale_ += 1;  // Sharmin
      }
    }

    if (!isScaleRefined_ && numStatesToRefineScale_ > 5) {
      const bool hasEnoughScaleData =
          (ransac2d2d_R_WS.size() > 1) && (ransac2d2d_t_WC.size() == ransac2d2d_R_WS.size()) &&
          (ransac2d2d_t_SC.size() == ransac2d2d_R_WS.size()) &&
          (imu_interal_dt.size() >= ransac2d2d_R_WS.size() - 1) &&
          (imu_interal_deltaP.size() >= ransac2d2d_R_WS.size() - 1) &&
          (imu_interal_deltaV.size() >= ransac2d2d_R_WS.size() - 1);

      if (!hasEnoughScaleData) {
        LOG(WARNING) << "Skipping scale refinement due to insufficient/intermittent data buffers.";
      } else {
        int n_state = numStatesToRefineScale_ * 3 + 3 + 1;

        Eigen::MatrixXd A{n_state, n_state};
        A.setZero();
        Eigen::VectorXd b{n_state};
        b.setZero();

        for (size_t i = 0; i < ransac2d2d_R_WS.size() - 1; i++) {
          Eigen::MatrixXd tmp_A(6, 10);
          tmp_A.setZero();
          Eigen::VectorXd tmp_b(6);
          tmp_b.setZero();

          double dt = imu_interal_dt.at(i);

          tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
          tmp_A.block<3, 3>(0, 6) = ransac2d2d_R_WS.at(i).transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
          tmp_A.block<3, 1>(0, 9) =
              ransac2d2d_R_WS.at(i).transpose() * (ransac2d2d_t_WC.at(i + 1) - ransac2d2d_t_WC.at(i)) / 100.0;
          tmp_b.block<3, 1>(0, 0) =
              imu_interal_deltaP.at(i) +
              ransac2d2d_R_WS.at(i).transpose() * ransac2d2d_R_WS.at(i + 1) * ransac2d2d_t_SC.at(i + 1) -
              ransac2d2d_t_SC.at(i);
          tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
          tmp_A.block<3, 3>(3, 3) = ransac2d2d_R_WS.at(i).transpose() * ransac2d2d_R_WS.at(i + 1);
          tmp_A.block<3, 3>(3, 6) = ransac2d2d_R_WS.at(i).transpose() * dt * Eigen::Matrix3d::Identity();
          tmp_b.block<3, 1>(3, 0) = imu_interal_deltaV.at(i);

          Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
          cov_inv.setIdentity();

          Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
          Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

          A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
          b.segment<6>(i * 3) += r_b.head<6>();

          A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
          b.tail<4>() += r_b.tail<4>();

          A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
          A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
        }

        A = A * 1000.0;
        b = b * 1000.0;
        Eigen::VectorXd x = A.ldlt().solve(b);
        double s = x(n_state - 1) / 100.0;

        std::cout << "================= Scale =================== " << std::endl;
        std::cout << "estimated scale: " << s << std::endl;

        isScaleRefined_ = true;
      }
    }
  }

  // rotationOnly = rotationOnly_tmp;

  /***********   End Scale Refinement: Added by Sharmin ****************/

  // TODO(test): for more than 2 cameras check that there were no duplications!

  // TODO(test): ensure 1-1 matching.

  // TODO(test): no RANSAC ?

  for (size_t im = 0; im < camNumber; im++) {
    const size_t ksize = multiFrame->numKeypoints(im);
    for (size_t k = 0; k < ksize; ++k) {
      if (multiFrame->landmarkId(im, k) != 0) {
        continue;  // already identified correspondence
      }
      multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());
    }
  }

  // Added by Sharmin
  /*for (size_t im = 0; im < camNumber; im++) {
      const size_t ksize = multiFrame->contour_numKeypoints(im);
      for (size_t k = 0; k < ksize; ++k) {
        if (multiFrame->landmarkId(im, k) != 0) {
          continue;  // already identified correspondence
        }
        multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());
      }
   }*/
  // End Added by Sharmin
}

// Perform 3D/2D RANSAC.
int Frontend::runRansac3d2d(okvis::Estimator& estimator,
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers,
                            bool lastFrameDiagnostics) {
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    return 1;
  }

  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(estimator, nCameraSystem, currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  const bool diagnosticsEnabled = featureDistributionDiagnosticsOptIn() && lastFrameDiagnostics && isInitialized_;
  const size_t finiteSupport = diagnosticsEnabled ? finiteSupportInCurrentFrame(estimator, currentFrame) : 0;
  const int64_t diagnosticSecond = static_cast<int64_t>(std::floor(currentFrame->timestamp().toSec()));
  const bool emitDiagnostic = ransac3d2dDiagnosticRateLimiter_.shouldEmit(
      diagnosticsEnabled, currentFrame->id(), diagnosticSecond, finiteSupport);
  if (numCorrespondences < 5) {
    if (emitDiagnostic) {
      LOG(INFO) << "[RANSAC_POSE_MODEL_DIAGNOSTIC] family=3d2d context=last_frame"
                << " timestamp=" << std::setprecision(15) << currentFrame->timestamp().toSec()
                << " frame=" << currentFrame->id() << " finite_support=" << finiteSupport
                << " correspondences=" << numCorrespondences << " model=none inliers=0 ratio=0"
                << " eligible=0 eligibility_reason=few_correspondences";
    }
    return numCorrespondences;
  }

  // create a RelativePoseSac problem and RANSAC
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 9;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  const bool modelSuccess = ransac.computeModel(0);

  // assign transformation
  numInliers = ransac.inliers_.size();
  if (emitDiagnostic) {
    double rotationDeltaDeg = std::numeric_limits<double>::quiet_NaN();
    double translationDeltaMeters = std::numeric_limits<double>::quiet_NaN();
    Eigen::Quaterniond modelQuaternion = Eigen::Quaterniond::Identity();
    Eigen::Vector3d modelTranslation = Eigen::Vector3d::Zero();
    if (modelSuccess) {
      modelQuaternion = Eigen::Quaterniond(ransac.model_coefficients_.leftCols<3>());
      modelTranslation = ransac.model_coefficients_.col(3);
      okvis::kinematics::Transformation T_WS;
      estimator.get_T_WS(currentFrame->id(), T_WS);
      rotationDeltaDeg = ransac_pose_diagnostics::rotationDeltaDegrees(
          ransac.model_coefficients_.leftCols<3>(), T_WS.C());
      translationDeltaMeters = (modelTranslation - T_WS.r()).norm();
    }
    const double inlierRatio = safeRatio(numInliers, numCorrespondences);
    const auto eligibility = ransac_pose_diagnostics::absolutePoseEligibility(
        modelSuccess, numCorrespondences, numInliers, inlierRatio, rotationDeltaDeg,
        translationDeltaMeters);
    LOG(INFO) << "[RANSAC_POSE_MODEL_DIAGNOSTIC] family=3d2d context=last_frame"
              << " timestamp=" << std::setprecision(15) << currentFrame->timestamp().toSec()
              << " frame=" << currentFrame->id() << " finite_support=" << finiteSupport
              << " correspondences=" << numCorrespondences << " model=gp3p"
              << " model_success=" << modelSuccess << " inliers=" << numInliers
              << " ratio=" << inlierRatio << " model_tx=" << modelTranslation.x()
              << " model_ty=" << modelTranslation.y() << " model_tz=" << modelTranslation.z()
              << " model_qx=" << modelQuaternion.x() << " model_qy=" << modelQuaternion.y()
              << " model_qz=" << modelQuaternion.z() << " model_qw=" << modelQuaternion.w()
              << " propagated_delta_rotation_deg=" << rotationDeltaDeg
              << " propagated_delta_translation_m=" << translationDeltaMeters
              << " eligible=" << eligibility.eligible
              << " eligibility_reason=" << eligibility.reason;
  }
  if (numInliers >= 10) {
    // kick out outliers:
    std::vector<bool> inliers(numCorrespondences, false);
    for (size_t k = 0; k < ransac.inliers_.size(); ++k) {
      inliers.at(ransac.inliers_.at(k)) = true;
    }

    for (size_t k = 0; k < numCorrespondences; ++k) {
      if (!inliers[k]) {
        // get the landmark id:
        size_t camIdx = adapter.camIndex(k);
        size_t keypointIdx = adapter.keypointIndex(k);
        uint64_t lmId = currentFrame->landmarkId(camIdx, keypointIdx);

        // reset ID:
        currentFrame->setLandmarkId(camIdx, keypointIdx, 0);

        // remove observation
        if (removeOutliers) {
          estimator.removeObservation(lmId, currentFrame->id(), camIdx, keypointIdx);
        }
      }
    }
  }
  return numInliers;
}

// Added by Sharmin
// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2dToRefineScale(okvis::Estimator& estimator,
                                         const okvis::VioParameters& params,
                                         uint64_t currentFrameId,
                                         uint64_t olderFrameId,
                                         bool initializePose,
                                         bool removeOutliers,
                                         bool& rotationOnly) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  if (numCameras < 2) {
    return 0;
  }

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  // for (size_t im = 0; im < numCameras; ++im) {

  // relative pose adapter for Kneip toolchain
  // Sharmin: to get relative pose in a stereo pair
  opengv::relative_pose::FrameRelativeAdapter adapter(
      estimator, params.nCameraSystem, olderFrameId, 0, currentFrameId, 1);

  size_t numCorrespondences = adapter.getNumberCorrespondences();

  if (numCorrespondences < 10)
    return 0;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!

  // try both the rotation-only RANSAC and the relative one:

  // create a RelativePoseSac problem and RANSAC
  typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
  opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
  std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(new FrameRotationOnlySacProblem(adapter, false));
  rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
  rotation_only_ransac.threshold_ = 9;
  rotation_only_ransac.max_iterations_ = 50;

  // run the ransac
  rotation_only_ransac.computeModel(0);

  // get quality
  int rotation_only_inliers = rotation_only_ransac.inliers_.size();
  float rotation_only_ratio = static_cast<float>(rotation_only_inliers) / static_cast<float>(numCorrespondences);

  // now the rel_pose one:
  typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
  opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
  std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
      new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS, false));
  rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
  rel_pose_ransac.threshold_ = 9;  // (1.0 - cos(0.5/600));
  rel_pose_ransac.max_iterations_ = 50;

  // run the ransac
  rel_pose_ransac.computeModel(0);

  // assess success
  int rel_pose_inliers = rel_pose_ransac.inliers_.size();
  float rel_pose_ratio = static_cast<float>(rel_pose_inliers) / static_cast<float>(numCorrespondences);

  // decide on success and fill inliers
  std::vector<bool> inliers(numCorrespondences, false);
  if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8) {
    if (rotation_only_inliers > 10) {
      rotation_only_success = true;
    }
    rotationOnly = true;
    totalInlierNumber += rotation_only_inliers;
    for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
      inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
    }
  } else {
    if (rel_pose_inliers > 10) {
      rel_pose_success = true;
    }
    totalInlierNumber += rel_pose_inliers;
    for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
      inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
    }
  }

  // failure?
  if (!rel_pose_success) {
    return 0;
  }

  // Sharmin: No need to kick out outliers!
  // Sharmin: This matching/ransac only for computing scale. This is not added/removed from estimator

  // initialize pose if necessary
  if (initializePose && !isInitialized_) {
    if (rel_pose_success) LOG(INFO) << "Refining scale from 2D-2D RANSAC";

    Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();

    okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
    uint64_t idA = olderFrameId;  // idA, id0 same
    uint64_t id0 = currentFrameId;
    estimator.getCameraSensorStates(idA, 0, T_SCA);  // Sharmin: camIndex = 0
    estimator.get_T_WS(idA, T_WSA);
    estimator.getCameraSensorStates(id0, 1, T_SC0);  // Sharmin: camIndex = 1
    estimator.get_T_WS(id0, T_WS0);
    if (rel_pose_success) {
      // update pose
      // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
      T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;

      // initialize with projected length according to motion prior.

      // CA==C1, C0==C2
      okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse() * T_WSA.inverse() * T_WS0 * T_SC0;
      T_C1C2_mat.topRightCorner<3, 1>() =
          T_C1C2_mat.topRightCorner<3, 1>() *
          std::max(0.0, static_cast<double>(T_C1C2_mat.topRightCorner<3, 1>().transpose() * T_C1C2.r()));
    }
    okvis::kinematics::Transformation T_WS_ransac2d2d =
        T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat) * T_SC0.inverse();
    ransac2d2d_R_WS.push_back(T_WS_ransac2d2d.q().toRotationMatrix());

    okvis::kinematics::Transformation T_WCA_ransac = T_WS_ransac2d2d * T_SCA;
    ransac2d2d_t_WC.push_back(T_WCA_ransac.r());

    okvis::kinematics::Transformation T_SCA_ransac =
        T_SCA * okvis::kinematics::Transformation(T_C1C2_mat) * T_SC0.inverse() * T_SCA;
    ransac2d2d_t_SC.push_back(T_SCA_ransac.r());

    Eigen::Vector3d del_p, del_v;
    double del_t;
    // estimator.getImuPreIntegral(idA, del_p, del_v, del_t);
    // imu_interal_deltaP.push_back(del_p);
    // imu_interal_deltaV.push_back(del_v);
    // imu_interal_dt.push_back(del_t);

    // set.
    // TODO(Sharmin)
    // estimator.set_T_WS(
    //    id0, T_WS_ransac2d2d);
  }
  //}

  if (rel_pose_success) {
    return totalInlierNumber;
  } else {
    // rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2d(okvis::Estimator& estimator,
                            const okvis::VioParameters& params,
                            uint64_t currentFrameId,
                            uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly,
                            bool lastFrameDiagnostics) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im) {
    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::FrameRelativeAdapter adapter(
        estimator, params.nCameraSystem, olderFrameId, im, currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();

    const std::shared_ptr<okvis::MultiFrame> diagnosticFrame = estimator.multiFrame(currentFrameId);
    const bool diagnosticsEnabled = featureDistributionDiagnosticsOptIn() && lastFrameDiagnostics && isInitialized_;
    const size_t finiteSupport = diagnosticsEnabled ? finiteSupportInCurrentFrame(estimator, diagnosticFrame) : 0;
    const int64_t diagnosticSecond = static_cast<int64_t>(std::floor(diagnosticFrame->timestamp().toSec()));
    const bool emitDiagnostic = ransac2d2dDiagnosticRateLimiter_.shouldEmit(
        diagnosticsEnabled, currentFrameId, diagnosticSecond, finiteSupport);
    PendingIntensityCounts pendingCounts;
    if (emitDiagnostic) {
      for (size_t k = 0; k < numCorrespondences; ++k) {
        const size_t idxB = adapter.getMatchKeypointIdxB(k);
        const uint64_t id = diagnosticFrame->landmarkId(im, idxB);
        if (id != 0 && !estimator.isLandmarkAdded(id)) {
          incrementPendingIntensity(diagnosticFrame->keypointIntensityFlags(im, idxB), &pendingCounts);
        }
      }
    }

    if (numCorrespondences < 10) {
      if (emitDiagnostic) {
        LOG(INFO) << "[RANSAC_POSE_MODEL_DIAGNOSTIC] family=2d2d context=last_frame"
                  << " timestamp=" << std::setprecision(15) << diagnosticFrame->timestamp().toSec()
                  << " frame_a=" << olderFrameId << " frame_b=" << currentFrameId
                  << " camera=" << im << " finite_support=" << finiteSupport
                  << " correspondences=" << numCorrespondences
                  << " pending_correspondences=" << pendingCounts.total
                  << " pending_original_bright=" << pendingCounts.originalBright
                  << " pending_original_dark=" << pendingCounts.originalDark
                  << " pending_original_unknown=" << pendingCounts.originalUnknown
                  << " model=none inliers=0 ratio=0 eligible=0"
                  << " eligibility_reason=few_correspondences translation_semantics=nonmetric_direction_only";
      }
      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
    }

    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(new FrameRotationOnlySacProblem(adapter, false));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    const bool rotationModelSuccess = rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = static_cast<float>(rotation_only_inliers) / static_cast<float>(numCorrespondences);

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
        new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS, false));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;  // (1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    const bool relativeModelSuccess = rel_pose_ransac.computeModel(0);

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = static_cast<float>(rel_pose_inliers) / static_cast<float>(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    const bool choseRotationOnly = rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8;
    if (choseRotationOnly) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rotationOnly = true;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    } else {
      if (rel_pose_inliers > 10) {
        rel_pose_success = true;
      }
      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
        inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
      }
    }

    if (emitDiagnostic) {
      Eigen::Matrix3d modelRotation = Eigen::Matrix3d::Identity();
      Eigen::Vector3d modelTranslation = Eigen::Vector3d::Zero();
      const bool chosenModelSuccess = choseRotationOnly ? rotationModelSuccess : relativeModelSuccess;
      const size_t chosenInliers = choseRotationOnly ? rotation_only_inliers : rel_pose_inliers;
      const double chosenRatio = choseRotationOnly ? rotation_only_ratio : rel_pose_ratio;
      if (chosenModelSuccess) {
        modelRotation = choseRotationOnly ? rotation_only_ransac.model_coefficients_
                                          : rel_pose_ransac.model_coefficients_.leftCols<3>();
        if (!choseRotationOnly) modelTranslation = rel_pose_ransac.model_coefficients_.col(3);
      }
      okvis::kinematics::Transformation T_WSa, T_SCa, T_WSb, T_SCb;
      estimator.get_T_WS(olderFrameId, T_WSa);
      estimator.getCameraSensorStates(olderFrameId, im, T_SCa);
      estimator.get_T_WS(currentFrameId, T_WSb);
      estimator.getCameraSensorStates(currentFrameId, im, T_SCb);
      const okvis::kinematics::Transformation propagatedRelative =
          ransac_pose_diagnostics::estimatorRelativeCameraPose(T_WSa, T_SCa, T_WSb, T_SCb);
      const double rotationDeltaDeg = chosenModelSuccess
          ? ransac_pose_diagnostics::rotationDeltaDegrees(modelRotation, propagatedRelative.C())
          : std::numeric_limits<double>::quiet_NaN();
      const double directionAgreement = (!choseRotationOnly && chosenModelSuccess)
          ? ransac_pose_diagnostics::directionCosine(modelTranslation, propagatedRelative.r())
          : std::numeric_limits<double>::quiet_NaN();
      const auto eligibility = ransac_pose_diagnostics::relativeRotationEligibility(
          chosenModelSuccess, numCorrespondences, chosenInliers, chosenRatio, rotationDeltaDeg);
      const Eigen::Quaterniond modelQuaternion(modelRotation);
      LOG(INFO) << "[RANSAC_POSE_MODEL_DIAGNOSTIC] family=2d2d context=last_frame"
                << " timestamp=" << std::setprecision(15) << diagnosticFrame->timestamp().toSec()
                << " frame_a=" << olderFrameId << " frame_b=" << currentFrameId
                << " camera=" << im << " finite_support=" << finiteSupport
                << " correspondences=" << numCorrespondences
                << " pending_correspondences=" << pendingCounts.total
                << " pending_original_bright=" << pendingCounts.originalBright
                << " pending_original_dark=" << pendingCounts.originalDark
                << " pending_original_unknown=" << pendingCounts.originalUnknown
                << " rotation_inliers=" << rotation_only_inliers
                << " rotation_ratio=" << rotation_only_ratio
                << " relative_inliers=" << rel_pose_inliers << " relative_ratio=" << rel_pose_ratio
                << " model=" << (choseRotationOnly ? "rotation_only" : "relative_pose")
                << " model_success=" << chosenModelSuccess << " inliers=" << chosenInliers
                << " ratio=" << chosenRatio << " model_qx=" << modelQuaternion.x()
                << " model_qy=" << modelQuaternion.y() << " model_qz=" << modelQuaternion.z()
                << " model_qw=" << modelQuaternion.w()
                << " propagated_delta_rotation_deg=" << rotationDeltaDeg
                << " translation_direction_cosine=" << directionAgreement
                << " translation_semantics=nonmetric_direction_only"
                << " propagated_translation_norm_m=" << propagatedRelative.r().norm()
                << " eligible=" << eligibility.eligible
                << " eligibility_reason=" << eligibility.reason;
    }

    // failure?
    if (!rotation_only_success && !rel_pose_success) {
      continue;
    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(currentFrameId);

    for (size_t k = 0; k < numCorrespondences; ++k) {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (!inliers[k]) {
        uint64_t lmId = multiFrame->landmarkId(im, idxB);
        // reset ID:
        multiFrame->setLandmarkId(im, idxB, 0);
        // remove observation
        if (removeOutliers) {
          if (lmId != 0 && estimator.isLandmarkAdded(lmId)) {
            estimator.removeObservation(lmId, currentFrameId, im, idxB);
          }
        }
      }
    }

    // initialize pose if necessary
    if (initializePose && !isInitialized_) {
      if (rel_pose_success)
        LOG(INFO) << "Initializing pose from 2D-2D RANSAC";
      else
        LOG(INFO) << "Initializing pose from 2D-2D RANSAC: orientation only";

      Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();

      okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
      uint64_t idA = olderFrameId;
      uint64_t id0 = currentFrameId;
      estimator.getCameraSensorStates(idA, im, T_SCA);
      estimator.get_T_WS(idA, T_WSA);
      estimator.getCameraSensorStates(id0, im, T_SC0);
      estimator.get_T_WS(id0, T_WS0);
      if (rel_pose_success) {
        // update pose
        // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
        T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;

        // initialize with projected length according to motion prior.

        okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse() * T_WSA.inverse() * T_WS0 * T_SC0;
        T_C1C2_mat.topRightCorner<3, 1>() =
            T_C1C2_mat.topRightCorner<3, 1>() *
            std::max(0.0, static_cast<double>(T_C1C2_mat.topRightCorner<3, 1>().transpose() * T_C1C2.r()));
      } else {
        // rotation only assigned...
        T_C1C2_mat.topLeftCorner<3, 3>() = rotation_only_ransac.model_coefficients_;
      }

      // set.
      estimator.set_T_WS(id0, T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat) * T_SC0.inverse());
    }
  }

  if (rel_pose_success || rotation_only_success) {
    return totalInlierNumber;
  } else {
    rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void Frontend::initialiseBriskFeatureDetectors() {
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->lock();
  }
  featureDetectors_.clear();
  descriptorExtractors_.clear();
  size_t detectorMaximumKeypoints = briskDetectionMaximumKeypoints_;
  if (spatialBalancingParameters_.enable) {
    const size_t multiplier = static_cast<size_t>(std::max(1, spatialBalancingParameters_.candidateMultiplier));
    if (detectorMaximumKeypoints <= std::numeric_limits<size_t>::max() / multiplier) {
      detectorMaximumKeypoints *= multiplier;
    }
  }
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectors_.push_back(std::shared_ptr<cv::FeatureDetector>(
#ifdef __ARM_NEON__
        new cv::GridAdaptedFeatureDetector(new cv::FastFeatureDetector(briskDetectionThreshold_),
                                           detectorMaximumKeypoints,
                                           7,
                                           4)));  // from config file, except the 7x4...
#else
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(briskDetectionThreshold_,
                                                                           briskDetectionOctaves_,
                                                                           briskDetectionAbsoluteThreshold_,
                                                                           detectorMaximumKeypoints)));
    std::cout << "briskDetectionThreshold_: " << briskDetectionThreshold_ << std::endl;
    std::cout << "briskDetectionOctaves_: " << briskDetectionOctaves_ << std::endl;
    std::cout << "briskDetectionAbsoluteThreshold_: " << briskDetectionAbsoluteThreshold_ << std::endl;
    std::cout << "briskDetectionMaximumKeypoints_: " << briskDetectionMaximumKeypoints_
              << " (candidate cap " << detectorMaximumKeypoints << ")" << std::endl;
#endif
    descriptorExtractors_.push_back(std::shared_ptr<cv::DescriptorExtractor>(
        new brisk::BriskDescriptorExtractor(briskDescriptionRotationInvariance_, briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->unlock();
  }
}

}  // namespace okvis
