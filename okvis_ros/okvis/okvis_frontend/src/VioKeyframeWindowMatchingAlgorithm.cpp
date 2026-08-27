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
 *  Created on: Oct 17, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file VioKeyframeWindowMatchingAlgorithm.cpp
 * @brief Source file for the VioKeyframeWindowMatchingAlgorithm class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/ceres/ReprojectionError.hpp>

// cameras and distortions
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <Eigen/Eigenvalues>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <okvis/cameras/DoubleSphereCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <opencv2/features2d/features2d.hpp>  // for cv::KeyPoint
/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {

bool retentionHammingProbeWindow(const okvis::Time& timestamp) {
  const char* value = std::getenv("SVIN2_ENABLE_RETENTION_ELIGIBILITY_DIAGNOSTICS");
  if (value == nullptr || std::strcmp(value, "1") != 0) return false;

  const double seconds = timestamp.toSec();
  return (seconds >= 11265.0 && seconds <= 11275.0) ||
         (seconds >= 11281.0 && seconds <= 11291.0) ||
         (seconds >= 11336.0 && seconds <= 11346.0);
}

bool promotionDiagnosticsOptIn(bool configuredDefault) {
  const char* value = std::getenv("SVIN2_ENABLE_PROMOTION_DIAGNOSTICS");
  return value == nullptr ? configuredDefault : std::strcmp(value, "1") == 0;
}

}  // namespace

// Constructor.
// Sharmin: useSCM = false
template <class CAMERA_GEOMETRY_T>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::VioKeyframeWindowMatchingAlgorithm(
    okvis::Estimator& estimator,  // NOLINT
    int matchingType,
    float distanceThreshold,
    bool usePoseUncertainty,
    bool useSCM,
    bool promotionDiagnostics) {
  matchingType_ = matchingType;
  distanceThreshold_ = distanceThreshold;
  estimator_ = &estimator;
  usePoseUncertainty_ = usePoseUncertainty;
  useSCM_ = useSCM;
  promotionDiagnosticsEnabled_ = promotionDiagnosticsOptIn(promotionDiagnostics);
  // std::cout << "USE SCM: "<< useSCM <<std::endl;
}

template <class CAMERA_GEOMETRY_T>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::~VioKeyframeWindowMatchingAlgorithm() {}

// Set which frames to match.
template <class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setFrames(uint64_t mfIdA,
                                                                      uint64_t mfIdB,
                                                                      size_t camIdA,
                                                                      size_t camIdB) {
  OKVIS_ASSERT_TRUE(Exception, !(mfIdA == mfIdB && camIdA == camIdB), "trying to match identical frames.");

  // remember indices
  mfIdA_ = mfIdA;
  mfIdB_ = mfIdB;
  camIdA_ = camIdA;
  camIdB_ = camIdB;
  // frames and related information
  frameA_ = estimator_->multiFrame(mfIdA_);
  frameB_ = estimator_->multiFrame(mfIdB_);

  // focal length
  fA_ = frameA_->geometryAs<CAMERA_GEOMETRY_T>(camIdA_)->focalLengthU();
  fB_ = frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->focalLengthU();

  // calculate the relative transformations and uncertainties
  // TODO(donno) if and what we need here - I'll see
  estimator_->getCameraSensorStates(mfIdA_, camIdA, T_SaCa_);
  estimator_->getCameraSensorStates(mfIdB_, camIdB, T_SbCb_);
  estimator_->get_T_WS(mfIdA_, T_WSa_);
  estimator_->get_T_WS(mfIdB_, T_WSb_);
  T_SaW_ = T_WSa_.inverse();
  T_SbW_ = T_WSb_.inverse();
  T_WCa_ = T_WSa_ * T_SaCa_;
  T_WCb_ = T_WSb_ * T_SbCb_;
  T_CaW_ = T_WCa_.inverse();
  T_CbW_ = T_WCb_.inverse();
  T_CaCb_ = T_WCa_.inverse() * T_WCb_;
  T_CbCa_ = T_CaCb_.inverse();

  validRelativeUncertainty_ = false;
}

// Set the matching type.
template <class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setMatchingType(int matchingType) {
  matchingType_ = matchingType;
}

// This will be called exactly once for each call to DenseMatcher::match().
template <class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::doSetup() {
  // setup stereo triangulator
  // first, let's get the relative uncertainty.
  okvis::kinematics::Transformation T_CaCb;
  Eigen::Matrix<double, 6, 6> UOplus = Eigen::Matrix<double, 6, 6>::Zero();
  if (usePoseUncertainty_) {
    OKVIS_THROW(Exception, "No pose uncertainty use currently supported");
  } else {
    UOplus.setIdentity();
    UOplus.bottomRightCorner<3, 3>() *= 1e-8;
    uint64_t currentId = estimator_->currentFrameId();
    if (estimator_->isInImuWindow(currentId) && (mfIdA_ != mfIdB_)) {
      okvis::SpeedAndBias speedAndBias;
      estimator_->getSpeedAndBias(currentId, 0, speedAndBias);
      double scale = std::max(1.0, speedAndBias.head<3>().norm());
      UOplus.topLeftCorner<3, 3>() *= (scale * scale) * 1.0e-2;
    } else {
      UOplus.topLeftCorner<3, 3>() *= 4e-8;
    }
  }

  // now set the frames and uncertainty
  probabilisticStereoTriangulator_.resetFrames(frameA_, frameB_, camIdA_, camIdB_, T_CaCb_, UOplus);

  // reset the match counter
  numMatches_ = 0;
  numUncertainMatches_ = 0;
  numBearingOnlyMatches_ = 0;
  numFiniteMatches_ = 0;
  numPromotions_ = 0;
  numPromotionAttempts_ = 0;
  numPromotionDeferredObservations_ = 0;
  numPromotionDeferredFrames_ = 0;
  numPromotionDeferredParallax_ = 0;
  rejectedCandidates_.store(0, std::memory_order_relaxed);

  const size_t numA = frameA_->numKeypoints(camIdA_);
  skipA_.clear();
  skipA_.resize(numA, false);
  raySigmasA_.resize(numA);
  retentionHammingProbeEnabled_ =
      matchingType_ == Match3D2D && retentionHammingProbeWindow(frameB_->timestamp());
  retentionHammingProbes_.clear();
  retentionHammingSourceLandmarkIds_.clear();
  if (retentionHammingProbeEnabled_) {
    retentionHammingSourceLandmarkIds_.assign(numA, 0);
    retentionHammingProbes_.resize(numA);
    for (size_t k = 0; k < numA; ++k) {
      const uint64_t landmarkId = frameA_->landmarkId(camIdA_, k);
      if (landmarkId != 0 && estimator_->isLandmarkAdded(landmarkId) &&
          estimator_->isLandmarkInitialized(landmarkId)) {
        retentionHammingSourceLandmarkIds_[k] = landmarkId;
      }
    }
  }
  // calculate projections only once
  if (matchingType_ == Match3D2D) {
    // allocate a matrix to store projections
    projectionsIntoB_ = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sizeA(), 2);
    projectionsIntoBUncertainties_ = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(sizeA() * 2, 2);

    // do the projections for each keypoint, if applicable
    for (size_t k = 0; k < numA; ++k) {
      uint64_t lm_id = frameA_->landmarkId(camIdA_, k);

      if (lm_id == 0 || !estimator_->isLandmarkAdded(lm_id)) {
        // this can happen, if you called the 2D-2D version just before,
        // without inserting the landmark into the graph
        skipA_[k] = true;
        continue;
      }

      okvis::MapPoint landmark;
      estimator_->getLandmark(lm_id, landmark);
      Eigen::Vector4d hp_W = landmark.point;

      if (!estimator_->isLandmarkInitialized(lm_id)) {
        skipA_[k] = true;
        continue;
      }

      // project (distorted)
      Eigen::Vector2d kptB;
      const Eigen::Vector4d hp_Cb = T_CbW_ * hp_W;
      if (frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(hp_Cb, &kptB) !=
          okvis::cameras::CameraBase::ProjectionStatus::Successful) {
        skipA_[k] = true;
        continue;
      }

      if (landmark.observations.size() < 2) {
        estimator_->setLandmarkInitialized(lm_id, false);
        skipA_[k] = true;
        continue;
      }

      // project and get uncertainty
      Eigen::Matrix<double, 2, 4> jacobian;
      Eigen::Matrix4d P_C = Eigen::Matrix4d::Zero();
      P_C.topLeftCorner<3, 3>() = UOplus.topLeftCorner<3, 3>();  // get from before -- velocity scaled
      frameB_->geometryAs<CAMERA_GEOMETRY_T>(camIdB_)->projectHomogeneous(hp_Cb, &kptB, &jacobian);
      projectionsIntoBUncertainties_.block<2, 2>(2 * k, 0) = jacobian * P_C * jacobian.transpose();
      projectionsIntoB_.row(k) = kptB;

      // precalculate ray uncertainties
      double keypointAStdDev;
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;  // (sqrt(MeasurementCovariance.norm()) / _fA)
    }
  } else {
    for (size_t k = 0; k < numA; ++k) {
      double keypointAStdDev;
      frameA_->getKeypointSize(camIdA_, k, keypointAStdDev);
      keypointAStdDev = 0.8 * keypointAStdDev / 12.0;
      raySigmasA_[k] = sqrt(sqrt(2)) * keypointAStdDev / fA_;
      if (frameA_->landmarkId(camIdA_, k) == 0) {
        continue;
      }
      if (estimator_->isLandmarkAdded(frameA_->landmarkId(camIdA_, k))) {
        if (estimator_->isLandmarkInitialized(frameA_->landmarkId(camIdA_, k))) {
          skipA_[k] = true;
        }
      }
    }
  }
  const size_t numB = frameB_->numKeypoints(camIdB_);
  skipB_.clear();
  skipB_.reserve(numB);
  raySigmasB_.resize(numB);
  // do the projections for each keypoint, if applicable
  if (matchingType_ == Match3D2D) {
    for (size_t k = 0; k < numB; ++k) {
      okvis::MapPoint landmark;
      if (frameB_->landmarkId(camIdB_, k) != 0 && estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k))) {
        estimator_->getLandmark(frameB_->landmarkId(camIdB_, k), landmark);
        skipB_.push_back(landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, k)) !=
                         landmark.observations.end());
      } else {
        skipB_.push_back(false);
      }
      double keypointBStdDev;
      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;
    }
  } else {
    for (size_t k = 0; k < numB; ++k) {
      double keypointBStdDev;
      frameB_->getKeypointSize(camIdB_, k, keypointBStdDev);
      keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
      raySigmasB_[k] = sqrt(sqrt(2)) * keypointBStdDev / fB_;

      if (frameB_->landmarkId(camIdB_, k) == 0) {
        skipB_.push_back(false);
        continue;
      }
      if (estimator_->isLandmarkAdded(frameB_->landmarkId(camIdB_, k))) {
        skipB_.push_back(estimator_->isLandmarkInitialized(frameB_->landmarkId(camIdB_, k)));  // old: isSet - check.
      } else {
        skipB_.push_back(false);
      }
    }
  }
}

// What is the size of list A?
template <class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::sizeA() const {
  return frameA_->numKeypoints(camIdA_);
}
// What is the size of list B?
template <class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::sizeB() const {
  return frameB_->numKeypoints(camIdB_);
}

// Set the distance threshold for which matches exceeding it will not be returned as matches.
template <class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setDistanceThreshold(float distanceThreshold) {
  distanceThreshold_ = distanceThreshold;
}

// Get the distance threshold for which matches exceeding it will not be returned as matches.
template <class CAMERA_GEOMETRY_T>
float VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::distanceThreshold() const {
  return distanceThreshold_;
}

template <class CAMERA_GEOMETRY_T>
std::vector<okvis::KeypointIdentifier>
VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::trackObservations(uint64_t trackId) const {
  std::vector<okvis::KeypointIdentifier> observations;
  for (size_t age = 0; age < estimator_->numFrames(); ++age) {
    const uint64_t frameId = estimator_->frameIdByAge(age);
    const std::shared_ptr<okvis::MultiFrame> multiFrame = estimator_->multiFrame(frameId);
    for (size_t cameraIndex = 0; cameraIndex < multiFrame->numFrames(); ++cameraIndex) {
      for (size_t keypointIndex = 0; keypointIndex < multiFrame->numKeypoints(cameraIndex); ++keypointIndex) {
        if (multiFrame->landmarkId(cameraIndex, keypointIndex) == trackId) {
          observations.emplace_back(frameId, cameraIndex, keypointIndex);
        }
      }
    }
  }
  return observations;
}

template <class CAMERA_GEOMETRY_T>
bool VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::mergePendingTracks(uint64_t targetId,
                                                                                uint64_t sourceId) {
  if (targetId == sourceId) return true;
  if (estimator_->isLandmarkAdded(targetId) || estimator_->isLandmarkAdded(sourceId)) return false;

  const std::vector<okvis::KeypointIdentifier> targetObservations = trackObservations(targetId);
  const std::vector<okvis::KeypointIdentifier> sourceObservations = trackObservations(sourceId);
  std::set<std::pair<uint64_t, size_t>> occupiedImages;
  for (const okvis::KeypointIdentifier& observation : targetObservations) {
    occupiedImages.emplace(observation.frameId, observation.cameraIndex);
  }
  for (const okvis::KeypointIdentifier& observation : sourceObservations) {
    if (occupiedImages.count(std::make_pair(observation.frameId, observation.cameraIndex)) != 0) {
      return false;
    }
  }

  for (const okvis::KeypointIdentifier& observation : sourceObservations) {
    estimator_->multiFrame(observation.frameId)
        ->setLandmarkId(observation.cameraIndex, observation.keypointIndex, targetId);
  }
  return true;
}

template <class CAMERA_GEOMETRY_T>
bool VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::observationIsValid(
    const okvis::KeypointIdentifier& observation,
    const Eigen::Vector4d& homogeneousPoint_W) const {
  if (!homogeneousPoint_W.allFinite()) return false;

  okvis::kinematics::Transformation T_WS;
  okvis::kinematics::Transformation T_SC;
  estimator_->get_T_WS(observation.frameId, T_WS);
  estimator_->getCameraSensorStates(observation.frameId, observation.cameraIndex, T_SC);
  const Eigen::Vector4d homogeneousPoint_C = T_SC.inverse() * T_WS.inverse() * homogeneousPoint_W;

  const std::shared_ptr<okvis::MultiFrame> multiFrame = estimator_->multiFrame(observation.frameId);
  Eigen::Vector2d prediction;
  const cameras::CameraBase::ProjectionStatus status =
      multiFrame->geometryAs<camera_geometry_t>(observation.cameraIndex)
          ->projectHomogeneous(homogeneousPoint_C, &prediction);
  if (status != cameras::CameraBase::ProjectionStatus::Successful || !prediction.allFinite()) return false;

  Eigen::Vector2d measurement;
  double keypointSize = 0.0;
  if (!multiFrame->getKeypoint(observation.cameraIndex, observation.keypointIndex, measurement) ||
      !multiFrame->getKeypointSize(observation.cameraIndex, observation.keypointIndex, keypointSize) ||
      !measurement.allFinite() || !std::isfinite(keypointSize)) {
    return false;
  }
  const double keypointStdDev = 0.8 * keypointSize / 12.0;
  if (!(keypointStdDev > 0.0) || !std::isfinite(keypointStdDev)) return false;
  const double chi2 = (prediction - measurement).squaredNorm() / (keypointStdDev * keypointStdDev);
  return std::isfinite(chi2) && chi2 <= 4.0;
}

template <class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::promoteTrack(
    uint64_t trackId,
    const Eigen::Vector4d& /*homogeneousPoint_W*/) {
  constexpr size_t kMinimumPromotionObservations = 3;
  constexpr size_t kMinimumPromotionFrames = 3;
  constexpr double kMinimumPromotionParallaxRadians = 0.25 * M_PI / 180.0;

  ++numPromotionAttempts_;
  const std::vector<okvis::KeypointIdentifier> observations = trackObservations(trackId);

  struct BearingObservation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    okvis::KeypointIdentifier identifier;
    Eigen::Vector3d cameraCenter_W;
    Eigen::Vector3d ray_W;
  };
  std::vector<BearingObservation, Eigen::aligned_allocator<BearingObservation>> bearingObservations;
  bearingObservations.reserve(observations.size());
  std::set<uint64_t> distinctFrames;
  double minimumTimestamp = std::numeric_limits<double>::infinity();
  double maximumTimestamp = -std::numeric_limits<double>::infinity();

  for (const okvis::KeypointIdentifier& observation : observations) {
    const std::shared_ptr<okvis::MultiFrame> multiFrame = estimator_->multiFrame(observation.frameId);
    okvis::kinematics::Transformation T_WS;
    okvis::kinematics::Transformation T_SC;
    Eigen::Vector2d measurement;
    Eigen::Vector3d ray_C;
    if (!estimator_->get_T_WS(observation.frameId, T_WS) ||
        !estimator_->getCameraSensorStates(observation.frameId, observation.cameraIndex, T_SC)) {
      continue;
    }
    if (!multiFrame->getKeypoint(observation.cameraIndex, observation.keypointIndex, measurement) ||
        !measurement.allFinite() ||
        !multiFrame->geometryAs<camera_geometry_t>(observation.cameraIndex)
             ->backProject(measurement, &ray_C) ||
        !ray_C.allFinite() || !(ray_C.norm() > 1.0e-12)) {
      continue;
    }
    const okvis::kinematics::Transformation T_WC = T_WS * T_SC;
    const Eigen::Vector3d cameraCenter_W = T_WC.r();
    const Eigen::Vector3d ray_W = (T_WC.C() * ray_C).normalized();
    if (!cameraCenter_W.allFinite() || !ray_W.allFinite()) continue;

    bearingObservations.push_back({observation, cameraCenter_W, ray_W});
    distinctFrames.insert(observation.frameId);
    const double timestamp = multiFrame->timestamp().toSec();
    minimumTimestamp = std::min(minimumTimestamp, timestamp);
    maximumTimestamp = std::max(maximumTimestamp, timestamp);
  }

  double maximumBaseline = 0.0;
  double maximumParallax = 0.0;
  for (size_t i = 0; i < bearingObservations.size(); ++i) {
    for (size_t j = i + 1; j < bearingObservations.size(); ++j) {
      maximumBaseline = std::max(
          maximumBaseline,
          (bearingObservations[i].cameraCenter_W - bearingObservations[j].cameraCenter_W).norm());
      const double crossNorm = bearingObservations[i].ray_W.cross(bearingObservations[j].ray_W).norm();
      const double dot = std::max(
          -1.0, std::min(1.0, bearingObservations[i].ray_W.dot(bearingObservations[j].ray_W)));
      maximumParallax = std::max(maximumParallax, std::atan2(crossNorm, dot));
    }
  }

  // Find the point closest to all viewing rays. This is a joint multiview
  // estimate; the latest two-view triangulation only triggers this attempt and
  // is deliberately not used as the depth seed.
  const auto triangulateRays = [&](const std::vector<size_t>& indices,
                                   Eigen::Vector3d& point_W) {
    Eigen::Matrix3d normalMatrix = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rightHandSide = Eigen::Vector3d::Zero();
    for (const size_t index : indices) {
      const BearingObservation& observation = bearingObservations[index];
      const Eigen::Matrix3d perpendicular =
          Eigen::Matrix3d::Identity() - observation.ray_W * observation.ray_W.transpose();
      normalMatrix += perpendicular;
      rightHandSide += perpendicular * observation.cameraCenter_W;
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(normalMatrix);
    if (eigenSolver.info() != Eigen::Success || !eigenSolver.eigenvalues().allFinite() ||
        !(eigenSolver.eigenvalues().minCoeff() > 1.0e-12)) {
      return false;
    }
    point_W = eigenSolver.eigenvectors() *
              eigenSolver.eigenvalues().cwiseInverse().asDiagonal() *
              eigenSolver.eigenvectors().transpose() * rightHandSide;
    return point_W.allFinite();
  };

  std::vector<size_t> allIndices(bearingObservations.size());
  std::iota(allIndices.begin(), allIndices.end(), 0);
  Eigen::Vector3d point_W = Eigen::Vector3d::Zero();
  bool finitePoint = triangulateRays(allIndices, point_W);
  Eigen::Vector4d homogeneousPoint_W(point_W.x(), point_W.y(), point_W.z(), 1.0);

  std::vector<size_t> inlierIndices;
  std::vector<okvis::KeypointIdentifier> validObservations;
  if (finitePoint) {
    inlierIndices.reserve(bearingObservations.size());
    for (size_t i = 0; i < bearingObservations.size(); ++i) {
      const double signedDistance =
          (point_W - bearingObservations[i].cameraCenter_W).dot(bearingObservations[i].ray_W);
      if (std::isfinite(signedDistance) && signedDistance > 0.0 &&
          observationIsValid(bearingObservations[i].identifier, homogeneousPoint_W)) {
        inlierIndices.push_back(i);
      }
    }

    // Refit once using only geometrically consistent observations, then make
    // the final inlier decision against the refined point.
    Eigen::Vector3d refinedPoint_W;
    if (inlierIndices.size() >= kMinimumPromotionObservations &&
        triangulateRays(inlierIndices, refinedPoint_W)) {
      point_W = refinedPoint_W;
      homogeneousPoint_W << point_W, 1.0;
      validObservations.clear();
      for (const size_t index : inlierIndices) {
        const double signedDistance =
            (point_W - bearingObservations[index].cameraCenter_W)
                .dot(bearingObservations[index].ray_W);
        if (std::isfinite(signedDistance) && signedDistance > 0.0 &&
            observationIsValid(bearingObservations[index].identifier, homogeneousPoint_W)) {
          validObservations.push_back(bearingObservations[index].identifier);
        }
      }
    }
  }

  double minimumRange = std::numeric_limits<double>::infinity();
  double maximumRange = 0.0;
  if (finitePoint) {
    for (const BearingObservation& observation : bearingObservations) {
      const double range = (point_W - observation.cameraCenter_W).norm();
      if (std::isfinite(range) && range > 0.0) {
        minimumRange = std::min(minimumRange, range);
        maximumRange = std::max(maximumRange, range);
      }
    }
  }

  const double frameSpanSeconds =
      std::isfinite(minimumTimestamp) && std::isfinite(maximumTimestamp)
          ? maximumTimestamp - minimumTimestamp
          : 0.0;
  const double baselineRangeRatio =
      maximumRange > 0.0 ? maximumBaseline / maximumRange : 0.0;

  const char* decision = "promoted";
  bool canPromote = true;
  if (!finitePoint || bearingObservations.size() != observations.size()) {
    decision = "invalid_geometry";
    canPromote = false;
    ++numPromotionDeferredObservations_;
  } else if (bearingObservations.size() < kMinimumPromotionObservations) {
    decision = "insufficient_observations";
    canPromote = false;
    ++numPromotionDeferredObservations_;
  } else if (distinctFrames.size() < kMinimumPromotionFrames) {
    decision = "insufficient_distinct_frames";
    canPromote = false;
    ++numPromotionDeferredFrames_;
  } else if (!(maximumParallax >= kMinimumPromotionParallaxRadians)) {
    decision = "insufficient_multiview_parallax";
    canPromote = false;
    ++numPromotionDeferredParallax_;
  } else if (validObservations.size() < kMinimumPromotionObservations) {
    decision = "insufficient_multiview_inliers";
    canPromote = false;
    ++numPromotionDeferredObservations_;
  }

  if (promotionDiagnosticsEnabled_) {
    static std::atomic<uint64_t> promotionDiagnosticCount{0};
    const uint64_t diagnosticCount =
        promotionDiagnosticCount.fetch_add(1, std::memory_order_relaxed) + 1;
    if (diagnosticCount <= 50 || diagnosticCount % 1000 == 0) {
      LOG(INFO) << "[PROMOTION_DIAGNOSTIC] event=" << diagnosticCount
                << " track_id=" << trackId << " decision=" << decision
                << " observations_total=" << observations.size()
                << " bearing_observations=" << bearingObservations.size()
                << " observations_valid=" << validObservations.size()
                << " distinct_frames=" << distinctFrames.size()
                << " frame_span_s=" << frameSpanSeconds
                << " baseline_m=" << maximumBaseline
                << " range_min_m=" << (std::isfinite(minimumRange) ? minimumRange : 0.0)
                << " range_max_m=" << maximumRange
                << " baseline_range_ratio=" << baselineRangeRatio
                << " parallax_deg=" << maximumParallax * 180.0 / M_PI
                << " required_parallax_deg=" << kMinimumPromotionParallaxRadians * 180.0 / M_PI;
    }
  }
  if (!canPromote) return 0;

  // Associations are intentionally preserved for every deferred attempt. Once
  // promotion succeeds, discard only observations rejected by the joint fit.
  const std::set<okvis::KeypointIdentifier> validObservationSet(
      validObservations.begin(), validObservations.end());
  for (const okvis::KeypointIdentifier& observation : observations) {
    if (validObservationSet.count(observation) == 0) {
      estimator_->multiFrame(observation.frameId)
          ->setLandmarkId(observation.cameraIndex, observation.keypointIndex, 0);
    }
  }

  if (!estimator_->isLandmarkAdded(trackId)) {
    if (!estimator_->addLandmark(trackId, homogeneousPoint_W)) return 0;
  } else {
    estimator_->setLandmark(trackId, homogeneousPoint_W);
  }
  estimator_->setLandmarkInitialized(trackId, true);
  for (const okvis::KeypointIdentifier& observation : validObservations) {
    estimator_->addObservation<camera_geometry_t>(
        trackId, observation.frameId, observation.cameraIndex, observation.keypointIndex);
  }
  return validObservations.size();
}

// Geometric verification of a match.
template <class CAMERA_GEOMETRY_T>
bool VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::verifyMatch(size_t indexA, size_t indexB) const {
  if (matchingType_ == Match2D2D) {
    // potential 2d2d match - verify by triangulation
    Eigen::Vector4d hP;
    bool isParallel;
    bool valid = probabilisticStereoTriangulator_.stereoTriangulate(
        indexA, indexB, hP, isParallel, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (valid) {
      return true;
    }
    rejectedCandidates_.fetch_add(1, std::memory_order_relaxed);
  } else {
    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);

    // uncertainty
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
    Eigen::Matrix2d U = Eigen::Matrix2d::Identity() * keypointBStdDev * keypointBStdDev +
                        projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);
    Eigen::Vector2d err = kptB - keypointBMeasurement;
    const int chi2 = err.transpose() * U.inverse() * err;

    if (chi2 < 4.0) {
      return true;
    }
  }
  return false;
}

// A function that tells you how many times setMatching() will be called.
template <class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::reserveMatches(size_t /*numMatches*/) {
  // DenseMatcher calls this only after waitForEmptyQueue(), so all per-source
  // probe updates made by its workers are complete before this deterministic
  // aggregation/logging pass.
  if (!retentionHammingProbeEnabled_) return;

  struct HammingAggregate {
    uint32_t bestHamming = std::numeric_limits<uint32_t>::max();
    size_t comparisonCount = 0;
    size_t geometryPassCount = 0;
    size_t sourceKeypointCount = 0;
  };
  std::map<uint64_t, HammingAggregate> aggregates;
  for (size_t indexA = 0; indexA < retentionHammingProbes_.size(); ++indexA) {
    const uint64_t landmarkId = retentionHammingSourceLandmarkIds_[indexA];
    if (landmarkId == 0) continue;

    HammingAggregate& aggregate = aggregates[landmarkId];
    const RetentionHammingProbe& probe = retentionHammingProbes_[indexA];
    aggregate.bestHamming = std::min(aggregate.bestHamming, probe.bestHamming);
    aggregate.comparisonCount += probe.comparisonCount;
    aggregate.geometryPassCount += probe.geometryPassCount;
    ++aggregate.sourceKeypointCount;
  }

  const double newestFrameTime = frameB_ ? frameB_->timestamp().toSec() : 0.0;
  for (const auto& entry : aggregates) {
    const uint64_t landmarkId = entry.first;
    const HammingAggregate& aggregate = entry.second;
    const bool hasBest = aggregate.bestHamming != std::numeric_limits<uint32_t>::max();
    LOG(INFO) << std::setprecision(17) << "[RETENTION_HAMMING_PROBE]"
              << " newest_frame_id=" << mfIdB_ << " newest_frame_time_s=" << newestFrameTime
              << " landmark_id=" << landmarkId
              << " source_keypoint_count=" << aggregate.sourceKeypointCount
              << " comparison_count=" << aggregate.comparisonCount
              << " best_hamming=" << (hasBest ? static_cast<int>(aggregate.bestHamming) : -1)
              << " best_hamming_lt_60=" << (hasBest && aggregate.bestHamming < 60)
              << " geometry_pass_count=" << aggregate.geometryPassCount;
  }
}

// Get the number of matches.
template <class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numMatches() {
  return numMatches_;
}

// Get the number of matches.
/*template<class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::scm_numMatches() {
  return scm_numMatches_;
}*/

// Get the number of uncertain matches.
template <class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numUncertainMatches() {
  return numUncertainMatches_;
}

template <class CAMERA_GEOMETRY_T>
size_t VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::numActivePendingTracks() const {
  std::set<uint64_t> pendingTrackIds;
  for (size_t age = 0; age < estimator_->numFrames(); ++age) {
    const std::shared_ptr<okvis::MultiFrame> multiFrame = estimator_->multiFrame(estimator_->frameIdByAge(age));
    for (size_t cameraIndex = 0; cameraIndex < multiFrame->numFrames(); ++cameraIndex) {
      for (size_t keypointIndex = 0; keypointIndex < multiFrame->numKeypoints(cameraIndex); ++keypointIndex) {
        const uint64_t trackId = multiFrame->landmarkId(cameraIndex, keypointIndex);
        if (trackId != 0 && !estimator_->isLandmarkAdded(trackId)) pendingTrackIds.insert(trackId);
      }
    }
  }
  return pendingTrackIds.size();
}

// At the end of the matching step, this function is called once
// for each pair of matches discovered.
template <class CAMERA_GEOMETRY_T>
void VioKeyframeWindowMatchingAlgorithm<CAMERA_GEOMETRY_T>::setBestMatch(size_t indexA,
                                                                         size_t indexB,
                                                                         double distance) {
  // assign correspondences
  uint64_t lmIdA = frameA_->landmarkId(camIdA_, indexA);
  uint64_t lmIdB = frameB_->landmarkId(camIdB_, indexB);

  if (matchingType_ == Match2D2D) {
    Eigen::Vector4d hP_Ca;
    bool canBeInitialized;
    const bool valid = probabilisticStereoTriangulator_.stereoTriangulate(
        indexA, indexB, hP_Ca, canBeInitialized, std::max(raySigmasA_[indexA], raySigmasB_[indexB]));
    if (!valid) return;

    if (canBeInitialized) {
      Eigen::Matrix3d pointUOplus_A;
      probabilisticStereoTriangulator_.getUncertainty(indexA, indexB, hP_Ca, pointUOplus_A, canBeInitialized);
    }

    uint64_t trackId = 0;
    if (lmIdA == 0 && lmIdB == 0) {
      trackId = okvis::IdProvider::instance().newId();
    } else if (lmIdA == 0) {
      trackId = lmIdB;
    } else if (lmIdB == 0) {
      trackId = lmIdA;
    } else if (lmIdA == lmIdB) {
      trackId = lmIdA;
    } else {
      const bool landmarkAExists = estimator_->isLandmarkAdded(lmIdA);
      const bool landmarkBExists = estimator_->isLandmarkAdded(lmIdB);
      if (landmarkAExists || landmarkBExists || !mergePendingTracks(lmIdA, lmIdB)) return;
      trackId = lmIdA;
    }

    // A weak 2D-2D match must never attach new observations to a landmark
    // that is already in the graph. The 3D-2D matching path owns that case.
    if (!canBeInitialized && estimator_->isLandmarkAdded(trackId)) return;

    frameA_->setLandmarkId(camIdA_, indexA, trackId);
    frameB_->setLandmarkId(camIdB_, indexB, trackId);

    if (!canBeInitialized) {
      ++numBearingOnlyMatches_;
      ++numMatches_;
      return;
    }

    const bool wasPending = !estimator_->isLandmarkAdded(trackId);
    const Eigen::Vector4d homogeneousPoint_W = T_WCa_ * hP_Ca;
    const size_t observationCount = promoteTrack(trackId, homogeneousPoint_W);
    if (observationCount < 2) return;
    ++numFiniteMatches_;
    if (wasPending) ++numPromotions_;

  } else {
    OKVIS_ASSERT_TRUE_DBG(Exception, lmIdB == 0, "bug. Id in frame B already set.");

    // get projection into B
    Eigen::Vector2d kptB = projectionsIntoB_.row(indexA);
    Eigen::Vector2d keypointBMeasurement;
    frameB_->getKeypoint(camIdB_, indexB, keypointBMeasurement);

    Eigen::Vector2d err = kptB - keypointBMeasurement;
    double keypointBStdDev;
    frameB_->getKeypointSize(camIdB_, indexB, keypointBStdDev);
    keypointBStdDev = 0.8 * keypointBStdDev / 12.0;
    Eigen::Matrix2d U_tot = Eigen::Matrix2d::Identity() * keypointBStdDev * keypointBStdDev +
                            projectionsIntoBUncertainties_.block<2, 2>(2 * indexA, 0);

    const double chi2 = err.transpose().eval() * U_tot.inverse() * err;

    if (chi2 > 4.0) {
      return;
    }

    // saturate allowed image uncertainty
    if (U_tot.norm() > 25.0 / (keypointBStdDev * keypointBStdDev * sqrt(2))) {
      numUncertainMatches_++;
      // return;
    }

    frameB_->setLandmarkId(camIdB_, indexB, lmIdA);
    lmIdB = lmIdA;
    okvis::MapPoint landmark;
    estimator_->getLandmark(lmIdA, landmark);

    // initialize in graph
    if (landmark.observations.find(okvis::KeypointIdentifier(mfIdB_, camIdB_, indexB)) ==
        landmark.observations.end()) {  // ensure no double observations...
      OKVIS_ASSERT_TRUE(Exception, estimator_->isLandmarkAdded(lmIdB), "not added");
      estimator_->addObservation<camera_geometry_t>(lmIdB, mfIdB_, camIdB_, indexB);
    }
  }
  if (retentionHammingProbeEnabled_ && matchingType_ == Match3D2D && lmIdA != 0) {
    const double newestFrameTime = frameB_ ? frameB_->timestamp().toSec() : 0.0;
    LOG(INFO) << std::setprecision(17) << "[RETENTION_HAMMING_SELECTED]"
              << " newest_frame_id=" << mfIdB_ << " newest_frame_time_s=" << newestFrameTime
              << " landmark_id=" << lmIdA << " source_keypoint_index=" << indexA
              << " target_camera_index=" << camIdB_ << " target_keypoint_index=" << indexB
              << " selected_distance=" << distance;
  }
  numMatches_++;
  /*if (useSCM_) // Added by Sharmin
          scm_numMatches_++;*/
}

// TODO: add support for double sphere camera here.
template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> >;

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> >;

template class VioKeyframeWindowMatchingAlgorithm<
    okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> >;

template class VioKeyframeWindowMatchingAlgorithm<
  okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion> >;

}  // namespace okvis
