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
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Estimator.cpp
 * @brief Source file for the Estimator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <set>

#include <okvis/Estimator.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/DepthError.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SonarError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
/// \brief okvis Main namespace of this package.
namespace okvis {

namespace {
bool environmentFlagOrDefault(const char* name, bool defaultValue) {
  const char* value = std::getenv(name);
  if (value == nullptr) return defaultValue;
  return std::strcmp(value, "1") == 0;
}
}  // namespace

// Constructor if a ceres map is already available.
Estimator::Estimator(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : mapPtr_(mapPtr),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0) {}

// The default constructor.
Estimator::Estimator()
    : mapPtr_(new okvis::ceres::Map()),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0) {}

Estimator::~Estimator() {}

// Add a camera to the configuration. Sensors can only be added and never removed.
int Estimator::addCamera(const ExtrinsicsEstimationParameters& extrinsicsEstimationParameters) {
  extrinsicsEstimationParametersVec_.push_back(extrinsicsEstimationParameters);
  return extrinsicsEstimationParametersVec_.size() - 1;
}

// Add an IMU to the configuration.
int Estimator::addImu(const ImuParameters& imuParameters) {
  if (imuParametersVec_.size() > 1) {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return imuParametersVec_.size() - 1;
}

// Remove all cameras from the configuration
void Estimator::clearCameras() { extrinsicsEstimationParametersVec_.clear(); }

// Remove all IMUs from the configuration.
void Estimator::clearImus() { imuParametersVec_.clear(); }

// Add a pose to the state.
bool Estimator::addStates(okvis::MultiFramePtr multiFrame,
                          const okvis::ImuMeasurementDeque& imuMeasurements,
                          bool asKeyframe,
                          const okvis::SonarMeasurementDeque& sonarMeasurements, /* @Sharmin */
                          const okvis::DepthMeasurementDeque& depthMeasurements,
                          double firstDepth) {
  // Note Sharmin: this is for imu propagation no matter isScaleRefined_ is true/false.
  // TODO(Sharmin): Start actual optimization when isScaleRefined_ = true.

  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    bool success0 = initPoseFromImu(imuMeasurements, T_WS);
    OKVIS_ASSERT_TRUE_DBG(Exception, success0, "pose could not be initialized from imu measurements.");
    if (!success0) return false;

    // Sharmin
    if (multiFrame->numKeypoints() > 10) {
      LOG(INFO) << "Initialized!! As enough keypoints are found";
      LOG(INFO) << "Initial T_WS: " << T_WS.parameters();
    } else {
      LOG(WARNING) << "Not enought multiframe Points: " << multiFrame->numKeypoints();
      return false;
    }
    // End Sharmin

    speedAndBias.setZero();
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else {
    // get the previous states
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;
    uint64_t speedAndBias_id =
        statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(
        Exception, mapPtr_->parameterBlockExists(T_WS_id), "this is an okvis bug. previous pose does not exist.");
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(mapPtr_->parameterBlockPtr(T_WS_id))->estimate();
    // OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias =
        std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(mapPtr_->parameterBlockPtr(speedAndBias_id))
            ->estimate();

    Eigen::Vector3d acc_doubleinteg;
    Eigen::Vector3d acc_integ;
    double Del_t;

    int numUsedImuMeasurements = ceres::ImuError::propagation(imuMeasurements,
                                                              imuParametersVec_.at(0),
                                                              T_WS,
                                                              speedAndBias,
                                                              statesMap_.rbegin()->second.timestamp,
                                                              multiFrame->timestamp(),
                                                              0,
                                                              0,
                                                              acc_doubleinteg,
                                                              acc_integ,
                                                              Del_t);

    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1, "propagation failed");
    if (numUsedImuMeasurements < 1) {
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }

    // Added by Sharmin
    setImuPreIntegral(multiFrame->id(), acc_doubleinteg, acc_integ, Del_t);
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // Added by Sharmin
  stateCount_ = stateCount_ + 1;

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(
      Exception, statesMap_.find(states.id) == statesMap_.end(), "pose ID" << states.id << " was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id, multiFrame->timestamp()));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;

  if (statesMap_.empty()) {
    referencePoseId_ = states.id;  // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  } else {
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  }

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;

  // initialize new sensor states
  // cameras:
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists = true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists = false;
    if (((extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation < 1e-12) ||
         (extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation < 1e-12)) &&
        (statesMap_.size() > 1)) {
      // use the same block...
      cameraInfos.at(CameraSensorStates::T_SCi).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id;
    } else {
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(
          new okvis::ceres::PoseParameterBlock(T_SC, id, multiFrame->timestamp()));
      if (!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr, ceres::Map::Pose6d)) {
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
    }
    // update the states info
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(
        new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if (!mapPtr_->addParameterBlock(speedAndBiasParameterBlock)) {
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  // @Sharmin
  // Depth
  if (depthMeasurements.size() != 0) {
    // Though there should not be more than one depth data
    double mean_depth = 0.0;
    for (auto depthMeasurements_it = depthMeasurements.begin(); depthMeasurements_it != depthMeasurements.end();
         ++depthMeasurements_it) {
      mean_depth += depthMeasurements_it->measurement.depth;
    }
    mean_depth = mean_depth / depthMeasurements.size();

    double information_depth = 5.0;  // TODO(Sharmin) doublre check with the manual

    std::shared_ptr<ceres::DepthError> depthError(new ceres::DepthError(mean_depth, information_depth, firstDepth));
    mapPtr_->addResidualBlock(depthError, NULL, poseParameterBlock);
    std::cout << "Residual block z: " << (*poseParameterBlock->parameters()) + 2 << std::endl;
  }

  // @Sharmin
  if (sonarMeasurements.size() != 0) {
    // Sonar
    std::vector<Eigen::Vector3d> landmarkSubset;
    Eigen::Vector3d sonar_landmark;
    double range = 0.0, heading = 0.0;
    auto last_sonarMeasurement_it = sonarMeasurements.rbegin();

    // Taking the nearest range value to the n+1 th frame
    range = last_sonarMeasurement_it->measurement.range;
    heading = last_sonarMeasurement_it->measurement.heading;

    okvis::kinematics::Transformation T_WSo = T_WS * sonarParameters_.T_SSo;

    okvis::kinematics::Transformation sonar_point(Eigen::Vector3d(range * cos(heading), range * sin(heading), 0.0),
                                                  Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
    okvis::kinematics::Transformation T_WSo_point = T_WSo * sonar_point;

    sonar_landmark = T_WSo_point.r();
    // std::cout << "T_WSo: " << T_WSo.r() << std::endl;
    // std::cout << "T_WSo_point: " << sonar_landmark << std::endl;

    // may be the reverse searching is faster
    for (PointMap::const_reverse_iterator rit = landmarksMap_.rbegin(); rit != landmarksMap_.rend(); ++rit) {
      Eigen::Vector3d visual_landmark;
      if (fabs(rit->second.point[3]) > 1.0e-8) {
        visual_landmark = (rit->second.point / rit->second.point[3]).head<3>();
        // LOG (INFO) << "Visual Landmark: " << visual_landmark;
      }
      // double distance_from_sonar = (sonar_landmark - visual_landmark).norm();  //Euclidean distance
      if (fabs(sonar_landmark[0] - visual_landmark[0]) < 0.1 && fabs(sonar_landmark[1] - visual_landmark[1]) < 0.1 &&
          fabs(sonar_landmark[2] - visual_landmark[2]) < 0.1) {
        // TODO(sharmin) parameter!!
        // searching around 10 cm of sonar landmark

        landmarkSubset.push_back(visual_landmark);
      }
    }

    // std::cout << "Size of visual patch: "<<landmarkSubset.size() << std::endl;

    if (landmarkSubset.size() > 0) {
      // LOG (INFO) << " Sonar added for ceres optimization";
      // @Sharmin
      // add sonarError and related addResidualBlock
      double information_sonar = 1.0;  // TODO(sharmin) calculate properly?

      std::shared_ptr<ceres::SonarError> sonarError(
          new ceres::SonarError(sonarParameters_, range, heading, information_sonar, landmarkSubset));
      mapPtr_->addResidualBlock(sonarError, NULL, poseParameterBlock);
    }
    // End @Sharmin
  }

  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  if (statesMap_.size() == 1) {
    // let's add a prior
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(5, 5) = 1.0e8;
    information(0, 0) = 1.0e8;
    information(1, 1) = 1.0e8;
    information(2, 2) = 1.0e8;
    std::shared_ptr<ceres::PoseError> poseError(new ceres::PoseError(T_WS, information));
    /*auto id2= */ mapPtr_->addResidualBlock(poseError, NULL, poseParameterBlock);
    // mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      double translationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_translation;
      double translationVariance = translationStdev * translationStdev;
      double rotationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_orientation;
      double rotationVariance = rotationStdev * rotationStdev;
      if (translationVariance > 1.0e-16 && rotationVariance > 1.0e-16) {
        const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
        std::shared_ptr<ceres::PoseError> cameraPoseError(
            new ceres::PoseError(T_SC, translationVariance, rotationVariance));
        // add to map
        mapPtr_->addResidualBlock(
            cameraPoseError,
            NULL,
            mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        // mapPtr_->isJacobianCorrect(id,1.0e-6);
      } else {
        mapPtr_->setParameterBlockConstant(
            states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id);
      }
    }
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      Eigen::Matrix<double, 6, 1> variances;
      // get these from parameter file
      const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
      const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
      std::shared_ptr<ceres::SpeedAndBiasError> speedAndBiasError(
          new ceres::SpeedAndBiasError(speedAndBias, 1.0, sigma_bg * sigma_bg, sigma_ba * sigma_ba));
      // add to map
      mapPtr_->addResidualBlock(
          speedAndBiasError,
          NULL,
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      // mapPtr_->isJacobianCorrect(id,1.0e-6);
    }

  } else {
    // add IMU error terms
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      std::shared_ptr<ceres::ImuError> imuError(new ceres::ImuError(
          imuMeasurements, imuParametersVec_.at(i), lastElementIterator->second.timestamp, states.timestamp));
      /*::ceres::ResidualBlockId id = */ mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),
          mapPtr_->parameterBlockPtr(
              lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id),
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      // imuError->setRecomputeInformation(false);
      // mapPtr_->isJacobianCorrect(id,1.0e-9);
      // imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      if (lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id !=
          states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id) {
        // i.e. they are different estimated variables, so link them with a temporal error term
        double dt = (states.timestamp - lastElementIterator->second.timestamp).toSec();
        double translationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation;
        double translationVariance = translationSigmaC * translationSigmaC * dt;
        double rotationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation;
        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError(
            new ceres::RelativePoseError(translationVariance, rotationVariance));
        mapPtr_->addResidualBlock(
            relativeExtrinsicsError,
            NULL,
            mapPtr_->parameterBlockPtr(
                lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id),
            mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        // mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
    }
    // only camera. this is slightly inconsistent, since the IMU error term contains both
    // a term for global states as well as for the sensor-internal ones (i.e. biases).
    // TODO: magnetometer, pressure, ...
  }

  return true;
}

// Add a landmark.
bool Estimator::addLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock(
      new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));
  if (!mapPtr_->addParameterBlock(pointParameterBlock, okvis::ceres::Map::HomogeneousPoint)) {
    return false;
  }

  // remember
  double dist = std::numeric_limits<double>::max();
  if (fabs(landmark[3]) > 1.0e-8) {
    dist = (landmark / landmark[3]).head<3>().norm();  // euclidean distance
  }
  landmarksMap_.insert(std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

// Remove an observation from a landmark.
bool Estimator::removeObservation(::ceres::ResidualBlockId residualBlockId) {
  const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator it = mapPoint.observations.begin();
       it != mapPoint.observations.end();) {
    if (it->second == uint64_t(residualBlockId)) {
      it = mapPoint.observations.erase(it);
    } else {
      it++;
    }
  }
  // remove residual block
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Remove an observation from a landmark, if available.
bool Estimator::removeObservation(uint64_t landmarkId, uint64_t poseId, size_t camIdx, size_t keypointIdx) {
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    for (PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      LOG(INFO) << it->first << ", no. obs = " << it->second.observations.size();
    }
    LOG(INFO) << landmarksMap_.at(landmarkId).id;
  }
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");

  okvis::KeypointIdentifier kid(poseId, camIdx, keypointIdx);
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t>::iterator it = mapPoint.observations.find(kid);
  if (it == landmarksMap_.at(landmarkId).observations.end()) {
    return false;  // observation not present
  }

  // remove residual block
  mapPtr_->removeResidualBlock(reinterpret_cast< ::ceres::ResidualBlockId>(it->second));

  // remove also in local map
  mapPoint.observations.erase(it);

  return true;
}

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template <class T>
bool vectorContains(const std::vector<T>& vector, const T& query) {
  for (size_t i = 0; i < vector.size(); ++i) {
    if (vector[i] == query) {
      return true;
    }
  }
  return false;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool Estimator::applyMarginalizationStrategy(size_t numKeyframes,
                                             size_t numImuFrames,
                                             okvis::MapPointVector& removedLandmarks) {
  const bool retentionDiagnosticsEnabled = environmentFlagOrDefault(
      "SVIN2_ENABLE_RETENTION_ELIGIBILITY_DIAGNOSTICS", retentionDiagnosticsEnabled_);
  const double retentionDiagnosticTimestamp =
      statesMap_.empty() ? 0.0 : statesMap_.rbegin()->second.timestamp.toSec();
  const bool retentionDiagnosticWindow =
      (retentionDiagnosticTimestamp >= 11265.0 &&
       retentionDiagnosticTimestamp <= 11275.0) ||
      (retentionDiagnosticTimestamp >= 11281.0 &&
       retentionDiagnosticTimestamp <= 11291.0) ||
      (retentionDiagnosticTimestamp >= 11336.0 &&
       retentionDiagnosticTimestamp <= 11346.0);
  size_t retentionDiagnosticsEmitted = 0;
  constexpr size_t kMaximumRetentionDiagnosticsPerMarginalization = 512;

  const bool retentionPolicyEnabled = environmentFlagOrDefault(
      "SVIN2_ENABLE_FINITE_LANDMARK_RETENTION", finiteLandmarkRetentionEnabled_);
  const double retentionPolicyTimestamp =
      statesMap_.empty() ? 0.0 : statesMap_.rbegin()->second.timestamp.toSec();
  constexpr size_t kMaximumNewRetentionsPerMarginalization = 16;
  constexpr size_t kMaximumTrackedRetentions = 64;
  constexpr double kMaximumRetentionAgeSeconds = 1.0;
  size_t retentionPolicyNewlySelected = 0;
  size_t retentionPolicyCandidates = 0;
  size_t retentionPolicyRemoved = 0;
  size_t retentionPolicyExpired = 0;
  size_t retentionPolicyRecovered = 0;
  size_t retentionPolicyIneligible = 0;
  size_t retentionPolicyCapacityRejected = 0;
  size_t retentionPolicyRemovedObservations = 0;
  std::set<uint64_t> retentionReleasedThisCall;
  std::set<uint64_t> retentionExpiredThisCall;

  auto releaseRetentionTracking = [&](uint64_t landmarkId, const char* reason) {
    std::map<uint64_t, double>::iterator retainedIt =
        retainedLandmarkFirstRetentionTime_.find(landmarkId);
    if (retainedIt == retainedLandmarkFirstRetentionTime_.end()) return;
    retainedLandmarkFirstRetentionTime_.erase(retainedIt);
    retentionReleasedThisCall.insert(landmarkId);
    if (retentionPolicyEnabled && retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
      LOG(INFO) << "[RETENTION_POLICY_DECISION]"
                << " landmark_id=" << landmarkId
                << " decision=release"
                << " reason=" << reason;
    }
  };

  auto logRetentionPolicySummary = [&]() {
    if (!retentionPolicyEnabled || !retentionDiagnosticsEnabled) return;
    LOG(INFO) << std::setprecision(17)
              << "[RETENTION_POLICY_SUMMARY]"
              << " newest_time_s=" << retentionPolicyTimestamp
              << " candidates=" << retentionPolicyCandidates
              << " newly_selected=" << retentionPolicyNewlySelected
              << " removed_observations=" << retentionPolicyRemovedObservations
              << " tracked_total=" << retainedLandmarkFirstRetentionTime_.size()
              << " released_removed=" << retentionPolicyRemoved
              << " released_expired=" << retentionPolicyExpired
              << " released_recovered=" << retentionPolicyRecovered
              << " released_ineligible=" << retentionPolicyIneligible
              << " capacity_rejected=" << retentionPolicyCapacityRejected;
  };

  if (retentionPolicyEnabled) {
    for (std::map<uint64_t, double>::iterator retainedIt =
             retainedLandmarkFirstRetentionTime_.begin();
         retainedIt != retainedLandmarkFirstRetentionTime_.end();) {
      const uint64_t landmarkId = retainedIt->first;
      const double firstRetentionTimestamp = retainedIt->second;
      const bool removed = landmarksMap_.find(landmarkId) == landmarksMap_.end() ||
                           !mapPtr_->parameterBlockExists(landmarkId);
      const bool expired = !std::isfinite(firstRetentionTimestamp) ||
                           !std::isfinite(retentionPolicyTimestamp) ||
                           retentionPolicyTimestamp - firstRetentionTimestamp >
                               kMaximumRetentionAgeSeconds;
      if (removed || expired) {
        ++(removed ? retentionPolicyRemoved : retentionPolicyExpired);
        retainedIt = retainedLandmarkFirstRetentionTime_.erase(retainedIt);
        retentionReleasedThisCall.insert(landmarkId);
        if (expired && !removed) retentionExpiredThisCall.insert(landmarkId);
        if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
          LOG(INFO) << "[RETENTION_POLICY_DECISION]"
                    << " landmark_id=" << landmarkId
                    << " decision=release"
                    << " reason=" << (removed ? "removed" : "expired");
        }
      } else {
        ++retainedIt;
      }
    }
  }

  // keep the newest numImuFrames
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  for (size_t k = 0; k < numImuFrames; k++) {
    rit++;
    if (rit == statesMap_.rend()) {
      // nothing to do.
      logRetentionPolicySummary();
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  if (marginalizationErrorPtr_ && marginalizationResidualId_) {
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
    OKVIS_ASSERT_TRUE_DBG(Exception, success, "could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success) {
      logRetentionPolicySummary();
      return false;
    }
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<bool> keepParameterBlocks;

  if (!marginalizationErrorPtr_) {
    marginalizationErrorPtr_.reset(new ceres::MarginalizationError(*mapPtr_.get()));
  }

  // A retained landmark must remain represented only by its explicit
  // reprojection residuals.  Reject any point that is already a parameter of
  // the incoming marginalization prior, even though the prior residual itself
  // has just been detached from Map above.
  std::set<uint64_t> existingMarginalizationPriorParameterIds;
  if (marginalizationErrorPtr_) {
    std::vector<std::shared_ptr<okvis::ceres::ParameterBlock>> priorParameters;
    marginalizationErrorPtr_->getParameterBlockPtrs(priorParameters);
    for (const auto& parameter : priorParameters) {
      if (parameter) existingMarginalizationPriorParameterIds.insert(parameter->id());
    }
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeFrames;
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  while (rit != statesMap_.rend()) {
    if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) {
      removeFrames.push_back(rit->second.id);
    } else {
      countedKeyframes++;
    }
    removeAllButPose.push_back(rit->second.id);
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;  // check the next frame
  }

  struct RetentionEligibility {
    bool finiteInitialized = false;
    bool observationGraphValid = true;
    bool hasRemovalNonReprojection = false;
    bool inExistingMarginalizationPrior = false;
    size_t survivingUsableObservations = 0;
    size_t newestFrameProjectionSuccess = 0;
    double newestSurvivingObservationTime = -std::numeric_limits<double>::infinity();
    double newestObservationAge = std::numeric_limits<double>::infinity();
    bool eligible = false;
  };

  auto evaluateRetentionEligibility = [&](uint64_t landmarkId,
                                           const okvis::MapPoint& mapPoint) {
    RetentionEligibility result;
    if (!mapPoint.point.allFinite() || std::abs(mapPoint.point[3]) <= 1.0e-8 ||
        !mapPtr_->parameterBlockExists(landmarkId)) {
      result.finiteInitialized = false;
      return result;
    }
    result.finiteInitialized = isLandmarkInitialized(landmarkId);
    if (!result.finiteInitialized || statesMap_.empty()) return result;
    result.inExistingMarginalizationPrior =
        existingMarginalizationPriorParameterIds.count(landmarkId) != 0;

    const uint64_t newestFrameId = statesMap_.rbegin()->first;
    const auto newestFrameIt = multiFramePtrMap_.find(newestFrameId);
    okvis::kinematics::Transformation newestT_WS;
    if (newestFrameIt != multiFramePtrMap_.end() && newestFrameIt->second &&
        get_T_WS(newestFrameId, newestT_WS)) {
      for (size_t cameraIndex = 0;
           cameraIndex < newestFrameIt->second->numFrames(); ++cameraIndex) {
        okvis::kinematics::Transformation T_SC;
        const auto geometry = newestFrameIt->second->geometry(cameraIndex);
        if (!geometry || !getCameraSensorStates(newestFrameId, cameraIndex, T_SC)) {
          continue;
        }
        Eigen::Vector2d projection;
        if (geometry->projectHomogeneous(
                    T_SC.inverse() * newestT_WS.inverse() * mapPoint.point,
                    &projection) == cameras::CameraBase::ProjectionStatus::Successful &&
            projection.allFinite()) {
          ++result.newestFrameProjectionSuccess;
        }
      }
    }

    const ceres::Map::ResidualBlockCollection landmarkResiduals =
        mapPtr_->residuals(landmarkId);
    for (size_t r = 0; r < landmarkResiduals.size(); ++r) {
      const ceres::Map::ParameterBlockCollection residualParameters =
          mapPtr_->parameters(landmarkResiduals[r].residualBlockId);
      if (residualParameters.empty()) {
        result.observationGraphValid = false;
        continue;
      }
      const std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
              landmarkResiduals[r].errorInterfacePtr);
      const bool removalPose = vectorContains(removeFrames, residualParameters[0].first);
      if (removalPose && !reprojectionError) {
        result.hasRemovalNonReprojection = true;
      } else if (!removalPose) {
        for (size_t parameterIndex = 0;
             parameterIndex < residualParameters.size(); ++parameterIndex) {
          if (residualParameters[parameterIndex].first != landmarkId &&
              vectorContains(paremeterBlocksToBeMarginalized,
                             residualParameters[parameterIndex].first)) {
            result.observationGraphValid = false;
          }
        }
      }
    }

    for (const auto& observation : mapPoint.observations) {
      const okvis::KeypointIdentifier& keypoint = observation.first;
      const auto frameIt = multiFramePtrMap_.find(keypoint.frameId);
      const auto stateIt = statesMap_.find(keypoint.frameId);
      if (frameIt == multiFramePtrMap_.end() || stateIt == statesMap_.end() ||
          !frameIt->second || keypoint.cameraIndex >= frameIt->second->numFrames() ||
          keypoint.keypointIndex >= frameIt->second->numKeypoints(keypoint.cameraIndex)) {
        result.observationGraphValid = false;
        continue;
      }

      const ::ceres::ResidualBlockId residualId =
          reinterpret_cast<::ceres::ResidualBlockId>(observation.second);
      const auto& residualSpecs = mapPtr_->residualBlockId2ResidualBlockSpecMap();
      const auto residualSpecIt = residualSpecs.find(residualId);
      if (residualSpecIt == residualSpecs.end()) {
        result.observationGraphValid = false;
        continue;
      }
      const ceres::Map::ParameterBlockCollection residualParameters =
          mapPtr_->parameters(residualId);
      const std::shared_ptr<ceres::ReprojectionError2dBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionError2dBase>(
              residualSpecIt->second.errorInterfacePtr);
      if (!reprojectionError || residualParameters.size() < 3 ||
          residualParameters[0].first != keypoint.frameId ||
          residualParameters[1].first != landmarkId ||
          reprojectionError->cameraId() != keypoint.cameraIndex) {
        result.observationGraphValid = false;
        continue;
      }
      const Eigen::Vector2d& measurement = reprojectionError->measurement();
      const Eigen::Matrix2d& information = reprojectionError->information();
      if (!measurement.allFinite() || !information.allFinite()) {
        result.observationGraphValid = false;
        continue;
      }

      const bool survives = !vectorContains(removeFrames, keypoint.frameId);
      if (survives) {
        const double observationTime = frameIt->second->timestamp().toSec();
        if (std::isfinite(observationTime)) {
          result.newestSurvivingObservationTime =
              std::max(result.newestSurvivingObservationTime, observationTime);
        }
      }

      okvis::kinematics::Transformation T_WS;
      okvis::kinematics::Transformation T_SC;
      const auto geometry = frameIt->second->geometry(keypoint.cameraIndex);
      if (!geometry || !get_T_WS(keypoint.frameId, T_WS) ||
          !getCameraSensorStates(keypoint.frameId, keypoint.cameraIndex, T_SC)) {
        continue;
      }
      const Eigen::Vector4d point_C =
          T_SC.inverse() * T_WS.inverse() * mapPoint.point;
      Eigen::Vector2d prediction;
      const auto projectionStatus = geometry->projectHomogeneous(point_C, &prediction);
      const bool positiveDepth = point_C.allFinite() &&
                                 std::abs(point_C[3]) > 1.0e-8 &&
                                 point_C[2] / point_C[3] > 0.0;
      const double chiSquared =
          projectionStatus == cameras::CameraBase::ProjectionStatus::Successful &&
                  prediction.allFinite()
              ? (measurement - prediction).transpose() * information *
                    (measurement - prediction)
              : std::numeric_limits<double>::infinity();
      if (survives && positiveDepth && std::isfinite(chiSquared) && chiSquared <= 4.0) {
        ++result.survivingUsableObservations;
      }
    }

    result.newestObservationAge =
        retentionPolicyTimestamp - result.newestSurvivingObservationTime;
    result.eligible =
        result.observationGraphValid && !result.hasRemovalNonReprojection &&
        !result.inExistingMarginalizationPrior &&
        result.survivingUsableObservations >= 2 &&
        result.newestFrameProjectionSuccess > 0 &&
        std::isfinite(result.newestObservationAge) &&
        result.newestObservationAge >= 0.0 &&
        result.newestObservationAge <= kMaximumRetentionAgeSeconds;
    return result;
  };

  // marginalize everything but pose:
  for (size_t k = 0; k < removeAllButPose.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);
    for (size_t i = 0; i < it->second.global.size(); ++i) {
      if (i == GlobalStates::T_WS) {
        continue;  // we do not remove the pose here.
      }
      if (!it->second.global[i].exists) {
        continue;  // if it doesn't exist, we don't do anything.
      }
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.global[i].exists && checkit->second.global[i].id == it->second.global[i].id) {
        continue;
      }
      it->second.global[i].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) {
            continue;  // we do not remove the extrinsics pose here.
          }
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if (checkit->second.sensors[i][j][k].exists &&
              checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
            continue;
          }
          it->second.sensors[i][j][k].exists = false;  // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
            if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
      }
    }
  }
  // marginalize ONLY pose now:
  bool reDoFixation = false;
  for (size_t k = 0; k < removeFrames.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

    // schedule removal - but always keep the very first frame.
    // if(it != statesMap_.begin()){
    if (true) {                                              /////DEBUG
      it->second.global[GlobalStates::T_WS].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
    }

    // add remaing error terms
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r) {
      if (std::dynamic_pointer_cast<ceres::PoseError>(
              residuals[r].errorInterfacePtr)) {  // avoids linearising initial pose error
        mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
        reDoFixation = true;
        continue;
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
      if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
      }
    }

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.sensors[i][j][k].exists &&
          checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
        continue;
      }
      it->second.sensors[i][j][k].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // now finally we treat all the observations.
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size() > 0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);

    {
      for (PointMap::iterator pit = landmarksMap_.begin(); pit != landmarksMap_.end();) {
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t, bool> visibleInFrame;
        size_t obsCount = 0;
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            // since we have implemented the linearisation to account for robustification,
            // we don't kick out bad measurements here any more like
            // if(vectorContains(allLinearizedFrames,poseId)){ ...
            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
            // }
            if (vectorContains(removeFrames, poseId)) {
              skipLandmark = false;
            }
            if (poseId >= currentKfId) {
              marginalize = false;
              hasNewObservations = true;
            }
            if (vectorContains(allLinearizedFrames, poseId)) {
              visibleInFrame.insert(std::pair<uint64_t, bool>(poseId, true));
              obsCount++;
            }
          }
        }

        // The retention lifetime is a bound on graph influence, not merely on
        // accounting.  If a retained point reaches the one-second deadline
        // without acquiring a genuinely new observation, remove its genuine
        // observations through the ownership-safe API and retire the point.
        if (retentionPolicyEnabled &&
            retentionExpiredThisCall.count(pit->first) &&
            !hasNewObservations) {
          for (const auto& residual : residuals) {
            if (std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                    residual.errorInterfacePtr) &&
                removeObservation(residual.residualBlockId)) {
              ++retentionPolicyRemovedObservations;
            }
          }
          if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
            LOG(INFO) << std::setprecision(17)
                      << "[RETENTION_POLICY_DECISION]"
                      << " landmark_id=" << pit->first
                      << " remove_frame=" << removeFrames[k]
                      << " decision=retire"
                      << " reason=retention_age_limit";
          }
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        const bool retentionTracked =
            retentionPolicyEnabled &&
            retainedLandmarkFirstRetentionTime_.find(pit->first) !=
                retainedLandmarkFirstRetentionTime_.end();
        RetentionEligibility retentionEligibility;
        bool retentionEligibilityComputed = false;
        if (retentionPolicyEnabled && !hasNewObservations &&
            (!skipLandmark || retentionTracked)) {
          ++retentionPolicyCandidates;
          retentionEligibility = evaluateRetentionEligibility(pit->first, pit->second);
          retentionEligibilityComputed = true;
        }
        if (retentionPolicyEnabled && retentionTracked) {
          if (hasNewObservations) {
            ++retentionPolicyRecovered;
            releaseRetentionTracking(pit->first, "recovered_new_observation");
          } else if (!retentionEligibilityComputed || !retentionEligibility.eligible) {
            ++retentionPolicyIneligible;
            releaseRetentionTracking(pit->first, "ineligible");
          }
        }

        if (residuals.size() == 0) {
          if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
            LOG(INFO) << std::setprecision(17)
                      << "[RETENTION_TERMINAL_REMOVAL_DIAGNOSTIC]"
                      << " landmark_id=" << pit->first
                      << " remove_frame=" << removeFrames[k]
                      << " remove_time_s=" << it->second.timestamp.toSec()
                      << " newest_frame=" << statesMap_.rbegin()->first
                      << " newest_time_s=" << statesMap_.rbegin()->second.timestamp.toSec()
                      << " terminal_reason=no_active_residuals"
                      << " observations_before=" << pit->second.observations.size();
          }
          releaseRetentionTracking(pit->first, "removed");
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        if (skipLandmark) {
          pit++;
          continue;
        }

        // Behavior-neutral A1 probe.  This snapshots the landmark immediately
        // before the existing marginalization loop mutates observations or
        // parameter ownership.  It deliberately does not retain, rehost, add,
        // remove, or otherwise alter estimator state.
        if (retentionDiagnosticsEnabled && retentionDiagnosticWindow &&
            retentionDiagnosticsEmitted < kMaximumRetentionDiagnosticsPerMarginalization &&
            pit->second.point.allFinite() && std::abs(pit->second.point[3]) > 1.0e-8 &&
            mapPtr_->parameterBlockExists(pit->first) && isLandmarkInitialized(pit->first)) {
          const uint64_t newestFrameId = statesMap_.empty() ? 0 : statesMap_.rbegin()->first;
          const okvis::Time newestTimestamp = statesMap_.empty()
                                                    ? okvis::Time(0, 0)
                                                    : statesMap_.rbegin()->second.timestamp;
          size_t activeObservations = 0;
          size_t survivingObservations = 0;
          size_t activeUsableObservations = 0;
          size_t survivingUsableObservations = 0;
          size_t survivingPositiveDepth = 0;
          size_t exactResidualObservations = 0;
          size_t invalidResidualObservations = 0;
          size_t originalBrightObservations = 0;
          size_t originalDarkObservations = 0;
          size_t unknownIntensityObservations = 0;
          std::set<uint64_t> activeFrames;
          std::set<uint64_t> survivingFrames;
          std::set<size_t> activeCameras;
          std::set<size_t> survivingCameras;
          std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
              survivingRays_W;
          double oldestObservationTime = std::numeric_limits<double>::infinity();
          double newestObservationTime = -std::numeric_limits<double>::infinity();
          double maximumParallaxDegrees = 0.0;
          double activeRobustCost = 0.0;
          double survivingRobustCost = 0.0;
          Eigen::Matrix3d rayNormal = Eigen::Matrix3d::Zero();

          for (const auto& observation : pit->second.observations) {
            const okvis::KeypointIdentifier& keypoint = observation.first;
            const auto frameIt = multiFramePtrMap_.find(keypoint.frameId);
            const auto stateIt = statesMap_.find(keypoint.frameId);
            if (frameIt == multiFramePtrMap_.end() || stateIt == statesMap_.end() ||
                !frameIt->second || keypoint.cameraIndex >= frameIt->second->numFrames() ||
                keypoint.keypointIndex >=
                    frameIt->second->numKeypoints(keypoint.cameraIndex)) {
              continue;
            }

            ++activeObservations;
            activeFrames.insert(keypoint.frameId);
            activeCameras.insert(keypoint.cameraIndex);
            const double observationTime = frameIt->second->timestamp().toSec();
            oldestObservationTime = std::min(oldestObservationTime, observationTime);
            newestObservationTime = std::max(newestObservationTime, observationTime);
            const bool survives = !vectorContains(removeFrames, keypoint.frameId);
            if (survives) {
              ++survivingObservations;
              survivingFrames.insert(keypoint.frameId);
              survivingCameras.insert(keypoint.cameraIndex);
            }

            const uint8_t intensityFlags = frameIt->second->keypointIntensityFlags(
                keypoint.cameraIndex, keypoint.keypointIndex);
            if ((intensityFlags & okvis::Frame::PreHistogramIntensityValid) == 0) {
              ++unknownIntensityObservations;
            } else if ((intensityFlags & okvis::Frame::PreHistogramDark) != 0) {
              ++originalDarkObservations;
            } else {
              ++originalBrightObservations;
            }

            okvis::kinematics::Transformation T_WS;
            okvis::kinematics::Transformation T_SC;
            if (!get_T_WS(keypoint.frameId, T_WS) ||
                !getCameraSensorStates(keypoint.frameId, keypoint.cameraIndex, T_SC)) {
              ++invalidResidualObservations;
              continue;
            }

            // Recover the exact active graph factor recorded for this
            // observation.  Eligibility must use the factor's unrobustified
            // Mahalanobis chi-square, not a reconstructed keypoint-scale
            // approximation.  Robust loss is reported separately below.
            const ::ceres::ResidualBlockId residualId =
                reinterpret_cast<::ceres::ResidualBlockId>(observation.second);
            const auto& residualSpecs =
                mapPtr_->residualBlockId2ResidualBlockSpecMap();
            const auto residualSpecIt = residualSpecs.find(residualId);
            if (residualSpecIt == residualSpecs.end()) {
              ++invalidResidualObservations;
              continue;
            }
            const ceres::Map::ParameterBlockCollection residualParameters =
                mapPtr_->parameters(residualId);
            const std::shared_ptr<ceres::ReprojectionError2dBase>
                reprojectionError =
                    std::dynamic_pointer_cast<ceres::ReprojectionError2dBase>(
                        residualSpecIt->second.errorInterfacePtr);
            if (!reprojectionError || residualParameters.size() < 3 ||
                residualParameters[0].first != keypoint.frameId ||
                residualParameters[1].first != pit->first ||
                reprojectionError->cameraId() != keypoint.cameraIndex) {
              ++invalidResidualObservations;
              continue;
            }
            const Eigen::Vector2d& measurement = reprojectionError->measurement();
            const Eigen::Matrix2d& information = reprojectionError->information();
            if (!measurement.allFinite() || !information.allFinite()) {
              ++invalidResidualObservations;
              continue;
            }

            const Eigen::Vector4d point_C =
                T_SC.inverse() * T_WS.inverse() * pit->second.point;
            Eigen::Vector2d prediction;
            const auto projectionStatus =
                frameIt->second->geometry(keypoint.cameraIndex)
                    ->projectHomogeneous(point_C, &prediction);
            const bool positiveDepth = point_C.allFinite() &&
                                       std::abs(point_C[3]) > 1.0e-8 &&
                                       point_C[2] / point_C[3] > 0.0;
            const double chiSquared =
                projectionStatus == cameras::CameraBase::ProjectionStatus::Successful &&
                        prediction.allFinite()
                    ? (measurement - prediction).transpose() * information *
                          (measurement - prediction)
                    : std::numeric_limits<double>::infinity();
            ++exactResidualObservations;
            if (std::isfinite(chiSquared)) {
              double rho[3] = {chiSquared, 1.0, 0.0};
              if (residualSpecIt->second.lossFunctionPtr != nullptr) {
                residualSpecIt->second.lossFunctionPtr->Evaluate(chiSquared, rho);
              }
              activeRobustCost += 0.5 * rho[0];
              if (survives) survivingRobustCost += 0.5 * rho[0];
            }
            const bool usable = positiveDepth && std::isfinite(chiSquared) &&
                                chiSquared <= 4.0;
            if (usable) ++activeUsableObservations;
            if (survives && positiveDepth) ++survivingPositiveDepth;
            if (survives && usable) ++survivingUsableObservations;

            if (survives) {
              Eigen::Vector3d bearing_C;
              if (frameIt->second->geometry(keypoint.cameraIndex)
                      ->backProject(measurement, &bearing_C) &&
                  bearing_C.allFinite() && bearing_C.norm() > 1.0e-12) {
                const Eigen::Vector3d bearing_W =
                    (T_WS * T_SC).C() * bearing_C.normalized();
                if (bearing_W.allFinite()) {
                  survivingRays_W.push_back(bearing_W.normalized());
                  rayNormal += Eigen::Matrix3d::Identity() -
                               bearing_W.normalized() * bearing_W.normalized().transpose();
                }
              }
            }
          }

          for (size_t first = 0; first < survivingRays_W.size(); ++first) {
            for (size_t second = first + 1; second < survivingRays_W.size(); ++second) {
              const double cosine = std::clamp(
                  survivingRays_W[first].dot(survivingRays_W[second]), -1.0, 1.0);
              maximumParallaxDegrees = std::max(
                  maximumParallaxDegrees, std::acos(cosine) * 180.0 / M_PI);
            }
          }
          double rayNormalRelativeMinimum = 0.0;
          if (!survivingRays_W.empty()) {
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(
                0.5 * (rayNormal + rayNormal.transpose()));
            if (eigenSolver.info() == Eigen::Success &&
                eigenSolver.eigenvalues().allFinite() &&
                eigenSolver.eigenvalues()[2] > 0.0) {
              rayNormalRelativeMinimum = std::max(
                  0.0, eigenSolver.eigenvalues()[0] / eigenSolver.eigenvalues()[2]);
            }
          }

          size_t newestFrameProjectionSuccess = 0;
          const auto newestFrameIt = multiFramePtrMap_.find(newestFrameId);
          okvis::kinematics::Transformation newestT_WS;
          if (newestFrameIt != multiFramePtrMap_.end() && newestFrameIt->second &&
              get_T_WS(newestFrameId, newestT_WS)) {
            for (size_t cameraIndex = 0;
                 cameraIndex < newestFrameIt->second->numFrames(); ++cameraIndex) {
              okvis::kinematics::Transformation T_SC;
              if (!getCameraSensorStates(newestFrameId, cameraIndex, T_SC)) continue;
              Eigen::Vector2d projection;
              if (newestFrameIt->second->geometry(cameraIndex)->projectHomogeneous(
                      T_SC.inverse() * newestT_WS.inverse() * pit->second.point,
                      &projection) == cameras::CameraBase::ProjectionStatus::Successful &&
                  projection.allFinite()) {
                ++newestFrameProjectionSuccess;
              }
            }
          }

          const char* cohort = "unknown";
          if (originalBrightObservations > 0 && originalDarkObservations == 0 &&
              unknownIntensityObservations == 0) {
            cohort = "original_bright";
          } else if (originalDarkObservations > 0 &&
                     originalBrightObservations == 0 &&
                     unknownIntensityObservations == 0) {
            cohort = "original_dark";
          }
          const char* predictedPath = hasNewObservations
                                          ? "drop_old_observations_keep_landmark"
                                          : (marginalize && obsCount >= 2
                                                 ? "marginalize_landmark"
                                                 : "delete_after_observation_removal");
          const bool losesFinalUsableObservation =
              activeUsableObservations > 0 && survivingUsableObservations == 0;
          const bool wouldRemainProjectable = newestFrameProjectionSuccess > 0;
          const bool eligible =
              survivingUsableObservations > 0 && wouldRemainProjectable;
          const double observationSpan =
              std::isfinite(oldestObservationTime) &&
                      std::isfinite(newestObservationTime)
                  ? newestObservationTime - oldestObservationTime
                  : 0.0;
          const double age = std::isfinite(newestObservationTime)
                                 ? newestTimestamp.toSec() - newestObservationTime
                                 : 0.0;

          LOG(INFO) << std::setprecision(17)
                    << "[RETENTION_ELIGIBILITY_DIAGNOSTIC]"
                    << " landmark_id=" << pit->first
                    << " remove_frame=" << removeFrames[k]
                    << " remove_time_s=" << it->second.timestamp.toSec()
                    << " newest_frame=" << newestFrameId
                    << " newest_time_s=" << newestTimestamp.toSec()
                    << " predicted_path=" << predictedPath
                    << " cohort=" << cohort
                    << " finite=1 initialized=1"
                    << " active_observations=" << activeObservations
                    << " active_distinct_frames=" << activeFrames.size()
                    << " active_distinct_cameras=" << activeCameras.size()
                    << " active_usable_observations=" << activeUsableObservations
                    << " exact_residual_observations=" << exactResidualObservations
                    << " invalid_residual_observations=" << invalidResidualObservations
                    << " active_robust_cost=" << activeRobustCost
                    << " surviving_observations=" << survivingObservations
                    << " surviving_distinct_frames=" << survivingFrames.size()
                    << " surviving_distinct_cameras=" << survivingCameras.size()
                    << " surviving_positive_depth=" << survivingPositiveDepth
                    << " surviving_usable_observations=" << survivingUsableObservations
                    << " surviving_robust_cost=" << survivingRobustCost
                    << " observation_span_s=" << observationSpan
                    << " age_s=" << age
                    << " max_parallax_deg=" << maximumParallaxDegrees
                    << " ray_normal_relative_min=" << rayNormalRelativeMinimum
                    << " newest_projection_success=" << newestFrameProjectionSuccess
                    << " loses_final_usable_observation="
                    << losesFinalUsableObservation
                    << " would_remain_projectable=" << wouldRemainProjectable
                    << " chi2_source=stored_residual_information"
                    << " eligible=" << eligible;
          ++retentionDiagnosticsEmitted;
        }

        if (retentionPolicyEnabled && !hasNewObservations &&
            !retentionReleasedThisCall.count(pit->first) &&
            retentionEligibilityComputed && retentionEligibility.eligible) {
          const bool alreadyTracked =
              retainedLandmarkFirstRetentionTime_.find(pit->first) !=
              retainedLandmarkFirstRetentionTime_.end();
          const bool perCallCapacityAvailable =
              alreadyTracked ||
              retentionPolicyNewlySelected < kMaximumNewRetentionsPerMarginalization;
          const bool totalCapacityAvailable =
              alreadyTracked ||
              retainedLandmarkFirstRetentionTime_.size() < kMaximumTrackedRetentions;
          if (!perCallCapacityAvailable || !totalCapacityAvailable) {
            ++retentionPolicyCapacityRejected;
            if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
              LOG(INFO) << std::setprecision(17)
                        << "[RETENTION_POLICY_DECISION]"
                        << " landmark_id=" << pit->first
                        << " remove_frame=" << removeFrames[k]
                        << " decision=skip"
                        << " reason="
                        << (!perCallCapacityAvailable ? "per_call_limit" : "tracked_limit")
                        << " surviving_usable_observations="
                        << retentionEligibility.survivingUsableObservations
                        << " newest_projection_success="
                        << retentionEligibility.newestFrameProjectionSuccess
                        << " age_s=" << retentionEligibility.newestObservationAge;
            }
          } else {
            if (!alreadyTracked) {
              retainedLandmarkFirstRetentionTime_.insert(
                  std::make_pair(pit->first, retentionPolicyTimestamp));
              ++retentionPolicyNewlySelected;
            }
            if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
              LOG(INFO) << std::setprecision(17)
                        << "[RETENTION_POLICY_DECISION]"
                        << " landmark_id=" << pit->first
                        << " remove_frame=" << removeFrames[k]
                        << " decision=retain"
                        << " newly_selected=" << (!alreadyTracked)
                        << " surviving_usable_observations="
                        << retentionEligibility.survivingUsableObservations
                        << " newest_projection_success="
                        << retentionEligibility.newestFrameProjectionSuccess
                        << " age_s=" << retentionEligibility.newestObservationAge
                        << " tracked_total="
                        << retainedLandmarkFirstRetentionTime_.size();
            }

            // Drop every reprojection attached to a frame scheduled for
            // removal.  The remaining genuine residuals stay explicit and
            // therefore never enter the marginalization prior through this
            // landmark path.
            for (size_t r = 0; r < residuals.size(); ++r) {
              std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                  std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                      residuals[r].errorInterfacePtr);
              if (!reprojectionError) continue;
              const ceres::Map::ParameterBlockCollection residualParameters =
                  mapPtr_->parameters(residuals[r].residualBlockId);
              if (!residualParameters.empty() &&
                  vectorContains(removeFrames, residualParameters[0].first)) {
                if (removeObservation(residuals[r].residualBlockId)) {
                  ++retentionPolicyRemovedObservations;
                }
              }
            }
            pit++;
            continue;
          }
        } else if (retentionPolicyEnabled && !hasNewObservations &&
                   !retentionReleasedThisCall.count(pit->first) &&
                   !retentionTracked && !skipLandmark &&
                   retentionEligibilityComputed && !retentionEligibility.eligible) {
          ++retentionPolicyIneligible;
          if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
            const char* reason = "ineligible";
            if (!retentionEligibility.finiteInitialized) {
              reason = "invalid_point";
            } else if (retentionEligibility.inExistingMarginalizationPrior) {
              reason = "already_in_marginalization_prior";
            } else if (!retentionEligibility.observationGraphValid ||
                       retentionEligibility.hasRemovalNonReprojection) {
              reason = "invalid_observation_graph";
            } else if (retentionEligibility.survivingUsableObservations < 2) {
              reason = "insufficient_surviving_support";
            } else if (retentionEligibility.newestFrameProjectionSuccess == 0) {
              reason = "not_projectable_in_newest_frame";
            } else if (!std::isfinite(retentionEligibility.newestObservationAge) ||
                       retentionEligibility.newestObservationAge < 0.0 ||
                       retentionEligibility.newestObservationAge >
                           kMaximumRetentionAgeSeconds) {
              reason = "observation_too_old";
            }
            LOG(INFO) << std::setprecision(17)
                      << "[RETENTION_POLICY_DECISION]"
                      << " landmark_id=" << pit->first
                      << " remove_frame=" << removeFrames[k]
                      << " decision=skip"
                      << " reason=" << reason
                      << " surviving_usable_observations="
                      << retentionEligibility.survivingUsableObservations
                      << " newest_projection_success="
                      << retentionEligibility.newestFrameProjectionSuccess
                      << " age_s=" << retentionEligibility.newestObservationAge;
          }
        }

        // so, we need to consider it.
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            if ((vectorContains(removeFrames, poseId) && hasNewObservations) ||
                (!vectorContains(allLinearizedFrames, poseId) && marginalize)) {
              // ok, let's ignore the observation.
              removeObservation(residuals[r].residualBlockId);
              residuals.erase(residuals.begin() + r);
              r--;
            } else if (marginalize && vectorContains(allLinearizedFrames, poseId)) {
              // TODO: consider only the sensible ones for marginalization
              if (obsCount < 2) {  // visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);
                residuals.erase(residuals.begin() + r);
                r--;
              } else {
                // add information to be considered in marginalization later.
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId, false);
              }
            }
            // check anything left
            if (residuals.size() == 0) {
              justDelete = true;
              marginalize = false;
            }
          }
        }

        if (justDelete) {
          if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
            LOG(INFO) << std::setprecision(17)
                      << "[RETENTION_TERMINAL_REMOVAL_DIAGNOSTIC]"
                      << " landmark_id=" << pit->first
                      << " remove_frame=" << removeFrames[k]
                      << " remove_time_s=" << it->second.timestamp.toSec()
                      << " newest_frame=" << statesMap_.rbegin()->first
                      << " newest_time_s=" << statesMap_.rbegin()->second.timestamp.toSec()
                      << " terminal_reason=all_residuals_deleted"
                      << " observations_before=" << pit->second.observations.size();
          }
          releaseRetentionTracking(pit->first, "removed");
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }
        if (marginalize && errorTermAdded) {
          if (retentionDiagnosticsEnabled && retentionDiagnosticWindow) {
            LOG(INFO) << std::setprecision(17)
                      << "[RETENTION_TERMINAL_REMOVAL_DIAGNOSTIC]"
                      << " landmark_id=" << pit->first
                      << " remove_frame=" << removeFrames[k]
                      << " remove_time_s=" << it->second.timestamp.toSec()
                      << " newest_frame=" << statesMap_.rbegin()->first
                      << " newest_time_s=" << statesMap_.rbegin()->second.timestamp.toSec()
                      << " terminal_reason=marginalized_with_error_term"
                      << " observations_before=" << pit->second.observations.size();
          }
          releaseRetentionTracking(pit->first, "removed");
          paremeterBlocksToBeMarginalized.push_back(pit->first);
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        pit++;
      }
    }

    // update book-keeping and go to the next frame
    // if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if (true) {  ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);
    }
  }

  // now apply the actual marginalization
  if (paremeterBlocksToBeMarginalized.size() > 0) {
    std::vector< ::ceres::ResidualBlockId> addedPriors;
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if (paremeterBlocksToBeMarginalized.size() > 0) {
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if (marginalizationErrorPtr_->num_residuals() == 0) {
    marginalizationErrorPtr_.reset();
  }
  if (marginalizationErrorPtr_) {
    std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
    marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
    marginalizationResidualId_ = mapPtr_->addResidualBlock(marginalizationErrorPtr_, NULL, parameterBlockPtrs);
    OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_, "could not add marginalization error");
    if (!marginalizationResidualId_) {
      logRetentionPolicySummary();
      return false;
    }
  }

  if (reDoFixation) {
    // finally fix the first pose properly
    // mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
    okvis::kinematics::Transformation T_WS_0;
    get_T_WS(statesMap_.begin()->first, T_WS_0);
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(5, 5) = 1.0e14;
    information(0, 0) = 1.0e14;
    information(1, 1) = 1.0e14;
    information(2, 2) = 1.0e14;
    std::shared_ptr<ceres::PoseError> poseError(new ceres::PoseError(T_WS_0, information));
    mapPtr_->addResidualBlock(poseError, NULL, mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
  }

  logRetentionPolicySummary();
  return true;
}

// Prints state information to buffer.
void Estimator::printStates(uint64_t poseId, std::ostream& buffer) const {
  buffer << "GLOBAL: ";
  for (size_t i = 0; i < statesMap_.at(poseId).global.size(); ++i) {
    if (statesMap_.at(poseId).global.at(i).exists) {
      uint64_t id = statesMap_.at(poseId).global.at(i).id;
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
      buffer << "id=" << id << ":";
      buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
      buffer << ", ";
    }
  }
  buffer << "SENSOR: ";
  for (size_t i = 0; i < statesMap_.at(poseId).sensors.size(); ++i) {
    for (size_t j = 0; j < statesMap_.at(poseId).sensors.at(i).size(); ++j) {
      for (size_t k = 0; k < statesMap_.at(poseId).sensors.at(i).at(j).size(); ++k) {
        if (statesMap_.at(poseId).sensors.at(i).at(j).at(k).exists) {
          uint64_t id = statesMap_.at(poseId).sensors.at(i).at(j).at(k).id;
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
          buffer << "id=" << id << ":";
          buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
          buffer << ", ";
        }
      }
    }
  }
  buffer << std::endl;
}

// Initialise pose from IMU measurements. For convenience as static.
bool Estimator::initPoseFromImu(const okvis::ImuMeasurementDeque& imuMeasurements,
                                okvis::kinematics::Transformation& T_WS) {
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  if (imuMeasurements.size() == 0) return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin(); it < imuMeasurements.end(); ++it) {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= static_cast<double>(imuMeasurements.size());
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);

  return true;
}

// Start ceres optimization.
void Estimator::optimize(size_t numIter, size_t numThreads, bool verbose) {
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  // mapPtr_->options.initial_trust_region_radius = 1.0e4;
  // mapPtr_->options.initial_trust_region_radius = 2.0e6;
  // mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
  // mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
  // mapPtr_->options.use_nonmonotonic_steps = true;
  // mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
  // mapPtr_->options.function_tolerance = 1e-12;
  // mapPtr_->options.gradient_tolerance = 1e-12;
  // mapPtr_->options.jacobi_scaling = false;
  mapPtr_->options.num_threads = numThreads;
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  mapPtr_->solve();

  // update landmarks
  {
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      Eigen::MatrixXd H(3, 3);
      mapPtr_->getLhs(it->first, H);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
      if (smallest < 1.0e-12) {
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest) / sqrt(largest);
      }

      // update coordinates
      it->second.point =
          std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(it->first))
              ->estimate();
    }
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

Estimator::OptimizationDiagnostics Estimator::optimizationDiagnostics() const {
  OptimizationDiagnostics diagnostics;
  diagnostics.iterations = static_cast<int>(mapPtr_->summary.iterations.size());
  diagnostics.successfulSteps = mapPtr_->summary.num_successful_steps;
  diagnostics.unsuccessfulSteps = mapPtr_->summary.num_unsuccessful_steps;
  diagnostics.terminationType = static_cast<int>(mapPtr_->summary.termination_type);
  diagnostics.initialCost = mapPtr_->summary.initial_cost;
  diagnostics.finalCost = mapPtr_->summary.final_cost;
  diagnostics.totalTimeSeconds = mapPtr_->summary.total_time_in_seconds;
  diagnostics.solutionUsable = mapPtr_->summary.IsSolutionUsable();
  return diagnostics;
}

// Set a time limit for the optimization process.
bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) {
  if (ceresCallback_ != nullptr) {
    if (timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(mapPtr_->options.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  } else if (timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(
        new okvis::ceres::CeresIterationCallback(timeLimit, minIterations));
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

// getters
// Get a specific landmark.
bool Estimator::getLandmark(uint64_t landmarkId, MapPoint& mapPoint) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception, "landmark with id = " << landmarkId << " does not exist.")
    return false;
  }
  mapPoint = landmarksMap_.at(landmarkId);
  return true;
}

// Checks whether the landmark is initialized.
bool Estimator::isLandmarkInitialized(uint64_t landmarkId) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");
  return std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))
      ->initialized();
}

// Get a copy of all the landmarks as a PointMap.
size_t Estimator::getLandmarks(PointMap& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks = landmarksMap_;
  return landmarksMap_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy support.
// Use getLandmarks(okvis::PointMap&) if possible.
size_t Estimator::getLandmarks(MapPointVector& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks.clear();
  landmarks.reserve(landmarksMap_.size());
  for (PointMap::const_iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    landmarks.push_back(it->second);
  }
  return landmarksMap_.size();
}

// Get pose for a given pose ID.
bool Estimator::get_T_WS(uint64_t poseId, okvis::kinematics::Transformation& T_WS) const {
  if (!getGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Added by Sharmin
bool Estimator::getImuPreIntegral(uint64_t poseId,
                                  Eigen::Vector3d& acc_doubleintegral,
                                  Eigen::Vector3d& acc_integral,
                                  double& Delta_t) const {
  std::map<uint64_t, imu_integrals>::const_iterator it = imuIntegralsMap_.find(poseId);
  if (it != imuIntegralsMap_.end()) {
    acc_doubleintegral = it->second.acc_doubleintegral;
    acc_integral = it->second.acc_integral;
    Delta_t = it->second.Delta_t;

    // std::cout <<"Accessing Imu integral Values: "<< it->second.acc_doubleintegral << ", "
    //<< it->second.acc_integral<< ", " << it->second.Delta_t <<std::endl;
    return true;
  }
  return false;
}

// Feel free to implement caching for them...
// Get speeds and IMU biases for a given pose ID.
bool Estimator::getSpeedAndBias(uint64_t poseId, uint64_t imuIdx, okvis::SpeedAndBias& speedAndBias) const {
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
          poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias)) {
    return false;
  }
  return true;
}

// Get camera states for a given pose ID.
bool Estimator::getCameraSensorStates(uint64_t poseId,
                                      size_t cameraIdx,
                                      okvis::kinematics::Transformation& T_SCi) const {
  return getSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Get the ID of the current keyframe.
uint64_t Estimator::currentKeyframeId() const {
  for (std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin(); rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      return rit->first;
    }
  }
  OKVIS_THROW_DBG(Exception, "no keyframes existing...");
  return 0;
}

// Get the ID of an older frame.
uint64_t Estimator::frameIdByAge(size_t age) const {
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  for (size_t i = 0; i < age; ++i) {
    ++rit;
    OKVIS_ASSERT_TRUE_DBG(Exception, rit != statesMap_.rend(), "requested age " << age << " out of range.");
  }
  return rit->first;
}

// Get the ID of the newest frame added to the state.
uint64_t Estimator::currentFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size() > 0, "no frames added yet.")
  return statesMap_.rbegin()->first;
}

// Checks if a particular frame is still in the IMU window
bool Estimator::isInImuWindow(uint64_t frameId) const {
  if (statesMap_.at(frameId).sensors.at(SensorStates::Imu).size() == 0) {
    return false;  // no IMU added
  }
  return statesMap_.at(frameId).sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).exists;
}

// Set pose for a given pose ID.
bool Estimator::set_T_WS(uint64_t poseId, const okvis::kinematics::Transformation& T_WS) {
  if (!setGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Added by Sharmin
void Estimator::setImuPreIntegral(uint64_t poseId,
                                  Eigen::Vector3d& acc_doubleintegral,
                                  Eigen::Vector3d& acc_integral,
                                  double& Delta_t) {
  imu_integrals imu_int(acc_doubleintegral, acc_integral, Delta_t);
  imuIntegralsMap_.insert(std::pair<uint64_t, imu_integrals>(poseId, imu_int));
  // std::cout <<"Imu integral Values: "<< imu_int.acc_doubleintegral << ", "
  //<< imu_int.acc_integral<< ", " << imu_int.Delta_t <<std::endl;
}

// Set the speeds and IMU biases for a given pose ID.
bool Estimator::setSpeedAndBias(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias& speedAndBias) {
  return setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias);
}

// Set the transformation from sensor to camera frame for a given pose ID.
bool Estimator::setCameraSensorStates(uint64_t poseId,
                                      size_t cameraIdx,
                                      const okvis::kinematics::Transformation& T_SCi) {
  return setSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Set the homogeneous coordinates for a landmark.
bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(landmarkId);
#ifndef NDEBUG
  std::shared_ptr<ceres::HomogeneousPointParameterBlock> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);
#else
  std::static_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr)->setEstimate(landmark);
#endif

  // also update in map
  landmarksMap_.at(landmarkId).point = landmark;
  return true;
}

// Set the landmark initialization state.
void Estimator::setLandmarkInitialized(uint64_t landmarkId, bool initialized) {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");
  std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))
      ->setInitialized(initialized);
}

// private stuff
// getters
bool Estimator::getGlobalStateParameterBlockPtr(uint64_t poseId,
                                                int stateType,
                                                std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW(Exception, "pose with id = " << id << " does not exist.")
    return false;
  }

  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateParameterBlockAs(uint64_t poseId,
                                               int stateType,
                                               PARAMETER_BLOCK_T& stateParameterBlock) const {
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    LOG(INFO) << "--" << parameterBlockPtr->typeInfo();
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,
                    "wrong pointer type requested: requested " << info->typeInfo() << " but is of type"
                                                               << parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
#endif
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateEstimateAs(uint64_t poseId,
                                         int stateType,
                                         typename PARAMETER_BLOCK_T::estimate_t& state) const {
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

bool Estimator::getSensorStateParameterBlockPtr(uint64_t poseId,
                                                int sensorIdx,
                                                int sensorType,
                                                int stateType,
                                                std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }
  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateParameterBlockAs(uint64_t poseId,
                                               int sensorIdx,
                                               int sensorType,
                                               int stateType,
                                               PARAMETER_BLOCK_T& stateParameterBlock) const {
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,
                    "wrong pointer type requested: requested " << info->typeInfo() << " but is of type"
                                                               << parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
#endif
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateEstimateAs(uint64_t poseId,
                                         int sensorIdx,
                                         int sensorType,
                                         int stateType,
                                         typename PARAMETER_BLOCK_T::estimate_t& state) const {
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getSensorStateParameterBlockAs(poseId, sensorIdx, sensorType, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

template <class PARAMETER_BLOCK_T>
bool Estimator::setGlobalStateEstimateAs(uint64_t poseId,
                                         int stateType,
                                         const typename PARAMETER_BLOCK_T::estimate_t& state) {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

template <class PARAMETER_BLOCK_T>
bool Estimator::setSensorStateEstimateAs(uint64_t poseId,
                                         int sensorIdx,
                                         int sensorType,
                                         int stateType,
                                         const typename PARAMETER_BLOCK_T::estimate_t& state) {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

}  // namespace okvis
