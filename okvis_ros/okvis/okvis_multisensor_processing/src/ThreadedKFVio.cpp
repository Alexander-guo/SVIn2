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
 *  Created on: Aug 21, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ThreadedKFVio.cpp
 * @brief Source file for the ThreadedKFVio class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <okvis/ThreadedKFVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/cameras/PinholeCamera.hpp>               // Sharmin
#include <okvis/cameras/RadialTangentialDistortion.hpp>  // Sharmin
#include <okvis/ceres/ImuError.hpp>
#include <utility>
#include <vector>
/// @Sharmin
#include <okvis/IdProvider.hpp>

int nNextId = 0;  // FIXME Sharmin To remove this global variable.
int numKF = 0;    // Sharmin: Number of keyframe

// Sharmin
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived>& ypr) {
  typedef typename Derived::Scalar Scalar_t;

  Scalar_t y = ypr(0) / 180.0 * M_PI;
  Scalar_t p = ypr(1) / 180.0 * M_PI;
  Scalar_t r = ypr(2) / 180.0 * M_PI;

  Eigen::Matrix<Scalar_t, 3, 3> Rz;
  Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;

  Eigen::Matrix<Scalar_t, 3, 3> Ry;
  Ry << cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p);

  Eigen::Matrix<Scalar_t, 3, 3> Rx;
  Rx << 1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r);

  return Rz * Ry * Rx;
}

/// \brief okvis Main namespace of this package.
namespace okvis {

uint64_t frameCnt = 0;  // Sharmin

static const int max_camera_input_queue_size = 10;
static const okvis::Duration temporal_imu_data_overlap(
    0.02);  // overlap of imu data before and after two consecutive frames [seconds]

namespace {
bool driftDiagnosticsOptIn() {
  const char* value = std::getenv("SVIN2_ENABLE_DRIFT_DIAGNOSTICS");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

double quaternionDistanceDegrees(const Eigen::Quaterniond& lhs, const Eigen::Quaterniond& rhs) {
  const double cosineHalfAngle = std::clamp(std::abs(lhs.dot(rhs)), 0.0, 1.0);
  return 2.0 * std::acos(cosineHalfAngle) * 180.0 / M_PI;
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::round(std::clamp(fraction, 0.0, 1.0) * static_cast<double>(values.size() - 1)));
  return values[index];
}

std::string commaSeparated(const std::vector<size_t>& values) {
  std::ostringstream stream;
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) stream << ',';
    stream << values[index];
  }
  return stream.str();
}

std::string commaSeparatedDoubles(const std::vector<double>& values) {
  std::ostringstream stream;
  stream << std::setprecision(17);
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) stream << ',';
    if (std::isfinite(values[index])) stream << values[index];
    else stream << "nan";
  }
  return stream.str();
}

double percentileOrNaN(std::vector<double> values, double fraction) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::round(std::clamp(fraction, 0.0, 1.0) * static_cast<double>(values.size() - 1)));
  return values[index];
}
}  // namespace

#ifdef USE_MOCK
// Constructor for gmock.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters,
                             okvis::MockVioBackendInterface& estimator,
                             okvis::MockVioFrontendInterface& frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      featureStateReadyThrough_(okvis::Time(0, 0)),
      shutdownRequested_(false),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      featureStateReadyThrough_(okvis::Time(0, 0)),
      shutdownRequested_(false),
      estimator_(),
      frontend_(parameters.nCameraSystem.numCameras()),
      parameters_(parameters),
      maxImuInputQueueSize_(2 * max_camera_input_queue_size * parameters.imu.rate /
                            parameters.sensors_information.cameraRate) {
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void ThreadedKFVio::init() {
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;
  kf_index_ = 0;  // Sharmin

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
  frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
  frontend_.setBriskDetectionAbsoluteThreshold(parameters_.optimization.detectionAbsoluteThreshold);
  frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);

  lastOptimizedStateTimestamp_ =
      okvis::Time(0.0) +
      temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ =
      okvis::Time(0.0) +
      temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  estimator_.addImu(parameters_.imu);
  for (size_t i = 0; i < numCameras_; ++i) {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    estimator_.addCamera(parameters_.camera_extrinsics);
    cameraMeasurementsReceived_.emplace_back(
        std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement>>>(
            new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement>>()));
  }

  // set up windows so things don't crash on Mac OS
  if (parameters_.visualization.displayImages) {
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
      cv::namedWindow(windowname.str());
    }
  }

  startThreads();
}

// Start all threads.
void ThreadedKFVio::startThreads() {
  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);

  // Sharmin
  if (parameters_.sensorList.isSonarUsed) {
    sonarConsumerThread_ = std::thread(&ThreadedKFVio::sonarConsumerLoop, this);  // @Sharmin
  }
  // Sharmin
  if (parameters_.sensorList.isDepthUsed) {
    depthConsumerThread_ = std::thread(&ThreadedKFVio::depthConsumerLoop, this);  // @Sharmin
  }

  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop, this);
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ = std::thread(&ThreadedKFVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ = std::thread(&ThreadedKFVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this);
  optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this);
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
ThreadedKFVio::~ThreadedKFVio() {
  shutdownRequested_ = true;
  featureStateNotification_.notify_all();
  for (size_t i = 0; i < numCameras_; ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  // Sharmin
  if (parameters_.sensorList.isSonarUsed) {
    sonarMeasurementsReceived_.Shutdown();  // @Sharmin
  }
  // Sharmin
  if (parameters_.sensorList.isDepthUsed) {
    depthMeasurementsReceived_.Shutdown();  // @Sharmin
  }

  optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.at(i).join();
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  // Sharmin
  if (parameters_.sensorList.isSonarUsed) {
    sonarConsumerThread_.join();
  }
  // Sharmin
  if (parameters_.sensorList.isDepthUsed) {
    depthConsumerThread_.join();
  }
  // Sharmin

  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  publisherThread_.join();

  /*okvis::kinematics::Transformation endPosition;
  estimator_.get_T_WS(estimator_.currentFrameId(), endPosition);
  std::stringstream s;
  s << endPosition.r();
  LOG(INFO) << "Sensor end position:\n" << s.str();
  LOG(INFO) << "Distance to origin: " << endPosition.r().norm();*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

// Add a new image.
bool ThreadedKFVio::addImage(const okvis::Time& stamp,
                             size_t cameraIndex,
                             const cv::Mat& image,
                             const std::vector<cv::KeyPoint>* keypoints,
                             bool* /*asKeyframe*/) {
  assert(cameraIndex < numCameras_);

  if (lastAddedImageTimestamp_ > stamp &&
      fabs((lastAddedImageTimestamp_ - stamp).toSec()) > parameters_.sensors_information.frameTimestampTolerance) {
    LOG(ERROR) << "Received image from the past. Dropping the image.";
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (keypoints != nullptr) {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  } else {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_) {
    return cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    // return true;
  } else {
    const bool droppedOldest =
        cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(frame, max_camera_input_queue_size);
    if (droppedOldest) {
      DLOG(WARNING) << "oldest in the queue frame dropped";
      return false;
    }
    // return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
    return true;
  }
}

// Add an abstracted image observation.
bool ThreadedKFVio::addKeypoints(const okvis::Time& /*stamp*/,
                                 size_t /*cameraIndex*/,
                                 const std::vector<cv::KeyPoint>& /*keypoints*/,
                                 const std::vector<uint64_t>& /*landmarkIds*/,
                                 const cv::Mat& /*descriptors*/,
                                 bool* /*asKeyframe*/) {
  OKVIS_THROW(Exception,
              "ThreadedKFVio::addKeypoints() not implemented anymore since changes to _keypointMeasurements queue.");
  return false;
}

// @Sharmin
// Add depth measurement
bool ThreadedKFVio::addDepthMeasurement(const okvis::Time& stamp, double depth) {
  okvis::DepthMeasurement depth_measurement;

  // For storing the first depth data
  if (isFirstDepth_) {
    firstDepth_ = depth;
    isFirstDepth_ = false;

    LOG(INFO) << "First depth: " << depth;
  }

  depth_measurement.timeStamp = stamp;
  depth_measurement.measurement.depth = depth;

  if (blocking_) {
    depthMeasurementsReceived_.PushBlockingIfFull(depth_measurement, 1);
    return true;
  } else {
    depthMeasurementsReceived_.PushNonBlockingDroppingIfFull(depth_measurement, maxDepthInputQueueSize_);
    return depthMeasurementsReceived_.Size() == 1;
  }
}
// @Sharmin
// Add a Sonar measurement.
bool ThreadedKFVio::addSonarMeasurement(const okvis::Time& stamp, double range, double heading) {
  okvis::SonarMeasurement sonar_measurement;
  sonar_measurement.timeStamp = stamp;
  sonar_measurement.measurement.range = range;
  sonar_measurement.measurement.heading = heading;

  if (blocking_) {
    sonarMeasurementsReceived_.PushBlockingIfFull(sonar_measurement, 1);
    return true;
  } else {
    sonarMeasurementsReceived_.PushNonBlockingDroppingIfFull(sonar_measurement,
                                                             maxImuInputQueueSize_);  // Running same rate as imu
    return sonarMeasurementsReceived_.Size() == 1;
  }
}

// Add an IMU measurement.
bool ThreadedKFVio::addImuMeasurement(const okvis::Time& stamp,
                                      const Eigen::Vector3d& alpha,
                                      const Eigen::Vector3d& omega) {
  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  if (blocking_) {
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
    return true;
  } else {
    const bool droppedOldest =
        imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(imu_measurement, maxImuInputQueueSize_);
    return imuMeasurementsReceived_.Size() == 1;
  }
}

// Add a position measurement.
void ThreadedKFVio::addPositionMeasurement(const okvis::Time& stamp,
                                           const Eigen::Vector3d& position,
                                           const Eigen::Vector3d& positionOffset,
                                           const Eigen::Matrix3d& positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionOffset = positionOffset;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// Add a GPS measurement.
void ThreadedKFVio::addGpsMeasurement(const okvis::Time&,
                                      double,
                                      double,
                                      double,
                                      const Eigen::Vector3d&,
                                      const Eigen::Matrix3d&) {
  OKVIS_THROW(Exception, "GPS measurements not supported")
}

// Add a magnetometer measurement.
void ThreadedKFVio::addMagnetometerMeasurement(const okvis::Time&, const Eigen::Vector3d&, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void ThreadedKFVio::addBarometerMeasurement(const okvis::Time&, double, double) {
  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void ThreadedKFVio::addDifferentialPressureMeasurement(const okvis::Time&, double, double) {
  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedKFVio::setBlocking(bool blocking) {
  blocking_ = blocking;
  // disable time limit for optimization
  if (blocking_) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_.setOptimizationTimeLimit(-1.0, parameters_.optimization.max_iterations);
  }
}

bool ThreadedKFVio::waitForFeatureState(const okvis::Time& requiredTimestamp) {
  if (requiredTimestamp.isZero()) return !shutdownRequested_.load();

  std::unique_lock<std::mutex> lock(lastState_mutex_);
  featureStateNotification_.wait(lock, [this, &requiredTimestamp]() {
    return shutdownRequested_.load() || featureStateReadyThrough_ >= requiredTimestamp;
  });
  return !shutdownRequested_.load();
}

void ThreadedKFVio::markFeatureStateReady(const okvis::Time& timestamp) {
  {
    std::lock_guard<std::mutex> lock(lastState_mutex_);
    if (featureStateReadyThrough_ < timestamp) featureStateReadyThrough_ = timestamp;
  }
  featureStateNotification_.notify_all();
}

// Loop to process frames from camera with index cameraIndex
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex) {
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  okvis::Time previousFeatureTimestamp(0, 0);
  okvis::startup_camera_imu::AdmissionGate startupAdmissionGate(
      parameters_.sensors_information.cameraRate, temporal_imu_data_overlap);
  bool startupWarmupReady = false;
  bool startupInitializationComplete = false;
  TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer(
      "1.1.1 waitForFrameSynchronizerMutex" + std::to_string(cameraIndex), true);
  TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex" + std::to_string(cameraIndex),
                                                  true);
  TimerSwitchable propagationTimer("1.1.4 propagationTimer" + std::to_string(cameraIndex), true);
  TimerSwitchable detectTimer("1.2 detectAndDescribe" + std::to_string(cameraIndex), true);
  TimerSwitchable afterDetectTimer("1.3 afterDetect" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2(
      "1.3.1 waitForFrameSynchronizerMutex2" + std::to_string(cameraIndex), true);
  TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread" + std::to_string(cameraIndex), true);

  for (;;) {
    // get data and check for termination request
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
      return;
    }
    if (!waitForFeatureState(previousFeatureTimestamp)) return;
    beforeDetectTimer.start();

    // The first replayed camera callback can arrive before the IMU callback
    // stream has established a usable history.  Hold the consumer at a
    // timestamp-derived boundary so the frames used for initialization and
    // the lower IMU-window boundary do not depend on callback scheduling.
    if (!startupInitializationComplete) {
      if (!startupAdmissionGate.initialized()) {
        startupAdmissionGate.initialize(frame->timeStamp);
      }
      if (!startupWarmupReady) {
        startupWarmupReady = imuFrameSynchronizer_.waitForUpToDateImuData(startupAdmissionGate.imuReadyUntil());
        if (!startupWarmupReady) {
          beforeDetectTimer.stop();
          return;
        }
      }
    }

    {  // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame containing it
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
      // std::cout<< "lastOptimizedStateTimestamp_: "<< lastOptimizedStateTimestamp_ << std::endl;
    }

    // @Sharmin
    // Depth
    if (parameters_.sensorList.isDepthUsed) {
      okvis::Time depthDataEndTime = multiFrame->timestamp();
      okvis::Time depthDataBeginTime = lastTimestamp;

      OKVIS_ASSERT_TRUE_DBG(
          Exception, depthDataBeginTime < depthDataEndTime, "Depth data end time is smaller than begin time.");

      // wait until all relevant depth messages have arrived and check for termination request
      // if (depthFrameSynchronizer_.waitForUpToDateDepthData(okvis::Time(depthDataEndTime)) == false) {
      //   return;
      // }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            depthDataEndTime < depthMeasurements_.back().timeStamp,
                            "Waiting for up to date depth data seems to have failed!");

      okvis::DepthMeasurementDeque depthData = getDepthMeasurements(depthDataBeginTime, depthDataEndTime);

      // if depth_data is empty, either end_time > begin_time or
      // no measurements in timeframe, should not happen, as we waited for measurements
      if (depthData.size() == 0) {
        beforeDetectTimer.stop();
        continue;
      }

      if (depthData.front().timeStamp > frame->timeStamp) {
        LOG(WARNING) << "Frame is newer than oldest Depth measurement. Dropping it.";
        beforeDetectTimer.stop();
        continue;
      }
    }
    // Sonar
    if (parameters_.sensorList.isSonarUsed) {
      // -- get relevant sonar messages for new state
      okvis::Time sonarDataEndTime = multiFrame->timestamp();
      okvis::Time sonarDataBeginTime = lastTimestamp;

      OKVIS_ASSERT_TRUE_DBG(
          Exception, sonarDataBeginTime < sonarDataEndTime, "sonar data end time is smaller than begin time.");

      // wait until all relevant sonar messages have arrived and check for termination request
      // if (sonarFrameSynchronizer_.waitForUpToDateSonarData(okvis::Time(sonarDataEndTime)) == false) {
      //   return;
      // }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            sonarDataEndTime < sonarMeasurements_.back().timeStamp,
                            "Waiting for up to date sonar data seems to have failed!");

      okvis::SonarMeasurementDeque sonarData = getSonarMeasurements(sonarDataBeginTime, sonarDataEndTime);

      // if sonar_data is empty, either end_time > begin_time or
      // no measurements in timeframe, should not happen, as we waited for measurements
      if (sonarData.size() == 0) {
        beforeDetectTimer.stop();
        continue;
      }

      if (sonarData.front().timeStamp > frame->timeStamp) {
        LOG(WARNING) << "Frame is newer than oldest Sonar measurement. Dropping it.";
        beforeDetectTimer.stop();
        continue;
      }

      okvis::kinematics::Transformation T_WSo = T_WS * parameters_.sonar.T_SSo;

      // LOG (INFO) << "StereoRig V2 translation T_SSo: " << T_SSo.r();
      // LOG (INFO) << "StereoRig V2 rot T_SSo: " << T_SSo.q().toRotationMatrix();

      // TODO(sharmin) check it
      // Add sonar landmark (in world frame) to the graph
      for (okvis::SonarMeasurementDeque::const_iterator it = sonarData.begin(); it != sonarData.end(); ++it) {
        double range = it->measurement.range;
        double heading = it->measurement.heading;
        uint64_t lmId = okvis::IdProvider::instance().newId();

        okvis::kinematics::Transformation sonar_point(Eigen::Vector3d(range * cos(heading), range * sin(heading), 0.0),
                                                      Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
        okvis::kinematics::Transformation T_WSo_point = T_WSo * sonar_point;

        Eigen::Vector3d sonar_landmark = T_WSo_point.r();
      }
    }

    bool estimatorHasFrames = false;
    {
      std::lock_guard<std::mutex> lock(estimator_mutex_);
      estimatorHasFrames = estimator_.numFrames() != 0;
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = multiFrame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;
    const bool useStartupInitializationWindow = !startupInitializationComplete && !estimatorHasFrames;
    if (useStartupInitializationWindow) {
      // The pre-fix lower bound was the zero last-state timestamp.  For the
      // first estimator state use a timestamp window beginning at this
      // candidate camera frame; this excludes the scheduling-dependent IMU
      // prefix while retaining the real gravity initializer.
      imuDataBeginTime = multiFrame->timestamp();
      imuDataEndTime = multiFrame->timestamp() + temporal_imu_data_overlap;
    }

    OKVIS_ASSERT_TRUE_DBG(
        Exception, imuDataBeginTime < imuDataEndTime, "imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    const bool imuReady = imuFrameSynchronizer_.waitForUpToDateImuData(okvis::Time(imuDataEndTime));
    if (!imuReady) {
      return;
    }
    // ImuError::propagation() integrates from lastTimestamp, not from the
    // extra temporal-overlap prefix used for readiness.  Select the latest
    // timestamp at or before that physical start so callback-dependent IMU
    // history before the state cannot alter the propagation input.
    okvis::Time imuSelectionBeginTime = estimatorHasFrames ? lastTimestamp : imuDataBeginTime;
    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuSelectionBeginTime, imuDataEndTime);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (!estimatorHasFrames) {
      // first frame ever
      bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.setZero();
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success, "pose could not be initialized from imu measurements.");
      if (!success) {
        beforeDetectTimer.stop();
        continue;
      }
      startupInitializationComplete = true;
    } else {
      // get old T_WS
      propagationTimer.start();
      okvis::ceres::ImuError::propagation(
          imuData, parameters_.imu, T_WS, speedAndBiases, lastTimestamp, multiFrame->timestamp());
      propagationTimer.stop();
    }
    okvis::kinematics::Transformation T_WC = T_WS * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();

    detectTimer.start();
    frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
    detectTimer.stop();

    afterDetectTimer.start();

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(multiFrame->id())) {
        // LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push) {
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        return;
      }
      waitForMatchingThreadTimer.stop();
      previousFeatureTimestamp = multiFrame->timestamp();
    }
  }
}

// Loop that matches frames with existing frames.
void ThreadedKFVio::matchingLoop() {
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState", true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization", true);
  TimerSwitchable addStateTimer("2.3 addState", true);
  TimerSwitchable matchingTimer("2.4 matching", true);
  size_t imuWindowDiagnosticCounter = 0;

  for (;;) {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    if (keypointMeasurements_.PopBlocking(&frame) == false) return;

    prepareToAddStateTimer.start();
    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = frame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastAddedStateTimestamp_ - temporal_imu_data_overlap;
    const bool firstEstimatorState = imuDataBeginTime.isZero();
    if (firstEstimatorState) {
      // Keep Estimator::addStates() on the same timestamp-defined 22-sample
      // initialization window used by frameConsumerLoop.  The zero value is
      // only the pre-state sentinel, not a physical IMU propagation start.
      imuDataBeginTime = frame->timestamp();
    }

    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataBeginTime < imuDataEndTime,
                          "imu data end time is smaller than begin time."
                              << "current frametimestamp " << frame->timestamp() << " (id: " << frame->id()
                              << "last timestamp         " << lastAddedStateTimestamp_
                              << " (id: " << estimator_.currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(okvis::Time(imuDataEndTime)) == false) return;
    // TODO(Sharmin): check if needed to wait until all relevant sonar messages

    // For the first matching state imuDataBeginTime was changed above to the
    // candidate frame timestamp.  Once an estimator state exists,
    // addStates()/ImuError::propagation() starts at lastAddedStateTimestamp_,
    // so the overlap prefix must not determine the selected predecessor.
    okvis::Time imuSelectionBeginTime = firstEstimatorState ? imuDataBeginTime : lastAddedStateTimestamp_;
    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuSelectionBeginTime, imuDataEndTime);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) continue;

    if (parameters_.visualization.displayImages || debugImgCallback_ || driftDiagnosticsOptIn()) {
      double maximumGap = 0.0;
      for (size_t index = 1; index < imuData.size(); ++index) {
        maximumGap = std::max(maximumGap, (imuData[index].timeStamp - imuData[index - 1].timeStamp).toSec());
      }
      const double predecessorMargin = (imuSelectionBeginTime - imuData.front().timeStamp).toSec();
      const double successorMargin = (imuData.back().timeStamp - imuDataEndTime).toSec();
      const double configuredImuRate = parameters_.imu.rate;
      const double gapThreshold = std::isfinite(configuredImuRate) && configuredImuRate > 0.0
                                      ? 2.5 / configuredImuRate
                                      : 0.02;
      const double configuredCameraRate = parameters_.sensors_information.cameraRate;
      const size_t baselinePeriod = std::isfinite(configuredCameraRate) && configuredCameraRate >= 1.0
                                        ? static_cast<size_t>(std::llround(configuredCameraRate))
                                        : 1;
      ++imuWindowDiagnosticCounter;
      if (maximumGap > gapThreshold || predecessorMargin < 0.0 || successorMargin < 0.0 ||
          imuWindowDiagnosticCounter % baselinePeriod == 0) {
        LOG(INFO) << std::setprecision(17) << "[IMU_WINDOW_DIAGNOSTIC]"
                  << " timestamp=" << frame->timestamp().toSec() << " frame=" << frame->id()
                  << " samples=" << imuData.size() << " selection_begin=" << imuSelectionBeginTime.toSec()
                  << " selection_end=" << imuDataEndTime.toSec()
                  << " first_sample=" << imuData.front().timeStamp.toSec()
                  << " last_sample=" << imuData.back().timeStamp.toSec()
                  << " predecessor_margin=" << predecessorMargin
                  << " successor_margin=" << successorMargin << " max_gap=" << maximumGap
                  << " expected_period="
                  << (std::isfinite(configuredImuRate) && configuredImuRate > 0.0 ? 1.0 / configuredImuRate : 0.0);
      }
    }

    // @Sharmin
    // Sonar Data
    okvis::SonarMeasurementDeque sonarData;
    if (parameters_.sensorList.isSonarUsed) {
      // -- get relevant sonar messages for new state
      okvis::Time sonarDataEndTime = frame->timestamp();
      okvis::Time sonarDataBeginTime = lastAddedStateTimestamp_;

      OKVIS_ASSERT_TRUE_DBG(
          Exception, sonarDataBeginTime < sonarDataEndTime, "sonar data end time is smaller than begin time.");

      // wait until all relevant sonar messages have arrived and check for termination request
      // if (sonarFrameSynchronizer_.waitForUpToDateSonarData(okvis::Time(sonarDataEndTime)) == false) {
      //   return;
      // }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            sonarDataEndTime < sonarMeasurements_.back().timeStamp,
                            "Waiting for up to date sonar data seems to have failed!");

      sonarData = getSonarMeasurements(sonarDataBeginTime, sonarDataEndTime);
      // prepareToAddStateTimer.stop();

      // if sonar_data is empty, either end_time > begin_time or
      // no measurements in timeframe, should not happen, as we waited for measurements
      if (sonarData.size() == 0) continue;
    }
    // Depth data
    okvis::DepthMeasurementDeque depthData;
    if (parameters_.sensorList.isDepthUsed) {
      // -- get relevant depth message for new state
      okvis::Time depthDataEndTime = frame->timestamp();
      okvis::Time depthDataBeginTime = lastAddedStateTimestamp_;

      OKVIS_ASSERT_TRUE_DBG(
          Exception, depthDataBeginTime < depthDataEndTime, "Depth data end time is smaller than begin time.");

      // wait until all relevant depth messages have arrived and check for termination request
      // if (depthFrameSynchronizer_.waitForUpToDateDepthData(okvis::Time(depthDataEndTime)) == false) {
      //   return;
      // }
      OKVIS_ASSERT_TRUE_DBG(Exception,
                            depthDataEndTime < depthMeasurements_.back().timeStamp,
                            "Waiting for up to date depth data seems to have failed!");

      depthData = getDepthMeasurements(depthDataBeginTime, depthDataEndTime);
      prepareToAddStateTimer.stop();

      // if depth_data is empty, either end_time > begin_time or
      // no measurements in timeframe, should not happen, as we waited for measurements
      if (depthData.size() == 0) {
        LOG(WARNING) << "NO DEPTH DATA!!!";
        continue;
      }
    }

    // End @sharmin

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_) optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;
      // @Sharmin
      if (estimator_.addStates(frame, imuData, asKeyframe, sonarData, depthData, firstDepth_)) {
        lastAddedStateTimestamp_ = frame->timestamp();
        addStateTimer.stop();
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        markFeatureStateReady(frame->timestamp());
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      okvis::kinematics::Transformation T_WS;
      estimator_.get_T_WS(frame->id(), T_WS);
      matchingTimer.start();
      frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_, map_, frame, &asKeyframe);
      matchingTimer.stop();
      if (asKeyframe) estimator_.setKeyframe(frame->id(), asKeyframe);
      if (!blocking_) {
        double timeLimit =
            parameters_.optimization.timeLimitForMatchingAndOptimization - (okvis::Time::now() - t0Matching).toSec();
        estimator_.setOptimizationTimeLimit(std::max<double>(0.0, timeLimit), parameters_.optimization.min_iterations);
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false) return;
  }
}

// Loop to process IMU measurements.
void ThreadedKFVio::imuConsumerLoop() {
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements", true);
  for (;;) {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false) return;
    processImuTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    okvis::ImuMeasurementDeque imuMeasurementsSnapshot;
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      if (!imuMeasurements_.empty() && !(imuMeasurements_.back().timeStamp < data.timeStamp)) {
        LOG(WARNING) << "Dropping out-of-order IMU measurement: last="
                     << imuMeasurements_.back().timeStamp.toSec() << ", new=" << data.timeStamp.toSec();
        processImuTimer.stop();
        continue;
      }

      if (parameters_.publishing.publishImuPropagatedState) {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_) {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else
          start = okvis::Time(0, 0);
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);
      imuMeasurementsSnapshot = imuMeasurements_;
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);

    if (parameters_.publishing.publishImuPropagatedState) {
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

      frontend_.propagation(imuMeasurementsSnapshot,
                            imu_params_,
                            T_WS_propagated_,
                            speedAndBiases_propagated_,
                            start,
                            *end,
                            &covariance,
                            &jacobian);
      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S =
          imuMeasurementsSnapshot.back().measurement.gyroscopes - speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(okvis::kinematics::Transformation(*parameters_.nCameraSystem.T_SC(i)));
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result, 1);
    }
    processImuTimer.stop();
  }
}

// @Sharmin
// Loop to process depth measurements.
void ThreadedKFVio::depthConsumerLoop() {
  okvis::DepthMeasurement data;
  TimerSwitchable processDepthTimer("0 processDepthMeasurements", true);
  for (;;) {
    // get data and check for termination request
    if (depthMeasurementsReceived_.PopBlocking(&data) == false) return;
    processDepthTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> depthLock(depthMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        depthMeasurements_.empty() || depthMeasurements_.back().timeStamp < data.timeStamp,
                        "Depth measurement from the past received");
      if (depthMeasurements_.size() > 0)
        start = depthMeasurements_.back().timeStamp;
      else
        start = okvis::Time(0, 0);
      end = &data.timeStamp;
      depthMeasurements_.push_back(data);
    }  // unlock depthMeasurements_mutex_

    // notify other threads that depth data with timeStamp is here.
    // depthFrameSynchronizer_.gotDepthData(data.timeStamp);

    processDepthTimer.stop();
  }
}

// @Sharmin
// Loop to process sonar measurements.
void ThreadedKFVio::sonarConsumerLoop() {
  okvis::SonarMeasurement data;
  TimerSwitchable processSonarTimer("0 processSonarMeasurements", true);
  for (;;) {
    // get data and check for termination request
    if (sonarMeasurementsReceived_.PopBlocking(&data) == false) return;
    processSonarTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> sonarLock(sonarMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        sonarMeasurements_.empty() || sonarMeasurements_.back().timeStamp < data.timeStamp,
                        "Sonar measurement from the past received");
      if (sonarMeasurements_.size() > 0)
        start = sonarMeasurements_.back().timeStamp;
      else
        start = okvis::Time(0, 0);
      end = &data.timeStamp;
      sonarMeasurements_.push_back(data);
    }  // unlock sonarMeasurements_mutex_

    // notify other threads that sonar data with timeStamp is here.
    // sonarFrameSynchronizer_.gotSonarData(data.timeStamp);

    processSonarTimer.stop();
  }
}

// Loop to process position measurements.
void ThreadedKFVio::positionConsumerLoop() {
  okvis::PositionMeasurement data;
  for (;;) {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false) return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void ThreadedKFVio::gpsConsumerLoop() {}

// Loop to process magnetometer measurements.
void ThreadedKFVio::magnetometerConsumerLoop() {}

// Loop to process differential pressure measurements.
void ThreadedKFVio::differentialConsumerLoop() {}

// Loop that visualizes completed frames.
void ThreadedKFVio::visualizationLoop() {
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;) {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false) return;
    // visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      out_images[i] = visualizer_.drawMatches(new_data, i);
    }
    displayImages_.PushNonBlockingDroppingIfFull(out_images, 1);
  }
}

// trigger display (needed because OSX won't allow threaded display)
void ThreadedKFVio::display() {
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0) return;
  if (displayImages_.PopBlocking(&out_images) == false) return;
  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
    if (debugImgCallback_) {
      debugImgCallback_(lastOptimizedStateTimestamp_, im, out_images[im]);
    }

    if (parameters_.visualization.displayImages) {
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
      if (!out_images[im].empty()) {
        cv::imshow(windowname.str(), out_images[im]);  // Prevent crashes from display
      }
    }
  }
  cv::waitKey(1);
}

// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(okvis::Time& imuDataBeginTime,
                                                            okvis::Time& imuDataEndTime) {
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  // Keep the producer, marginalization cleanup, and this snapshot on one
  // synchronization path.  In particular, do not inspect back() before the
  // lock: the camera and matching consumers can run while the IMU consumer
  // appends or marginalization erases measurements.
  if (imuMeasurements_.empty() || imuDataEndTime < imuDataBeginTime ||
      imuDataBeginTime > imuMeasurements_.back().timeStamp) {
    return okvis::ImuMeasurementDeque();
  }
  return okvis::startup_camera_imu::selectWindow(imuMeasurements_, imuDataBeginTime, imuDataEndTime);
}

// @Sharmin
// Get the depth measurement in-between/nearest to start and end. Depth sensor has a slowed rate, 1 Hz.
okvis::DepthMeasurementDeque ThreadedKFVio::getDepthMeasurements(okvis::Time& beginTime, okvis::Time& endTime) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest sonar time, return empty queue.
  if (endTime < beginTime || beginTime > depthMeasurements_.back().timeStamp) return okvis::DepthMeasurementDeque();

  std::lock_guard<std::mutex> lock(depthMeasurements_mutex_);
  // get iterator to depth data before previous frame
  okvis::DepthMeasurementDeque::iterator first_depth_package = depthMeasurements_.begin();
  okvis::DepthMeasurementDeque::iterator last_depth_package = depthMeasurements_.end();

  // TODO(sharmin) go backwards through queue. Is probably faster.
  // TODO(Sharmin) check it
  for (auto iter = depthMeasurements_.begin(); iter != depthMeasurements_.end(); ++iter) {
    // move depth_package iterator back until iter->timeStamp is higher than requested begintime
    if (iter->timeStamp <= beginTime) first_depth_package = iter;

    if (iter->timeStamp >= endTime) {
      last_depth_package = iter;
      ++last_depth_package;
      break;
    }
  }

  // create copy of depth buffer
  return okvis::DepthMeasurementDeque(first_depth_package, last_depth_package);
}

// @Sharmin
// Get a subset of the recorded Sonar measurements.
okvis::SonarMeasurementDeque ThreadedKFVio::getSonarMeasurements(okvis::Time& sonarDataBeginTime,
                                                                 okvis::Time& sonarDataEndTime) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest sonar time, return empty queue.
  if (sonarDataEndTime < sonarDataBeginTime || sonarDataBeginTime > sonarMeasurements_.back().timeStamp)
    return okvis::SonarMeasurementDeque();

  std::lock_guard<std::mutex> lock(sonarMeasurements_mutex_);
  // get iterator to sonar data before previous frame
  okvis::SonarMeasurementDeque::iterator first_sonar_package = sonarMeasurements_.begin();
  okvis::SonarMeasurementDeque::iterator last_sonar_package = sonarMeasurements_.end();
  // TODO(sharmin) go backwards through queue. Is probably faster.
  for (auto iter = sonarMeasurements_.begin(); iter != sonarMeasurements_.end(); ++iter) {
    // move first_sonar_package iterator back until iter->timeStamp is higher than requested begintime
    if (iter->timeStamp <= sonarDataBeginTime) first_sonar_package = iter;

    // set last_sonar_package iterator as soon as we hit first timeStamp higher than requested endtime & break
    if (iter->timeStamp >= sonarDataEndTime) {
      last_sonar_package = iter;
      // since we want to include this last sonar measurement in returned Deque we
      // increase last_sonar_package iterator once.
      ++last_sonar_package;
      break;
    }
  }

  // create copy of sonar buffer
  return okvis::SonarMeasurementDeque(first_sonar_package, last_sonar_package);
}

// Remove IMU measurements from the internal buffer.
int ThreadedKFVio::deleteImuMeasurements(const okvis::Time& eraseUntil) {
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  if (imuMeasurements_.empty()) return 0;
  if (imuMeasurements_.front().timeStamp > eraseUntil) return 0;

  const int removed = static_cast<int>(okvis::startup_camera_imu::eraseBeforeKeepingPredecessor(
      imuMeasurements_, eraseUntil));

  return removed;
}

// Loop that performs the optimization and marginalisation.
void ThreadedKFVio::optimizationLoop() {
  TimerSwitchable optimizationTimer("3.1 optimization", true);
  TimerSwitchable marginalizationTimer("3.2 marginalization", true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization", true);
  okvis::Time driftDiagnosticEpoch(0, 0);
  size_t driftDiagnosticFrameCounter = 0;
  size_t previousFiniteSupport = 0;
  size_t driftDiagnosticBurstRemaining = 0;
  bool driftDiagnosticEventActive = false;

  for (;;) {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    std::string driftDiagnosticLine;
    bool driftDiagnosticShouldEmit = false;
    if (matchedFrames_.PopBlocking(&frame_pairs) == false) return;
    OptimizationResults result;
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      const bool driftDiagnosticsEnabled = parameters_.visualization.displayImages || debugImgCallback_ ||
                                           driftDiagnosticsOptIn();
      okvis::kinematics::Transformation preOptimizationT_WS;
      okvis::SpeedAndBias preOptimizationSpeedAndBias = okvis::SpeedAndBias::Zero();
      okvis::kinematics::Transformation previousOptimizedT_WS;
      okvis::Time previousOptimizedTimestamp(0, 0);
      bool havePreOptimizationState = false;
      bool havePreviousOptimizedState = false;
      std::vector<double> preOptimizationChi2;
      size_t preOptimizationInvalidProjections = 0;
      if (driftDiagnosticsEnabled) {
        havePreOptimizationState = estimator_.get_T_WS(frame_pairs->id(), preOptimizationT_WS) &&
                                   estimator_.getSpeedAndBias(
                                       frame_pairs->id(), 0, preOptimizationSpeedAndBias);
        std::lock_guard<std::mutex> stateLock(lastState_mutex_);
        if (lastOptimizedStateTimestamp_ > okvis::Time(0, 0)) {
          previousOptimizedT_WS = lastOptimized_T_WS_;
          previousOptimizedTimestamp = lastOptimizedStateTimestamp_;
          havePreviousOptimizedState = true;
        }
      }
      if (havePreOptimizationState) {
        for (size_t cameraIndex = 0; cameraIndex < frame_pairs->numFrames(); ++cameraIndex) {
          for (size_t keypointIndex = 0; keypointIndex < frame_pairs->numKeypoints(cameraIndex); ++keypointIndex) {
            const uint64_t landmarkId = frame_pairs->landmarkId(cameraIndex, keypointIndex);
            if (landmarkId == 0 || !estimator_.isLandmarkAdded(landmarkId) ||
                !estimator_.isLandmarkInitialized(landmarkId)) {
              continue;
            }
            okvis::MapPoint landmark;
            estimator_.getLandmark(landmarkId, landmark);
            Eigen::Vector2d measurement;
            double keypointSize = 0.0;
            frame_pairs->getKeypoint(cameraIndex, keypointIndex, measurement);
            frame_pairs->getKeypointSize(cameraIndex, keypointIndex, keypointSize);
            const okvis::kinematics::Transformation T_CW =
                frame_pairs->T_SC(cameraIndex)->inverse() * preOptimizationT_WS.inverse();
            Eigen::Vector2d prediction;
            const cameras::CameraBase::ProjectionStatus status =
                frame_pairs->geometry(cameraIndex)->projectHomogeneous(T_CW * landmark.point, &prediction);
            if (status != cameras::CameraBase::ProjectionStatus::Successful || !prediction.allFinite() ||
                !measurement.allFinite() || !std::isfinite(keypointSize) || keypointSize <= 0.0) {
              ++preOptimizationInvalidProjections;
              continue;
            }
            preOptimizationChi2.push_back(
                (prediction - measurement).squaredNorm() * 64.0 / (keypointSize * keypointSize));
          }
        }
      }
      optimizationTimer.start();
      // if(frontend_.isInitialized()){
      //TODO(GUO): make this num of threads for optimization configurable
      estimator_.optimize(parameters_.optimization.max_iterations, 2, false);
      //}
      /*if (estimator_.numFrames() > 0 && !frontend_.isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_.numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_.get_T_WS(estimator_.frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_.set_T_WS(estimator_.frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_.getSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_.setSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      optimizationTimer.stop();

      const size_t preMarginalizationFrames = estimator_.numFrames();
      const size_t preMarginalizationLandmarks = estimator_.numLandmarks();

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there
      // (if not keyframe)
      if (estimator_.numFrames() > size_t(parameters_.optimization.numImuFrames)) {
        deleteImuMeasurementsUntil =
            estimator_.multiFrame(estimator_.frameIdByAge(parameters_.optimization.numImuFrames))->timestamp() -
            temporal_imu_data_overlap;
      }

      marginalizationTimer.start();
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes, parameters_.optimization.numImuFrames, result.transferredLandmarks);
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_.getSpeedAndBias(frame_pairs->id(), 0, lastOptimizedSpeedAndBiases_);
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();
        if (featureStateReadyThrough_ < frame_pairs->timestamp()) {
          featureStateReadyThrough_ = frame_pairs->timestamp();
        }

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        } else {
          result.onlyPublishLandmarks = true;
        }
        estimator_.getLandmarks(result.landmarksVector);

        repropagationNeeded_ = true;

        //*********************   Added by Sharmin*******************************************//

        // To publish the topics needed by pose-graph

        if (estimator_.isKeyframe(frame_pairs->id())) {
          // publish keyframe image, pose, and points
          if (keyframeCallback_ && !result.landmarksVector.empty()) {
            {
              std::lock_guard<std::mutex> lock(kf_f_map_mutex_);
              kf_f_map_.insert(std::make_pair(kf_index_, frame_pairs->id()));
            }

            // @Reloc: it's redundant, but reduces a lot of hassels
            PointMap lmMap;  // get a copy of landmarksMap_
            estimator_.getLandmarks(lmMap);

            const size_t CamIndexA = 0;                                 // for left camera
            cv::Mat image_l = frame_pairs->frames_[CamIndexA].image();  // image to publish

            std::vector<std::list<std::vector<double>>> kf_points;
            int num_keypoint = 0;

            for (auto cit = lmMap.begin(); cit != lmMap.end(); ++cit) {
              std::map<okvis::KeypointIdentifier, uint64_t> observations =
                  cit->second.observations;  // result.landmarksVector.at(l).observations;
              for (auto mit = observations.begin(); mit != observations.end(); ++mit) {
                if ((mit->first).frameId == frame_pairs->id()) {
                  std::list<std::vector<double>> ptList;
                  int obs_num = 0;

                  Eigen::Vector4d landmark = cit->second.point;  // 3D point to publish
                  std::vector<double> pt3d;
                  pt3d.push_back(landmark[0] / landmark[3]);
                  pt3d.push_back(landmark[1] / landmark[3]);
                  pt3d.push_back(landmark[2] / landmark[3]);
                  ptList.push_back(pt3d);

                  obs_num++;

                  cv::KeyPoint cvkeypoint;  // Associated 2D point in left image to publish
                  if (mit->first.cameraIndex != CamIndexA) continue;
                  if ((mit->first).keypointIndex >= frame_pairs->numKeypoints(CamIndexA)) continue;

                  frame_pairs->getCvKeypoint(CamIndexA, (mit->first).keypointIndex, cvkeypoint);

                  std::vector<double> pt_id_w_uv;
                  if (isnan(cvkeypoint.pt.x) || isnan(cvkeypoint.pt.y) ||
                      isnan(cvkeypoint.size))  // TODO(Sharmin): Better way to fix this?
                    continue;

                  // @Reloc
                  pt_id_w_uv.push_back(cit->first);                  // landmarkId
                  pt_id_w_uv.push_back(frame_pairs->id());           // poseId or MultiFrameId
                  pt_id_w_uv.push_back((mit->first).keypointIndex);  // keypointIdx
                  pt_id_w_uv.push_back(cit->second.quality);         // u

                  // cv::keypoint. Size 8
                  pt_id_w_uv.push_back(kf_index_);
                  pt_id_w_uv.push_back(cvkeypoint.pt.x);
                  pt_id_w_uv.push_back(cvkeypoint.pt.y);

                  pt_id_w_uv.push_back(cvkeypoint.size);
                  pt_id_w_uv.push_back(cvkeypoint.angle);
                  pt_id_w_uv.push_back(static_cast<double>(cvkeypoint.octave));
                  pt_id_w_uv.push_back(cvkeypoint.response);
                  pt_id_w_uv.push_back(static_cast<double>(cvkeypoint.class_id));

                  ptList.push_back(pt_id_w_uv);

                  obs_num++;

                  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator mMPit = observations.begin();
                       mMPit != observations.end();
                       ++mMPit) {
                    if ((mMPit->first).frameId == frame_pairs->id()) continue;

                    {
                      std::lock_guard<std::mutex> lock(kf_f_map_mutex_);
                      for (std::map<int, size_t>::iterator mKfId = kf_f_map_.begin(); mKfId != kf_f_map_.end();
                           ++mKfId) {
                        if ((mMPit->first).frameId == mKfId->second) {
                          ptList.push_back({static_cast<double>(mKfId->first)});

                          obs_num++;
                          break;
                        }
                      }
                    }
                  }
                  kf_points.push_back(ptList);

                  num_keypoint++;
                  break;
                }
              }
            }

            okvis::kinematics::Transformation T_WCa =
                lastOptimized_T_WS_ * (*parameters_.nCameraSystem.T_SC(CamIndexA));
            keyframeCallback_(lastOptimizedStateTimestamp_, image_l, T_WCa, kf_points);

            std::cout << "Keyframe Index: " << kf_index_ << std::endl;
            kf_index_++;
          }
        }

        //*********** End Added by Sharmin *******//
      }

      if (parameters_.visualization.displayImages || debugImgCallback_ || driftDiagnosticsOptIn()) {
        // fill in information that requires access to estimator.
        visualizationDataPtr = std::make_shared<VioVisualizer::VisualizationData>();
        VioVisualizer::VisualizationData::DriftDiagnostics& drift = visualizationDataPtr->drift;
        drift.frameId = frame_pairs->id();
        drift.timestamp = frame_pairs->timestamp().toSec();
        if (driftDiagnosticEpoch == okvis::Time(0, 0)) driftDiagnosticEpoch = frame_pairs->timestamp();
        drift.relativeTimestamp = (frame_pairs->timestamp() - driftDiagnosticEpoch).toSec();
        drift.isKeyframe = estimator_.isKeyframe(frame_pairs->id());
        drift.activeFrames = estimator_.numFrames();
        for (size_t age = 0; age < estimator_.numFrames(); ++age) {
          if (estimator_.isKeyframe(estimator_.frameIdByAge(age))) ++drift.activeKeyframes;
        }
        drift.activeLandmarks = result.landmarksVector.size();
        drift.preMarginalizationFrames = preMarginalizationFrames;
        drift.preMarginalizationLandmarks = preMarginalizationLandmarks;
        drift.marginalizedLandmarks = result.transferredLandmarks.size();
        for (const okvis::MapPoint& mapPoint : result.landmarksVector) {
          if (estimator_.isLandmarkInitialized(mapPoint.id)) ++drift.activeInitializedLandmarks;
        }
        drift.position = lastOptimized_T_WS_.r();
        drift.orientation = lastOptimized_T_WS_.q();
        drift.velocity = lastOptimizedSpeedAndBiases_.head<3>();
        drift.gyroBias = lastOptimizedSpeedAndBiases_.segment<3>(3);
        drift.accelBias = lastOptimizedSpeedAndBiases_.segment<3>(6);
        if (havePreOptimizationState) {
          drift.optimizationTranslation = (lastOptimized_T_WS_.r() - preOptimizationT_WS.r()).norm();
          drift.optimizationRotationDegrees =
              quaternionDistanceDegrees(lastOptimized_T_WS_.q(), preOptimizationT_WS.q());
          drift.optimizationVelocity =
              (lastOptimizedSpeedAndBiases_.head<3>() - preOptimizationSpeedAndBias.head<3>()).norm();
          drift.optimizationGyroBias =
              (lastOptimizedSpeedAndBiases_.segment<3>(3) - preOptimizationSpeedAndBias.segment<3>(3)).norm();
          drift.optimizationAccelBias =
              (lastOptimizedSpeedAndBiases_.segment<3>(6) - preOptimizationSpeedAndBias.segment<3>(6)).norm();
        }
        if (havePreviousOptimizedState) {
          drift.deltaTime = (lastOptimizedStateTimestamp_ - previousOptimizedTimestamp).toSec();
          drift.frameTranslation = (lastOptimized_T_WS_.r() - previousOptimizedT_WS.r()).norm();
          drift.frameRotationDegrees =
              quaternionDistanceDegrees(lastOptimized_T_WS_.q(), previousOptimizedT_WS.q());
        }
        const Estimator::OptimizationDiagnostics optimizer = estimator_.optimizationDiagnostics();
        drift.optimizationIterations = optimizer.iterations;
        drift.optimizationSuccessfulSteps = optimizer.successfulSteps;
        drift.optimizationUnsuccessfulSteps = optimizer.unsuccessfulSteps;
        drift.optimizationTerminationType = optimizer.terminationType;
        drift.optimizationInitialCost = optimizer.initialCost;
        drift.optimizationFinalCost = optimizer.finalCost;
        drift.optimizationTimeSeconds = optimizer.totalTimeSeconds;
        drift.optimizationUsable = optimizer.solutionUsable;
        drift.cameraKeypoints.assign(frame_pairs->numFrames(), 0);
        drift.cameraUnassociated.assign(frame_pairs->numFrames(), 0);
        drift.cameraPending.assign(frame_pairs->numFrames(), 0);
        drift.cameraFinite.assign(frame_pairs->numFrames(), 0);
        drift.cameraUninitialized.assign(frame_pairs->numFrames(), 0);
        drift.cameraInvalidInitialized.assign(frame_pairs->numFrames(), 0);
        drift.cameraFiniteCoverageCells.assign(frame_pairs->numFrames(), 0);
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        auto it = visualizationDataPtr->observations.begin();
        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames(); ++camIndex) {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k) {
            OKVIS_ASSERT_TRUE_DBG(
                Exception, it != visualizationDataPtr->observations.end(), "Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            ++drift.keypoints;
            ++drift.cameraKeypoints[camIndex];
            if (it->landmarkId == 0) {
              ++drift.unassociated;
              ++drift.cameraUnassociated[camIndex];
            }
            if (estimator_.isLandmarkAdded(it->landmarkId)) {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.point;
              if (estimator_.isLandmarkInitialized(it->landmarkId)) {
                it->isInitialized = true;
                if (landmark.point.allFinite() && std::abs(landmark.point[3]) > 1.0e-8) {
                  ++drift.finite;
                  ++drift.cameraFinite[camIndex];
                } else {
                  ++drift.invalidInitialized;
                  ++drift.cameraInvalidInitialized[camIndex];
                }
              } else {
                it->isInitialized = false;
                ++drift.uninitialized;
                ++drift.cameraUninitialized[camIndex];
              }
            } else {
              it->landmark_W =
                  Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
              if (it->landmarkId != 0) {
                ++drift.pending;
                ++drift.cameraPending[camIndex];
              }
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(), visualizationDataPtr->T_WS_keyFrame);

        // Geometry-only diagnostics.  The lambda is invoked only when the
        // existing drift-record cadence selects this frame below.  It reads
        // copies already made for this frame and never alters landmarks,
        // observations, thresholds, factors, or solver state.
        auto populateGeometryDiagnostics = [&]() {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const size_t cameraCount = frame_pairs->numFrames();
        drift.geometryCurrentCameraRangeValidCount.assign(cameraCount, 0);
        drift.geometryCurrentCameraRangeP10.assign(cameraCount, nan);
        drift.geometryCurrentCameraRangeP50.assign(cameraCount, nan);
        drift.geometryCurrentCameraRangeP90.assign(cameraCount, nan);
        drift.geometryCurrentCameraDepthValidCount.assign(cameraCount, 0);
        drift.geometryCurrentCameraDepthP10.assign(cameraCount, nan);
        drift.geometryCurrentCameraDepthP50.assign(cameraCount, nan);
        drift.geometryCurrentCameraDepthP90.assign(cameraCount, nan);
        drift.geometryActiveObservationCountValid = 0;
        drift.geometryActiveObservationCountP10 = nan;
        drift.geometryActiveObservationCountP50 = nan;
        drift.geometryActiveObservationCountP90 = nan;
        drift.geometryActiveFrameSpanValid = 0;
        drift.geometryActiveFrameSpanP10 = nan;
        drift.geometryActiveFrameSpanP50 = nan;
        drift.geometryActiveFrameSpanP90 = nan;
        drift.geometryParallaxValid = 0;
        drift.geometryParallaxP10 = nan;
        drift.geometryParallaxP50 = nan;
        drift.geometryParallaxP90 = nan;
        drift.geometryTranslationInfoObservationCount = 0;
        drift.geometryTranslationInfoEigenMin = nan;
        drift.geometryTranslationInfoEigenMedian = nan;
        drift.geometryTranslationInfoEigenMax = nan;
        drift.geometryTranslationInfoEffectiveRank = 0;
        drift.geometryTranslationInfoConditionNumber = nan;

        // Build an explicit active-frame index.  Observations referring to a
        // marginalized/unavailable frame or keypoint are excluded rather than
        // treated as zero-good samples.  The map is bounded by the estimator
        // window and avoids repeated multiFrame lookups per observation.
        struct ActiveFrameState {
          double timestamp = 0.0;
          okvis::kinematics::Transformation T_WS;
          okvis::MultiFramePtr frame;
        };
        std::map<uint64_t, ActiveFrameState> activeFrameStates;
        for (size_t age = 0; age < estimator_.numFrames(); ++age) {
          const uint64_t activeFrameId = estimator_.frameIdByAge(age);
          if (activeFrameId == 0) continue;
          okvis::MultiFramePtr activeFrame = estimator_.multiFrame(activeFrameId);
          okvis::kinematics::Transformation activeT_WS;
          if (!activeFrame || !estimator_.get_T_WS(activeFrameId, activeT_WS)) continue;
          activeFrameStates.emplace(activeFrameId,
                                    ActiveFrameState{activeFrame->timestamp().toSec(), activeT_WS, activeFrame});
        }

        // result.landmarksVector is a unique active-map copy.  Keep one entry
        // per ID so repeated current-frame observations cannot duplicate a
        // landmark in the per-camera depth/range summaries.
        std::map<uint64_t, const okvis::MapPoint*> finiteLandmarks;
        for (const okvis::MapPoint& mapPoint : result.landmarksVector) {
          if (mapPoint.id == 0 || !mapPoint.point.allFinite() || std::abs(mapPoint.point[3]) <= 1.0e-8 ||
              !estimator_.isLandmarkInitialized(mapPoint.id)) {
            continue;
          }
          const Eigen::Vector3d point_W = mapPoint.point.head<3>() / mapPoint.point[3];
          if (point_W.allFinite()) finiteLandmarks.emplace(mapPoint.id, &mapPoint);
        }

        // The nested sets are the unique current-frame camera+landmark keys;
        // repeated keypoints for one landmark in one camera contribute once.
        std::vector<std::set<uint64_t>> currentCameraFiniteLandmarks(cameraCount);
        std::set<uint64_t> currentVisibleFiniteLandmarks;
        for (const okvis::Observation& observation : visualizationDataPtr->observations) {
          if (!observation.isInitialized || observation.cameraIdx >= cameraCount || observation.landmarkId == 0 ||
              finiteLandmarks.find(observation.landmarkId) == finiteLandmarks.end()) {
            continue;
          }
          currentCameraFiniteLandmarks[observation.cameraIdx].insert(observation.landmarkId);
          currentVisibleFiniteLandmarks.insert(observation.landmarkId);
        }

        // Current-camera depth/range uses the optimized current pose and only
        // finite initialized landmarks observed in that camera.  Depth is the
        // positive camera-frame z coordinate; range is the Euclidean camera
        // frame norm.  Quantiles are per camera and retain nan/count when empty.
        Eigen::Matrix3d translationInfoSurrogate = Eigen::Matrix3d::Zero();
        size_t translationInfoObservationCount = 0;
        for (size_t cameraIndex = 0; cameraIndex < cameraCount; ++cameraIndex) {
          std::vector<double> ranges;
          std::vector<double> depths;
          const okvis::kinematics::Transformation T_CW =
              frame_pairs->T_SC(cameraIndex)->inverse() * lastOptimized_T_WS_.inverse();
          for (const uint64_t landmarkId : currentCameraFiniteLandmarks[cameraIndex]) {
            const auto landmarkIt = finiteLandmarks.find(landmarkId);
            if (landmarkIt == finiteLandmarks.end()) continue;
            const Eigen::Vector4d point_C_h = T_CW * landmarkIt->second->point;
            if (!point_C_h.allFinite() || std::abs(point_C_h[3]) <= 1.0e-8) continue;
            const Eigen::Vector3d point_C = point_C_h.head<3>() / point_C_h[3];
            const double range = point_C.norm();
            if (std::isfinite(range) && range > 1.0e-9) ranges.push_back(range);
            if (std::isfinite(point_C.z()) && point_C.z() > 1.0e-9) depths.push_back(point_C.z());
            if (std::isfinite(range) && range > 1.0e-9) {
              // This is the current-frame translation-information surrogate.
              // Each unique current camera+landmark observation contributes in
              // the common world frame, using the current optimized pose/ray.
              const okvis::kinematics::Transformation T_WC =
                  lastOptimized_T_WS_ * (*parameters_.nCameraSystem.T_SC(cameraIndex));
              const Eigen::Vector3d unitBearing_C = point_C / range;
              const Eigen::Vector3d unitBearing_W = T_WC * unitBearing_C;
              if (unitBearing_W.allFinite()) {
                translationInfoSurrogate +=
                    (Eigen::Matrix3d::Identity() - unitBearing_W * unitBearing_W.transpose()) /
                    (range * range);
                ++translationInfoObservationCount;
              }
            }
          }
          drift.geometryCurrentCameraRangeValidCount[cameraIndex] = ranges.size();
          drift.geometryCurrentCameraRangeP10[cameraIndex] = percentileOrNaN(ranges, 0.10);
          drift.geometryCurrentCameraRangeP50[cameraIndex] = percentileOrNaN(ranges, 0.50);
          drift.geometryCurrentCameraRangeP90[cameraIndex] = percentileOrNaN(ranges, 0.90);
          drift.geometryCurrentCameraDepthValidCount[cameraIndex] = depths.size();
          drift.geometryCurrentCameraDepthP10[cameraIndex] = percentileOrNaN(depths, 0.10);
          drift.geometryCurrentCameraDepthP50[cameraIndex] = percentileOrNaN(depths, 0.50);
          drift.geometryCurrentCameraDepthP90[cameraIndex] = percentileOrNaN(depths, 0.90);
        }

        std::vector<double> activeObservationCounts;
        std::vector<double> activeFrameSpans;
        std::vector<double> parallaxAnglesDegrees;
        for (const uint64_t landmarkId : currentVisibleFiniteLandmarks) {
          const auto landmarkIt = finiteLandmarks.find(landmarkId);
          if (landmarkIt == finiteLandmarks.end()) continue;
          const okvis::MapPoint& mapPoint = *landmarkIt->second;
          const Eigen::Vector3d point_W = mapPoint.point.head<3>() / mapPoint.point[3];
          std::vector<double> observationTimes;
          std::vector<Eigen::Vector3d> observationCenters;
          std::vector<Eigen::Vector3d> observationRays;
          for (const auto& observationEntry : mapPoint.observations) {
            const okvis::KeypointIdentifier& observation = observationEntry.first;
            const auto frameIt = activeFrameStates.find(observation.frameId);
            if (frameIt == activeFrameStates.end() || !frameIt->second.frame ||
                observation.cameraIndex >= cameraCount ||
                observation.keypointIndex >= frameIt->second.frame->numKeypoints(observation.cameraIndex)) {
              continue;
            }
            const okvis::kinematics::Transformation T_WC =
                frameIt->second.T_WS * (*parameters_.nCameraSystem.T_SC(observation.cameraIndex));
            const Eigen::Vector3d ray_W = point_W - T_WC.r();
            const double range = ray_W.norm();
            if (!std::isfinite(range) || range <= 1.0e-9 || !ray_W.allFinite()) continue;
            const Eigen::Vector3d unitBearing_W = ray_W / range;
            if (!unitBearing_W.allFinite()) continue;
            observationTimes.push_back(frameIt->second.timestamp);
            observationCenters.push_back(T_WC.r());
            observationRays.push_back(unitBearing_W);
          }
          if (observationTimes.empty()) continue;
          activeObservationCounts.push_back(static_cast<double>(observationTimes.size()));
          const auto timeBounds = std::minmax_element(observationTimes.begin(), observationTimes.end());
          activeFrameSpans.push_back(*timeBounds.second - *timeBounds.first);
          for (size_t first = 0; first < observationCenters.size(); ++first) {
            for (size_t second = first + 1; second < observationCenters.size(); ++second) {
              if ((observationCenters[first] - observationCenters[second]).norm() <= 1.0e-9) continue;
              const double cosine = std::clamp(observationRays[first].dot(observationRays[second]), -1.0, 1.0);
              parallaxAnglesDegrees.push_back(std::acos(cosine) * 180.0 / M_PI);
            }
          }
        }
        drift.geometryActiveObservationCountValid = activeObservationCounts.size();
        drift.geometryActiveObservationCountP10 = percentileOrNaN(activeObservationCounts, 0.10);
        drift.geometryActiveObservationCountP50 = percentileOrNaN(activeObservationCounts, 0.50);
        drift.geometryActiveObservationCountP90 = percentileOrNaN(activeObservationCounts, 0.90);
        drift.geometryActiveFrameSpanValid = activeFrameSpans.size();
        drift.geometryActiveFrameSpanP10 = percentileOrNaN(activeFrameSpans, 0.10);
        drift.geometryActiveFrameSpanP50 = percentileOrNaN(activeFrameSpans, 0.50);
        drift.geometryActiveFrameSpanP90 = percentileOrNaN(activeFrameSpans, 0.90);
        drift.geometryParallaxValid = parallaxAnglesDegrees.size();
        drift.geometryParallaxP10 = percentileOrNaN(parallaxAnglesDegrees, 0.10);
        drift.geometryParallaxP50 = percentileOrNaN(parallaxAnglesDegrees, 0.50);
        drift.geometryParallaxP90 = percentileOrNaN(parallaxAnglesDegrees, 0.90);
        drift.geometryTranslationInfoObservationCount = translationInfoObservationCount;
        if (translationInfoObservationCount > 0) {
          translationInfoSurrogate =
              0.5 * (translationInfoSurrogate + translationInfoSurrogate.transpose());
          const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(translationInfoSurrogate);
          if (eigenSolver.info() == Eigen::Success && eigenSolver.eigenvalues().allFinite()) {
            const Eigen::Vector3d eigenvalues = eigenSolver.eigenvalues();
            drift.geometryTranslationInfoEigenMin = eigenvalues[0];
            drift.geometryTranslationInfoEigenMedian = eigenvalues[1];
            drift.geometryTranslationInfoEigenMax = eigenvalues[2];
            // Effective rank uses a documented relative 1e-3 threshold on
            // lambda_max; rank deficiency remains visible instead of becoming
            // a misleading finite/zero-good scalar.
            const double relativeRankThreshold = 1.0e-3 * eigenvalues[2];
            drift.geometryTranslationInfoEffectiveRank = static_cast<size_t>(
                (eigenvalues.array() > relativeRankThreshold).count());
            if (eigenvalues[0] > 0.0) {
              drift.geometryTranslationInfoConditionNumber = eigenvalues[2] / eigenvalues[0];
            }
          }
        }
        };

        std::vector<bool> occupiedCells(parameters_.nCameraSystem.numCameras() * 16, false);
        for (const okvis::Observation& observation : visualizationDataPtr->observations) {
          if (!observation.isInitialized || observation.cameraIdx >= frame_pairs->numFrames()) continue;
          const cv::Mat& image = frame_pairs->image(observation.cameraIdx);
          if (image.empty() || !observation.keypointMeasurement.allFinite()) continue;
          const int column = std::clamp(
              static_cast<int>(4.0 * observation.keypointMeasurement[0] / image.cols), 0, 3);
          const int row = std::clamp(
              static_cast<int>(4.0 * observation.keypointMeasurement[1] / image.rows), 0, 3);
          occupiedCells[observation.cameraIdx * 16 + static_cast<size_t>(row * 4 + column)] = true;
        }
        drift.finiteCoverageCells = static_cast<size_t>(std::count(occupiedCells.begin(), occupiedCells.end(), true));
        drift.finiteCoverageCellsPossible = occupiedCells.size();
        for (size_t cameraIndex = 0; cameraIndex < frame_pairs->numFrames(); ++cameraIndex) {
          drift.cameraFiniteCoverageCells[cameraIndex] = static_cast<size_t>(
              std::count(occupiedCells.begin() + cameraIndex * 16,
                         occupiedCells.begin() + (cameraIndex + 1) * 16,
                         true));
        }

        const auto reprojectionChi2 = [&](const okvis::kinematics::Transformation& T_WS,
                                          std::vector<double>& values,
                                          size_t& invalid) {
          values.clear();
          invalid = 0;
          for (const okvis::Observation& observation : visualizationDataPtr->observations) {
            if (!observation.isInitialized || observation.cameraIdx >= frame_pairs->numFrames()) continue;
            const okvis::kinematics::Transformation T_CW =
                frame_pairs->T_SC(observation.cameraIdx)->inverse() * T_WS.inverse();
            const Eigen::Vector4d point_C = T_CW * observation.landmark_W;
            Eigen::Vector2d prediction;
            const cameras::CameraBase::ProjectionStatus status =
                frame_pairs->geometry(observation.cameraIdx)->projectHomogeneous(point_C, &prediction);
            if (status != cameras::CameraBase::ProjectionStatus::Successful || !prediction.allFinite() ||
                !observation.keypointMeasurement.allFinite() || !std::isfinite(observation.keypointSize) ||
                observation.keypointSize <= 0.0) {
              ++invalid;
              continue;
            }
            const double squaredError = (prediction - observation.keypointMeasurement).squaredNorm();
            values.push_back(squaredError * 64.0 /
                             (observation.keypointSize * observation.keypointSize));
          }
        };
        std::vector<double> postChi2;
        drift.reprojectionPreCount = preOptimizationChi2.size();
        drift.reprojectionPreInvalid = preOptimizationInvalidProjections;
        drift.reprojectionPreP50 = percentile(preOptimizationChi2, 0.50);
        drift.reprojectionPreP90 = percentile(preOptimizationChi2, 0.90);
        reprojectionChi2(lastOptimized_T_WS_, postChi2, drift.reprojectionPostInvalid);
        drift.reprojectionPostCount = postChi2.size();
        drift.reprojectionPostP50 = percentile(postChi2, 0.50);
        drift.reprojectionPostP90 = percentile(postChi2, 0.90);
        drift.reprojectionPostMax = percentile(postChi2, 1.0);
        if (!postChi2.empty()) {
          drift.reprojectionPostFractionOver4 =
              static_cast<double>(std::count_if(postChi2.begin(), postChi2.end(),
                                                [](double value) { return value > 4.0; })) /
              postChi2.size();
          drift.reprojectionPostFractionOver9 =
              static_cast<double>(std::count_if(postChi2.begin(), postChi2.end(),
                                                [](double value) { return value > 9.0; })) /
              postChi2.size();
        }

        // Decide the exact existing baseline/burst emission cadence before
        // running the expensive geometry summaries.  State updates remain
        // single-shot here; the log itself is still formatted and emitted
        // outside estimator_mutex_.
        const double configuredCameraRate = parameters_.sensors_information.cameraRate;
        const size_t baselinePeriod = std::isfinite(configuredCameraRate) && configuredCameraRate >= 1.0
                                          ? static_cast<size_t>(std::llround(configuredCameraRate))
                                          : 1;
        const bool supportCollapsed = drift.finite < 30;
        const bool supportDropped = previousFiniteSupport > 0 && drift.finite * 2 < previousFiniteSupport;
        const bool estimatorConflict = drift.reprojectionPostP90 > 4.0 || drift.optimizationTranslation > 0.2 ||
                                       !drift.optimizationUsable;
        ++driftDiagnosticFrameCounter;
        const bool eventActive = supportCollapsed || supportDropped || estimatorConflict;
        if (eventActive && !driftDiagnosticEventActive) {
          driftDiagnosticBurstRemaining = std::max<size_t>(1, 3 * baselinePeriod);
        }
        if (!eventActive) driftDiagnosticEventActive = false;
        else driftDiagnosticEventActive = true;
        const bool inBurst = driftDiagnosticBurstRemaining > 0;
        if (inBurst) --driftDiagnosticBurstRemaining;
        driftDiagnosticShouldEmit = inBurst || driftDiagnosticFrameCounter % baselinePeriod == 0;
        previousFiniteSupport = drift.finite;
        if (driftDiagnosticShouldEmit) populateGeometryDiagnostics();

        std::ostringstream diagnostic;
        diagnostic << std::setprecision(17) << "[ESTIMATOR_DRIFT_DIAGNOSTIC]"
                   << " timestamp=" << drift.timestamp << " relative_timestamp=" << drift.relativeTimestamp
                   << " frame=" << drift.frameId
                   << " keyframe=" << drift.isKeyframe << " dt=" << drift.deltaTime
                   << " position_x=" << drift.position.x() << " position_y=" << drift.position.y()
                   << " position_z=" << drift.position.z()
                   << " quaternion_x=" << drift.orientation.x() << " quaternion_y=" << drift.orientation.y()
                   << " quaternion_z=" << drift.orientation.z() << " quaternion_w=" << drift.orientation.w()
                   << " velocity_x=" << drift.velocity.x() << " velocity_y=" << drift.velocity.y()
                   << " velocity_z=" << drift.velocity.z()
                   << " gyro_bias_x=" << drift.gyroBias.x() << " gyro_bias_y=" << drift.gyroBias.y()
                   << " gyro_bias_z=" << drift.gyroBias.z()
                   << " accel_bias_x=" << drift.accelBias.x() << " accel_bias_y=" << drift.accelBias.y()
                   << " accel_bias_z=" << drift.accelBias.z()
                   << " frame_translation=" << drift.frameTranslation
                   << " frame_rotation_deg=" << drift.frameRotationDegrees
                   << " opt_translation=" << drift.optimizationTranslation
                   << " opt_rotation_deg=" << drift.optimizationRotationDegrees
                   << " opt_velocity=" << drift.optimizationVelocity
                   << " opt_gyro_bias=" << drift.optimizationGyroBias
                   << " opt_accel_bias=" << drift.optimizationAccelBias
                   << " support_keypoints=" << drift.keypoints
                   << " support_finite=" << drift.finite
                   << " support_pending=" << drift.pending
                   << " support_uninitialized=" << drift.uninitialized
                   << " support_invalid_initialized=" << drift.invalidInitialized
                   << " support_unassociated=" << drift.unassociated
                   << " camera_keypoints=" << commaSeparated(drift.cameraKeypoints)
                   << " camera_finite=" << commaSeparated(drift.cameraFinite)
                   << " camera_pending=" << commaSeparated(drift.cameraPending)
                   << " camera_uninitialized=" << commaSeparated(drift.cameraUninitialized)
                   << " camera_invalid_initialized=" << commaSeparated(drift.cameraInvalidInitialized)
                   << " camera_unassociated=" << commaSeparated(drift.cameraUnassociated)
                   << " camera_coverage_cells=" << commaSeparated(drift.cameraFiniteCoverageCells)
                   << " geometry_current_camera_range_valid_count="
                   << commaSeparated(drift.geometryCurrentCameraRangeValidCount)
                   << " geometry_current_camera_range_p10="
                   << commaSeparatedDoubles(drift.geometryCurrentCameraRangeP10)
                   << " geometry_current_camera_range_p50="
                   << commaSeparatedDoubles(drift.geometryCurrentCameraRangeP50)
                   << " geometry_current_camera_range_p90="
                   << commaSeparatedDoubles(drift.geometryCurrentCameraRangeP90)
                   << " geometry_current_camera_depth_valid_count="
                   << commaSeparated(drift.geometryCurrentCameraDepthValidCount)
                   << " geometry_current_camera_depth_p10="
                   << commaSeparatedDoubles(drift.geometryCurrentCameraDepthP10)
                   << " geometry_current_camera_depth_p50="
                   << commaSeparatedDoubles(drift.geometryCurrentCameraDepthP50)
                   << " geometry_current_camera_depth_p90="
                   << commaSeparatedDoubles(drift.geometryCurrentCameraDepthP90)
                   << " geometry_active_observation_count_valid="
                   << drift.geometryActiveObservationCountValid
                   << " geometry_active_observation_count_p10="
                   << drift.geometryActiveObservationCountP10
                   << " geometry_active_observation_count_p50="
                   << drift.geometryActiveObservationCountP50
                   << " geometry_active_observation_count_p90="
                   << drift.geometryActiveObservationCountP90
                   << " geometry_active_frame_span_valid=" << drift.geometryActiveFrameSpanValid
                   << " geometry_active_frame_span_s_p10=" << drift.geometryActiveFrameSpanP10
                   << " geometry_active_frame_span_s_p50=" << drift.geometryActiveFrameSpanP50
                   << " geometry_active_frame_span_s_p90=" << drift.geometryActiveFrameSpanP90
                   << " geometry_multiview_parallax_valid=" << drift.geometryParallaxValid
                   << " geometry_multiview_parallax_deg_p10=" << drift.geometryParallaxP10
                   << " geometry_multiview_parallax_deg_p50=" << drift.geometryParallaxP50
                   << " geometry_multiview_parallax_deg_p90=" << drift.geometryParallaxP90
                   << " geometry_translation_info_surrogate_observation_count="
                   << drift.geometryTranslationInfoObservationCount
                   << " geometry_translation_info_surrogate_eigen_min="
                   << drift.geometryTranslationInfoEigenMin
                   << " geometry_translation_info_surrogate_eigen_median="
                   << drift.geometryTranslationInfoEigenMedian
                   << " geometry_translation_info_surrogate_eigen_max="
                   << drift.geometryTranslationInfoEigenMax
                   << " geometry_translation_info_surrogate_effective_rank="
                   << drift.geometryTranslationInfoEffectiveRank
                   << " geometry_translation_info_surrogate_condition_number="
                   << drift.geometryTranslationInfoConditionNumber
                   << " finite_coverage_cells=" << drift.finiteCoverageCells
                   << " finite_coverage_possible=" << drift.finiteCoverageCellsPossible
                   << " estimator_reproj_pre_p50=" << drift.reprojectionPreP50
                   << " estimator_reproj_pre_p90=" << drift.reprojectionPreP90
                   << " estimator_reproj_pre_count=" << drift.reprojectionPreCount
                   << " estimator_reproj_pre_invalid=" << drift.reprojectionPreInvalid
                   << " estimator_reproj_post_p50=" << drift.reprojectionPostP50
                   << " estimator_reproj_post_p90=" << drift.reprojectionPostP90
                   << " estimator_reproj_post_max=" << drift.reprojectionPostMax
                   << " estimator_reproj_post_over4=" << drift.reprojectionPostFractionOver4
                   << " estimator_reproj_post_over9=" << drift.reprojectionPostFractionOver9
                   << " estimator_reproj_post_count=" << drift.reprojectionPostCount
                   << " estimator_reproj_post_invalid=" << drift.reprojectionPostInvalid
                   << " window_frames=" << drift.activeFrames
                   << " window_keyframes=" << drift.activeKeyframes
                   << " pre_marg_frames=" << drift.preMarginalizationFrames
                   << " pre_marg_landmarks=" << drift.preMarginalizationLandmarks
                   << " marginalized_landmarks=" << drift.marginalizedLandmarks
                   << " map_landmarks=" << drift.activeLandmarks
                   << " map_initialized=" << drift.activeInitializedLandmarks
                   << " opt_initial_cost=" << drift.optimizationInitialCost
                   << " opt_final_cost=" << drift.optimizationFinalCost
                   << " opt_iterations=" << drift.optimizationIterations
                   << " opt_successful_steps=" << drift.optimizationSuccessfulSteps
                   << " opt_unsuccessful_steps=" << drift.optimizationUnsuccessfulSteps
                   << " opt_termination=" << drift.optimizationTerminationType
                   << " opt_usable=" << drift.optimizationUsable
                   << " opt_time_s=" << drift.optimizationTimeSeconds;
        driftDiagnosticLine = diagnostic.str();
      }

      optimizationDone_ = true;
    }  // unlock mutex
    if (driftDiagnosticShouldEmit && !driftDiagnosticLine.empty() && visualizationDataPtr) {
      LOG(INFO) << driftDiagnosticLine;
    }
    optimizationNotification_.notify_all();
    featureStateNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState) {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.emplace_back(*parameters_.nCameraSystem.T_SC(i));
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not access estimator
    if ((parameters_.visualization.displayImages || debugImgCallback_) && visualizationDataPtr) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    afterOptimizationTimer.stop();
  }
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop() {
  for (;;) {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false) return;

    // call all user callbacks
    if (stateCallback_ && !result.onlyPublishLandmarks) stateCallback_(result.stamp, result.T_WS);

    // Modified by Sharmin
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
      fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases, result.omega_S);

    if (fullStateCallbackWithExtrinsics_ && !result.onlyPublishLandmarks)
      fullStateCallbackWithExtrinsics_(
          result.stamp, result.T_WS, result.speedAndBiases, result.omega_S, result.vector_of_T_SCi);
    if (landmarksCallback_ && !result.landmarksVector.empty())
      landmarksCallback_(result.stamp,
                         result.landmarksVector,
                         result.transferredLandmarks);  // TODO(gohlp): why two maps?
  }
}

}  // namespace okvis
