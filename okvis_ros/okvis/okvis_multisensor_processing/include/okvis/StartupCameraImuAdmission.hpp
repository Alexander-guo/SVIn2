/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the above copyright notice,
 *  this list of conditions and the following disclaimer are retained.
 *********************************************************************************/

/**
 * @file StartupCameraImuAdmission.hpp
 * @brief Deterministic startup camera/IMU boundary and measurement selection.
 */

#ifndef INCLUDE_OKVIS_STARTUP_CAMERA_IMU_ADMISSION_HPP_
#define INCLUDE_OKVIS_STARTUP_CAMERA_IMU_ADMISSION_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <okvis/Measurements.hpp>
#include <okvis/Time.hpp>

namespace okvis {
namespace startup_camera_imu {

/**
 * @brief Bounded startup boundary derived from the first camera timestamp.
 *
 * The first camera callback can race the first IMU callbacks when a replay is
 * started.  The gate gives the IMU stream a fixed number of camera periods to
 * establish history.  Camera admission itself remains timestamp based: a
 * candidate is accepted only when the selected IMU window reaches the camera
 * timestamp, rather than because a callback happened to arrive first.
 */
class AdmissionGate {
 public:
  static constexpr size_t kWarmupCameraPeriods = 5;

  AdmissionGate(int cameraRate, const okvis::Duration& temporalOverlap)
      : cameraRate_(cameraRate > 0 ? cameraRate : 1), temporalOverlap_(temporalOverlap) {}

  void initialize(const okvis::Time& firstCameraTimestamp) {
    firstCameraTimestamp_ = firstCameraTimestamp;
    const double warmupSeconds = static_cast<double>(kWarmupCameraPeriods) / static_cast<double>(cameraRate_);
    firstAcceptedTimestamp_ = firstCameraTimestamp_;
    initializationBegin_ = firstCameraTimestamp_;
    imuReadyUntil_ = firstCameraTimestamp_ + okvis::Duration(warmupSeconds) + temporalOverlap_;
    initialized_ = true;
  }

  bool initialized() const { return initialized_; }

  bool beforeFirstAccepted(const okvis::Time& cameraTimestamp) const {
    return initialized_ && cameraTimestamp < firstAcceptedTimestamp_;
  }

  const okvis::Time& firstCameraTimestamp() const { return firstCameraTimestamp_; }
  const okvis::Time& firstAcceptedTimestamp() const { return firstAcceptedTimestamp_; }
  const okvis::Time& initializationBegin() const { return initializationBegin_; }
  const okvis::Time& imuReadyUntil() const { return imuReadyUntil_; }

 private:
  int cameraRate_;
  okvis::Duration temporalOverlap_;
  bool initialized_ = false;
  okvis::Time firstCameraTimestamp_;
  okvis::Time firstAcceptedTimestamp_;
  okvis::Time initializationBegin_;
  okvis::Time imuReadyUntil_;
};

/**
 * @brief Select the spanning IMU window using the same inclusive boundaries
 * as ThreadedKFVio::getImuMeasurments().
 */
inline okvis::ImuMeasurementDeque selectWindow(const okvis::ImuMeasurementDeque& measurements,
                                               const okvis::Time& begin,
                                               const okvis::Time& end) {
  // ImuError::propagation() interpolates from the sample at or before begin
  // and therefore requires both sides of a real requested interval to be
  // present.  Returning an unbracketed front sample would make the first
  // integration step an extrapolation whose result depends on cleanup timing.
  // A zero begin is the existing matching-loop sentinel used before the first
  // estimator state exists; that path does not call ImuError::propagation() and
  // must retain its historical front-to-end initialization window.
  const bool initializationSentinel = begin.isZero();
  if (measurements.empty() || end < begin ||
      (!initializationSentinel && measurements.front().timeStamp > begin) ||
      measurements.back().timeStamp < end) {
    return okvis::ImuMeasurementDeque();
  }

  okvis::ImuMeasurementDeque::const_iterator first = measurements.begin();
  okvis::ImuMeasurementDeque::const_iterator last = measurements.end();
  for (okvis::ImuMeasurementDeque::const_iterator it = measurements.begin(); it != measurements.end(); ++it) {
    if (it->timeStamp <= begin) first = it;
    if (it->timeStamp >= end) {
      last = it;
      ++last;
      break;
    }
  }
  return okvis::ImuMeasurementDeque(first, last);
}

/**
 * @brief Erase old history while retaining the predecessor needed for the
 *        next timestamp-defined propagation interval.
 *
 * The propagation selector includes the newest sample at or before its begin
 * time so ImuError can interpolate the state at that begin time.  Cleanup
 * must retain that predecessor; keeping only the first sample at or after the
 * cleanup boundary makes the selected window depend on cleanup scheduling.
 */
inline size_t eraseBeforeKeepingPredecessor(okvis::ImuMeasurementDeque& measurements,
                                            const okvis::Time& eraseUntil) {
  if (measurements.empty()) return 0;

  okvis::ImuMeasurementDeque::iterator predecessor = measurements.begin();
  for (okvis::ImuMeasurementDeque::iterator it = measurements.begin(); it != measurements.end(); ++it) {
    if (it->timeStamp > eraseUntil) break;
    predecessor = it;
  }

  if (predecessor == measurements.begin()) return 0;
  const size_t removed = static_cast<size_t>(std::distance(measurements.begin(), predecessor));
  measurements.erase(measurements.begin(), predecessor);
  return removed;
}

inline uint64_t doubleBits(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

/**
 * @brief Bit-exact FNV-1a content hash for a selected IMU window.
 *
 * The timestamp sec/nsec fields and all gyro/accelerometer double bit patterns
 * are included in deque order.
 */
inline uint64_t contentHash(const okvis::ImuMeasurementDeque& measurements) {
  uint64_t hash = 1469598103934665603ULL;
  const auto addByte = [&hash](uint8_t value) {
    hash ^= static_cast<uint64_t>(value);
    hash *= 1099511628211ULL;
  };
  const auto addU64 = [&addByte](uint64_t value) {
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
      addByte(static_cast<uint8_t>(value >> (8U * byte)));
    }
  };

  addU64(0x5356494d55574831ULL);  // "SVIMUWH1", version/tag.
  addU64(static_cast<uint64_t>(measurements.size()));
  for (const okvis::ImuMeasurement& measurement : measurements) {
    addU64(static_cast<uint64_t>(measurement.timeStamp.sec));
    addU64(static_cast<uint64_t>(measurement.timeStamp.nsec));
    for (int axis = 0; axis < 3; ++axis) addU64(doubleBits(measurement.measurement.gyroscopes[axis]));
    for (int axis = 0; axis < 3; ++axis) addU64(doubleBits(measurement.measurement.accelerometers[axis]));
  }
  return hash;
}

}  // namespace startup_camera_imu
}  // namespace okvis

#endif  // INCLUDE_OKVIS_STARTUP_CAMERA_IMU_ADMISSION_HPP_
