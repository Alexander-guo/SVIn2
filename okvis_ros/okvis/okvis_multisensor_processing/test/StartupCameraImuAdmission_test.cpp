#include <gtest/gtest.h>

#include <okvis/StartupCameraImuAdmission.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <vector>

namespace {

okvis::ImuMeasurement makeMeasurement(const okvis::Time& timestamp, double value) {
  okvis::ImuMeasurement measurement;
  measurement.timeStamp = timestamp;
  measurement.measurement.gyroscopes = Eigen::Vector3d(value, value + 1.0, value + 2.0);
  measurement.measurement.accelerometers = Eigen::Vector3d(value + 3.0, value + 4.0, value + 5.0);
  return measurement;
}

okvis::ImuMeasurementDeque makePropagationHistory() {
  okvis::ImuMeasurementDeque history;
  for (uint32_t nanoseconds = 131766000; nanoseconds <= 205766000; nanoseconds += 1000000) {
    history.push_back(makeMeasurement(okvis::Time(100, nanoseconds), nanoseconds * 1e-9));
  }
  return history;
}

okvis::ImuParameters makePropagationParameters() {
  okvis::ImuParameters parameters{};
  parameters.a_max = 1000.0;
  parameters.g_max = 1000.0;
  parameters.sigma_g_c = 0.01;
  parameters.sigma_a_c = 0.01;
  parameters.sigma_gw_c = 0.001;
  parameters.sigma_aw_c = 0.001;
  parameters.g = 9.81;
  parameters.a0.setZero();
  return parameters;
}

// Test-only reconstruction of the pre-fix inclusive selector.  It is kept
// here to make the regression's old callback-dependent result explicit.
okvis::ImuMeasurementDeque legacySelectWindow(const okvis::ImuMeasurementDeque& measurements,
                                              const okvis::Time& begin,
                                              const okvis::Time& end) {
  if (measurements.empty() || end < begin) return okvis::ImuMeasurementDeque();
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

TEST(StartupCameraImuAdmissionTest, EqualityBoundarySelectsTheFirstEligibleCameraTimestamp) {
  okvis::startup_camera_imu::AdmissionGate gate(30, okvis::Duration(0.02));
  const okvis::Time firstCamera(100, 18577391);
  gate.initialize(firstCamera);

  EXPECT_EQ(gate.firstAcceptedTimestamp(), firstCamera);
  EXPECT_TRUE(gate.beforeFirstAccepted(gate.firstAcceptedTimestamp() - okvis::Duration(1e-9)));
  EXPECT_FALSE(gate.beforeFirstAccepted(gate.firstAcceptedTimestamp()));
  EXPECT_EQ(gate.initializationBegin(), firstCamera);
  EXPECT_EQ(gate.imuReadyUntil(), firstCamera + okvis::Duration(5.0 / 30.0 + 0.02));
}

TEST(StartupCameraImuAdmissionTest, CameraCallbackOrderDoesNotChangeTimestampBoundary) {
  const okvis::Time firstCamera(100, 18577391);
  const std::vector<okvis::Time> cameraTimestamps = {
      firstCamera,
      firstCamera + okvis::Duration(1.0 / 30.0),
      firstCamera + okvis::Duration(2.0 / 30.0),
      firstCamera + okvis::Duration(3.0 / 30.0),
      firstCamera + okvis::Duration(4.0 / 30.0),
      firstCamera + okvis::Duration(5.0 / 30.0),
      firstCamera + okvis::Duration(6.0 / 30.0),
  };
  okvis::startup_camera_imu::AdmissionGate forward(30, okvis::Duration(0.02));
  okvis::startup_camera_imu::AdmissionGate reverse(30, okvis::Duration(0.02));
  forward.initialize(cameraTimestamps.front());
  reverse.initialize(cameraTimestamps.front());

  EXPECT_EQ(forward.firstAcceptedTimestamp(), reverse.firstAcceptedTimestamp());
  EXPECT_EQ(forward.initializationBegin(), reverse.initializationBegin());
  EXPECT_EQ(forward.imuReadyUntil(), reverse.imuReadyUntil());
  for (size_t index = 0; index < cameraTimestamps.size(); ++index) {
    EXPECT_EQ(forward.beforeFirstAccepted(cameraTimestamps[index]),
              reverse.beforeFirstAccepted(cameraTimestamps[index]));
  }
}

TEST(StartupCameraImuAdmissionTest, LowerBoundMakesDifferentCallbackStartHistoriesSelectSameWindow) {
  const okvis::Time firstCamera(100, 18577391);
  okvis::startup_camera_imu::AdmissionGate gate(30, okvis::Duration(0.02));
  gate.initialize(firstCamera);

  okvis::ImuMeasurementDeque historyWithEarlierCallbacks;
  okvis::ImuMeasurementDeque historyStartingLater;
  for (uint32_t nanoseconds = 16000000; nanoseconds <= 60000000; nanoseconds += 1000000) {
    const okvis::Time timestamp(100, nanoseconds);
    const okvis::ImuMeasurement measurement = makeMeasurement(timestamp, nanoseconds * 1e-9);
    historyWithEarlierCallbacks.push_back(measurement);
    if (nanoseconds >= 18000000) historyStartingLater.push_back(measurement);
  }

  const okvis::ImuMeasurementDeque selectedA = okvis::startup_camera_imu::selectWindow(
      historyWithEarlierCallbacks, gate.initializationBegin(), gate.firstAcceptedTimestamp() + okvis::Duration(0.02));
  const okvis::ImuMeasurementDeque selectedB = okvis::startup_camera_imu::selectWindow(
      historyStartingLater, gate.initializationBegin(), gate.firstAcceptedTimestamp() + okvis::Duration(0.02));

  ASSERT_FALSE(selectedA.empty());
  ASSERT_FALSE(selectedB.empty());
  EXPECT_EQ(selectedA.size(), selectedB.size());
  EXPECT_EQ(selectedA.front().timeStamp, selectedB.front().timeStamp);
  EXPECT_EQ(selectedA.back().timeStamp, selectedB.back().timeStamp);
  EXPECT_EQ(okvis::startup_camera_imu::contentHash(selectedA), okvis::startup_camera_imu::contentHash(selectedB));
}

TEST(StartupCameraImuAdmissionTest, IMUWindowIncludesExactBeginAndEndSamples) {
  const okvis::Time begin(200, 100000000);
  const okvis::Time end(200, 200000000);
  okvis::ImuMeasurementDeque measurements;
  measurements.push_back(makeMeasurement(okvis::Time(200, 100000000), 1.0));
  measurements.push_back(makeMeasurement(okvis::Time(200, 150000000), 2.0));
  measurements.push_back(makeMeasurement(okvis::Time(200, 200000000), 3.0));

  const okvis::ImuMeasurementDeque selected = okvis::startup_camera_imu::selectWindow(measurements, begin, end);
  ASSERT_EQ(selected.size(), 3u);
  EXPECT_EQ(selected.front().timeStamp, begin);
  EXPECT_EQ(selected.back().timeStamp, end);
}

TEST(StartupCameraImuAdmissionTest, ZeroBeginSentinelRetainsInitializationWindow) {
  okvis::ImuMeasurementDeque measurements;
  measurements.push_back(makeMeasurement(okvis::Time(200, 100000000), 1.0));
  measurements.push_back(makeMeasurement(okvis::Time(200, 200000000), 2.0));

  // The matching loop uses begin=0 only before the first estimator state and
  // passes this front-to-end window to initialization, not propagation.
  const okvis::ImuMeasurementDeque selected = okvis::startup_camera_imu::selectWindow(
      measurements, okvis::Time(0.0), okvis::Time(200, 200000000));
  ASSERT_EQ(selected.size(), measurements.size());
  EXPECT_EQ(selected.front().timeStamp, measurements.front().timeStamp);
  EXPECT_EQ(selected.back().timeStamp, measurements.back().timeStamp);
}

TEST(StartupCameraImuAdmissionTest, RejectsWindowWithoutInterpolationPredecessor) {
  const okvis::Time begin(100, 132044057);
  const okvis::Time end(100, 205410724);
  okvis::ImuMeasurementDeque history;
  history.push_back(makeMeasurement(okvis::Time(100, 133765999), 1.0));
  history.push_back(makeMeasurement(okvis::Time(100, 150765999), 2.0));
  history.push_back(makeMeasurement(okvis::Time(100, 205598999), 3.0));

  // Before this fix, selectWindow() returned the front sample here even
  // though it was newer than begin.  ImuError::propagation() then received
  // an unbracketed interval and its first interpolation depended on cleanup.
  EXPECT_TRUE(okvis::startup_camera_imu::selectWindow(history, begin, end).empty());
}

TEST(StartupCameraImuAdmissionTest, CleanupKeepsTheTimestampPredecessor) {
  const okvis::Time begin(100, 132044057);
  okvis::ImuMeasurementDeque history;
  history.push_back(makeMeasurement(okvis::Time(100, 120766000), 1.0));
  history.push_back(makeMeasurement(okvis::Time(100, 131766000), 2.0));
  history.push_back(makeMeasurement(okvis::Time(100, 133765999), 3.0));

  EXPECT_EQ(okvis::startup_camera_imu::eraseBeforeKeepingPredecessor(history, begin), 1u);
  ASSERT_FALSE(history.empty());
  EXPECT_EQ(history.front().timeStamp, okvis::Time(100, 131766000));
  EXPECT_FALSE(okvis::startup_camera_imu::selectWindow(history, begin, okvis::Time(100, 133765999)).empty());
}

TEST(StartupCameraImuAdmissionTest, PhysicalPropagationStartIgnoresVariableOverlapPrefix) {
  const okvis::Time requestedBegin(100, 132044057);
  const okvis::Time physicalPropagationBegin(100, 152044057);
  const okvis::Time end(100, 205410724);
  const okvis::ImuMeasurementDeque historyWithEarlyPrefix = makePropagationHistory();
  okvis::ImuMeasurementDeque historyWithLatePrefix = historyWithEarlyPrefix;
  historyWithLatePrefix.pop_front();
  historyWithLatePrefix.pop_front();

  // Selecting from the overlap lower bound reproduces the pre-fix
  // callback-history dependency, while selecting from the actual ImuError
  // start timestamp chooses the same interpolation predecessor.
  const okvis::ImuMeasurementDeque oldStyleEarly = legacySelectWindow(historyWithEarlyPrefix, requestedBegin, end);
  const okvis::ImuMeasurementDeque oldStyleLate = legacySelectWindow(historyWithLatePrefix, requestedBegin, end);
  ASSERT_FALSE(oldStyleEarly.empty());
  ASSERT_FALSE(oldStyleLate.empty());
  EXPECT_NE(okvis::startup_camera_imu::contentHash(oldStyleEarly),
            okvis::startup_camera_imu::contentHash(oldStyleLate));

  const okvis::ImuMeasurementDeque selectedEarly = okvis::startup_camera_imu::selectWindow(
      historyWithEarlyPrefix, physicalPropagationBegin, end);
  const okvis::ImuMeasurementDeque selectedLate = okvis::startup_camera_imu::selectWindow(
      historyWithLatePrefix, physicalPropagationBegin, end);
  ASSERT_FALSE(selectedEarly.empty());
  ASSERT_FALSE(selectedLate.empty());
  EXPECT_EQ(okvis::startup_camera_imu::contentHash(selectedEarly),
            okvis::startup_camera_imu::contentHash(selectedLate));
  EXPECT_EQ(selectedEarly.front().timeStamp, okvis::Time(100, 151766000));
  EXPECT_EQ(selectedLate.front().timeStamp, selectedEarly.front().timeStamp);

  const okvis::ImuParameters imuParameters = makePropagationParameters();
  okvis::kinematics::Transformation poseEarly;
  okvis::kinematics::Transformation poseLate;
  poseEarly.setIdentity();
  poseLate.setIdentity();
  okvis::SpeedAndBias speedEarly = okvis::SpeedAndBias::Zero();
  okvis::SpeedAndBias speedLate = okvis::SpeedAndBias::Zero();
  const int stepsEarly = okvis::ceres::ImuError::propagation(
      selectedEarly, imuParameters, poseEarly, speedEarly, physicalPropagationBegin, end);
  const int stepsLate = okvis::ceres::ImuError::propagation(
      selectedLate, imuParameters, poseLate, speedLate, physicalPropagationBegin, end);
  EXPECT_EQ(stepsEarly, stepsLate);
  for (int index = 0; index < 4; ++index) EXPECT_DOUBLE_EQ(poseEarly.q().coeffs()[index], poseLate.q().coeffs()[index]);
  for (int index = 0; index < 3; ++index) EXPECT_DOUBLE_EQ(poseEarly.r()[index], poseLate.r()[index]);
  for (int index = 0; index < speedEarly.size(); ++index) EXPECT_DOUBLE_EQ(speedEarly[index], speedLate[index]);
}

TEST(StartupCameraImuAdmissionTest, ExtraHistoryDoesNotChangeSelectedPropagationOrOutput) {
  const okvis::Time begin(100, 132044057);
  const okvis::Time end(100, 205410724);
  const okvis::ImuMeasurementDeque baseHistory = makePropagationHistory();
  okvis::ImuMeasurementDeque historyWithExtraLeadingAndTrailing = baseHistory;
  historyWithExtraLeadingAndTrailing.push_front(makeMeasurement(okvis::Time(100, 100766000), -1.0));
  historyWithExtraLeadingAndTrailing.push_back(makeMeasurement(okvis::Time(100, 220766000), 100.0));

  const okvis::ImuMeasurementDeque selectedBase =
      okvis::startup_camera_imu::selectWindow(baseHistory, begin, end);
  const okvis::ImuMeasurementDeque selectedWithExtra =
      okvis::startup_camera_imu::selectWindow(historyWithExtraLeadingAndTrailing, begin, end);
  ASSERT_FALSE(selectedBase.empty());
  ASSERT_FALSE(selectedWithExtra.empty());
  EXPECT_EQ(okvis::startup_camera_imu::contentHash(selectedBase),
            okvis::startup_camera_imu::contentHash(selectedWithExtra));
  EXPECT_EQ(selectedBase.size(), selectedWithExtra.size());

  const okvis::ImuParameters imuParameters = makePropagationParameters();
  okvis::kinematics::Transformation poseBase;
  okvis::kinematics::Transformation poseWithExtra;
  poseBase.setIdentity();
  poseWithExtra.setIdentity();
  okvis::SpeedAndBias speedBase = okvis::SpeedAndBias::Zero();
  okvis::SpeedAndBias speedWithExtra = okvis::SpeedAndBias::Zero();
  Eigen::Matrix<double, 15, 15> covarianceBase = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> covarianceWithExtra = Eigen::Matrix<double, 15, 15>::Zero();

  const int stepsBase = okvis::ceres::ImuError::propagation(
      selectedBase, imuParameters, poseBase, speedBase, begin, end, &covarianceBase, nullptr);
  const int stepsWithExtra = okvis::ceres::ImuError::propagation(
      selectedWithExtra, imuParameters, poseWithExtra, speedWithExtra, begin, end, &covarianceWithExtra, nullptr);
  EXPECT_EQ(stepsBase, stepsWithExtra);
  for (int index = 0; index < 4; ++index) EXPECT_DOUBLE_EQ(poseBase.q().coeffs()[index], poseWithExtra.q().coeffs()[index]);
  for (int index = 0; index < 3; ++index) EXPECT_DOUBLE_EQ(poseBase.r()[index], poseWithExtra.r()[index]);
  for (int index = 0; index < speedBase.size(); ++index) EXPECT_DOUBLE_EQ(speedBase[index], speedWithExtra[index]);
  for (int row = 0; row < covarianceBase.rows(); ++row) {
    for (int column = 0; column < covarianceBase.cols(); ++column) {
      EXPECT_DOUBLE_EQ(covarianceBase(row, column), covarianceWithExtra(row, column));
    }
  }
}

}  // namespace
