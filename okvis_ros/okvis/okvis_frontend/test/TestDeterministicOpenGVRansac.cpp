#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include <okvis/Estimator.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

namespace {

using Camera = okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>;
using RotationProblem = opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem;
using RelativeProblem = opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem;

class TracingRotationProblem : public RotationProblem {
 public:
  using RotationProblem::adapter_t;

  TracingRotationProblem(adapter_t& adapter, bool randomSeed)
      : RotationProblem(adapter, randomSeed) {}

  void getSamples(int& iterations, std::vector<int>& samples) override {
    RotationProblem::getSamples(iterations, samples);
    sampledSequence.push_back(samples);
  }

  std::vector<std::vector<int>> sampledSequence;
};

class TracingRelativeProblem : public RelativeProblem {
 public:
  using RelativeProblem::adapter_t;

  TracingRelativeProblem(adapter_t& adapter, algorithm_t algorithm, bool randomSeed)
      : RelativeProblem(adapter, algorithm, randomSeed) {}

  void getSamples(int& iterations, std::vector<int>& samples) override {
    RelativeProblem::getSamples(iterations, samples);
    sampledSequence.push_back(samples);
  }

  std::vector<std::vector<int>> sampledSequence;
};

void appendBytes(std::vector<uint8_t>& output, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  output.insert(output.end(), bytes, bytes + size);
}

void appendUInt64(std::vector<uint8_t>& output, uint64_t value) {
  appendBytes(output, &value, sizeof(value));
}

void appendIntVector(std::vector<uint8_t>& output, const std::vector<int>& values) {
  appendUInt64(output, static_cast<uint64_t>(values.size()));
  if (!values.empty()) appendBytes(output, values.data(), values.size() * sizeof(values.front()));
}

void appendSampleSequence(std::vector<uint8_t>& output,
                          const std::vector<std::vector<int>>& sequence) {
  appendUInt64(output, static_cast<uint64_t>(sequence.size()));
  for (const std::vector<int>& sample : sequence) appendIntVector(output, sample);
}

template <typename Problem>
std::vector<uint8_t> samplingBytes(const Problem& problem) {
  std::vector<uint8_t> output;
  appendSampleSequence(output, problem.sampledSequence);
  return output;
}

template <typename Problem>
std::vector<uint8_t> modelAndInlierBytes(const opengv::sac::Ransac<Problem>& ransac) {
  std::vector<uint8_t> output;
  appendUInt64(output, static_cast<uint64_t>(ransac.iterations_));
  appendIntVector(output, ransac.model_);
  appendIntVector(output, ransac.inliers_);
  appendBytes(output, ransac.model_coefficients_.data(),
              static_cast<size_t>(ransac.model_coefficients_.size()) * sizeof(double));
  return output;
}

struct RansacTrace {
  std::vector<uint8_t> rotationSampling;
  std::vector<uint8_t> rotationModelAndInliers;
  std::vector<uint8_t> relativeSampling;
  std::vector<uint8_t> relativeModelAndInliers;
  bool selectedRotationOnly = false;
  size_t rotationInlierCount = 0;
  size_t relativeInlierCount = 0;
};

class SyntheticFramePair {
 public:
  explicit SyntheticFramePair(double currentImageOffsetX = 0.0)
      : camera_(Camera::createTestObject()), startTime_(1000.0) {
    const std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
        new okvis::kinematics::Transformation());
    cameraSystem_.addCamera(T_SC, camera_, okvis::cameras::NCameraSystem::DistortionType::Equidistant);

    okvis::ExtrinsicsEstimationParameters extrinsics;
    extrinsics.sigma_absolute_translation = 0.0;
    extrinsics.sigma_absolute_orientation = 0.0;
    extrinsics.sigma_c_relative_translation = 0.0;
    extrinsics.sigma_c_relative_orientation = 0.0;
    EXPECT_EQ(estimator_.addCamera(extrinsics), 0);

    imuParameters_.a0.setZero();
    imuParameters_.g = 9.81;
    imuParameters_.a_max = 1000.0;
    imuParameters_.g_max = 1000.0;
    imuParameters_.rate = 100.0;
    imuParameters_.sigma_g_c = 6.0e-4;
    imuParameters_.sigma_a_c = 2.0e-3;
    imuParameters_.sigma_gw_c = 3.0e-6;
    imuParameters_.sigma_aw_c = 2.0e-5;
    imuParameters_.sigma_bg = 0.03;
    imuParameters_.sigma_ba = 0.1;
    imuParameters_.tau = 3600.0;
    EXPECT_EQ(estimator_.addImu(imuParameters_), 0);

    for (size_t i = 0; i <= 100; ++i) {
      const double dt = 0.01 * static_cast<double>(i);
      imuMeasurements_.emplace_back(
          startTime_ + okvis::Duration(dt),
          okvis::ImuSensorReadings(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, imuParameters_.g)));
    }

    std::vector<Eigen::Vector2d> measurements;
    std::vector<Eigen::Vector4d> points;
    for (size_t row = 0; row < 4; ++row) {
      for (size_t column = 0; column < 6; ++column) {
        const double x = -0.30 + 0.12 * static_cast<double>(column);
        const double y = -0.20 + 0.13 * static_cast<double>(row);
        const Eigen::Vector3d point(x, y, 1.0);
        Eigen::Vector2d measurement;
        EXPECT_EQ(camera_->project(point, &measurement),
                  okvis::cameras::CameraBase::ProjectionStatus::Successful);
        measurements.push_back(measurement);
        points.emplace_back(x, y, 1.0, 1.0);
      }
    }

    std::vector<cv::KeyPoint> keypointsA;
    std::vector<cv::KeyPoint> keypointsB;
    keypointsA.reserve(measurements.size());
    keypointsB.reserve(measurements.size());
    for (const Eigen::Vector2d& measurement : measurements) {
      keypointsA.emplace_back(static_cast<float>(measurement.x()), static_cast<float>(measurement.y()), 12.0f);
      keypointsB.emplace_back(static_cast<float>(measurement.x() + currentImageOffsetX),
                              static_cast<float>(measurement.y()), 12.0f);
    }

    frameA_.reset(new okvis::MultiFrame(
        cameraSystem_, startTime_, okvis::IdProvider::instance().newId()));
    frameB_.reset(new okvis::MultiFrame(
        cameraSystem_, startTime_ + okvis::Duration(0.1), okvis::IdProvider::instance().newId()));
    EXPECT_TRUE(frameA_->resetKeypoints(0, keypointsA));
    EXPECT_TRUE(frameB_->resetKeypoints(0, keypointsB));
    EXPECT_TRUE(estimator_.addStates(frameA_, imuMeasurements_, true));
    EXPECT_TRUE(estimator_.addStates(frameB_, imuMeasurements_, true));

    for (size_t index = 0; index < points.size(); ++index) {
      const uint64_t landmarkId = okvis::IdProvider::instance().newId();
      EXPECT_TRUE(estimator_.addLandmark(landmarkId, points[index]));
      estimator_.setLandmarkInitialized(landmarkId, true);
      frameA_->setLandmarkId(0, index, landmarkId);
      frameB_->setLandmarkId(0, index, landmarkId);
      EXPECT_NE(estimator_.addObservation<Camera>(landmarkId, frameA_->id(), 0, index), nullptr);
      EXPECT_NE(estimator_.addObservation<Camera>(landmarkId, frameB_->id(), 0, index), nullptr);
    }
  }

  std::unique_ptr<opengv::relative_pose::FrameRelativeAdapter> makeAdapter() const {
    return std::unique_ptr<opengv::relative_pose::FrameRelativeAdapter>(
        new opengv::relative_pose::FrameRelativeAdapter(
            estimator_, cameraSystem_, frameA_->id(), 0, frameB_->id(), 0));
  }

 private:
  std::shared_ptr<const okvis::cameras::CameraBase> camera_;
  okvis::cameras::NCameraSystem cameraSystem_;
  okvis::Estimator estimator_;
  okvis::ImuParameters imuParameters_;
  okvis::ImuMeasurementDeque imuMeasurements_;
  okvis::Time startTime_;
  okvis::MultiFramePtr frameA_;
  okvis::MultiFramePtr frameB_;
};

RansacTrace runRansac(const SyntheticFramePair& fixture) {
  std::unique_ptr<opengv::relative_pose::FrameRelativeAdapter> adapter = fixture.makeAdapter();
  if (adapter->getNumberCorrespondences() < 10u) {
    ADD_FAILURE() << "synthetic adapter did not contain enough correspondences";
    return RansacTrace();
  }

  opengv::sac::Ransac<TracingRotationProblem> rotationRansac;
  std::shared_ptr<TracingRotationProblem> rotationProblem(new TracingRotationProblem(*adapter, false));
  rotationRansac.sac_model_ = rotationProblem;
  rotationRansac.threshold_ = 9;
  rotationRansac.max_iterations_ = 50;
  if (!rotationRansac.computeModel(0)) {
    ADD_FAILURE() << "rotation-only OpenGV RANSAC did not produce a model";
    return RansacTrace();
  }

  opengv::sac::Ransac<TracingRelativeProblem> relativeRansac;
  std::shared_ptr<TracingRelativeProblem> relativeProblem(new TracingRelativeProblem(
      *adapter, RelativeProblem::STEWENIUS, false));
  relativeRansac.sac_model_ = relativeProblem;
  relativeRansac.threshold_ = 9;
  relativeRansac.max_iterations_ = 50;
  if (!relativeRansac.computeModel(0)) {
    ADD_FAILURE() << "relative-pose OpenGV RANSAC did not produce a model";
    return RansacTrace();
  }

  RansacTrace trace;
  trace.rotationSampling = samplingBytes(*rotationProblem);
  trace.rotationModelAndInliers = modelAndInlierBytes(rotationRansac);
  trace.relativeSampling = samplingBytes(*relativeProblem);
  trace.relativeModelAndInliers = modelAndInlierBytes(relativeRansac);
  trace.rotationInlierCount = rotationRansac.inliers_.size();
  trace.relativeInlierCount = relativeRansac.inliers_.size();
  const float rotationRatio = static_cast<float>(trace.rotationInlierCount) /
                              static_cast<float>(adapter->getNumberCorrespondences());
  const float relativeRatio = static_cast<float>(trace.relativeInlierCount) /
                              static_cast<float>(adapter->getNumberCorrespondences());
  trace.selectedRotationOnly = rotationRatio > relativeRatio || rotationRatio > 0.8f;
  return trace;
}

TEST(DeterministicOpenGVRansacTest, IdenticalInputsAndIrrelevantConcurrentHistoryAreByteIdentical) {
  const SyntheticFramePair firstFixture;
  const RansacTrace first = runRansac(firstFixture);

  const SyntheticFramePair targetFixture;
  const SyntheticFramePair unrelatedFixture;
  std::atomic<bool> startUnrelated{false};
  std::thread unrelated([&]() {
    while (!startUnrelated.load(std::memory_order_acquire)) std::this_thread::yield();
    for (int iteration = 0; iteration < 3; ++iteration) {
      const RansacTrace ignored = runRansac(unrelatedFixture);
      (void)ignored;
    }
  });
  startUnrelated.store(true, std::memory_order_release);
  const RansacTrace second = runRansac(targetFixture);
  unrelated.join();

  EXPECT_EQ(first.rotationSampling, second.rotationSampling);
  EXPECT_EQ(first.rotationModelAndInliers, second.rotationModelAndInliers);
  EXPECT_EQ(first.relativeSampling, second.relativeSampling);
  EXPECT_EQ(first.relativeModelAndInliers, second.relativeModelAndInliers);
  EXPECT_EQ(first.selectedRotationOnly, second.selectedRotationOnly);
  EXPECT_EQ(first.rotationInlierCount, second.rotationInlierCount);
  EXPECT_EQ(first.relativeInlierCount, second.relativeInlierCount);
}

TEST(DeterministicOpenGVRansacTest, InputContentChangesModelOutcomeWithoutChangingLocalSampleSchedule) {
  const SyntheticFramePair referenceFixture;
  const SyntheticFramePair changedFixture(20.0);
  const RansacTrace reference = runRansac(referenceFixture);
  const RansacTrace changed = runRansac(changedFixture);

  // RANSAC's adaptive stopping rule may make the sampled prefix shorter or
  // longer after the input changes; identical-input runs are covered above.
  EXPECT_TRUE(reference.rotationModelAndInliers != changed.rotationModelAndInliers ||
              reference.relativeModelAndInliers != changed.relativeModelAndInliers ||
              reference.selectedRotationOnly != changed.selectedRotationOnly);
}

}  // namespace
