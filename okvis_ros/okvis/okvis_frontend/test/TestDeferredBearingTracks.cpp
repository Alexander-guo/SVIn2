#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include <okvis/Estimator.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>

namespace {

using Camera = okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>;
using MatchingAlgorithm = okvis::VioKeyframeWindowMatchingAlgorithm<Camera>;

class DeferredBearingTracksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    camera_ = Camera::createTestObject();
    const std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
        new okvis::kinematics::Transformation());
    cameraSystem_.addCamera(T_SC, camera_, okvis::cameras::NCameraSystem::DistortionType::Equidistant);

    okvis::ExtrinsicsEstimationParameters extrinsics;
    extrinsics.sigma_absolute_translation = 0.0;
    extrinsics.sigma_absolute_orientation = 0.0;
    extrinsics.sigma_c_relative_translation = 0.0;
    extrinsics.sigma_c_relative_orientation = 0.0;
    estimator_.addCamera(extrinsics);

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
    estimator_.addImu(imuParameters_);

    startTime_ = okvis::Time::now();
    for (size_t i = 0; i <= 100; ++i) {
      const double dt = 0.01 * static_cast<double>(i);
      imuMeasurements_.emplace_back(
          startTime_ + okvis::Duration(dt),
          okvis::ImuSensorReadings(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, imuParameters_.g)));
    }
  }

  okvis::MultiFramePtr makeFrame(double seconds, const Eigen::Vector2d& trackedKeypoint) {
    okvis::MultiFramePtr frame(new okvis::MultiFrame(
        cameraSystem_, startTime_ + okvis::Duration(seconds), okvis::IdProvider::instance().newId()));
    std::vector<cv::KeyPoint> keypoints;
    keypoints.emplace_back(static_cast<float>(trackedKeypoint.x()),
                           static_cast<float>(trackedKeypoint.y()), 12.0f);
    for (size_t i = 1; i < 12; ++i) {
      keypoints.emplace_back(static_cast<float>(trackedKeypoint.x() + i),
                             static_cast<float>(trackedKeypoint.y() + i), 12.0f);
    }
    EXPECT_TRUE(frame->resetKeypoints(0, keypoints));
    EXPECT_TRUE(estimator_.addStates(frame, imuMeasurements_, true));
    return frame;
  }

  std::shared_ptr<const okvis::cameras::CameraBase> camera_;
  okvis::cameras::NCameraSystem cameraSystem_;
  okvis::Estimator estimator_;
  okvis::ImuParameters imuParameters_;
  okvis::ImuMeasurementDeque imuMeasurements_;
  okvis::Time startTime_;
};

TEST_F(DeferredBearingTracksTest, KeepsWeakMatchOutOfEstimatorAndExposesItToRelativeAdapter) {
  Eigen::Vector2d center;
  ASSERT_EQ(camera_->project(Eigen::Vector3d::UnitZ(), &center),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  const okvis::MultiFramePtr frameA = makeFrame(0.0, center);
  const okvis::MultiFramePtr frameB = makeFrame(0.1, center);

  MatchingAlgorithm matcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  matcher.setFrames(frameA->id(), frameB->id(), 0, 0);
  matcher.doSetup();
  matcher.setBestMatch(0, 0, 0.0);

  const uint64_t trackId = frameA->landmarkId(0, 0);
  ASSERT_NE(trackId, 0u);
  EXPECT_EQ(frameB->landmarkId(0, 0), trackId);
  EXPECT_FALSE(estimator_.isLandmarkAdded(trackId));
  EXPECT_EQ(matcher.numBearingOnlyMatches(), 1u);
  EXPECT_EQ(matcher.numFiniteMatches(), 0u);
  EXPECT_EQ(matcher.numActivePendingTracks(), 1u);

  opengv::relative_pose::FrameRelativeAdapter adapter(
      estimator_, cameraSystem_, frameA->id(), 0, frameB->id(), 0);
  ASSERT_EQ(adapter.getNumberCorrespondences(), 1u);
  EXPECT_TRUE(adapter.getBearingVector1(0).allFinite());
  EXPECT_TRUE(adapter.getBearingVector2(0).allFinite());
}

TEST_F(DeferredBearingTracksTest, PromotesPendingTrackWhenFiniteDepthBecomesObservable) {
  Eigen::Vector2d center;
  ASSERT_EQ(camera_->project(Eigen::Vector3d::UnitZ(), &center),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  const okvis::MultiFramePtr frameA = makeFrame(0.0, center);
  const okvis::MultiFramePtr frameB = makeFrame(0.1, center);

  MatchingAlgorithm weakMatcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  weakMatcher.setFrames(frameA->id(), frameB->id(), 0, 0);
  weakMatcher.doSetup();
  weakMatcher.setBestMatch(0, 0, 0.0);
  const uint64_t trackId = frameA->landmarkId(0, 0);
  ASSERT_NE(trackId, 0u);
  ASSERT_FALSE(estimator_.isLandmarkAdded(trackId));

  Eigen::Vector2d translatedMeasurement;
  ASSERT_EQ(camera_->project(Eigen::Vector3d(-0.1, 0.0, 1.0), &translatedMeasurement),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  const okvis::MultiFramePtr frameC = makeFrame(0.2, translatedMeasurement);
  estimator_.set_T_WS(
      frameC->id(),
      okvis::kinematics::Transformation(Eigen::Vector3d(0.1, 0.0, 0.0), Eigen::Quaterniond::Identity()));

  MatchingAlgorithm finiteMatcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  finiteMatcher.setFrames(frameA->id(), frameC->id(), 0, 0);
  finiteMatcher.doSetup();
  finiteMatcher.setBestMatch(0, 0, 0.0);

  EXPECT_EQ(frameC->landmarkId(0, 0), trackId);
  ASSERT_TRUE(estimator_.isLandmarkAdded(trackId));
  EXPECT_TRUE(estimator_.isLandmarkInitialized(trackId));
  EXPECT_EQ(finiteMatcher.numPromotions(), 1u);
  EXPECT_EQ(finiteMatcher.numFiniteMatches(), 1u);

  okvis::MapPoint landmark;
  ASSERT_TRUE(estimator_.getLandmark(trackId, landmark));
  EXPECT_GE(landmark.observations.size(), 3u);
  ASSERT_GT(std::abs(landmark.point.w()), 1.0e-12);
  EXPECT_NEAR((landmark.point.head<3>() / landmark.point.w() - Eigen::Vector3d(0.0, 0.0, 1.0)).norm(),
              0.0, 1.0e-6);
}

TEST_F(DeferredBearingTracksTest, DefersFiniteDepthUntilThreeFramesConfirmIt) {
  const Eigen::Vector3d point_W(0.0, 0.0, 1.0);
  Eigen::Vector2d measurementA;
  Eigen::Vector2d measurementB;
  Eigen::Vector2d measurementC;
  ASSERT_EQ(camera_->project(point_W, &measurementA),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  ASSERT_EQ(camera_->project(point_W - Eigen::Vector3d(0.1, 0.0, 0.0), &measurementB),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  ASSERT_EQ(camera_->project(point_W - Eigen::Vector3d(0.2, 0.0, 0.0), &measurementC),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);

  const okvis::MultiFramePtr frameA = makeFrame(0.0, measurementA);
  const okvis::MultiFramePtr frameB = makeFrame(0.1, measurementB);
  estimator_.set_T_WS(
      frameB->id(),
      okvis::kinematics::Transformation(Eigen::Vector3d(0.1, 0.0, 0.0),
                                        Eigen::Quaterniond::Identity()));

  MatchingAlgorithm twoFrameMatcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  twoFrameMatcher.setFrames(frameA->id(), frameB->id(), 0, 0);
  twoFrameMatcher.doSetup();
  twoFrameMatcher.setBestMatch(0, 0, 0.0);

  const uint64_t trackId = frameA->landmarkId(0, 0);
  ASSERT_NE(trackId, 0u);
  EXPECT_EQ(frameB->landmarkId(0, 0), trackId);
  EXPECT_FALSE(estimator_.isLandmarkAdded(trackId));
  EXPECT_EQ(twoFrameMatcher.numPromotionAttempts(), 1u);
  EXPECT_EQ(twoFrameMatcher.numPromotionDeferredObservations(), 1u);
  EXPECT_EQ(twoFrameMatcher.numPromotions(), 0u);

  const okvis::MultiFramePtr frameC = makeFrame(0.2, measurementC);
  estimator_.set_T_WS(
      frameC->id(),
      okvis::kinematics::Transformation(Eigen::Vector3d(0.2, 0.0, 0.0),
                                        Eigen::Quaterniond::Identity()));
  MatchingAlgorithm threeFrameMatcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  threeFrameMatcher.setFrames(frameA->id(), frameC->id(), 0, 0);
  threeFrameMatcher.doSetup();
  threeFrameMatcher.setBestMatch(0, 0, 0.0);

  EXPECT_EQ(frameC->landmarkId(0, 0), trackId);
  EXPECT_TRUE(estimator_.isLandmarkAdded(trackId));
  EXPECT_EQ(threeFrameMatcher.numPromotionAttempts(), 1u);
  EXPECT_EQ(threeFrameMatcher.numPromotions(), 1u);
}

TEST_F(DeferredBearingTracksTest, PreservesPendingAssociationsWhenJointPromotionFails) {
  const Eigen::Vector3d point_W(0.0, 0.0, 1.0);
  Eigen::Vector2d measurementA;
  Eigen::Vector2d measurementB;
  Eigen::Vector2d measurementC;
  ASSERT_EQ(camera_->project(point_W, &measurementA),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  ASSERT_EQ(camera_->project(point_W - Eigen::Vector3d(0.05, 0.0, 0.0), &measurementB),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  ASSERT_EQ(camera_->project(point_W - Eigen::Vector3d(0.1, 0.0, 0.0), &measurementC),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  // Make the historical observation inconsistent with the track. The joint
  // estimate must be deferred without destroying its pending association.
  measurementB.x() += 100.0;

  const okvis::MultiFramePtr frameA = makeFrame(0.0, measurementA);
  const okvis::MultiFramePtr frameB = makeFrame(0.1, measurementB);
  const okvis::MultiFramePtr frameC = makeFrame(0.2, measurementC);
  estimator_.set_T_WS(
      frameB->id(),
      okvis::kinematics::Transformation(Eigen::Vector3d(0.05, 0.0, 0.0),
                                        Eigen::Quaterniond::Identity()));
  estimator_.set_T_WS(
      frameC->id(),
      okvis::kinematics::Transformation(Eigen::Vector3d(0.1, 0.0, 0.0),
                                        Eigen::Quaterniond::Identity()));

  const uint64_t trackId = okvis::IdProvider::instance().newId();
  frameA->setLandmarkId(0, 0, trackId);
  frameB->setLandmarkId(0, 0, trackId);

  MatchingAlgorithm matcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  matcher.setFrames(frameA->id(), frameC->id(), 0, 0);
  matcher.doSetup();
  matcher.setBestMatch(0, 0, 0.0);

  EXPECT_FALSE(estimator_.isLandmarkAdded(trackId));
  EXPECT_EQ(matcher.numPromotions(), 0u);
  EXPECT_EQ(frameA->landmarkId(0, 0), trackId);
  EXPECT_EQ(frameB->landmarkId(0, 0), trackId);
  EXPECT_EQ(frameC->landmarkId(0, 0), trackId);
}

TEST_F(DeferredBearingTracksTest, DoesNotMergeTracksThatConflictInOneImage) {
  Eigen::Vector2d center;
  ASSERT_EQ(camera_->project(Eigen::Vector3d::UnitZ(), &center),
            okvis::cameras::CameraBase::ProjectionStatus::Successful);
  const okvis::MultiFramePtr frameA = makeFrame(0.0, center);
  const okvis::MultiFramePtr frameB = makeFrame(0.1, center);
  const uint64_t trackA = okvis::IdProvider::instance().newId();
  const uint64_t trackB = okvis::IdProvider::instance().newId();
  frameA->setLandmarkId(0, 0, trackA);
  frameA->setLandmarkId(0, 1, trackB);
  frameB->setLandmarkId(0, 0, trackB);

  MatchingAlgorithm matcher(estimator_, MatchingAlgorithm::Match2D2D, 80.0f, false);
  matcher.setFrames(frameA->id(), frameB->id(), 0, 0);
  matcher.doSetup();
  matcher.setBestMatch(0, 0, 0.0);

  EXPECT_EQ(frameA->landmarkId(0, 0), trackA);
  EXPECT_EQ(frameA->landmarkId(0, 1), trackB);
  EXPECT_EQ(frameB->landmarkId(0, 0), trackB);
  EXPECT_EQ(matcher.numMatches(), 0u);
  EXPECT_FALSE(estimator_.isLandmarkAdded(trackA));
  EXPECT_FALSE(estimator_.isLandmarkAdded(trackB));
}

}  // namespace
