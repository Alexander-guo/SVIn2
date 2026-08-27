#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <okvis/RansacPoseDiagnostics.hpp>

namespace {
using okvis::ransac_pose_diagnostics::RateLimiter;

TEST(RansacPoseDiagnostics, RelativeTransformConventionAndDeltas) {
  using okvis::kinematics::Transformation;
  const Transformation T_WSa(Eigen::Vector3d(1.0, 2.0, 0.0),
                              Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ())));
  const Transformation T_SCa(Eigen::Vector3d(0.1, 0.0, 0.0), Eigen::Quaterniond::Identity());
  const Transformation T_CaCb(Eigen::Vector3d(0.3, -0.2, 0.1),
                              Eigen::Quaterniond(Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitY())));
  const Transformation T_SCb(Eigen::Vector3d(-0.05, 0.0, 0.0), Eigen::Quaterniond::Identity());
  const Transformation T_WSb = T_WSa * T_SCa * T_CaCb * T_SCb.inverse();
  const Transformation recovered =
      okvis::ransac_pose_diagnostics::estimatorRelativeCameraPose(T_WSa, T_SCa, T_WSb, T_SCb);
  EXPECT_NEAR((recovered.r() - T_CaCb.r()).norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(okvis::ransac_pose_diagnostics::rotationDeltaDegrees(recovered.C(), T_CaCb.C()),
              0.0, 1.0e-6);
  EXPECT_NEAR(okvis::ransac_pose_diagnostics::directionCosine(recovered.r(), T_CaCb.r()), 1.0,
              1.0e-12);
  EXPECT_TRUE(std::isnan(okvis::ransac_pose_diagnostics::directionCosine(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX())));
}

TEST(RansacPoseDiagnostics, AbsoluteModelUsesWorldSensorConvention) {
  const Eigen::Matrix3d propagated =
      Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d model =
      Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()).toRotationMatrix() * propagated;
  EXPECT_NEAR(okvis::ransac_pose_diagnostics::rotationDeltaDegrees(model, propagated),
              0.05 * 180.0 / M_PI, 1.0e-10);
}

TEST(RansacPoseDiagnostics, DiagnosticsOffHasNoEmissionOrEligibility) {
  EXPECT_FALSE(okvis::ransac_pose_diagnostics::enabledFromEnvironmentValue(nullptr));
  EXPECT_FALSE(okvis::ransac_pose_diagnostics::enabledFromEnvironmentValue("0"));
  EXPECT_TRUE(okvis::ransac_pose_diagnostics::enabledFromEnvironmentValue("1"));
  RateLimiter limiter;
  EXPECT_FALSE(limiter.shouldEmit(false, 1, 10, 0));
  EXPECT_FALSE(okvis::ransac_pose_diagnostics::relativeRotationEligibility(
                   false, 100, 90, 0.9, 1.0)
                   .eligible);
}

TEST(RansacPoseDiagnostics, LowSupportAndRecoveryBurstAreDeterministic) {
  RateLimiter limiter;
  EXPECT_TRUE(limiter.shouldEmit(true, 100, 10, 15));
  for (int i = 0; i < 10; ++i) EXPECT_TRUE(limiter.shouldEmit(true, 101 + i, 10, 16));
  EXPECT_TRUE(limiter.shouldEmit(true, 111, 10, 16));  // first normal-rate sample
  EXPECT_FALSE(limiter.shouldEmit(true, 112, 10, 16));
  EXPECT_TRUE(limiter.shouldEmit(true, 113, 11, 16));
}
}  // namespace
