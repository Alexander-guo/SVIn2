#include <gtest/gtest.h>

#include <limits>

#include <okvis/OpposingMulticamKeyframePolicy.hpp>

namespace {

using Decision = okvis::OpposingMulticamKeyframeDecision;

TEST(OpposingMulticamKeyframePolicy, EnforcesExactFrameAndTimeBounds) {
  const okvis::KeyframeSelectionParameters policy;
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 2, true, 0.20, false, 2),
            Decision::MinSpacing);
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 3, true, 0.099, false, 2),
            Decision::MinSpacing);
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 3, true, 0.10, false, 2),
            Decision::QualityDegraded);
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 14, true, 0.75, true, 0),
            Decision::MaxSpacing);
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 15, true, 0.40, true, 0),
            Decision::MaxSpacing);
}

TEST(OpposingMulticamKeyframePolicy, ConfirmsDegradationAndTreatsEitherCameraSymmetrically) {
  const okvis::KeyframeSelectionParameters policy;
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 6, true, 0.20, false, 1),
            Decision::DegradationPending);
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 6, true, 0.20, false, 2),
            Decision::QualityDegraded);
  // The state machine consumes only rig health. A healthy result from camera 0
  // or camera 1 is therefore exactly equivalent after symmetric per-camera tests.
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(policy, 6, true, 0.20, true, 0),
            Decision::HealthySkip);
}

TEST(OpposingMulticamKeyframePolicy, FrameMaximumPreventsStarvationWithoutValidTime) {
  const okvis::KeyframeSelectionParameters policy;
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(
                policy, 14, false, std::numeric_limits<double>::quiet_NaN(), false, 20),
            Decision::MinSpacing);
  EXPECT_EQ(okvis::opposingMulticamKeyframeDecision(
                policy, 15, false, std::numeric_limits<double>::quiet_NaN(), false, 20),
            Decision::MaxSpacing);
}

}  // namespace
