#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include <okvis/triangulation/stereo_triangulation.hpp>

namespace {

using okvis::triangulation::TriangulationDiagnostics;
using okvis::triangulation::TriangulationStatus;
using okvis::triangulation::TriangulationType;
using okvis::triangulation::triangulateFast;

struct Result {
  Eigen::Vector4d point;
  bool valid = false;
  bool parallel = false;
  TriangulationDiagnostics diagnostics;
};

Result triangulate(const Eigen::Vector3d& p1,
                   const Eigen::Vector3d& e1,
                   const Eigen::Vector3d& p2,
                   const Eigen::Vector3d& e2,
                   double sigma = 1.0e-4) {
  Result result;
  result.point = triangulateFast(p1, e1, p2, e2, sigma, result.valid, result.parallel, &result.diagnostics);
  return result;
}

void expectRejected(const Result& result, TriangulationStatus status) {
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.diagnostics.type, TriangulationType::Rejected);
  EXPECT_EQ(result.diagnostics.status, status);
  EXPECT_TRUE(result.point.allFinite());
  EXPECT_TRUE(result.point.isZero());
}

void expectBearingOnly(const Result& result, TriangulationStatus status) {
  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(result.parallel);
  EXPECT_EQ(result.diagnostics.type, TriangulationType::BearingOnly);
  EXPECT_EQ(result.diagnostics.status, status);
  EXPECT_TRUE(result.point.allFinite());
  EXPECT_DOUBLE_EQ(result.point.w(), 0.0);
  EXPECT_NEAR(result.point.head<3>().norm(), 1.0, 1.0e-12);
}

TEST(StereoTriangulation, PreservesExactAndNearlyParallelRaysAsBearings) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.1, 0.0, 0.0);

  Result exact = triangulate(p1, Eigen::Vector3d::UnitZ(), p2, Eigen::Vector3d::UnitZ());
  expectBearingOnly(exact, TriangulationStatus::InsufficientParallax);

  const double angle = 0.1 * M_PI / 180.0;
  Result nearly = triangulate(p1, Eigen::Vector3d::UnitZ(), p2,
                              Eigen::Vector3d(std::sin(angle), 0.0, std::cos(angle)));
  expectBearingOnly(nearly, TriangulationStatus::InsufficientParallax);
}

TEST(StereoTriangulation, UsesRelaxedNoiseAdaptiveThresholdWithAngularFloor) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.1, 0.0, 0.0);
  const double lowSigma = 1.0e-4;
  const double highSigma = 0.005;

  Result floorResult = triangulate(p1, Eigen::Vector3d::UnitZ(), p2,
                                   Eigen::Vector3d(std::sin(0.1 * M_PI / 180.0), 0.0,
                                                   std::cos(0.1 * M_PI / 180.0)),
                                   lowSigma);
  EXPECT_NEAR(floorResult.diagnostics.minimumParallaxAngle, 0.25 * M_PI / 180.0, 1.0e-12);

  Result noiseResult = triangulate(p1, Eigen::Vector3d::UnitZ(), p2,
                                   Eigen::Vector3d(std::sin(0.1 * M_PI / 180.0), 0.0,
                                                   std::cos(0.1 * M_PI / 180.0)),
                                   highSigma);
  EXPECT_NEAR(noiseResult.diagnostics.minimumParallaxAngle, 2.0 * highSigma, 1.0e-12);
}

TEST(StereoTriangulation, RejectsInvalidInputs) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.1, 0.0, 0.0);
  expectRejected(triangulate(p1, Eigen::Vector3d::Zero(), p2, Eigen::Vector3d::UnitZ()),
                 TriangulationStatus::InvalidInput);

  Eigen::Vector3d nonfinite = Eigen::Vector3d::UnitZ();
  nonfinite.x() = std::numeric_limits<double>::quiet_NaN();
  expectRejected(triangulate(p1, nonfinite, p2, Eigen::Vector3d::UnitZ()),
                 TriangulationStatus::InvalidInput);
}

TEST(StereoTriangulation, AcceptsSmallBaselineForNearbyPoint) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.05, 0.0, 0.0);
  const Eigen::Vector3d expectedPoint(0.0, 0.0, 1.0);
  Result result = triangulate(p1, expectedPoint - p1, p2, expectedPoint - p2);

  ASSERT_TRUE(result.valid);
  EXPECT_FALSE(result.parallel);
  EXPECT_EQ(result.diagnostics.type, TriangulationType::Finite);
  EXPECT_EQ(result.diagnostics.status, TriangulationStatus::Success);
  ASSERT_GT(std::abs(result.point.w()), 1.0e-12);
  EXPECT_TRUE((result.point.head<3>() / result.point.w() - expectedPoint).norm() < 1.0e-9);
}

TEST(StereoTriangulation, PreservesDistantPointAsBearingForSameBaseline) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.05, 0.0, 0.0);
  const Eigen::Vector3d distantPoint(0.0, 0.0, 100.0);
  Result result = triangulate(p1, distantPoint - p1, p2, distantPoint - p2);
  expectBearingOnly(result, TriangulationStatus::InsufficientParallax);
}

TEST(StereoTriangulation, RejectsIntersectionBehindViewingRays) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.1, 0.0, 0.0);
  const Eigen::Vector3d point(0.0, 0.0, 1.0);
  Result result = triangulate(p1, p1 - point, p2, p2 - point);
  expectRejected(result, TriangulationStatus::BehindRay);
}

TEST(StereoTriangulation, RejectsInconsistentSkewRays) {
  const Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d p2(0.1, 0.0, 0.0);
  Result result = triangulate(p1, Eigen::Vector3d(0.0, 0.0, 1.0), p2,
                              Eigen::Vector3d(-0.1, 0.02, 1.0));
  expectRejected(result, TriangulationStatus::RayMismatch);
}

TEST(StereoTriangulation, PreservesWellConditionedIntersection) {
  const Eigen::Vector3d p1(-0.2, 0.0, 0.0);
  const Eigen::Vector3d p2(0.2, 0.0, 0.0);
  const Eigen::Vector3d expectedPoint(0.1, -0.05, 2.0);
  Result result = triangulate(p1, expectedPoint - p1, p2, expectedPoint - p2);

  ASSERT_TRUE(result.valid);
  ASSERT_GT(std::abs(result.point.w()), 1.0e-12);
  EXPECT_TRUE((result.point.head<3>() / result.point.w() - expectedPoint).norm() < 1.0e-9);
}

}  // namespace
