#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>

#include "okvis/cameras/DoubleSphereCamera.hpp"
#include "okvis/cameras/NoDistortion.hpp"

TEST(DoubleSphereCamera, functions) {
  const size_t NUM_POINTS = 100;

  // instantiate test camera
  auto camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>::createTestObject();

  for (size_t i = 0; i < NUM_POINTS; ++i) {
    // create a random point in the field of view:
    Eigen::Vector2d imagePoint = camera->createRandomImagePoint();

    // backProject
    Eigen::Vector3d ray;
    EXPECT_TRUE(camera->backProject(imagePoint, &ray));

    // randomise distance
    ray.normalize();
    ray *= (0.2 + 8 * (Eigen::Vector2d::Random()[0] + 1.0));

    // project
    Eigen::Vector2d imagePoint2;
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Matrix2Xd J_intrinsics;
    EXPECT_TRUE(camera->project(ray, &imagePoint2, &J, &J_intrinsics) ==
                okvis::cameras::CameraBase::ProjectionStatus::Successful);

    // check they are the same
    EXPECT_TRUE((imagePoint2 - imagePoint).norm() < 0.01);

    // check point Jacobian vs. NumDiff
    const double dp = 1.0e-7;
    Eigen::Matrix<double, 2, 3> J_numDiff;
    for (size_t d = 0; d < 3; ++d) {
      Eigen::Vector3d point_p = ray + Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0);
      Eigen::Vector3d point_m = ray - Eigen::Vector3d(d == 0 ? dp : 0, d == 1 ? dp : 0, d == 2 ? dp : 0);
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      camera->project(point_p, &imagePoint_p);
      camera->project(point_m, &imagePoint_m);
      J_numDiff.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }
    EXPECT_TRUE((J_numDiff - J).norm() < 0.001);

    // check intrinsics Jacobian
    const int numIntrinsics = camera->noIntrinsicsParameters();
    Eigen::VectorXd intrinsics;
    camera->getIntrinsics(intrinsics);
    Eigen::Matrix2Xd J_numDiff_intrinsics;
    J_numDiff_intrinsics.resize(2, numIntrinsics);
    for (int d = 0; d < numIntrinsics; ++d) {
      Eigen::VectorXd di;
      di.resize(numIntrinsics);
      di.setZero();
      di[d] = dp;
      Eigen::Vector2d imagePoint_p;
      Eigen::Vector2d imagePoint_m;
      Eigen::VectorXd intrinsics_p = intrinsics + di;
      Eigen::VectorXd intrinsics_m = intrinsics - di;
      camera->projectWithExternalParameters(ray, intrinsics_p, &imagePoint_p);
      camera->projectWithExternalParameters(ray, intrinsics_m, &imagePoint_m);
      J_numDiff_intrinsics.col(d) = (imagePoint_p - imagePoint_m) / (2 * dp);
    }
    EXPECT_TRUE((J_numDiff_intrinsics - J_intrinsics).norm() < 0.001);
  }
}

TEST(DoubleSphereCamera, backProjectHomogeneous) {
  auto camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>::createTestObject();
  const Eigen::Vector2d imagePoint = camera->createRandomImagePoint();

  Eigen::Vector3d ray;
  ASSERT_TRUE(camera->backProject(imagePoint, &ray));

  Eigen::Vector4d homogeneousDirection = Eigen::Vector4d::Constant(-1.0);
  ASSERT_TRUE(camera->backProjectHomogeneous(imagePoint, &homogeneousDirection));
  EXPECT_TRUE(homogeneousDirection.head<3>().isApprox(ray));
  EXPECT_DOUBLE_EQ(homogeneousDirection[3], 1.0);

  Eigen::Vector4d homogeneousDirectionWithJacobian = Eigen::Vector4d::Constant(-1.0);
  Eigen::Matrix<double, 4, 2> homogeneousJacobian = Eigen::Matrix<double, 4, 2>::Constant(-1.0);
  ASSERT_TRUE(camera->backProjectHomogeneous(
      imagePoint, &homogeneousDirectionWithJacobian, &homogeneousJacobian));
  EXPECT_TRUE(homogeneousDirectionWithJacobian.head<3>().isApprox(ray));
  EXPECT_DOUBLE_EQ(homogeneousDirectionWithJacobian[3], 1.0);
  EXPECT_TRUE(homogeneousJacobian.row(3).isZero());
}

TEST(DoubleSphereCamera, rejectsInvalidHomogeneousPoint) {
  auto camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>::createTestObject();
  const auto invalid = okvis::cameras::CameraBase::ProjectionStatus::Invalid;
  const Eigen::Vector4d zeroPoint = Eigen::Vector4d::Zero();

  Eigen::Vector2d imagePoint = Eigen::Vector2d::Zero();
  EXPECT_EQ(camera->projectHomogeneous(zeroPoint, &imagePoint), invalid);
  EXPECT_FALSE(imagePoint.allFinite());

  Eigen::Matrix<double, 2, 4> pointJacobian = Eigen::Matrix<double, 2, 4>::Ones();
  Eigen::Matrix2Xd intrinsicsJacobian = Eigen::Matrix2Xd::Ones(2, camera->noIntrinsicsParameters());
  EXPECT_EQ(camera->projectHomogeneous(zeroPoint, &imagePoint, &pointJacobian, &intrinsicsJacobian), invalid);
  EXPECT_FALSE(imagePoint.allFinite());
  EXPECT_TRUE(pointJacobian.isZero());
  EXPECT_TRUE(intrinsicsJacobian.isZero());

  Eigen::VectorXd intrinsics;
  camera->getIntrinsics(intrinsics);
  pointJacobian.setOnes();
  intrinsicsJacobian.setOnes();
  EXPECT_EQ(camera->projectHomogeneousWithExternalParameters(
                zeroPoint, intrinsics, &imagePoint, &pointJacobian, &intrinsicsJacobian),
            invalid);
  EXPECT_FALSE(imagePoint.allFinite());
  EXPECT_TRUE(pointJacobian.isZero());
  EXPECT_TRUE(intrinsicsJacobian.isZero());

  const Eigen::Vector4d zeroDirectionWithFiniteScale(0.0, 0.0, 0.0, 1.0);
  EXPECT_EQ(camera->projectHomogeneous(zeroDirectionWithFiniteScale, &imagePoint), invalid);
}
