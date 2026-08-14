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
 *  Created on: Sep 3, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <memory>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Time.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/HomogeneousPointError.hpp>
#include <okvis/ceres/HomogeneousPointManifold.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/PoseManifold.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/kinematics/Transformation.hpp>

#include "ceres/ceres.h"
#include "glog/logging.h"

namespace {

class ControllableCameraGeometry {
 public:
  using ProjectionStatus = okvis::cameras::CameraBase::ProjectionStatus;

  ControllableCameraGeometry(ProjectionStatus status, bool finiteOutput)
      : status_(status), finiteOutput_(finiteOutput) {}

  ProjectionStatus projectHomogeneous(const Eigen::Vector4d&, Eigen::Vector2d* imagePoint) const {
    setProjection(imagePoint, nullptr);
    return status_;
  }

  ProjectionStatus projectHomogeneous(const Eigen::Vector4d&,
                                       Eigen::Vector2d* imagePoint,
                                       Eigen::Matrix<double, 2, 4>* pointJacobian) const {
    setProjection(imagePoint, pointJacobian);
    return status_;
  }

  void getIntrinsics(Eigen::VectorXd& intrinsics) const {
    intrinsics.resize(6);
    intrinsics << 200.0, 200.0, 320.0, 320.0, 0.0, 0.6;
  }

  uint32_t imageWidth() const { return 640; }
  uint32_t imageHeight() const { return 640; }
  void setProjectionStatus(ProjectionStatus status) { status_ = status; }

 private:
  void setProjection(Eigen::Vector2d* imagePoint, Eigen::Matrix<double, 2, 4>* pointJacobian) const {
    if (finiteOutput_) {
      *imagePoint << 4.0, 5.0;
      if (pointJacobian != nullptr) {
        pointJacobian->setConstant(0.25);
      }
      return;
    }

    imagePoint->setConstant(std::numeric_limits<double>::quiet_NaN());
    if (pointJacobian != nullptr) {
      pointJacobian->setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }

  ProjectionStatus status_;
  bool finiteOutput_;
};

void expectZeroReprojectionFactor(ControllableCameraGeometry::ProjectionStatus status, bool finiteOutput) {
  auto cameraGeometry = std::make_shared<const ControllableCameraGeometry>(status, finiteOutput);
  const Eigen::Vector2d measurement(10.0, 20.0);
  const Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
  okvis::ceres::ReprojectionError<ControllableCameraGeometry> error(
      cameraGeometry, 0, measurement, information);

  const std::array<double, 7> pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 4> point = {0.0, 0.0, 1.0, 1.0};
  const std::array<double, 7> extrinsics = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const double* parameters[3] = {pose.data(), point.data(), extrinsics.data()};

  std::array<double, 2> residuals = {9.0, 9.0};
  EXPECT_TRUE(error.Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_DOUBLE_EQ(residuals[0], 0.0);
  EXPECT_DOUBLE_EQ(residuals[1], 0.0);

  std::array<double, 14> J0;
  std::array<double, 8> J1;
  std::array<double, 14> J2;
  std::array<double, 12> J0Minimal;
  std::array<double, 6> J1Minimal;
  std::array<double, 12> J2Minimal;
  J0.fill(9.0);
  J1.fill(9.0);
  J2.fill(9.0);
  J0Minimal.fill(9.0);
  J1Minimal.fill(9.0);
  J2Minimal.fill(9.0);
  double* jacobians[3] = {J0.data(), J1.data(), J2.data()};
  double* jacobiansMinimal[3] = {J0Minimal.data(), J1Minimal.data(), J2Minimal.data()};

  residuals = {9.0, 9.0};
  EXPECT_TRUE(error.EvaluateWithMinimalJacobians(
      parameters, residuals.data(), jacobians, jacobiansMinimal));
  EXPECT_DOUBLE_EQ(residuals[0], 0.0);
  EXPECT_DOUBLE_EQ(residuals[1], 0.0);
  for (double value : J0) EXPECT_DOUBLE_EQ(value, 0.0);
  for (double value : J1) EXPECT_DOUBLE_EQ(value, 0.0);
  for (double value : J2) EXPECT_DOUBLE_EQ(value, 0.0);
  for (double value : J0Minimal) EXPECT_DOUBLE_EQ(value, 0.0);
  for (double value : J1Minimal) EXPECT_DOUBLE_EQ(value, 0.0);
  for (double value : J2Minimal) EXPECT_DOUBLE_EQ(value, 0.0);
}

}  // namespace

TEST(okvisTestSuite, ReprojectionErrorDeactivatesInvalidProjections) {
  using ProjectionStatus = okvis::cameras::CameraBase::ProjectionStatus;
  expectZeroReprojectionFactor(ProjectionStatus::Behind, true);
  expectZeroReprojectionFactor(ProjectionStatus::Invalid, true);
  expectZeroReprojectionFactor(ProjectionStatus::Masked, true);
  expectZeroReprojectionFactor(ProjectionStatus::OutsideImage, true);
  expectZeroReprojectionFactor(ProjectionStatus::Successful, false);
}

TEST(okvisTestSuite, ReprojectionErrorPreservesSuccessfulProjection) {
  using ProjectionStatus = okvis::cameras::CameraBase::ProjectionStatus;
  auto cameraGeometry =
      std::make_shared<const ControllableCameraGeometry>(ProjectionStatus::Successful, true);
  okvis::ceres::ReprojectionError<ControllableCameraGeometry> error(
      cameraGeometry, 0, Eigen::Vector2d(10.0, 20.0), Eigen::Matrix2d::Identity());

  const std::array<double, 7> pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 4> point = {0.0, 0.0, 1.0, 1.0};
  const std::array<double, 7> extrinsics = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  const double* parameters[3] = {pose.data(), point.data(), extrinsics.data()};
  std::array<double, 2> residuals = {0.0, 0.0};

  EXPECT_TRUE(error.Evaluate(parameters, residuals.data(), nullptr));
  EXPECT_DOUBLE_EQ(residuals[0], 6.0);
  EXPECT_DOUBLE_EQ(residuals[1], 15.0);
}

TEST(okvisTestSuite, ReprojectionErrorTracksObservationIdentityAndActiveState) {
  using Camera = ControllableCameraGeometry;
  using Error = okvis::ceres::ReprojectionError<Camera>;
  using ProjectionStatus = okvis::cameras::CameraBase::ProjectionStatus;
  auto cameraGeometry = std::make_shared<Camera>(ProjectionStatus::Successful, true);
  const uint64_t activeBefore = Error::activeFactorCount();
  const uint64_t deactivatedBefore = Error::deactivatedFactorCount();

  {
    Error error(cameraGeometry,
                2,
                Eigen::Vector2d(10.0, 20.0),
                Eigen::Matrix2d::Identity(),
                101,
                202,
                303);
    EXPECT_EQ(error.landmarkId(), 101);
    EXPECT_EQ(error.frameId(), 202);
    EXPECT_EQ(error.keypointId(), 303);

    const std::array<double, 7> pose = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const std::array<double, 4> point = {0.0, 0.0, 1.0, 1.0};
    const std::array<double, 7> extrinsics = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const double* parameters[3] = {pose.data(), point.data(), extrinsics.data()};
    std::array<double, 2> residuals{};

    ASSERT_TRUE(error.Evaluate(parameters, residuals.data(), nullptr));
    EXPECT_EQ(Error::activeFactorCount(), activeBefore + 1);
    EXPECT_EQ(Error::deactivatedFactorCount(), deactivatedBefore);

    cameraGeometry->setProjectionStatus(ProjectionStatus::Behind);
    ASSERT_TRUE(error.Evaluate(parameters, residuals.data(), nullptr));
    EXPECT_EQ(Error::activeFactorCount(), activeBefore);
    EXPECT_EQ(Error::deactivatedFactorCount(), deactivatedBefore + 1);
  }

  EXPECT_EQ(Error::activeFactorCount(), activeBefore);
  EXPECT_EQ(Error::deactivatedFactorCount(), deactivatedBefore);
}

TEST(okvisTestSuite, ReprojectionError) {
  // initialize random number generator
  // srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

  // Build the problem.
  ::ceres::Problem problem;

  // set up a random geometry
  std::cout << "set up a random geometry... " << std::flush;
  okvis::kinematics::Transformation T_WS;  // world to sensor
  T_WS.setRandom(10.0, M_PI);
  okvis::kinematics::Transformation T_disturb;
  T_disturb.setRandom(1, 0.01);
  okvis::kinematics::Transformation T_WS_init = T_WS * T_disturb;  // world to sensor
  okvis::kinematics::Transformation T_SC;                          // sensor to camera
  T_SC.setRandom(0.2, M_PI);
  okvis::ceres::PoseParameterBlock poseParameterBlock(T_WS_init, 1, okvis::Time(0));
  okvis::ceres::PoseParameterBlock extrinsicsParameterBlock(T_SC, 2, okvis::Time(0));
  problem.AddParameterBlock(poseParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  problem.AddParameterBlock(extrinsicsParameterBlock.parameters(), okvis::ceres::PoseParameterBlock::Dimension);
  problem.SetParameterBlockVariable(poseParameterBlock.parameters());        // optimize this...
  problem.SetParameterBlockConstant(extrinsicsParameterBlock.parameters());  // do not optimize this...
  std::cout << " [ OK ] " << std::endl;

  // set up a random camera geometry
  std::cout << "set up a random camera geometry... " << std::flush;
  typedef okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> DistortedPinholeCameraGeometry;
  std::shared_ptr<const DistortedPinholeCameraGeometry> cameraGeometry =
      std::static_pointer_cast<const DistortedPinholeCameraGeometry>(
          DistortedPinholeCameraGeometry::createTestObject());
  std::cout << " [ OK ] " << std::endl;

  // let's use our own local quaternion perturbation
  std::cout << "setting local parameterization for pose... " << std::flush;
  ::ceres::Manifold* poseManifold = new okvis::ceres::PoseManifold;

  problem.SetManifold(poseParameterBlock.parameters(), poseManifold);
  problem.SetManifold(extrinsicsParameterBlock.parameters(), poseManifold);
  std::cout << " [ OK ] " << std::endl;

  // and the parameterization for points:
  ::ceres::Manifold* homogeneousPointManifold = new okvis::ceres::HomogeneousPointManifold;

  // get some random points and build error terms
  const size_t N = 100;
  std::cout << "create N=" << N << " visible points and add respective reprojection error terms... " << std::flush;
  for (size_t i = 1; i < 100; ++i) {
    Eigen::Vector4d point = cameraGeometry->createRandomVisibleHomogeneousPoint(static_cast<double>(i % 10) * 3 + 2.0);
    okvis::ceres::HomogeneousPointParameterBlock* homogeneousPointParameterBlock_ptr =
        new okvis::ceres::HomogeneousPointParameterBlock(T_WS * T_SC * point, i + 2);
    problem.AddParameterBlock(homogeneousPointParameterBlock_ptr->parameters(),
                              okvis::ceres::HomogeneousPointParameterBlock::Dimension);
    problem.SetParameterBlockConstant(homogeneousPointParameterBlock_ptr->parameters());

    // get a randomized projection
    Eigen::Vector2d kp;
    cameraGeometry->projectHomogeneous(point, &kp);
    kp += Eigen::Vector2d::Random();

    // Set up the only cost function (also known as residual).
    Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
    ::ceres::CostFunction* cost_function =
        new okvis::ceres::ReprojectionError<DistortedPinholeCameraGeometry>(cameraGeometry, 1, kp, information);
    problem.AddResidualBlock(cost_function,
                             NULL,
                             poseParameterBlock.parameters(),
                             homogeneousPointParameterBlock_ptr->parameters(),
                             extrinsicsParameterBlock.parameters());

    // set the parameterization
    problem.SetManifold(homogeneousPointParameterBlock_ptr->parameters(), homogeneousPointManifold);
  }
  std::cout << " [ OK ] " << std::endl;

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ::ceres::Solver::Options options;
  // options.check_gradients=true;
  // options.numeric_derivative_relative_step_size = 1e-6;
  // options.gradient_check_relative_precision=1e-2;
  options.minimizer_progress_to_stdout = false;
  ::FLAGS_stderrthreshold = google::WARNING;  // enable console warnings (Jacobian verification)
  ::ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  // verify there are no errors in the Jacobians
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

  // print some infos about the optimization
  // std::cout << summary.BriefReport() << "\n";
  std::cout << "initial T_WS : " << T_WS_init.T() << "\n"
            << "optimized T_WS : " << poseParameterBlock.estimate().T() << "\n"
            << "correct T_WS : " << T_WS.T() << "\n";

  // make sure it converged
  OKVIS_ASSERT_TRUE(Exception,
                    2 * (T_WS.q() * poseParameterBlock.estimate().q().inverse()).vec().norm() < 1e-2,
                    "quaternions not close enough");
  OKVIS_ASSERT_TRUE(
      Exception, (T_WS.r() - poseParameterBlock.estimate().r()).norm() < 1e-1, "translation not close enough");
}
