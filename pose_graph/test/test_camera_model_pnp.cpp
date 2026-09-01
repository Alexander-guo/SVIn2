#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <map>
#include <vector>

#include "okvis/cameras/DoubleSphereCamera.hpp"
#include "okvis/cameras/NoDistortion.hpp"
#include "pose_graph/Keyframe.h"

namespace {

using DoubleSphereCamera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;

Parameters makeDoubleSphereParameters() {
  Parameters params;
  params.camera_calibration_.projection_type_ = "double_sphere";
  params.camera_calibration_.distortion_type_ = "none";
  params.camera_calibration_.image_dimension_ << 832, 832;
  params.camera_calibration_.focal_length_ << 245.3211158993786, 246.35906171785095;
  params.camera_calibration_.principal_point_ << 415.00183901843747, 407.9706681361295;
  params.camera_calibration_.double_sphere_params_ << 0.014379153239762127, 0.6864870568393587;
  params.loop_closure_params_.pnp_reprojection_thresh = 3.0;
  params.loop_closure_params_.pnp_ransac_iterations = 500;
  return params;
}

TEST(CameraModelPnP, RecoversDoubleSpherePoseWithWideFieldOfViewAndOutliers) {
  Parameters params = makeDoubleSphereParameters();
  const CameraCalibration& calibration = params.camera_calibration_;
  const DoubleSphereCamera camera(calibration.image_dimension_.x(),
                                  calibration.image_dimension_.y(),
                                  calibration.focal_length_.x(),
                                  calibration.focal_length_.y(),
                                  calibration.principal_point_.x(),
                                  calibration.principal_point_.y(),
                                  calibration.double_sphere_params_.x(),
                                  calibration.double_sphere_params_.y(),
                                  okvis::cameras::NoDistortion());

  const Eigen::Matrix3d R_w_c =
      (Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(-0.18, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  const Eigen::Vector3d T_w_c(1.2, -0.7, 0.4);

  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> world_points;
  for (int v = 90; v <= 750; v += 110) {
    for (int u = 90; u <= 750; u += 110) {
      Eigen::Vector3d bearing;
      if (!camera.backProject(Eigen::Vector2d(u, v), &bearing)) continue;
      const double depth = 3.0 + 0.003 * u + 0.002 * v;
      const Eigen::Vector3d point_w = R_w_c * (depth * bearing.normalized()) + T_w_c;
      image_points.emplace_back(static_cast<float>(u), static_cast<float>(v));
      world_points.emplace_back(point_w.x(), point_w.y(), point_w.z());
    }
  }
  ASSERT_GE(world_points.size(), 30U);

  const std::vector<cv::Point2f> valid_observations = image_points;
  for (size_t i = 0; i < 10; ++i) {
    image_points[i] = valid_observations[(i + 5) % 10];
  }
  image_points.emplace_back(0.0F, 0.0F);
  world_points.emplace_back(10.0F, 10.0F, 10.0F);

  Eigen::Vector3d initial_translation = Eigen::Vector3d::Zero();
  Eigen::Matrix3d initial_rotation = Eigen::Matrix3d::Identity();
  std::map<Keyframe*, int> covisibility;
  Keyframe keyframe(0, 0, initial_translation, initial_rotation, covisibility, 0, params, false);

  std::vector<uchar> status;
  Eigen::Vector3d estimated_translation;
  Eigen::Matrix3d estimated_rotation;
  bool threw_exception = false;
  ASSERT_TRUE(keyframe.PnPRANSAC(image_points,
                                world_points,
                                status,
                                estimated_translation,
                                estimated_rotation,
                                &threw_exception));
  EXPECT_FALSE(threw_exception);

  const size_t inliers = static_cast<size_t>(std::count(status.begin(), status.end(), uchar{1}));
  ASSERT_EQ(status.size(), world_points.size());
  EXPECT_EQ(status.back(), 0);
  EXPECT_GE(inliers, world_points.size() - 11);
  EXPECT_LT((estimated_translation - T_w_c).norm(), 0.05);
  const Eigen::Matrix3d rotation_error = estimated_rotation.transpose() * R_w_c;
  EXPECT_LT(Eigen::AngleAxisd(rotation_error).angle(), 0.01);
}

TEST(CameraModelPnP, PreservesPinholePosePath) {
  Parameters params;
  params.camera_calibration_.projection_type_ = "pinhole";
  params.camera_calibration_.focal_length_ << 420.0, 415.0;
  params.camera_calibration_.principal_point_ << 320.0, 240.0;
  params.camera_calibration_.distortion_coefficients_ = cv::Mat::zeros(4, 1, CV_64F);
  params.loop_closure_params_.pnp_reprojection_thresh = 3.0;
  params.loop_closure_params_.pnp_ransac_iterations = 500;

  const Eigen::Matrix3d R_w_c =
      (Eigen::AngleAxisd(-0.25, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitY()))
          .toRotationMatrix();
  const Eigen::Vector3d T_w_c(-0.8, 0.4, 1.1);

  std::vector<cv::Point2f> image_points;
  std::vector<cv::Point3f> world_points;
  for (int row = -3; row <= 3; ++row) {
    for (int column = -4; column <= 4; ++column) {
      const Eigen::Vector3d point_c(0.22 * column, 0.18 * row, 4.0 + 0.12 * (row + column + 7));
      const Eigen::Vector3d point_w = R_w_c * point_c + T_w_c;
      image_points.emplace_back(
          static_cast<float>(params.camera_calibration_.focal_length_.x() * point_c.x() / point_c.z() +
                             params.camera_calibration_.principal_point_.x()),
          static_cast<float>(params.camera_calibration_.focal_length_.y() * point_c.y() / point_c.z() +
                             params.camera_calibration_.principal_point_.y()));
      world_points.emplace_back(point_w.x(), point_w.y(), point_w.z());
    }
  }
  const std::vector<cv::Point2f> valid_observations = image_points;
  for (size_t i = 0; i < 8; ++i) image_points[i] = valid_observations[(i + 4) % 8];

  Eigen::Vector3d initial_translation = Eigen::Vector3d::Zero();
  Eigen::Matrix3d initial_rotation = Eigen::Matrix3d::Identity();
  std::map<Keyframe*, int> covisibility;
  Keyframe keyframe(0, 0, initial_translation, initial_rotation, covisibility, 0, params, false);

  std::vector<uchar> status;
  Eigen::Vector3d estimated_translation;
  Eigen::Matrix3d estimated_rotation;
  ASSERT_TRUE(
      keyframe.PnPRANSAC(image_points, world_points, status, estimated_translation, estimated_rotation, nullptr));
  EXPECT_GE(std::count(status.begin(), status.end(), uchar{1}), static_cast<int>(world_points.size() - 8));
  EXPECT_LT((estimated_translation - T_w_c).norm(), 0.05);
  EXPECT_LT(Eigen::AngleAxisd(estimated_rotation.transpose() * R_w_c).angle(), 0.01);
}

}  // namespace
