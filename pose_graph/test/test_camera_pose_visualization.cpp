#include <gtest/gtest.h>

#include "pose_graph/Publisher.h"
#include "utils/CameraPoseVisualization.h"

TEST(CameraPoseVisualization, AssignsDistinctMarkerIdsAndColors) {
  CameraPoseVisualization visualizer(1.0, 0.0, 0.0, 1.0);
  visualizer.add_pose(Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
  visualizer.setImageBoundaryColor(0.0, 0.7, 1.0);
  visualizer.setOpticalCenterConnectorColor(0.0, 0.7, 1.0);
  visualizer.add_pose(Eigen::Vector3d::UnitX(), Eigen::Quaterniond::Identity());

  const auto& markers = visualizer.cameraPoseMarkers();
  ASSERT_EQ(markers.size(), 2U);
  EXPECT_EQ(markers[0].id, 0);
  EXPECT_EQ(markers[1].id, 1);
  ASSERT_FALSE(markers[0].colors.empty());
  ASSERT_FALSE(markers[1].colors.empty());
  EXPECT_FLOAT_EQ(markers[0].colors.front().r, 1.0F);
  EXPECT_FLOAT_EQ(markers[0].colors.front().g, 0.0F);
  EXPECT_FLOAT_EQ(markers[1].colors.front().g, 0.7F);
  EXPECT_FLOAT_EQ(markers[1].colors.front().b, 1.0F);
}

TEST(CameraPoseVisualization, ConvertsPrimaryPoseUsingRigExtrinsics) {
  Eigen::Matrix4d T_WC0 = Eigen::Matrix4d::Identity();
  T_WC0.block<3, 1>(0, 3) << 1.0, 2.0, 3.0;

  Eigen::Matrix4d T_SC0 = Eigen::Matrix4d::Identity();
  T_SC0(0, 3) = 0.1;

  Eigen::Matrix4d T_SC1 = Eigen::Matrix4d::Identity();
  T_SC1.block<3, 3>(0, 0) =
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_SC1(0, 3) = -0.2;

  const Eigen::Matrix4d T_WC1 = Publisher::cameraPoseFromPrimary(T_WC0, T_SC0, T_SC1);
  EXPECT_TRUE(T_WC1.isApprox(T_WC0 * T_SC0.inverse() * T_SC1));
  EXPECT_NEAR(T_WC1(0, 3), 0.7, 1.0e-12);
}
