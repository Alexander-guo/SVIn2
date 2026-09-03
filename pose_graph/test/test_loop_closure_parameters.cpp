#include <gtest/gtest.h>

#include <filesystem>

#include "pose_graph/Parameters.h"

TEST(LoopClosureParameters, PreserveLegacyDefaults) {
  const Parameters params;

  EXPECT_EQ(params.loop_closure_params_.keyframe_queue_size, 5);
  EXPECT_DOUBLE_EQ(params.loop_closure_params_.max_yaw_diff, 25.0);
  EXPECT_DOUBLE_EQ(params.loop_closure_params_.max_position_diff, 15.0);
  EXPECT_FALSE(params.loop_closure_params_.multicamera_enabled);
  EXPECT_FALSE(params.loop_closure_params_.multicamera_diagnostics);
  EXPECT_EQ(params.loop_closure_params_.multicamera_matching_threads, 1);
  EXPECT_FALSE(params.lc_diagnostic_);
  EXPECT_FALSE(params.loopClosureDiagnosticsEnabled());
}

TEST(LoopClosureParameters, LoadQueueAndPoseGatesFromConfig) {
  const std::filesystem::path config_path =
      std::filesystem::path(__FILE__).parent_path() / "test_loop_closure_parameters.yaml";
  Parameters params;

  params.loadParameters(config_path.string());

  EXPECT_EQ(params.loop_closure_params_.keyframe_queue_size, -1);
  EXPECT_DOUBLE_EQ(params.loop_closure_params_.max_yaw_diff, 30.0);
  EXPECT_DOUBLE_EQ(params.loop_closure_params_.max_position_diff, 6.0);
  EXPECT_TRUE(params.loop_closure_params_.multicamera_enabled);
  EXPECT_TRUE(params.loop_closure_params_.multicamera_diagnostics);
  EXPECT_EQ(params.loop_closure_params_.multicamera_matching_threads, 4);
  EXPECT_TRUE(params.debug_mode_);
  EXPECT_TRUE(params.lc_diagnostic_);
  EXPECT_TRUE(params.loopClosureDiagnosticsEnabled());
}

TEST(LoopClosureParameters, RequireDebugAndLoopClosureDiagnosticSwitches) {
  Parameters params;

  params.debug_mode_ = true;
  EXPECT_FALSE(params.loopClosureDiagnosticsEnabled());

  params.debug_mode_ = false;
  params.lc_diagnostic_ = true;
  EXPECT_FALSE(params.loopClosureDiagnosticsEnabled());

  params.debug_mode_ = true;
  EXPECT_TRUE(params.loopClosureDiagnosticsEnabled());
}
