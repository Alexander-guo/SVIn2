
#include <glog/logging.h>

#include <filesystem>
#include <string>
#include <fstream>
#include <rclcpp/rclcpp.hpp>

#include "pose_graph/LoopClosure.h"
#include "pose_graph/Parameters.h"
#include "pose_graph/Publisher.h"
#include "pose_graph/Subscriber.h"
#include "utils/Utils.h"

void setupLoopClosureDebugOutputs(const std::string& base_path, bool initialize_diagnostic_csvs) {
  std::string output_dir = base_path + "/loop_candidates/";
  if (!std::filesystem::is_directory(output_dir) || !std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
    std::filesystem::remove_all(entry.path());
  }

  output_dir = base_path + "/descriptor_matched/";
  if (!std::filesystem::is_directory(output_dir) || !std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
    std::filesystem::remove_all(entry.path());
  }

  output_dir = base_path + "/pnp_verified/";
  if (!std::filesystem::is_directory(output_dir) || !std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
    std::filesystem::remove_all(entry.path());
  }
  std::filesystem::create_directories(output_dir + "/passed");
  std::filesystem::create_directories(output_dir + "/rejected");

  output_dir = base_path + "/loop_closure/";
  if (!std::filesystem::is_directory(output_dir) || !std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
    std::filesystem::remove_all(entry.path());
  }

  output_dir = base_path + "/geometric_verification/";
  if (!std::filesystem::is_directory(output_dir) || !std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
    std::filesystem::remove_all(entry.path());
  }

  std::string loop_closure_file = base_path + "/loop_closure.txt";
  if (std::filesystem::exists(loop_closure_file)) {
    std::filesystem::remove(loop_closure_file);
  }
  std::ofstream loop_path_file(loop_closure_file, std::ios::out);
  loop_path_file << "cur_kf_id"
                 << " "
                 << "cur_kf_ts"
                 << " "
                 << "matched_kf_id"
                 << " "
                 << "matched_kf_ts"
                 << " "
                 << "relative_tx"
                 << " "
                 << "relative_ty"
                 << " "
                 << "relative_tz"
                 << " "
                 << "relative_yaw"
                 << " "
                 << "relative_pitch"
                 << " "
                 << "relative_roll" << std::endl;
  loop_path_file.close();

  std::string loop_funnel_file = base_path + "/loop_closure_funnel.csv";
  if (std::filesystem::exists(loop_funnel_file)) {
    std::filesystem::remove(loop_funnel_file);
  }
  std::string dbow_funnel_file = base_path + "/loop_closure_dbow_funnel.csv";
  if (std::filesystem::exists(dbow_funnel_file)) {
    std::filesystem::remove(dbow_funnel_file);
  }

  if (initialize_diagnostic_csvs) {
    std::ofstream loop_funnel_stream(loop_funnel_file, std::ios::out);
    loop_funnel_stream
        << "current_kf_id,current_timestamp,candidate_kf_id,candidate_timestamp,camera_model,pnp_model,tracked_points,"
           "candidate_keypoints,descriptor_matches,brief_hamming_threshold,min_correspondences,pnp_attempted,"
           "pnp_solver_succeeded,pnp_exception,"
           "pnp_inliers,pnp_iterations,pnp_reprojection_threshold,relative_yaw_deg,relative_translation_m,max_yaw_deg,"
           "max_position_m,yaw_gate_passed,position_gate_passed,accepted,rejection_reason\n";
    loop_funnel_stream.close();

    std::ofstream dbow_funnel_stream(dbow_funnel_file, std::ios::out);
    dbow_funnel_stream << "current_kf_id,current_timestamp,min_neighbor_score,score_threshold,query_max_id,"
                          "query_result_count,passing_result_count,best_passing_candidate_id,best_passing_score,"
                          "selected_candidate_id,temporal_gap,decision\n";
    dbow_funnel_stream.close();
  }
}

void setupGeneralDebugOutputs(const std::string& base_path) {
  std::filesystem::create_directories(base_path);
  std::string switch_info_file = base_path + "/switch_info.txt";
  if (std::filesystem::exists(switch_info_file)) {
    std::filesystem::remove(switch_info_file);
  }
  std::ofstream switch_info_file_stream(switch_info_file, std::ios::out);
  switch_info_file_stream << "type"
                          << " "
                          << "vio_stamp"
                          << " "
                          << "prim_stamp"
                          << " "
                          << "uber_stamp" << std::endl;
  switch_info_file_stream.close();
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<rclcpp::Node>("pose_graph_node");

  // Initialize Google's logging library.
  google::InitGoogleLogging(argv[0]);

  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;

  // read parameters
  std::string config_file;

  node->declare_parameter<std::string>("config_file", "");
  node->get_parameter<std::string>("config_file", config_file);

  if (config_file.empty()) {
    LOG(ERROR) << "Config file not provided";
    return EXIT_FAILURE;
  }

  Parameters params;
  params.loadParameters(config_file);

  if (params.debug_mode_) {
    setupGeneralDebugOutputs(params.debug_output_path_);
    setupLoopClosureDebugOutputs(params.debug_output_path_, params.loopClosureDiagnosticsEnabled());
    FLAGS_v = 0;
  }

  auto subscriber = std::make_unique<Subscriber>(node, params);
  auto loop_closure = std::make_unique<LoopClosure>(params);
  auto publisher = std::make_unique<Publisher>(node, params.debug_mode_);

  loop_closure->setKeyframePoseCallback(
      std::bind(&Publisher::publishKeyframePath, publisher.get(), std::placeholders::_1, std::placeholders::_2));
  loop_closure->setLoopClosureCallback(
      std::bind(&Publisher::publishLoopClosurePath, publisher.get(), std::placeholders::_1, std::placeholders::_2));

  if (params.debug_mode_) {
    loop_closure->setPrimitivePublishCallback(
        std::bind(&Publisher::publishPrimitiveEstimator, publisher.get(), std::placeholders::_1));
  }

  subscriber->registerKeyframeCallback(
      std::bind(&LoopClosure::fillKeyframeTrackingQueue, loop_closure.get(), std::placeholders::_1));

  if (params.health_params_.enabled) {
    subscriber->registerPrimitiveEstimatorCallback(
        std::bind(&LoopClosure::fillPrimitiveEstimatorBuffer, loop_closure.get(), std::placeholders::_1));
  }

  rclcpp::TimerBase::SharedPtr timer;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr pointcloud_service;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_trajectory_service;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_kf_observations_service;

  // Save Trajectory service
  save_trajectory_service = node->create_service<std_srvs::srv::Trigger>(
      "save_trajectory",
      [&publisher, &params](const std::shared_ptr<rmw_request_id_t> /*req_header*/,
                           const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                           const std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        try {
          std::string save_path = params.svin_traj_path_ + "svin_" + Utils::getTimeStr();
          publisher->saveTrajectory(save_path);
          response->success = true;
          response->message = "Trajectory saved successfully!";
        } catch (const std::exception& e) {
          response->success = false;
          response->message = std::string("Failed to save trajectory: ") + e.what();
        }
      });
  
  // Save Keyframe Observations service (also saves keyframes)
  save_kf_observations_service = node->create_service<std_srvs::srv::Trigger>(
      "save_keyframe_observations",
      [&loop_closure, &publisher, &params](const std::shared_ptr<rmw_request_id_t> /*req_header*/,
                               const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                               const std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        try {
          // Ensure base directory exists
          std::string base_dir = params.output_path_;
          if (!std::filesystem::exists(base_dir)) {
            std::filesystem::create_directories(base_dir);
          }
          
          // Save keyframe observations
          std::string obs_subdir = base_dir + "/keyframe_observations";
          if (!std::filesystem::exists(obs_subdir)) {
            std::filesystem::create_directories(obs_subdir);
          }
          std::string obs_filename = obs_subdir + "/keyframe_observations_" + Utils::getTimeStr() + ".txt";
          bool obs_ok = loop_closure->saveKeyframeObservations(obs_filename);
          
          // Save keyframes
          std::string kf_subdir = base_dir + "/keyframes";
          if (!std::filesystem::exists(kf_subdir)) {
            std::filesystem::create_directories(kf_subdir);
          }
          std::string kf_filename = kf_subdir + "/keyframes_" + Utils::getTimeStr();
          std::vector<KeyframeDump> dump;
          loop_closure->getKeyframesDump(dump);
          bool kf_ok = publisher->saveKeyframes(kf_filename, dump);
          
          if (!obs_ok || !kf_ok) {
            response->success = false;
            response->message = "Failed to save some files. obs=" + std::string(obs_ok ? "ok" : "fail") + 
                               " kf=" + std::string(kf_ok ? "ok" : "fail");
            return;
          }
          response->success = true;
          response->message = "Observations: " + obs_filename + "\nKeyframes: " + kf_filename;
        } catch (const std::exception& e) {
          response->success = false;
          response->message = std::string("Exception while saving: ") + e.what();
        }
      });
  

  if (params.global_mapping_params_.enabled) {
    publisher->setGlobalPointCloudFunction(
        std::bind(&LoopClosure::getGlobalMap, loop_closure.get(), std::placeholders::_1));
    subscriber->registerImageCallback(
        std::bind(&LoopClosure::fillImageQueue, loop_closure.get(), std::placeholders::_1));
    timer = node->create_wall_timer(std::chrono::seconds(5),
                                    std::bind(&Publisher::updatePublishGlobalMap, publisher.get()));
    pointcloud_service = node->create_service<std_srvs::srv::Trigger>("save_pointcloud",
                                                                      std::bind(&Publisher::savePointCloud,
                                                                                publisher.get(),
                                                                                std::placeholders::_1,
                                                                                std::placeholders::_2,
                                                                                std::placeholders::_3));
  }

  auto process_thread = std::thread(&LoopClosure::run, loop_closure.get());

  rclcpp::Time last_print_time = node->now();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  while (rclcpp::ok()) {
    executor.spin_once();
    if ((node->now() - last_print_time).seconds() > 10.0 && !subscriber->isFrozen()) {
      last_print_time = node->now();
      LOG(INFO) << utils::Statistics::Print();
    }
  }

  // After rclcpp::ok() is false (shutdown signal received)
  std::string save_path = params.svin_traj_path_ + "svin_" + Utils::getTimeStr();
  publisher->saveTrajectory(save_path);
  LOG(INFO) << "Shutting down threads...";
  loop_closure->shutdown();

  return EXIT_SUCCESS;
}
