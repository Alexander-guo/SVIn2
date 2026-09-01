#include "pose_graph/Parameters.h"

#include <glog/logging.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <boost/filesystem.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "utils/Utils.h"

Parameters::Parameters() {
  // default values
  tic_ = Eigen::Vector3d::Zero();
  qic_ = Eigen::Matrix3d::Identity();
  health_params_.min_tracked_keypoints = 8;
  health_params_.kf_wait_time = 0.5;
  health_params_.consecutive_keyframes = 5;
  debug_mode_ = false;
  lc_diagnostic_ = false;
  image_delay_ = 0.0;

  // Enable loop closure by default
  loop_closure_params_.enabled = true;
  loop_closure_params_.min_correspondences = 25;
  loop_closure_params_.pnp_reprojection_thresh = 20.0;
  loop_closure_params_.pnp_ransac_iterations = 100;
  loop_closure_params_.keyframe_queue_size = 5;
  loop_closure_params_.max_yaw_diff = 25.0;
  loop_closure_params_.max_position_diff = 15.0;
  resize_factor_ = 1.0;
}

void Parameters::loadParameters(const std::string& config_file) {
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);

  if (!fsSettings.isOpened()) {
    LOG(FATAL) << "ERROR: Wrong path to settings" << std::endl;
  }

  camera_visual_size_ = fsSettings["visualize_camera_size"];

  std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("pose_graph");

  vocabulary_file_ = pkg_share_dir + "/Vocabulary/brief_k10L6.bin";
  std::cout << "vocabulary_file" << vocabulary_file_ << std::endl;

  brief_pattern_file_ = pkg_share_dir + "/Vocabulary/brief_pattern.yml";

  if (fsSettings["loop_closure_params"]["enable"].isInt()) {
    loop_closure_params_.enabled = static_cast<int>(fsSettings["loop_closure_params"]["enable"]);
    LOG(INFO) << "loop_closure_params.enable: " << loop_closure_params_.enabled;

    if (fsSettings["loop_closure_params"]["min_correspondences"].isInt() ||
        fsSettings["loop_closure_params"]["min_correspondences"].isReal()) {
      loop_closure_params_.min_correspondences =
          static_cast<int>(fsSettings["loop_closure_params"]["min_correspondences"]);
      LOG(INFO) << "Num of matched keypoints for Loop Detection:" << loop_closure_params_.min_correspondences;
    }

    if (fsSettings["loop_closure_params"]["pnp_reprojection_threshold"].isReal() ||
        fsSettings["loop_closure_params"]["pnp_reprojection_threshold"].isInt()) {
      loop_closure_params_.pnp_reprojection_thresh =
          static_cast<double>(fsSettings["loop_closure_params"]["pnp_reprojection_threshold"]);
      LOG(INFO) << "PnP reprojection threshold: " << loop_closure_params_.pnp_reprojection_thresh;
    }

    if (fsSettings["loop_closure_params"]["pnp_ransac_iterations"].isInt() ||
        fsSettings["loop_closure_params"]["pnp_ransac_iterations"].isReal()) {
      loop_closure_params_.pnp_ransac_iterations =
          static_cast<int>(fsSettings["loop_closure_params"]["pnp_ransac_iterations"]);
      LOG(INFO) << "PnP ransac iterations: " << loop_closure_params_.pnp_ransac_iterations;
    }
  }

  if (fsSettings["loop_closure_params"]["keyframe_queue"].isInt()) {
    const int keyframe_queue_size = static_cast<int>(fsSettings["loop_closure_params"]["keyframe_queue"]);
    if (keyframe_queue_size == -1 || keyframe_queue_size > 0) {
      loop_closure_params_.keyframe_queue_size = keyframe_queue_size;
    } else {
      LOG(WARNING) << "loop_closure_params.keyframe_queue must be -1 or greater than 0; keeping existing value "
                   << loop_closure_params_.keyframe_queue_size;
    }
    LOG(INFO) << "Loop-closure keyframe queue size: " << loop_closure_params_.keyframe_queue_size;
  }

  if (fsSettings["loop_closure_params"]["max_yaw_diff"].isReal() ||
      fsSettings["loop_closure_params"]["max_yaw_diff"].isInt()) {
    const double max_yaw_diff = static_cast<double>(fsSettings["loop_closure_params"]["max_yaw_diff"]);
    if (std::isfinite(max_yaw_diff) && max_yaw_diff > 0.0) {
      loop_closure_params_.max_yaw_diff = max_yaw_diff;
    } else {
      LOG(WARNING) << "loop_closure_params.max_yaw_diff must be greater than 0; keeping existing value "
                   << loop_closure_params_.max_yaw_diff;
    }
    LOG(INFO) << "Maximum loop-closure yaw difference: " << loop_closure_params_.max_yaw_diff << " deg";
  }

  if (fsSettings["loop_closure_params"]["max_position_diff"].isReal() ||
      fsSettings["loop_closure_params"]["max_position_diff"].isInt()) {
    const double max_position_diff = static_cast<double>(fsSettings["loop_closure_params"]["max_position_diff"]);
    if (std::isfinite(max_position_diff) && max_position_diff > 0.0) {
      loop_closure_params_.max_position_diff = max_position_diff;
    } else {
      LOG(WARNING) << "loop_closure_params.max_position_diff must be greater than 0; keeping existing value "
                   << loop_closure_params_.max_position_diff;
    }
    LOG(INFO) << "Maximum loop-closure position difference: " << loop_closure_params_.max_position_diff << " m";
  }

  // Determine source directory of this file
  std::filesystem::path this_file(__FILE__);
  std::filesystem::path package_src_dir = this_file.parent_path().parent_path().parent_path();  // Go up from src/pose_graph/

  output_path_ = (package_src_dir).string();

  if (fsSettings["output_params"]["output_dir"].isString()) {
    try {
      if (!std::filesystem::exists(output_path_)) {
        std::filesystem::create_directory(output_path_);
      }
    } catch (std::filesystem::filesystem_error& e) {
      LOG(ERROR) << e.what();
      output_path_ = "/tmp";
    }
    debug_output_path_ = output_path_ + "/debug_output";
    LOG(INFO) << "Output folder: " << output_path_;
    std::cout << "Output folder: " << output_path_ << std::endl;
    if (fsSettings["output_params"]["debug"].isInt()) {
      debug_mode_ = static_cast<int>(fsSettings["output_params"]["debug"]);
    }
  }

  const cv::FileNode lc_diagnostic_node = fsSettings["output_params"]["lc_diagnostic"];
  if (lc_diagnostic_node.isInt()) {
    lc_diagnostic_ = static_cast<int>(lc_diagnostic_node) != 0;
  } else if (lc_diagnostic_node.isString()) {
    const std::string value = static_cast<std::string>(lc_diagnostic_node);
    if (value == "true" || value == "True" || value == "TRUE") {
      lc_diagnostic_ = true;
    } else if (value == "false" || value == "False" || value == "FALSE") {
      lc_diagnostic_ = false;
    } else {
      LOG(WARNING) << "output_params.lc_diagnostic must be true or false; keeping disabled";
    }
  }
  LOG(INFO) << "Loop-closure diagnostics enabled: " << loopClosureDiagnosticsEnabled();

  if (fsSettings["global_map_params"]["enable"].isInt()) {
    global_mapping_params_.enabled = static_cast<int>(fsSettings["global_map_params"]["enable"]);
    LOG(INFO) << "global_map.enable: " << global_mapping_params_.enabled;

    if (fsSettings["global_map_params"]["min_landmark_quality"].isInt() ||
        fsSettings["global_map_params"]["min_landmark_quality"].isReal()) {
      global_mapping_params_.min_lmk_quality =
          static_cast<double>(fsSettings["global_map_params"]["min_landmark_quality"]);
      LOG(INFO) << "Minimum landmark quality to add to global map:" << global_mapping_params_.min_lmk_quality;
    }
  }

  if (fsSettings["health"]["enable"].isInt()) {
    health_params_.enabled = static_cast<int>(fsSettings["health"]["enable"]);
    LOG(INFO) << "health_params_.enable: " << health_params_.enabled;

    if (fsSettings["health"]["min_keypoints"].isInt() || fsSettings["health"]["min_keypoints"].isReal()) {
      health_params_.min_tracked_keypoints = static_cast<int>(fsSettings["health"]["min_keypoints"]);
      LOG(INFO) << "health_params_.min_tracked_keypoints :" << health_params_.min_tracked_keypoints;
    }

    if (fsSettings["health"]["consecutive_keyframes"].isReal() ||
        fsSettings["health"]["consecutive_keyframes"].isInt()) {
      health_params_.consecutive_keyframes = static_cast<double>(fsSettings["health"]["consecutive_keyframes"]);
      LOG(INFO) << "health_params_.consecutive keyframes check " << health_params_.consecutive_keyframes;
    }

    if (fsSettings["health"]["keyframe_wait_time"].isInt() || fsSettings["health"]["keyframe_wait_time"].isReal()) {
      health_params_.kf_wait_time = static_cast<float>(fsSettings["health"]["keyframe_wait_time"]);
      LOG(INFO) << "health_params_.keyframe wait time: " << health_params_.kf_wait_time;
    }

    if (fsSettings["health"]["kps_per_quadrant"].isInt() || fsSettings["health"]["kps_per_quadrant"].isReal()) {
      health_params_.kps_per_quadrant = static_cast<int>(fsSettings["health"]["kps_per_quadrant"]);
      LOG(INFO) << "health_params_.keypoints_per_quadrant: " << health_params_.kps_per_quadrant;
    }
  }

  fast_relocalization_ = fsSettings["fast_relocalization"];

  std::string results_path = output_path_ + "/svin_results/";
  if (!std::filesystem::exists(results_path)) {
    std::filesystem::create_directory(results_path);
  }
  svin_traj_path_ = results_path;

  // Read config file parameters
  if (fsSettings["resizeFactor"].isReal() || fsSettings["resizeFactor"].isInt()) {
    resize_factor_ = static_cast<double>(fsSettings["resizeFactor"]);
  }

  bool calibration_valid = getCalibrationViaConfig(camera_calibrations_, fsSettings["cameras"]);
  if (!calibration_valid || camera_calibrations_.empty()) {
    LOG(FATAL) << "Calibration not found in config file. Please provide calibration in config file.";
  }
  camera_calibration_ = camera_calibrations_.front();
  for (size_t cam_idx = 0; cam_idx < camera_calibrations_.size(); ++cam_idx) {
    std::cout << "Camera calibration [" << cam_idx << "]:" << std::endl;
    camera_calibrations_[cam_idx].print();
  }
  fsSettings.release();
}

// Get the camera calibration via the configuration file.
bool Parameters::getCalibrationViaConfig(CameraCalibration& calib, cv::FileNode camera_node) {
  std::vector<CameraCalibration, Eigen::aligned_allocator<CameraCalibration>> calibrations;
  if (!getCalibrationViaConfig(calibrations, camera_node) || calibrations.empty()) {
    return false;
  }
  calib = calibrations.front();
  return true;
}

bool Parameters::getCalibrationViaConfig(std::vector<CameraCalibration, Eigen::aligned_allocator<CameraCalibration>>& calibrations,
                                         cv::FileNode camera_node) {
  bool got_calibration = false;
  calibrations.clear();
  // first check if calibration is available in config file
  if (camera_node.isSeq() && camera_node.size() > 0) {
    calibrations.reserve(camera_node.size());
    for (size_t cam_idx = 0; cam_idx < camera_node.size(); ++cam_idx) {
      cv::FileNode camera = camera_node[cam_idx];
      CameraCalibration calib;

      if (camera.isMap() && camera["T_SC"].isSeq() && camera["image_dimension"].isSeq() &&
          camera["image_dimension"].size() == 2 && camera["distortion_coefficients"].isSeq() &&
          camera["distortion_coefficients"].size() >= 4 && camera["distortion_type"].isString() &&
          camera["projection_type"].isString() && (std::string)(camera["projection_type"]) == "pinhole" &&
          camera["focal_length"].isSeq() && camera["focal_length"].size() == 2 && camera["principal_point"].isSeq() &&
          camera["principal_point"].size() == 2) {
        LOG(INFO) << "Found pinhole calibration in configuration file for camera " << cam_idx;
      } else if (camera.isMap() && camera["T_SC"].isSeq() && camera["image_dimension"].isSeq() &&
                 camera["image_dimension"].size() == 2 && camera["projection_type"].isString() &&
                 (std::string)(camera["projection_type"]) == "double_sphere" && camera["focal_length"].isSeq() &&
                 camera["focal_length"].size() == 2 && camera["principal_point"].isSeq() &&
                 camera["principal_point"].size() == 2 && camera["xi"].isReal() && camera["alpha"].isReal()) {
        LOG(INFO) << "Found double sphere calibration in configuration file for camera " << cam_idx
                  << ", distortion types is set to 'none'.";
      } else {
        LOG(WARNING) << "Found incomplete calibration in configuration file for camera " << cam_idx
                     << ". Will not use the calibration from the configuration file.";
        calibrations.clear();
        return false;
      }

      cv::FileNode T_SC_node = camera["T_SC"];
      cv::FileNode image_dimension_node = camera["image_dimension"];
      cv::FileNode focal_length_node = camera["focal_length"];
      cv::FileNode principal_point_node = camera["principal_point"];

      cv::FileNode distortion_coefficient_node;
      if ((std::string)(camera["projection_type"]) == "double_sphere") {
        calib.projection_type_ = "double_sphere";
        calib.double_sphere_params_ << static_cast<double>(camera["xi"]), static_cast<double>(camera["alpha"]);
        distortion_coefficient_node = cv::FileNode();
        calib.distortion_type_ = "none";
      } else {
        calib.projection_type_ = "pinhole";
        distortion_coefficient_node = camera["distortion_coefficients"];
        calib.distortion_type_ = (std::string)(camera["distortion_type"]);
      }

      calib.T_imu_cam0_ << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3], T_SC_node[4], T_SC_node[5],
          T_SC_node[6], T_SC_node[7], T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11], T_SC_node[12],
          T_SC_node[13], T_SC_node[14], T_SC_node[15];

      calib.image_dimension_ << image_dimension_node[0], image_dimension_node[1];
      calib.image_dimension_(0) = static_cast<int>(static_cast<double>(calib.image_dimension_(0)) * resize_factor_);
      calib.image_dimension_(1) = static_cast<int>(static_cast<double>(calib.image_dimension_(1)) * resize_factor_);
      LOG(WARNING) << "Resize Factor: " << resize_factor_;
      LOG(WARNING) << calib.image_dimension_;

      calib.distortion_coefficients_ = cv::Mat::zeros(distortion_coefficient_node.size(), 1, CV_64F);
      for (size_t i = 0; i < distortion_coefficient_node.size(); ++i) {
        calib.distortion_coefficients_.at<double>(i, 0) = distortion_coefficient_node[i];
      }

      calib.focal_length_ << focal_length_node[0], focal_length_node[1];
      calib.focal_length_ = calib.focal_length_ * resize_factor_;

      calib.principal_point_ << principal_point_node[0], principal_point_node[1];
      calib.principal_point_ = calib.principal_point_ * resize_factor_;

      calibrations.push_back(calib);
      got_calibration = true;
    }
  } else {
    LOG(INFO) << "Did not find a calibration in the configuration file.";
  }
  return got_calibration;
}
