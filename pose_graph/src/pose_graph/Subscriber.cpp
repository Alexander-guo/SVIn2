#include "pose_graph/Subscriber.h"

#include <memory>
#include <opencv2/core/eigen.hpp>
#include <utility>
#include <vector>

#include "utils/Statistics.h"
#include "utils/Timer.h"
#include "utils/Utils.h"
#include "utils/UtilsOpenCV.h"

Subscriber::Subscriber(std::shared_ptr<rclcpp::Node> node, Parameters& params) : node_(node), params_(params) {
  // TODO(bjoshi): pass as params from roslaunch file
  kf_image_topic_ = "/keyframe_imageL";
  kf_pose_topic_ = "/keyframe_pose";
  kf_points_topic_ = "/keyframe_points";
  svin_reloc_odom_topic_ = "/relocalization_odometry";
  svin_health_topic_ = "/svin_health";
  primitive_estimator_topic_ = "/aqua_primitive_estimator/odometry";
  last_image_time_ = -1;

  rclcpp::QoS qos(10);
  auto rmw_qos_profile = qos.get_rmw_qos_profile();

  keyframe_points_subscriber_.subscribe(node_, kf_points_topic_, rmw_qos_profile);
  keyframe_pose_subscriber_.subscribe(node_, kf_pose_topic_, rmw_qos_profile);
  svin_health_subscriber_.subscribe(node_, svin_health_topic_, rmw_qos_profile);

  static constexpr size_t kMaxKeyframeSynchronizerQueueSize = 10u;
  keyframe_image_subscriber_.subscribe(node_, kf_image_topic_, rmw_qos_profile);
  if (params_.camera_calibrations_.size() == 2) {
    keyframe_image_camera1_subscriber_.subscribe(node_, "/keyframe_image_1", rmw_qos_profile);
    sync_multicamera_keyframe_ =
        std::make_unique<message_filters::Synchronizer<multicamera_keyframe_sync_policy>>(
            multicamera_keyframe_sync_policy(kMaxKeyframeSynchronizerQueueSize),
            keyframe_image_subscriber_,
            keyframe_image_camera1_subscriber_,
            keyframe_pose_subscriber_,
            keyframe_points_subscriber_,
            svin_health_subscriber_);
    sync_multicamera_keyframe_->registerCallback(std::bind(&Subscriber::multicameraKeyframeCallback,
                                                            this,
                                                            std::placeholders::_1,
                                                            std::placeholders::_2,
                                                            std::placeholders::_3,
                                                            std::placeholders::_4,
                                                            std::placeholders::_5));
  } else {
    sync_keyframe_ = std::make_unique<message_filters::Synchronizer<keyframe_sync_policy>>(
        keyframe_sync_policy(kMaxKeyframeSynchronizerQueueSize),
        keyframe_image_subscriber_,
        keyframe_pose_subscriber_,
        keyframe_points_subscriber_,
        svin_health_subscriber_);
    sync_keyframe_->registerCallback(std::bind(&Subscriber::keyframeCallback,
                                               this,
                                               std::placeholders::_1,
                                               std::placeholders::_2,
                                               std::placeholders::_3,
                                               std::placeholders::_4));
  }

  if (params_.global_mapping_params_.enabled) {
    sub_orig_images_.reserve(params_.camera_calibrations_.size());
    for (size_t camera_index = 0; camera_index < params_.camera_calibrations_.size(); ++camera_index) {
      const std::string topic = "/cam" + std::to_string(camera_index) + "/image_raw";
      sub_orig_images_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
          topic,
          100,
          [this, camera_index](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
            imageCallback(camera_index, msg);
          }));
    }
  }
  if (params_.health_params_.enabled) {
    sub_primitive_estimator_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        primitive_estimator_topic_,
        100,
        std::bind(&Subscriber::primitiveEstimatorCallback, this, std::placeholders::_1));
  }

  // Watchdog: parameter and timer setup (uses steady clock)
  try {
    if (!node_->has_parameter("freeze_timeout_sec")) {
      node_->declare_parameter<double>("freeze_timeout_sec", 60.0);
    }
    node_->get_parameter("freeze_timeout_sec", freeze_timeout_sec_);
  } catch (const std::exception&) {
    freeze_timeout_sec_ = 240.0;
  }
  last_keyframe_tp_ = std::chrono::steady_clock::now();
  frozen_ = false;
  seen_first_keyframe_ = false;
  using namespace std::chrono_literals;  // NOLINT
  watchdog_timer_ = node_->create_wall_timer(500ms, std::bind(&Subscriber::watchdogTick, this));
}

void Subscriber::primitiveEstimatorCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
  if (!primitive_estimator_callback_ && params_.health_params_.enabled) {
    LOG_EVERY_N(ERROR, 100) << "Primitive estimator callback not set";
  } else if (primitive_estimator_callback_) {
    Eigen::Matrix4d pose = Utils::rosPoseToMatrix(msg->pose.pose);
    cv::Mat cv_pose;
    cv::eigen2cv(pose, cv_pose);
    auto pose_with_timestamp =
        std::make_unique<std::pair<Timestamp, cv::Mat>>(std::make_pair(Utils::getHeaderStamp(msg->header), cv_pose));
    primitive_estimator_callback_(std::move(pose_with_timestamp));
  }
}

void Subscriber::imageCallback(size_t camera_index, const sensor_msgs::msg::Image::ConstSharedPtr image_msg) {
  cv::Mat image = UtilsOpenCV::readRosImage(image_msg, false);
  auto image_with_timestamp =
      std::make_unique<std::pair<Timestamp, cv::Mat>>(std::make_pair(Utils::getHeaderStamp(image_msg->header), image));
  // raw image callback is not compulsory
  if (raw_image_callback_) {
    raw_image_callback_(camera_index, std::move(image_with_timestamp));
  } else {
    LOG_EVERY_N(WARNING, 100) << "Raw image callback not set";
  }
}

nav_msgs::msg::Odometry::ConstSharedPtr Subscriber::getPrimitiveEstimatorPose(const int64_t& ros_stamp) {
  nav_msgs::msg::Odometry::ConstSharedPtr prim_estimator_pose = nullptr;
  // 25ms sync period while using primitive estimator publish rate of 20Hz
  while (!prim_estimator_odom_buffer_.empty() &&
         Utils::getHeaderStamp(prim_estimator_odom_buffer_.front()->header) < (ros_stamp - 25000000)) {
    prim_estimator_pose = prim_estimator_odom_buffer_.front();
    prim_estimator_odom_buffer_.pop();
  }

  if (!prim_estimator_odom_buffer_.empty()) {
    prim_estimator_pose = prim_estimator_odom_buffer_.front();
    prim_estimator_odom_buffer_.pop();
  }

  return prim_estimator_pose;
}

void Subscriber::getPrimitiveEstimatorPoses(const int64_t& ros_stamp,
                                            std::vector<nav_msgs::msg::Odometry::ConstSharedPtr>& poses) {
  nav_msgs::msg::Odometry::ConstSharedPtr prim_estimator_pose = nullptr;
  // 25ms sync period while using primitive estimator publish rate of 20Hz

  while (!prim_estimator_odom_buffer_.empty() &&
         Utils::getHeaderStamp(prim_estimator_odom_buffer_.front()->header) < (ros_stamp - 25000000)) {
    prim_estimator_odom_buffer_.pop();
  }

  while (!prim_estimator_odom_buffer_.empty()) {
    poses.emplace_back(prim_estimator_odom_buffer_.front());
    prim_estimator_odom_buffer_.pop();
  }
}

void Subscriber::keyframeCallback(const sensor_msgs::msg::Image::ConstSharedPtr kf_image_msg,
                                  const nav_msgs::msg::Odometry::ConstSharedPtr kf_odom,
                                  const sensor_msgs::msg::PointCloud::ConstSharedPtr kf_points,
                                  const okvis_ros::msg::SvinHealth::ConstSharedPtr svin_health) {
  processKeyframe({kf_image_msg}, kf_odom, kf_points, svin_health);
}

void Subscriber::multicameraKeyframeCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr kf_image0_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr kf_image1_msg,
    const nav_msgs::msg::Odometry::ConstSharedPtr kf_odom,
    const sensor_msgs::msg::PointCloud::ConstSharedPtr kf_points,
    const okvis_ros::msg::SvinHealth::ConstSharedPtr svin_health) {
  processKeyframe({kf_image0_msg, kf_image1_msg}, kf_odom, kf_points, svin_health);
}

void Subscriber::processKeyframe(
    const std::vector<sensor_msgs::msg::Image::ConstSharedPtr>& kf_image_msgs,
    const nav_msgs::msg::Odometry::ConstSharedPtr& kf_odom,
    const sensor_msgs::msg::PointCloud::ConstSharedPtr& kf_points,
    const okvis_ros::msg::SvinHealth::ConstSharedPtr& svin_health) {
  if (frozen_) {
    return;  // frozen: ignore keyframes
  }

  // Update last keyframe time for watchdog (steady clock)
  last_keyframe_tp_ = std::chrono::steady_clock::now();
  seen_first_keyframe_ = true;

  TrackingInfo tracking_info(Utils::getHeaderStamp(kf_odom->header),
                             svin_health->num_tracked_kps,
                             svin_health->new_kps,
                             svin_health->kps_per_quadrant,
                             svin_health->covisibilities,
                             svin_health->response_strengths,
                             svin_health->quality);

  Eigen::Vector3d translation =
      Eigen::Vector3d(kf_odom->pose.pose.position.x, kf_odom->pose.pose.position.y, kf_odom->pose.pose.position.z);
  Eigen::Matrix3d rotation = Eigen::Quaterniond(kf_odom->pose.pose.orientation.w,
                                                kf_odom->pose.pose.orientation.x,
                                                kf_odom->pose.pose.orientation.y,
                                                kf_odom->pose.pose.orientation.z)
                                 .toRotationMatrix();

  std::vector<cv::Point3f> keyframe_points;
  std::vector<float> point_qualities;
  std::vector<cv::KeyPoint> keypoint_observations;
  std::vector<Eigen::Vector3i> point_ids;
  std::vector<std::vector<int64_t>> kf_covisibilities;

  int64_t keyframe_index = -1;
  std::vector<cv::Mat> kf_images;
  kf_images.reserve(kf_image_msgs.size());
  for (const auto& image_msg : kf_image_msgs) {
    kf_images.push_back(UtilsOpenCV::readRosImage(image_msg));
  }
  std::vector<size_t> camera_indices;

  for (unsigned int i = 0; i < kf_points->points.size(); i++) {
    keyframe_index = kf_points->channels[i].values[4];

    cv::Point3f point_3d(kf_points->points[i].x, kf_points->points[i].y, kf_points->points[i].z);
    keyframe_points.push_back(point_3d);
    point_qualities.push_back(kf_points->channels[i].values[3]);

    // @Reloc landmarkId, poseId or MultiFrameId,  keypointIdx
    Eigen::Vector3i point_id(
        kf_points->channels[i].values[0], kf_points->channels[i].values[1], kf_points->channels[i].values[2]);
    point_ids.push_back(point_id);

    cv::KeyPoint p_2d_uv;
    p_2d_uv.pt.x = kf_points->channels[i].values[5];
    p_2d_uv.pt.y = kf_points->channels[i].values[6];
    p_2d_uv.size = kf_points->channels[i].values[7];
    p_2d_uv.angle = kf_points->channels[i].values[8];
    p_2d_uv.octave = kf_points->channels[i].values[9];
    p_2d_uv.response = kf_points->channels[i].values[10];
    p_2d_uv.class_id = kf_points->channels[i].values[11];

    keypoint_observations.push_back(p_2d_uv);

    const bool has_camera_index = params_.camera_calibrations_.size() > 1 &&
                                  kf_points->channels[i].values.size() >= 13;
    const size_t camera_index = has_camera_index
                                    ? static_cast<size_t>(kf_points->channels[i].values[12])
                                    : 0u;
    if (camera_index >= params_.camera_calibrations_.size()) {
      LOG(WARNING) << "Skipping keyframe observation with invalid camera index " << camera_index;
      keyframe_points.pop_back();
      point_qualities.pop_back();
      point_ids.pop_back();
      keypoint_observations.pop_back();
      continue;
    }
    if (camera_index >= kf_images.size()) {
      keyframe_points.pop_back();
      point_qualities.pop_back();
      point_ids.pop_back();
      keypoint_observations.pop_back();
      continue;
    }
    camera_indices.push_back(camera_index);

    std::vector<int64_t> covisible_kfs;
    const size_t covisibility_begin = has_camera_index ? 13u : 12u;
    for (size_t sz = covisibility_begin; sz < kf_points->channels[i].values.size(); sz++) {
      int observed_kf_index = kf_points->channels[i].values[sz];
      if (observed_kf_index != keyframe_index) {
        covisible_kfs.push_back(observed_kf_index);
      }
    }
    kf_covisibilities.push_back(covisible_kfs);
  }

  if (keyframe_index != -1) {
    std::unique_ptr<KeyframeInfo> keyframe_info = std::make_unique<KeyframeInfo>(keyframe_index,
                                                                                 kf_images,
                                                                                 translation,
                                                                                 rotation,
                                                                                 tracking_info,
                                                                                 keyframe_points,
                                                                                 point_qualities,
                                                                                 keypoint_observations,
                                                                                 camera_indices,
                                                                                 point_ids,
                                                                                 kf_covisibilities);

    keyframe_callback_(std::move(keyframe_info));
  } else {
    LOG(WARNING) << "Skipping keyframe. Does not contain any triangulated points.";
  }
}

// Watchdog tick: freeze when keyframe input stops longer than threshold
void Subscriber::watchdogTick() {
  if (frozen_) return;
  // Do not start watchdog until keyframe stream has been seen at least once
  if (!seen_first_keyframe_) return;
  const auto now = std::chrono::steady_clock::now();
  const double dt_kf = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_keyframe_tp_).count();
  if (dt_kf > freeze_timeout_sec_) {
    RCLCPP_ERROR(node_->get_logger(),
                 "PoseGraph watchdog: No keyframe data for %.2fs (timeout=%.2fs). Freezing node.",
                 dt_kf, freeze_timeout_sec_);
    frozen_ = true;
  }
}
