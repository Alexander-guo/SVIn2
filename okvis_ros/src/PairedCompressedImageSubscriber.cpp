/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the conditions in the project
 *  license are met.
 *********************************************************************************/

#include <okvis/PairedCompressedImageSubscriber.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <utility>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

namespace okvis {

PairedCompressedImageSubscriber::PairedCompressedImageSubscriber(
    const std::shared_ptr<rclcpp::Node>& node,
    const std::string& camera0Topic,
    const std::string& camera1Topic,
    std::size_t queueSize,
    ImageCallback imageCallback,
    bool logCounters)
    : node_(node),
      imageCallback_(std::move(imageCallback)),
      logCounters_(logCounters) {
  const auto qos = rclcpp::QoS(rclcpp::KeepLast(
      std::max<std::size_t>(queueSize, static_cast<std::size_t>(1))));
  const auto qosProfile = qos.get_rmw_qos_profile();
  camera0Subscriber_.subscribe(node_.get(), camera0Topic, qosProfile);
  camera1Subscriber_.subscribe(node_.get(), camera1Topic, qosProfile);

  // These count messages delivered to message_filters by ROS. They do not and
  // cannot claim to observe losses before this subscription (for example DDS
  // or publisher-side drops).
  camera0Subscriber_.registerCallback(
      &PairedCompressedImageSubscriber::camera0Received, this);
  camera1Subscriber_.registerCallback(
      &PairedCompressedImageSubscriber::camera1Received, this);

  synchronizer_ = std::make_unique<Synchronizer>(
      camera0Subscriber_, camera1Subscriber_,
      static_cast<uint32_t>(std::max<std::size_t>(queueSize, 1)));
  synchronizer_->registerCallback(
      std::bind(&PairedCompressedImageSubscriber::pairedCallback, this,
                std::placeholders::_1, std::placeholders::_2));
  synchronizer_->registerDropCallback(
      &PairedCompressedImageSubscriber::droppedTupleCallback, this);

  if (logCounters_) {
    diagnosticsTimer_ = node_->create_wall_timer(
        std::chrono::seconds(10),
        [this]() { this->logCounters("periodic"); });
  }

  RCLCPP_INFO(node_->get_logger(),
              "Using exact-time paired compressed camera input: [%s, %s] "
              "(queue size %zu)",
              camera0Topic.c_str(), camera1Topic.c_str(), queueSize);
}

PairedCompressedImageSubscriber::~PairedCompressedImageSubscriber() {
  if (diagnosticsTimer_) {
    diagnosticsTimer_->cancel();
  }
  if (logCounters_) {
    logCounters("final");
  }
}

PairedCompressedImageSubscriber::Counters
PairedCompressedImageSubscriber::counters() const {
  Counters result;
  result.receivedCamera0 = receivedCamera0_.load(std::memory_order_relaxed);
  result.receivedCamera1 = receivedCamera1_.load(std::memory_order_relaxed);
  result.matchedPairs = matchedPairs_.load(std::memory_order_relaxed);
  result.decodeSuccessPairs =
      decodeSuccessPairs_.load(std::memory_order_relaxed);
  result.decodeFailedPairs =
      decodeFailedPairs_.load(std::memory_order_relaxed);
  result.admissionSuccessPairs =
      admissionSuccessPairs_.load(std::memory_order_relaxed);
  result.admissionFailedPairs =
      admissionFailedPairs_.load(std::memory_order_relaxed);
  result.syncExpiredCamera0Only =
      syncExpiredCamera0Only_.load(std::memory_order_relaxed);
  result.syncExpiredCamera1Only =
      syncExpiredCamera1Only_.load(std::memory_order_relaxed);
  result.syncExpiredOther =
      syncExpiredOther_.load(std::memory_order_relaxed);
  return result;
}

void PairedCompressedImageSubscriber::camera0Received(
    const CompressedImage::ConstSharedPtr&) {
  receivedCamera0_.fetch_add(1, std::memory_order_relaxed);
}

void PairedCompressedImageSubscriber::camera1Received(
    const CompressedImage::ConstSharedPtr&) {
  receivedCamera1_.fetch_add(1, std::memory_order_relaxed);
}

void PairedCompressedImageSubscriber::pairedCallback(
    const CompressedImage::ConstSharedPtr& camera0Message,
    const CompressedImage::ConstSharedPtr& camera1Message) {
  matchedPairs_.fetch_add(1, std::memory_order_relaxed);
  Image::SharedPtr camera0Image;
  Image::SharedPtr camera1Image;
  try {
    const auto camera0CvImage = cv_bridge::toCvCopy(camera0Message);
    const auto camera1CvImage = cv_bridge::toCvCopy(camera1Message);
    camera0Image = camera0CvImage->toImageMsg();
    camera1Image = camera1CvImage->toImageMsg();

    // Make preservation of the source stamps explicit. TimeSynchronizer only
    // invokes this callback when these stamps are exactly equal.
    camera0Image->header = camera0Message->header;
    camera1Image->header = camera1Message->header;

  } catch (const cv_bridge::Exception& exception) {
    decodeFailedPairs_.fetch_add(1, std::memory_order_relaxed);
    RCLCPP_ERROR(node_->get_logger(),
                 "Dropping synchronized compressed image pair: decode failed: %s",
                 exception.what());
    return;
  } catch (const cv::Exception& exception) {
    decodeFailedPairs_.fetch_add(1, std::memory_order_relaxed);
    RCLCPP_ERROR(node_->get_logger(),
                 "Dropping synchronized compressed image pair: OpenCV decode failed: %s",
                 exception.what());
    return;
  }

  decodeSuccessPairs_.fetch_add(1, std::memory_order_relaxed);
  // Admit the complete rig frame together. No callback from another camera
  // can be interposed between these two calls.
  const bool camera0Admitted = imageCallback_(camera0Image, 0);
  const bool camera1Admitted = imageCallback_(camera1Image, 1);
  if (camera0Admitted && camera1Admitted) {
    admissionSuccessPairs_.fetch_add(1, std::memory_order_relaxed);
  } else {
    admissionFailedPairs_.fetch_add(1, std::memory_order_relaxed);
  }
}

void PairedCompressedImageSubscriber::droppedTupleCallback(
    const Synchronizer::M0Event& camera0Event,
    const Synchronizer::M1Event& camera1Event) {
  const bool hasCamera0 = static_cast<bool>(camera0Event.getMessage());
  const bool hasCamera1 = static_cast<bool>(camera1Event.getMessage());
  if (hasCamera0 && !hasCamera1) {
    syncExpiredCamera0Only_.fetch_add(1, std::memory_order_relaxed);
  } else if (!hasCamera0 && hasCamera1) {
    syncExpiredCamera1Only_.fetch_add(1, std::memory_order_relaxed);
  } else {
    syncExpiredOther_.fetch_add(1, std::memory_order_relaxed);
  }
}

void PairedCompressedImageSubscriber::logCounters(const char* phase) const {
  const Counters values = counters();
  RCLCPP_INFO(
      node_->get_logger(),
      "[PAIRED_COMPRESSED_INPUT] phase=%s received_camera0=%llu "
      "received_camera1=%llu matched_pairs=%llu decode_success_pairs=%llu "
      "decode_failed_pairs=%llu admission_success_pairs=%llu "
      "admission_failed_pairs=%llu sync_expired_camera0_only=%llu "
      "sync_expired_camera1_only=%llu sync_expired_other=%llu",
      phase,
      static_cast<unsigned long long>(values.receivedCamera0),
      static_cast<unsigned long long>(values.receivedCamera1),
      static_cast<unsigned long long>(values.matchedPairs),
      static_cast<unsigned long long>(values.decodeSuccessPairs),
      static_cast<unsigned long long>(values.decodeFailedPairs),
      static_cast<unsigned long long>(values.admissionSuccessPairs),
      static_cast<unsigned long long>(values.admissionFailedPairs),
      static_cast<unsigned long long>(values.syncExpiredCamera0Only),
      static_cast<unsigned long long>(values.syncExpiredCamera1Only),
      static_cast<unsigned long long>(values.syncExpiredOther));
}

}  // namespace okvis
