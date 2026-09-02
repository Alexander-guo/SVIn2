/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the conditions in the project
 *  license are met.
 *********************************************************************************/

#ifndef INCLUDE_OKVIS_PAIREDCOMPRESSEDIMAGESUBSCRIBER_HPP_
#define INCLUDE_OKVIS_PAIREDCOMPRESSEDIMAGESUBSCRIBER_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace okvis {

/// Exact-time, two-camera compressed image input for synchronized camera rigs.
class PairedCompressedImageSubscriber {
 public:
  using CompressedImage = sensor_msgs::msg::CompressedImage;
  using Image = sensor_msgs::msg::Image;
  using ImageCallback =
      std::function<bool(const Image::ConstSharedPtr&, unsigned int)>;

  struct Counters {
    uint64_t receivedCamera0 = 0;
    uint64_t receivedCamera1 = 0;
    uint64_t matchedPairs = 0;
    uint64_t decodeSuccessPairs = 0;
    uint64_t decodeFailedPairs = 0;
    uint64_t admissionSuccessPairs = 0;
    uint64_t admissionFailedPairs = 0;
    uint64_t syncExpiredCamera0Only = 0;
    uint64_t syncExpiredCamera1Only = 0;
    uint64_t syncExpiredOther = 0;
  };

  PairedCompressedImageSubscriber(
      const std::shared_ptr<rclcpp::Node>& node,
      const std::string& camera0Topic,
      const std::string& camera1Topic,
      std::size_t queueSize,
      ImageCallback imageCallback,
      bool logCounters = false);

  ~PairedCompressedImageSubscriber();

  /// Thread-safe snapshot of directly observable subscriber/synchronizer state.
  Counters counters() const;

 private:
  using Synchronizer =
      message_filters::TimeSynchronizer<CompressedImage, CompressedImage>;

  void camera0Received(const CompressedImage::ConstSharedPtr& message);
  void camera1Received(const CompressedImage::ConstSharedPtr& message);
  void pairedCallback(const CompressedImage::ConstSharedPtr& camera0Message,
                      const CompressedImage::ConstSharedPtr& camera1Message);
  void droppedTupleCallback(
      const Synchronizer::M0Event& camera0Event,
      const Synchronizer::M1Event& camera1Event);
  void logCounters(const char* phase) const;

  std::shared_ptr<rclcpp::Node> node_;
  ImageCallback imageCallback_;
  bool logCounters_ = false;
  message_filters::Subscriber<CompressedImage> camera0Subscriber_;
  message_filters::Subscriber<CompressedImage> camera1Subscriber_;
  std::unique_ptr<Synchronizer> synchronizer_;
  rclcpp::TimerBase::SharedPtr diagnosticsTimer_;

  std::atomic<uint64_t> receivedCamera0_{0};
  std::atomic<uint64_t> receivedCamera1_{0};
  std::atomic<uint64_t> matchedPairs_{0};
  std::atomic<uint64_t> decodeSuccessPairs_{0};
  std::atomic<uint64_t> decodeFailedPairs_{0};
  std::atomic<uint64_t> admissionSuccessPairs_{0};
  std::atomic<uint64_t> admissionFailedPairs_{0};
  std::atomic<uint64_t> syncExpiredCamera0Only_{0};
  std::atomic<uint64_t> syncExpiredCamera1Only_{0};
  std::atomic<uint64_t> syncExpiredOther_{0};
};

}  // namespace okvis

#endif  // INCLUDE_OKVIS_PAIREDCOMPRESSEDIMAGESUBSCRIBER_HPP_
