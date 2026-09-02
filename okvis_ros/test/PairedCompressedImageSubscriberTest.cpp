#include <okvis/PairedCompressedImageSubscriber.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

namespace {

using namespace std::chrono_literals;

sensor_msgs::msg::CompressedImage makeCompressedImage(int32_t second,
                                                       uint32_t nanosecond,
                                                       uint8_t value) {
  cv::Mat image(8, 8, CV_8UC1, cv::Scalar(value));
  sensor_msgs::msg::CompressedImage message;
  message.header.stamp.sec = second;
  message.header.stamp.nanosec = nanosecond;
  message.format = "jpeg";
  EXPECT_TRUE(cv::imencode(".jpg", image, message.data));
  return message;
}

sensor_msgs::msg::CompressedImage makeInvalidCompressedImage(int32_t second) {
  sensor_msgs::msg::CompressedImage message;
  message.header.stamp.sec = second;
  message.format = "jpeg";
  return message;
}

TEST(PairedCompressedImageSubscriber, DeliversOnlyExactPairsInCameraOrder) {
  rclcpp::init(0, nullptr);
  auto node = std::make_shared<rclcpp::Node>("paired_compressed_input_test");

  std::mutex mutex;
  std::condition_variable callbackCondition;
  std::vector<std::pair<unsigned int, builtin_interfaces::msg::Time>> callbacks;
  okvis::PairedCompressedImageSubscriber subscriber(
      node, "/test/camera0/compressed", "/test/camera1/compressed", 10,
      [&](const sensor_msgs::msg::Image::ConstSharedPtr& image,
          unsigned int cameraIndex) {
        std::lock_guard<std::mutex> lock(mutex);
        callbacks.emplace_back(cameraIndex, image->header.stamp);
        callbackCondition.notify_all();
        return !(image->header.stamp.sec == 30 && cameraIndex == 1u);
      });

  auto camera0Publisher =
      node->create_publisher<sensor_msgs::msg::CompressedImage>(
          "/test/camera0/compressed", 10);
  auto camera1Publisher =
      node->create_publisher<sensor_msgs::msg::CompressedImage>(
          "/test/camera1/compressed", 10);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinThread([&executor]() { executor.spin(); });

  const auto discoveryDeadline = std::chrono::steady_clock::now() + 2s;
  while ((camera0Publisher->get_subscription_count() == 0 ||
          camera1Publisher->get_subscription_count() == 0) &&
         std::chrono::steady_clock::now() < discoveryDeadline) {
    std::this_thread::sleep_for(10ms);
  }
  const bool discovered = camera0Publisher->get_subscription_count() == 1u &&
                          camera1Publisher->get_subscription_count() == 1u;
  EXPECT_TRUE(discovered);

  if (discovered) {
    camera0Publisher->publish(makeCompressedImage(10, 123u, 40u));
    camera1Publisher->publish(makeCompressedImage(10, 123u, 80u));
  }

  bool receivedPair = false;
  {
    std::unique_lock<std::mutex> lock(mutex);
    receivedPair = callbackCondition.wait_for(
        lock, 2s, [&callbacks]() { return callbacks.size() == 2; });
    EXPECT_TRUE(receivedPair);
    if (receivedPair) {
      EXPECT_EQ(callbacks[0].first, 0u);
      EXPECT_EQ(callbacks[1].first, 1u);
      EXPECT_EQ(callbacks[0].second.sec, 10);
      EXPECT_EQ(callbacks[0].second.nanosec, 123u);
      EXPECT_EQ(callbacks[1].second.sec, 10);
      EXPECT_EQ(callbacks[1].second.nanosec, 123u);
    }
  }

  if (discovered && receivedPair) {
    camera0Publisher->publish(makeCompressedImage(20, 0u, 40u));
    camera1Publisher->publish(makeCompressedImage(21, 0u, 80u));
    std::this_thread::sleep_for(100ms);
    {
      std::lock_guard<std::mutex> lock(mutex);
      EXPECT_EQ(callbacks.size(), 2u);
    }

    // A later complete tuple makes both older one-sided tuples observably
    // expire in ExactTime. Camera 1 rejects this pair to exercise pair-level
    // admission accounting.
    camera0Publisher->publish(makeCompressedImage(30, 0u, 40u));
    camera1Publisher->publish(makeCompressedImage(30, 0u, 80u));
    const auto admissionDeadline = std::chrono::steady_clock::now() + 2s;
    while (subscriber.counters().admissionFailedPairs != 1u &&
           std::chrono::steady_clock::now() < admissionDeadline) {
      std::this_thread::sleep_for(10ms);
    }

    // A synchronized corrupt pair is matched but neither decoded nor admitted.
    camera0Publisher->publish(makeInvalidCompressedImage(40));
    camera1Publisher->publish(makeInvalidCompressedImage(40));
    const auto decodeDeadline = std::chrono::steady_clock::now() + 2s;
    while (subscriber.counters().decodeFailedPairs != 1u &&
           std::chrono::steady_clock::now() < decodeDeadline) {
      std::this_thread::sleep_for(10ms);
    }

    const auto counters = subscriber.counters();
    EXPECT_EQ(counters.receivedCamera0, 4u);
    EXPECT_EQ(counters.receivedCamera1, 4u);
    EXPECT_EQ(counters.matchedPairs, 3u);
    EXPECT_EQ(counters.decodeSuccessPairs, 2u);
    EXPECT_EQ(counters.decodeFailedPairs, 1u);
    EXPECT_EQ(counters.admissionSuccessPairs, 1u);
    EXPECT_EQ(counters.admissionFailedPairs, 1u);
    EXPECT_EQ(counters.syncExpiredCamera0Only, 1u);
    EXPECT_EQ(counters.syncExpiredCamera1Only, 1u);
    EXPECT_EQ(counters.syncExpiredOther, 0u);
  }

  executor.cancel();
  spinThread.join();
  rclcpp::shutdown();
}

}  // namespace
