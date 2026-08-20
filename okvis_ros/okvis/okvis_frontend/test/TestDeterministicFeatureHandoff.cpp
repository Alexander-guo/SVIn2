#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include <okvis/Frontend.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/ImuFrameSynchronizer.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>

namespace {

using Camera = okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>;
constexpr size_t kDescriptorBytes = 48;

uint32_t floatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

cv::Mat makeDeterministicImage() {
  cv::Mat image(480, 752, CV_8UC1);
  for (int row = 0; row < image.rows; ++row) {
    unsigned char* pixels = image.ptr<unsigned char>(row);
    for (int column = 0; column < image.cols; ++column) {
      const uint32_t value = static_cast<uint32_t>(column * 37 + row * 91 +
                                                    ((column * row) % 251) * 13 +
                                                    ((column ^ row) & 31) * 17);
      pixels[column] = static_cast<unsigned char>(value & 0xffU);
    }
  }
  return image;
}

okvis::cameras::NCameraSystem makeCameraSystem() {
  okvis::cameras::NCameraSystem system;
  const std::shared_ptr<const okvis::kinematics::Transformation> T_SC(
      new okvis::kinematics::Transformation());
  system.addCamera(T_SC, Camera::createTestObject(), okvis::cameras::NCameraSystem::NoDistortion);
  return system;
}

struct ProducedFeatures {
  std::vector<uint8_t> bytes;
  std::vector<std::array<uint32_t, 7>> keypoints;
};

ProducedFeatures produce(const cv::Mat& image,
                          const okvis::cameras::NCameraSystem& cameraSystem,
                          const okvis::kinematics::Transformation& T_WC) {
  okvis::Frontend frontend(1);
  const okvis::Time timestamp(42, 0);
  okvis::MultiFrame frame(cameraSystem, timestamp, okvis::IdProvider::instance().newId());
  frame.setImage(0, image.clone());
  EXPECT_TRUE(frontend.detectAndDescribe(0, std::shared_ptr<okvis::MultiFrame>(&frame, [](okvis::MultiFrame*) {}),
                                         T_WC, nullptr));

  ProducedFeatures result;
  result.bytes.reserve(frame.numKeypoints(0) * (7 * sizeof(uint32_t) + kDescriptorBytes));
  result.keypoints.reserve(frame.numKeypoints(0));
  for (size_t index = 0; index < frame.numKeypoints(0); ++index) {
    cv::KeyPoint keypoint;
    EXPECT_TRUE(frame.getCvKeypoint(0, index, keypoint));
    result.keypoints.push_back({floatBits(keypoint.pt.x),
                                floatBits(keypoint.pt.y),
                                floatBits(keypoint.size),
                                floatBits(keypoint.response),
                                static_cast<uint32_t>(static_cast<int32_t>(keypoint.octave)),
                                static_cast<uint32_t>(static_cast<int32_t>(keypoint.class_id)),
                                floatBits(keypoint.angle)});
    const unsigned char* descriptor = frame.keypointDescriptor(0, index);
    EXPECT_NE(descriptor, nullptr);
    if (descriptor != nullptr) result.bytes.insert(result.bytes.end(), descriptor, descriptor + kDescriptorBytes);
  }
  return result;
}

TEST(DeterministicFeatureHandoffTest, SameGravityStateIsByteIdenticalAcrossConcurrentProducers) {
  const cv::Mat image = makeDeterministicImage();
  const okvis::cameras::NCameraSystem cameraSystem = makeCameraSystem();
  const Eigen::Quaterniond tilt(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
  const okvis::kinematics::Transformation T_WC(
      Eigen::Vector3d(0.0, 0.0, 0.0), tilt);

  std::vector<std::future<ProducedFeatures>> producers;
  for (size_t index = 0; index < 4; ++index) {
    producers.push_back(std::async(std::launch::async, [&image, &cameraSystem, &T_WC]() {
      return produce(image, cameraSystem, T_WC);
    }));
  }

  const ProducedFeatures reference = producers.front().get();
  ASSERT_FALSE(reference.bytes.empty());
  for (size_t index = 1; index < producers.size(); ++index) {
    const ProducedFeatures repeat = producers[index].get();
    EXPECT_EQ(repeat.keypoints, reference.keypoints);
    EXPECT_EQ(repeat.bytes, reference.bytes);
  }
}

TEST(DeterministicFeatureHandoffTest, PublishedImuBoundaryDoesNotDependOnNotificationTiming) {
  okvis::ImuFrameSynchronizer synchronizer;
  const okvis::Time frameStamp(10, 0);
  synchronizer.gotImuData(frameStamp);

  std::promise<bool> resultPromise;
  std::future<bool> result = resultPromise.get_future();
  std::thread waiter([&synchronizer, &frameStamp, &resultPromise]() {
    resultPromise.set_value(synchronizer.waitForUpToDateImuData(frameStamp));
  });

  const std::future_status status = result.wait_for(std::chrono::milliseconds(100));
  if (status != std::future_status::ready) synchronizer.shutdown();
  waiter.join();

  ASSERT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(result.get());
}

TEST(DeterministicFeatureHandoffTest, TranslationAndYawContextDoNotReplaceGravitySemantics) {
  const cv::Mat image = makeDeterministicImage();
  const okvis::cameras::NCameraSystem cameraSystem = makeCameraSystem();
  const Eigen::Quaterniond tilt(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
  const Eigen::Quaterniond yaw(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()));
  const okvis::kinematics::Transformation first(
      Eigen::Vector3d(0.0, 0.0, 0.0), tilt);
  const okvis::kinematics::Transformation sameGravity(
      Eigen::Vector3d(4.0, -2.0, 1.0), yaw * tilt);
  const okvis::kinematics::Transformation differentGravity(
      Eigen::Vector3d(4.0, -2.0, 1.0), Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY())));

  const ProducedFeatures firstOutput = produce(image, cameraSystem, first);
  const ProducedFeatures sameGravityOutput = produce(image, cameraSystem, sameGravity);
  const ProducedFeatures differentGravityOutput = produce(image, cameraSystem, differentGravity);

  ASSERT_FALSE(firstOutput.bytes.empty());
  EXPECT_EQ(firstOutput.keypoints, sameGravityOutput.keypoints);
  EXPECT_EQ(firstOutput.bytes, sameGravityOutput.bytes);
  EXPECT_NE(firstOutput.keypoints, differentGravityOutput.keypoints);
  EXPECT_NE(firstOutput.bytes, differentGravityOutput.bytes);
}

}  // namespace
