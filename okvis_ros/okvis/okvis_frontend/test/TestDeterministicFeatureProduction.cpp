#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <brisk/brisk.h>
#pragma GCC diagnostic pop

#include <okvis/Frame.hpp>
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

std::array<uint32_t, 6> featureCore(const cv::KeyPoint& keypoint) {
  return {floatBits(keypoint.pt.x),
          floatBits(keypoint.pt.y),
          floatBits(keypoint.size),
          floatBits(keypoint.response),
          static_cast<uint32_t>(static_cast<int32_t>(keypoint.octave)),
          static_cast<uint32_t>(static_cast<int32_t>(keypoint.class_id))};
}

std::vector<std::array<uint32_t, 6>> sortedFeatureCores(const std::vector<cv::KeyPoint>& keypoints) {
  std::vector<std::array<uint32_t, 6>> cores;
  cores.reserve(keypoints.size());
  for (const cv::KeyPoint& keypoint : keypoints) cores.push_back(featureCore(keypoint));
  std::sort(cores.begin(), cores.end());
  return cores;
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

std::shared_ptr<cv::FeatureDetector> makeDetector() {
#ifdef __ARM_NEON__
  return std::shared_ptr<cv::FeatureDetector>(new brisk::BriskFeatureDetector(34, 2));
#else
  return std::shared_ptr<cv::FeatureDetector>(
      new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(34, 2, 800, 450));
#endif
}

std::shared_ptr<cv::DescriptorExtractor> makeExtractor() {
  return std::shared_ptr<cv::DescriptorExtractor>(new cv::BriskDescriptorExtractor(true, false));
}

std::vector<uint8_t> serializeFrame(okvis::Frame& frame) {
  std::vector<uint8_t> bytes;
  const auto appendU32 = [&bytes](uint32_t value) {
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
      bytes.push_back(static_cast<uint8_t>((value >> (8U * byte)) & 0xffU));
    }
  };
  appendU32(static_cast<uint32_t>(frame.numKeypoints()));
  for (size_t index = 0; index < frame.numKeypoints(); ++index) {
    cv::KeyPoint keypoint;
    EXPECT_TRUE(frame.getCvKeypoint(index, keypoint));
    appendU32(floatBits(keypoint.pt.x));
    appendU32(floatBits(keypoint.pt.y));
    appendU32(floatBits(keypoint.size));
    appendU32(floatBits(keypoint.response));
    appendU32(static_cast<uint32_t>(static_cast<int32_t>(keypoint.octave)));
    appendU32(static_cast<uint32_t>(static_cast<int32_t>(keypoint.class_id)));
    appendU32(floatBits(keypoint.angle));
    const unsigned char* descriptor = frame.keypointDescriptor(index);
    EXPECT_NE(descriptor, nullptr);
    if (descriptor == nullptr) return {};
    bytes.insert(bytes.end(), descriptor, descriptor + kDescriptorBytes);
  }
  return bytes;
}

struct ProducedFeatures {
  std::vector<uint8_t> serialized;
  std::vector<cv::KeyPoint> detectorKeypoints;
  std::vector<cv::KeyPoint> handoffKeypoints;
};

ProducedFeatures produce(const cv::Mat& image) {
  std::shared_ptr<okvis::cameras::CameraBase> camera = Camera::createTestObject();
  std::shared_ptr<cv::FeatureDetector> detector = makeDetector();
  std::shared_ptr<cv::DescriptorExtractor> extractor = makeExtractor();
  okvis::Frame frame(image.clone(), camera, detector, extractor);
  EXPECT_GT(frame.detect(), 0);
  ProducedFeatures result;
  result.detectorKeypoints.reserve(frame.numKeypoints());
  for (size_t index = 0; index < frame.numKeypoints(); ++index) {
    cv::KeyPoint keypoint;
    EXPECT_TRUE(frame.getCvKeypoint(index, keypoint));
    result.detectorKeypoints.push_back(keypoint);
  }
  frame.describe(Eigen::Vector3d(0.0, 0.0, 1.0));
  result.handoffKeypoints.reserve(frame.numKeypoints());
  for (size_t index = 0; index < frame.numKeypoints(); ++index) {
    cv::KeyPoint keypoint;
    EXPECT_TRUE(frame.getCvKeypoint(index, keypoint));
    result.handoffKeypoints.push_back(keypoint);
  }
  result.serialized = serializeFrame(frame);
  return result;
}

std::vector<cv::KeyPoint> tiedKeypoints(bool reverse) {
  const std::array<cv::Point2f, 8> points = {
      cv::Point2f(200.0f, 160.0f), cv::Point2f(300.0f, 160.0f),
      cv::Point2f(400.0f, 160.0f), cv::Point2f(500.0f, 160.0f),
      cv::Point2f(200.0f, 280.0f), cv::Point2f(300.0f, 280.0f),
      cv::Point2f(400.0f, 280.0f), cv::Point2f(500.0f, 280.0f)};
  std::vector<cv::KeyPoint> keypoints;
  keypoints.reserve(points.size());
  for (const cv::Point2f& point : points) keypoints.emplace_back(point, 12.0f, 0.0f, 1.0f, 0, -1);
  if (reverse) std::reverse(keypoints.begin(), keypoints.end());
  return keypoints;
}

std::pair<std::vector<uint8_t>, std::vector<cv::KeyPoint>> describeTiedFeatures(
    const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints) {
  std::shared_ptr<okvis::cameras::CameraBase> camera = Camera::createTestObject();
  std::shared_ptr<cv::FeatureDetector> detector = makeDetector();
  std::shared_ptr<cv::DescriptorExtractor> extractor = makeExtractor();
  okvis::Frame frame(image.clone(), camera, detector, extractor);
  EXPECT_TRUE(frame.resetKeypoints(keypoints));
  frame.describe(Eigen::Vector3d(0.0, 0.0, 1.0));
  std::vector<cv::KeyPoint> output;
  output.reserve(frame.numKeypoints());
  for (size_t index = 0; index < frame.numKeypoints(); ++index) {
    cv::KeyPoint keypoint;
    EXPECT_TRUE(frame.getCvKeypoint(index, keypoint));
    output.push_back(keypoint);
  }
  return {serializeFrame(frame), output};
}

TEST(DeterministicFeatureProductionTest, IndependentConcurrentProducersAreByteIdentical) {
  const cv::Mat image = makeDeterministicImage();
  std::vector<std::future<ProducedFeatures>> producers;
  for (size_t index = 0; index < 4; ++index) {
    producers.push_back(std::async(std::launch::async, [&image]() { return produce(image); }));
  }

  const ProducedFeatures reference = producers.front().get();
  ASSERT_FALSE(reference.serialized.empty());
  ASSERT_GT(reference.detectorKeypoints.size(), 0u);
  ASSERT_GT(reference.handoffKeypoints.size(), 0u);

  for (size_t index = 1; index < producers.size(); ++index) {
    const ProducedFeatures repeat = producers[index].get();
    EXPECT_EQ(repeat.serialized, reference.serialized) << "independent producer " << index;
    EXPECT_EQ(sortedFeatureCores(repeat.detectorKeypoints), sortedFeatureCores(reference.detectorKeypoints));
    EXPECT_EQ(sortedFeatureCores(repeat.handoffKeypoints), sortedFeatureCores(reference.handoffKeypoints));
  }
}

TEST(DeterministicFeatureProductionTest, TiedFeatureInputOrderDoesNotChangeMembershipOrBytes) {
  const cv::Mat image = makeDeterministicImage();
  const std::vector<cv::KeyPoint> forward = tiedKeypoints(false);
  const std::vector<cv::KeyPoint> reverse = tiedKeypoints(true);
  const auto forwardOutput = describeTiedFeatures(image, forward);
  const auto reverseOutput = describeTiedFeatures(image, reverse);

  ASSERT_FALSE(forwardOutput.first.empty());
  EXPECT_EQ(forwardOutput.first, reverseOutput.first);
  EXPECT_EQ(sortedFeatureCores(forwardOutput.second), sortedFeatureCores(reverseOutput.second));
  EXPECT_EQ(sortedFeatureCores(forwardOutput.second),
            sortedFeatureCores(forwardOutput.second));
  for (size_t index = 1; index < forwardOutput.second.size(); ++index) {
    EXPECT_LE(featureCore(forwardOutput.second[index - 1]), featureCore(forwardOutput.second[index]));
  }
}

}  // namespace
