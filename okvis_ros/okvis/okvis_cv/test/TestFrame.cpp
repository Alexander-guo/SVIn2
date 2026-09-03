/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Mar 31, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <brisk/brisk.h>
#pragma GCC diagnostic pop
#include "okvis/Frame.hpp"
#include "okvis/cameras/CameraValidDomain.hpp"
#include "okvis/cameras/DoubleSphereCamera.hpp"
#include "okvis/cameras/EquidistantDistortion.hpp"
#include "okvis/cameras/NoDistortion.hpp"
#include "okvis/cameras/PinholeCamera.hpp"
#include "okvis/cameras/RadialTangentialDistortion.hpp"
#include <opencv2/imgproc.hpp>

TEST(Frame, functions) {
  // instantiate all possible versions of test cameras
  std::vector<std::shared_ptr<okvis::cameras::CameraBase> > cameras;
  cameras.push_back(okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>::createTestObject());
  cameras.push_back(okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>::createTestObject());
  cameras.push_back(okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());

  for (size_t c = 0; c < cameras.size(); ++c) {
#ifdef __ARM_NEON__
    std::shared_ptr<cv::FeatureDetector> detector(new brisk::BriskFeatureDetector(34, 2));
#else
    std::shared_ptr<cv::FeatureDetector> detector(
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(34, 2, 800, 450));
#endif

    std::shared_ptr<cv::DescriptorExtractor> extractor(new cv::BriskDescriptorExtractor(true, false));

    // create a stupid random image
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> eigenImage(752, 480);
    eigenImage.setRandom();
    cv::Mat image(480, 752, CV_8UC1, eigenImage.data());
    okvis::Frame frame(image, cameras.at(c), detector, extractor);

    // run
    frame.detect();
    frame.describe();
  }
}

TEST(CameraValidDomain, DoubleSphereAndExplicitCameraMaskAreCombined) {
  using Camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;
  auto camera = std::make_shared<Camera>(
      256, 256, 100.0, 100.0, 128.0, 128.0, 0.0, 0.8,
      okvis::cameras::NoDistortion::testObject());

  cv::Mat cameraMask = cv::Mat::zeros(256, 256, CV_8UC1);
  cameraMask.at<unsigned char>(128, 128) = 255;  // CameraBase: nonzero is invalid.
  ASSERT_TRUE(camera->setMask(cameraMask));

  const cv::Mat validMask = okvis::cameras::cameraValidDomainMask(camera, cv::Size(256, 256));
  EXPECT_EQ(validMask.type(), CV_8UC1);
  EXPECT_EQ(validMask.at<unsigned char>(0, 0), 0);
  EXPECT_EQ(validMask.at<unsigned char>(128, 128), 0);
  EXPECT_NE(validMask.at<unsigned char>(128, 129), 0);

  const cv::Mat source(256, 256, CV_8UC1, cv::Scalar(17));
  const cv::Mat detectionImage =
      okvis::cameras::cameraDomainMaskedDetectionImage(source, validMask);
  EXPECT_EQ(detectionImage.at<unsigned char>(0, 0), 0);
  EXPECT_EQ(detectionImage.at<unsigned char>(128, 128), 0);
  EXPECT_EQ(detectionImage.at<unsigned char>(128, 129), 17);
  EXPECT_EQ(source.at<unsigned char>(0, 0), 17);
}

TEST(CameraValidDomain, PinholeIsFullyValidWithoutExplicitMask) {
  auto camera = okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>::createTestObject();
  const cv::Size imageSize(static_cast<int>(camera->imageWidth()),
                           static_cast<int>(camera->imageHeight()));
  cv::Mat validMask = okvis::cameras::cameraValidDomainMask(camera, imageSize);
  EXPECT_EQ(cv::countNonZero(validMask), imageSize.area());

  cv::Mat cameraMask = cv::Mat::zeros(imageSize, CV_8UC1);
  cameraMask.at<unsigned char>(0, 0) = 255;
  ASSERT_TRUE(camera->setMask(cameraMask));
  validMask = okvis::cameras::cameraValidDomainMask(camera, imageSize);
  EXPECT_EQ(validMask.at<unsigned char>(0, 0), 0);
  EXPECT_EQ(cv::countNonZero(validMask), imageSize.area() - 1);
}

TEST(Frame, PlainDetectionRejectsInvalidDoubleSphereKeypointsWithoutChangingStoredImage) {
  using Camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;
  std::shared_ptr<okvis::cameras::CameraBase> camera = std::make_shared<Camera>(
      256, 256, 100.0, 100.0, 128.0, 128.0, 0.0, 0.8,
      okvis::cameras::NoDistortion::testObject());
  std::shared_ptr<cv::FeatureDetector> detector(
      new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(5.0, 0, 1.0, 1000));
  std::shared_ptr<cv::DescriptorExtractor> extractor(new cv::BriskDescriptorExtractor(true, false));

  cv::Mat image(256, 256, CV_8UC1);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<unsigned char>(y, x) = ((x / 12 + y / 12) % 2 == 0) ? 30 : 220;
    }
  }
  const cv::Mat original = image.clone();
  okvis::Frame frame(image, camera, detector, extractor);
  ASSERT_GT(frame.detect(), 0);

  const cv::Mat validMask = okvis::cameras::cameraValidDomainMask(camera, image.size());
  for (size_t index = 0; index < frame.numKeypoints(); ++index) {
    cv::KeyPoint keypoint;
    ASSERT_TRUE(frame.getCvKeypoint(index, keypoint));
    EXPECT_TRUE(okvis::cameras::isValidCameraKeypoint(keypoint, validMask, camera));
  }
  EXPECT_EQ(cv::norm(frame.image(), original, cv::NORM_INF), 0.0);
}

TEST(CameraValidDomain, ExactSubpixelCheckRejectsPointOutsideValidRasterCell) {
  using Camera = okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>;
  std::shared_ptr<const okvis::cameras::CameraBase> camera = std::make_shared<Camera>(
      256, 256, 100.0, 100.0, 128.0, 128.0, 0.0, 0.8,
      okvis::cameras::NoDistortion::testObject());
  const cv::Mat validMask = okvis::cameras::cameraValidDomainMask(camera, cv::Size(256, 256));

  bool foundBoundaryPoint = false;
  for (int y = 128; y + 1 < validMask.rows && !foundBoundaryPoint; ++y) {
    for (int x = 128; x + 1 < validMask.cols; ++x) {
      if (validMask.at<unsigned char>(y, x) == 0) continue;
      const cv::KeyPoint subpixelPoint(
          cv::Point2f(static_cast<float>(x) + 0.99f, static_cast<float>(y) + 0.99f), 8.0f);
      Eigen::Vector3d direction;
      const bool exactValid = camera->backProject(
                                  Eigen::Vector2d(subpixelPoint.pt.x, subpixelPoint.pt.y),
                                  &direction) &&
                              direction.allFinite();
      if (exactValid) continue;
      foundBoundaryPoint = true;
      EXPECT_FALSE(okvis::cameras::isValidCameraKeypoint(subpixelPoint, validMask, camera));
      break;
    }
  }
  EXPECT_TRUE(foundBoundaryPoint);
}

TEST(CameraValidDomain, RedOverlayChangesOnlyInvalidPixelsAtHalfOpacity) {
  cv::Mat image(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
  const cv::Mat original = image.clone();
  cv::Mat validMask(2, 2, CV_8UC1, cv::Scalar(255));
  validMask.at<unsigned char>(0, 0) = 0;

  cv::Mat redImage(image.size(), image.type(), cv::Scalar(0, 0, 255));
  cv::Mat expectedBlend;
  cv::addWeighted(original, 0.5, redImage, 0.5, 0.0, expectedBlend);
  okvis::cameras::overlayInvalidCameraDomain50Percent(&image, validMask);

  EXPECT_EQ(image.at<cv::Vec3b>(0, 0), expectedBlend.at<cv::Vec3b>(0, 0));
  EXPECT_EQ(image.at<cv::Vec3b>(0, 1), original.at<cv::Vec3b>(0, 1));
  EXPECT_EQ(image.at<cv::Vec3b>(1, 0), original.at<cv::Vec3b>(1, 0));
  EXPECT_EQ(image.at<cv::Vec3b>(1, 1), original.at<cv::Vec3b>(1, 1));
}

TEST(CameraValidDomain, ConfiguredMaskOverlayIsBlueAtThirtyPercentOpacity) {
  cv::Mat image(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
  const cv::Mat original = image.clone();
  cv::Mat validMask(2, 2, CV_8UC1, cv::Scalar(255));
  validMask.at<unsigned char>(1, 0) = 0;

  cv::Mat blueImage(image.size(), image.type(), cv::Scalar(255, 0, 0));
  cv::Mat expectedBlend;
  cv::addWeighted(original, 0.7, blueImage, 0.3, 0.0, expectedBlend);
  okvis::cameras::overlayConfiguredCameraMask30PercentBlue(&image, validMask);

  EXPECT_EQ(image.at<cv::Vec3b>(1, 0), expectedBlend.at<cv::Vec3b>(1, 0));
  EXPECT_EQ(image.at<cv::Vec3b>(0, 0), original.at<cv::Vec3b>(0, 0));
  EXPECT_EQ(image.at<cv::Vec3b>(0, 1), original.at<cv::Vec3b>(0, 1));
  EXPECT_EQ(image.at<cv::Vec3b>(1, 1), original.at<cv::Vec3b>(1, 1));
}
