/*
 * VioVisualizer_test.cpp
 *
 *  Created on: Sep 15, 2014
 *      Author: pascal
 */

#include <okvis/VioVisualizer.hpp>

#include <gtest/gtest.h>

namespace {

void expectColor(const cv::Scalar& actual, const cv::Scalar& expected) {
  for (int channel = 0; channel < 4; ++channel) EXPECT_DOUBLE_EQ(actual[channel], expected[channel]);
}

void expectPixel(const cv::Vec3b& actual, const cv::Scalar& expected) {
  for (int channel = 0; channel < 3; ++channel) EXPECT_EQ(actual[channel], expected[channel]);
}

TEST(VioVisualizer, ObservationColorsSeparateSingletonAndTemporalPendingIds) {
  okvis::Observation observation;

  expectColor(okvis::VioVisualizer::observationColor(observation), cv::Scalar(0, 0, 255));

  observation.landmarkId = 17;
  observation.activeWindowOccurrences = 1;
  observation.activeWindowDistinctFrames = 1;
  expectColor(okvis::VioVisualizer::observationColor(observation), cv::Scalar(0, 165, 255));

  observation.activeWindowOccurrences = 2;
  observation.activeWindowDistinctFrames = 2;
  expectColor(okvis::VioVisualizer::observationColor(observation), cv::Scalar(255, 0, 0));

  observation.landmarkInGraph = true;
  observation.isInitialized = false;
  observation.landmark_W = Eigen::Vector4d(1.0, 2.0, 3.0, 1.0);
  expectColor(okvis::VioVisualizer::observationColor(observation), cv::Scalar(0, 255, 255));

  observation.isInitialized = true;
  expectColor(okvis::VioVisualizer::observationColor(observation), cv::Scalar(0, 255, 0));

  observation.landmark_W[3] = 0.0;
  expectColor(okvis::VioVisualizer::observationColor(observation), cv::Scalar(0, 255, 255));
}

TEST(VioVisualizer, GenuineTrackingExcludesSyntheticSingletonIds) {
  okvis::Observation observation;
  observation.landmarkId = 23;
  observation.activeWindowOccurrences = 1;
  observation.activeWindowDistinctFrames = 1;
  EXPECT_FALSE(okvis::observationIsGenuinelyTracked(observation));

  // Multiple observations confined to one frame do not establish a temporal track.
  observation.activeWindowOccurrences = 2;
  EXPECT_FALSE(okvis::observationIsGenuinelyTracked(observation));

  observation.activeWindowDistinctFrames = 2;
  EXPECT_TRUE(okvis::observationIsGenuinelyTracked(observation));

  observation.activeWindowOccurrences = 1;
  observation.activeWindowDistinctFrames = 1;
  observation.landmarkInGraph = true;
  EXPECT_TRUE(okvis::observationIsGenuinelyTracked(observation));

  observation.landmarkId = 0;
  EXPECT_FALSE(okvis::observationIsGenuinelyTracked(observation));
}

TEST(VioVisualizer, AlphaBlendRectangleUsesFortyPercentOpacityAndClips) {
  cv::Mat image(20, 20, CV_8UC3, cv::Scalar::all(255));
  okvis::VioVisualizer::alphaBlendRectangle(
      image, cv::Rect(-5, -5, 15, 15), cv::Scalar::all(0), 0.4);
  expectPixel(image.at<cv::Vec3b>(0, 0), cv::Scalar(153, 153, 153));
  expectPixel(image.at<cv::Vec3b>(9, 9), cv::Scalar(153, 153, 153));
  expectPixel(image.at<cv::Vec3b>(10, 10), cv::Scalar(255, 255, 255));
}

TEST(VioVisualizer, LegendOccupiesLowerRightAndKeepsMarkersOpaque) {
  cv::Mat image(150, 300, CV_8UC3, cv::Scalar::all(255));
  okvis::VioVisualizer::drawObservationLegend(image);

  // The fixed 230x97 legend begins at (64,47) for this image.
  expectPixel(image.at<cv::Vec3b>(48, 65), cv::Scalar(153, 153, 153));
  expectPixel(image.at<cv::Vec3b>(10, 10), cv::Scalar(255, 255, 255));
  expectPixel(image.at<cv::Vec3b>(61, 74), cv::Scalar(0, 0, 255));
  expectPixel(image.at<cv::Vec3b>(78, 74), cv::Scalar(0, 165, 255));
  expectPixel(image.at<cv::Vec3b>(95, 74), cv::Scalar(255, 0, 0));
}

}  // namespace
