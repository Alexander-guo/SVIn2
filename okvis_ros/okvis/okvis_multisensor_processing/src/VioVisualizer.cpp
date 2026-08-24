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
 *  Created on: Sep 15, 2014
 *      Author: Pascal Gohl
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file VioVisualizer.cpp
 * @brief Source file for the VioVisualizer class.
 * @author Pascal Gohl
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include "okvis/VioVisualizer.hpp"

#include <glog/logging.h>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/kinematics/Transformation.hpp>

// cameras and distortions
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <memory>
#include <okvis/cameras/DoubleSphereCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <sstream>
#include <vector>

/// \brief okvis Main namespace of this package.
namespace okvis {

VioVisualizer::VioVisualizer(okvis::VioParameters& parameters) : parameters_(parameters) {
  if (parameters.nCameraSystem.numCameras() > 0) {
    init(parameters);
  }
}

VioVisualizer::~VioVisualizer() {}

void VioVisualizer::init(okvis::VioParameters& parameters) {
  parameters_ = parameters;
  diagnosticFrameCounters_.assign(parameters_.nCameraSystem.numCameras(), 0);
}

cv::Mat VioVisualizer::drawMatches(VisualizationData::Ptr& data, size_t image_number) {
  std::shared_ptr<okvis::MultiFrame> keyframe = data->keyFrames;
  std::shared_ptr<okvis::MultiFrame> frame = data->currentFrames;

  constexpr int kPanelHeight = 500;
  constexpr double kMinimumMarkerRadius = 1.5;
  constexpr double kMaximumMarkerRadius = 12.0;

  const cv::Mat& currentImage = frame->image(image_number);
  if (currentImage.empty()) return cv::Mat();

  const auto panelWidth = [](const cv::Mat& image) {
    return std::max(1, cvRound(static_cast<double>(image.cols) * kPanelHeight / image.rows));
  };
  const auto toDisplayImage = [&](const cv::Mat& image, cv::Mat& displayImage, int width) {
    cv::Mat colorImage;
    if (image.channels() == 1) {
      cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
    } else {
      colorImage = image;
    }
    cv::resize(colorImage, displayImage, cv::Size(width, kPanelHeight), 0.0, 0.0, cv::INTER_AREA);
  };

  cv::Mat grayImage;
  if (currentImage.channels() == 1) {
    grayImage = currentImage;
  } else {
    cv::cvtColor(currentImage, grayImage, cv::COLOR_BGR2GRAY);
  }
  cv::Scalar intensityMean;
  cv::Scalar intensityStdDev;
  cv::meanStdDev(grayImage, intensityMean, intensityStdDev);
  const double pixelCount = static_cast<double>(grayImage.total());
  const double darkFraction = pixelCount > 0.0 ? cv::countNonZero(grayImage < 20) / pixelCount : 0.0;
  const double saturatedFraction = pixelCount > 0.0 ? cv::countNonZero(grayImage > 245) / pixelCount : 0.0;
  cv::Mat laplacian;
  cv::Laplacian(grayImage, laplacian, CV_64F);
  cv::Scalar laplacianMean;
  cv::Scalar laplacianStdDev;
  cv::meanStdDev(laplacian, laplacianMean, laplacianStdDev);
  const double laplacianVariance = laplacianStdDev[0] * laplacianStdDev[0];

  const auto annotateDiagnostics = [&](cv::Mat& image) {
    if (image.empty()) return;
    const VisualizationData::DriftDiagnostics& drift = data->drift;
    std::vector<std::string> lines;
    std::ostringstream line;
    line << std::fixed << std::setprecision(2) << "t " << drift.timestamp << "  rel " << drift.relativeTimestamp
         << "  frame " << drift.frameId << (drift.isKeyframe ? "  KF" : "") << "  cam " << image_number;
    lines.push_back(line.str());
    line.str("");
    line.clear();
    const size_t cameraFinite = image_number < drift.cameraFinite.size() ? drift.cameraFinite[image_number] : 0;
    const size_t cameraPending = image_number < drift.cameraPending.size() ? drift.cameraPending[image_number] : 0;
    const size_t cameraUnassociated =
        image_number < drift.cameraUnassociated.size() ? drift.cameraUnassociated[image_number] : 0;
    const size_t cameraCoverage = image_number < drift.cameraFiniteCoverageCells.size()
                                      ? drift.cameraFiniteCoverageCells[image_number]
                                      : 0;
    line << "cam support finite/pending/unassoc " << cameraFinite << "/" << cameraPending << "/"
         << cameraUnassociated << "  coverage " << cameraCoverage << "/16";
    lines.push_back(line.str());
    line.str("");
    line.clear();
    line << std::fixed << std::setprecision(3) << "|v| " << drift.velocity.norm() << "  opt dp "
         << drift.optimizationTranslation << " m  dR " << drift.optimizationRotationDegrees << " deg";
    lines.push_back(line.str());
    line.str("");
    line.clear();
    line << std::fixed << std::setprecision(2) << "est reproj chi2 p50/p90 " << drift.reprojectionPostP50 << "/"
         << drift.reprojectionPostP90 << "  >4 " << 100.0 * drift.reprojectionPostFractionOver4 << "%";
    lines.push_back(line.str());
    line.str("");
    line.clear();
    line << std::fixed << std::setprecision(1) << "post image mean/std " << intensityMean[0] << "/"
         << intensityStdDev[0] << "  dark " << 100.0 * darkFraction << "%  blur " << laplacianVariance;
    lines.push_back(line.str());

    constexpr int kLineHeight = 18;
    const int boxHeight = static_cast<int>(lines.size()) * kLineHeight + 8;
    cv::rectangle(image, cv::Rect(0, 0, image.cols, std::min(boxHeight, image.rows)), cv::Scalar(0, 0, 0), -1);
    for (size_t index = 0; index < lines.size(); ++index) {
      cv::putText(image,
                  lines[index],
                  cv::Point(6, 17 + static_cast<int>(index) * kLineHeight),
                  cv::FONT_HERSHEY_SIMPLEX,
                  0.43,
                  cv::Scalar(255, 255, 255),
                  1,
                  cv::LINE_AA);
    }
  };

  const double configuredCameraRate = parameters_.sensors_information.cameraRate;
  const size_t baselinePeriod = std::isfinite(configuredCameraRate) && configuredCameraRate >= 1.0
                                    ? static_cast<size_t>(std::llround(configuredCameraRate))
                                    : 1;
  if (image_number < diagnosticFrameCounters_.size() &&
      ++diagnosticFrameCounters_[image_number] % baselinePeriod == 0) {
    LOG(INFO) << std::setprecision(17) << "[IMAGE_QUALITY_DIAGNOSTIC]"
              << " timestamp=" << data->drift.timestamp
              << " relative_timestamp=" << data->drift.relativeTimestamp << " frame=" << data->drift.frameId
              << " camera=" << image_number << " post_mean=" << intensityMean[0]
              << " post_stddev=" << intensityStdDev[0] << " post_dark_fraction=" << darkFraction
              << " post_saturated_fraction=" << saturatedFraction
              << " post_laplacian_variance=" << laplacianVariance;
  }

  const int currentWidth = panelWidth(currentImage);
  const double currentScale = static_cast<double>(kPanelHeight) / currentImage.rows;
  if (keyframe == nullptr) {
    cv::Mat currentDisplay;
    toDisplayImage(currentImage, currentDisplay, currentWidth);
    annotateDiagnostics(currentDisplay);
    return currentDisplay;
  }

  const cv::Mat& keyframeImage = keyframe->image(image_number);
  if (keyframeImage.empty()) return cv::Mat();
  const int keyframeWidth = panelWidth(keyframeImage);
  const double keyframeScale = static_cast<double>(kPanelHeight) / keyframeImage.rows;
  const int outputWidth = std::max(currentWidth, keyframeWidth);
  const int rowJump = kPanelHeight;

  cv::Mat outimg(2 * kPanelHeight, outputWidth, CV_8UC3, cv::Scalar::all(0));
  cv::Mat current = outimg(cv::Rect(0, rowJump, currentWidth, kPanelHeight));
  cv::Mat actKeyframe = outimg(cv::Rect(0, 0, keyframeWidth, kPanelHeight));
  toDisplayImage(currentImage, current, currentWidth);
  toDisplayImage(keyframeImage, actKeyframe, keyframeWidth);

  const auto scaledPoint = [](const Eigen::Vector2d& point, double scale, double yOffset = 0.0) {
    return cv::Point2f(static_cast<float>(point[0] * scale), static_cast<float>(point[1] * scale + yOffset));
  };
  const auto markerRadius = [&](double keypointSize, double scale) {
    if (!std::isfinite(keypointSize) || keypointSize <= 0.0) return kMinimumMarkerRadius;
    return std::clamp(0.5 * keypointSize * scale, kMinimumMarkerRadius, kMaximumMarkerRadius);
  };

  // the keyframe trafo
  Eigen::Vector2d keypoint;
  Eigen::Vector4d landmark;
  okvis::kinematics::Transformation lastKeyframeT_CW =
      parameters_.nCameraSystem.T_SC(image_number)->inverse() * data->T_WS_keyFrame.inverse();

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType = parameters_.nCameraSystem.distortionType(0);
  for (size_t i = 1; i < parameters_.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(Exception,
                      distortionType == parameters_.nCameraSystem.distortionType(i),
                      "mixed frame types are not supported yet");
  }

  for (auto it = data->observations.begin(); it != data->observations.end(); ++it) {
    if (it->cameraIdx != image_number) continue;

    cv::Scalar color;

    if (it->landmarkId != 0) {
      color = cv::Scalar(255, 0, 0);  // blue
    } else {
      color = cv::Scalar(0, 0, 255);  // red
    }

    // draw matches to keyframe
    keypoint = it->keypointMeasurement;
    if (fabs(it->landmark_W[3]) > 1.0e-8) {
      Eigen::Vector4d hPoint = it->landmark_W;
      if (it->isInitialized) {
        color = cv::Scalar(0, 255, 0);  // green
      } else {
        color = cv::Scalar(0, 255, 255);  // yellow
      }
      Eigen::Vector2d keyframePt;
      bool isVisibleInKeyframe = false;
      Eigen::Vector4d hP_C = lastKeyframeT_CW * hPoint;
      switch (distortionType) {
        case okvis::cameras::NCameraSystem::RadialTangential: {
          if (frame->geometryAs<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion>>(image_number)
                  ->projectHomogeneous(hP_C, &keyframePt) == okvis::cameras::CameraBase::ProjectionStatus::Successful)
            isVisibleInKeyframe = true;
          break;
        }
        case okvis::cameras::NCameraSystem::Equidistant: {
          if (frame->geometryAs<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>>(image_number)
                  ->projectHomogeneous(hP_C, &keyframePt) == okvis::cameras::CameraBase::ProjectionStatus::Successful)
            isVisibleInKeyframe = true;
          break;
        }
        case okvis::cameras::NCameraSystem::RadialTangential8: {
          if (frame
                  ->geometryAs<okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8>>(image_number)
                  ->projectHomogeneous(hP_C, &keyframePt) == okvis::cameras::CameraBase::ProjectionStatus::Successful)
            isVisibleInKeyframe = true;
          break;
        }
        case okvis::cameras::NCameraSystem::NoDistortion: {
          if (frame->geometryAs<okvis::cameras::DoubleSphereCamera<okvis::cameras::NoDistortion>>(image_number)
                  ->projectHomogeneous(hP_C, &keyframePt) == okvis::cameras::CameraBase::ProjectionStatus::Successful)
            isVisibleInKeyframe = true;
          break;
        }
        default:
          OKVIS_THROW(Exception, "Unsupported distortion type.")
          break;
      }
      if (fabs(hP_C[3]) > 1.0e-8) {
        if (hP_C[2] / hP_C[3] < 0.4) {
          isVisibleInKeyframe = false;
        }
      }

      if (isVisibleInKeyframe && keyframePt.allFinite() && keypoint.allFinite()) {
        // found in the keyframe. draw line
        cv::line(outimg,
                 scaledPoint(keyframePt, keyframeScale),
                 scaledPoint(keypoint, currentScale, rowJump),
                 color,
                 1,
                 cv::LINE_AA);
        cv::circle(actKeyframe,
                   scaledPoint(keyframePt, keyframeScale),
                   markerRadius(it->keypointSize, keyframeScale),
                   color,
                   1,
                   cv::LINE_AA);
      }
    }
    // draw keypoint
    if (!keypoint.allFinite()) continue;
    const double r = markerRadius(it->keypointSize, currentScale);
    const cv::Point2f currentPoint = scaledPoint(keypoint, currentScale);
    cv::circle(current, currentPoint, r, color, 1, cv::LINE_AA);
    cv::KeyPoint cvKeypoint;
    frame->getCvKeypoint(image_number, it->keypointIdx, cvKeypoint);
    if (std::isfinite(cvKeypoint.angle) && cvKeypoint.angle >= 0.0f) {
      const double angle = cvKeypoint.angle / 180.0 * M_PI;
      const cv::Point2f stackedCurrentPoint = scaledPoint(keypoint, currentScale, rowJump);
      cv::line(outimg,
               stackedCurrentPoint,
               stackedCurrentPoint + cv::Point2f(cos(angle), sin(angle)) * r,
               color,
               1,
               cv::LINE_AA);
    }
  }
  annotateDiagnostics(outimg);
  return outimg;
}

cv::Mat VioVisualizer::drawKeypoints(VisualizationData::Ptr& data, size_t cameraIndex) {
  std::shared_ptr<okvis::MultiFrame> currentFrames = data->currentFrames;
  const cv::Mat currentImage = currentFrames->image(cameraIndex);

  cv::Mat outimg;
  cv::cvtColor(currentImage, outimg, cv::COLOR_GRAY2BGR);
  cv::Scalar greenColor(0, 255, 0);  // green

  cv::KeyPoint keypoint;
  for (size_t k = 0; k < currentFrames->numKeypoints(cameraIndex); ++k) {
    currentFrames->getCvKeypoint(cameraIndex, k, keypoint);

    double radius = keypoint.size;
    double angle = keypoint.angle / 180.0 * M_PI;

    cv::circle(outimg, keypoint.pt, radius, greenColor);
    cv::line(outimg,
             keypoint.pt,
             cv::Point2f(keypoint.pt.x + radius * cos(angle), keypoint.pt.y - radius * sin(angle)),
             greenColor);
  }

  return outimg;
}

void VioVisualizer::showDebugImages(VisualizationData::Ptr& data) {
  std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
    out_images[i] = drawMatches(data, i);
  }

  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
    std::stringstream windowname;
    windowname << "OKVIS camera " << im;
    if (!out_images[im].empty()) {
      cv::imshow(windowname.str(), out_images[im]);  // Prevent crashes from display
    }
    cv::waitKey(1);
  }
}

} /* namespace okvis */
