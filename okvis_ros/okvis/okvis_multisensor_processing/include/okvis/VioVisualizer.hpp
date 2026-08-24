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
 * @file VioVisualizer.hpp
 * @brief Header file for the VioVisualizer class.
 * @author Pascal Gohl
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#ifndef INCLUDE_OKVIS_VIOVISUALIZER_HPP_
#define INCLUDE_OKVIS_VIOVISUALIZER_HPP_

#include <memory>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/assert_macros.hpp>
#include <opencv2/highgui.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief This class is responsible to visualize the matching results
 */
class VioVisualizer {
 public:
  /// @brief This struct contains the relevant data for visualizing
  struct VisualizationData {
    typedef std::shared_ptr<VisualizationData> Ptr;
    okvis::ObservationVector observations;             ///< Vector containing all the keypoint observations.
    std::shared_ptr<okvis::MultiFrame> currentFrames;  ///< Current multiframe.
    std::shared_ptr<okvis::MultiFrame> keyFrames;      ///< Current keyframe.
    okvis::kinematics::Transformation T_WS_keyFrame;   ///< Pose of the current keyframe

    struct DriftDiagnostics {
      uint64_t frameId = 0;
      double timestamp = 0.0;
      double relativeTimestamp = 0.0;
      bool isKeyframe = false;
      size_t activeFrames = 0;
      size_t activeKeyframes = 0;
      size_t activeLandmarks = 0;
      size_t activeInitializedLandmarks = 0;
      size_t preMarginalizationFrames = 0;
      size_t preMarginalizationLandmarks = 0;
      size_t marginalizedLandmarks = 0;
      size_t keypoints = 0;
      size_t unassociated = 0;
      size_t pending = 0;
      size_t finite = 0;
      size_t uninitialized = 0;
      size_t invalidInitialized = 0;
      std::vector<size_t> cameraKeypoints;
      std::vector<size_t> cameraUnassociated;
      std::vector<size_t> cameraPending;
      std::vector<size_t> cameraFinite;
      std::vector<size_t> cameraUninitialized;
      std::vector<size_t> cameraInvalidInitialized;
      std::vector<size_t> cameraFiniteCoverageCells;
      Eigen::Vector3d position = Eigen::Vector3d::Zero();
      Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
      Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
      Eigen::Vector3d gyroBias = Eigen::Vector3d::Zero();
      Eigen::Vector3d accelBias = Eigen::Vector3d::Zero();
      double deltaTime = 0.0;
      double frameTranslation = 0.0;
      double frameRotationDegrees = 0.0;
      double optimizationTranslation = 0.0;
      double optimizationRotationDegrees = 0.0;
      double optimizationVelocity = 0.0;
      double optimizationGyroBias = 0.0;
      double optimizationAccelBias = 0.0;
      size_t finiteCoverageCells = 0;
      size_t finiteCoverageCellsPossible = 0;
      double reprojectionPreP50 = 0.0;
      double reprojectionPreP90 = 0.0;
      double reprojectionPostP50 = 0.0;
      double reprojectionPostP90 = 0.0;
      double reprojectionPostMax = 0.0;
      double reprojectionPostFractionOver4 = 0.0;
      double reprojectionPostFractionOver9 = 0.0;
      size_t reprojectionPreCount = 0;
      size_t reprojectionPostCount = 0;
      size_t reprojectionPreInvalid = 0;
      size_t reprojectionPostInvalid = 0;
      // Geometry diagnostics are read-only summaries.  Current-camera
      // depth/range and translation-information fields use the unique finite
      // landmark IDs visible in the current frame; active-window observation
      // counts/spans and multiview parallax then use historical observations
      // only for that currently visible ID set.  Per-camera values are
      // comma-separated in the log; an empty sample is serialized as "nan" and
      // has an explicit zero count.
      std::vector<size_t> geometryCurrentCameraRangeValidCount;
      std::vector<double> geometryCurrentCameraRangeP10;
      std::vector<double> geometryCurrentCameraRangeP50;
      std::vector<double> geometryCurrentCameraRangeP90;
      std::vector<size_t> geometryCurrentCameraDepthValidCount;
      std::vector<double> geometryCurrentCameraDepthP10;
      std::vector<double> geometryCurrentCameraDepthP50;
      std::vector<double> geometryCurrentCameraDepthP90;
      size_t geometryActiveObservationCountValid = 0;
      double geometryActiveObservationCountP10 = 0.0;
      double geometryActiveObservationCountP50 = 0.0;
      double geometryActiveObservationCountP90 = 0.0;
      size_t geometryActiveFrameSpanValid = 0;
      double geometryActiveFrameSpanP10 = 0.0;
      double geometryActiveFrameSpanP50 = 0.0;
      double geometryActiveFrameSpanP90 = 0.0;
      size_t geometryParallaxValid = 0;
      double geometryParallaxP10 = 0.0;
      double geometryParallaxP50 = 0.0;
      double geometryParallaxP90 = 0.0;
      size_t geometryTranslationInfoObservationCount = 0;
      double geometryTranslationInfoEigenMin = 0.0;
      double geometryTranslationInfoEigenMedian = 0.0;
      double geometryTranslationInfoEigenMax = 0.0;
      size_t geometryTranslationInfoEffectiveRank = 0;
      double geometryTranslationInfoConditionNumber = 0.0;
      int optimizationIterations = 0;
      int optimizationSuccessfulSteps = 0;
      int optimizationUnsuccessfulSteps = 0;
      int optimizationTerminationType = -1;
      double optimizationInitialCost = 0.0;
      double optimizationFinalCost = 0.0;
      double optimizationTimeSeconds = 0.0;
      bool optimizationUsable = false;
    } drift;
  };

  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /**
   * @brief Constructor.
   * @param parameters Parameters and settings.
   */
  explicit VioVisualizer(okvis::VioParameters& parameters);  // NOLINT
  virtual ~VioVisualizer();

  /**
   * @brief Initialise parameters. Called in constructor.
   * @param parameters Parameters and settings.
   */
  void init(okvis::VioParameters& parameters);  // NOLINT
  /**
   * @brief Show the current frames with the current keyframe and all its matches.
   * @param data Visualization data containing all the info.
   */
  void showDebugImages(VisualizationData::Ptr& data);  // NOLINT

  /**
   * @brief Circles all keypoints in the current frame, links the matching ones to
   *        the current keyframe and returns the result.
   * @param data Visualization data.
   * @param image_number Index of the frame to display.
   * @return OpenCV matrix with the resulting image.
   */
  cv::Mat drawMatches(VisualizationData::Ptr& data, size_t image_number);  // NOLINT

 private:
  /**
   * @brief Circles all keypoints in the current frame and returns the result.
   * @param data Visualization data.
   * @param cameraIndex Index of the frame to display.
   * @return OpenCV matrix with the resulting image.
   */
  cv::Mat drawKeypoints(VisualizationData::Ptr& data, size_t cameraIndex);  // NOLINT

  /// Parameters and settings.
  okvis::VioParameters parameters_;
  /// Per-camera counters used to bound structured image-quality logging.
  std::vector<size_t> diagnosticFrameCounters_;
};

} /* namespace okvis */

#endif /* INCLUDE_OKVIS_VIOVISUALIZER_HPP_ */
