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
 *  Created on: Mar 10, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file stereo_triangulation.hpp
 * @brief File containing the triangulateFast method definition.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_OKVIS_TRIANGULATION_STEREO_TRIANGULATION_HPP_
#define INCLUDE_OKVIS_TRIANGULATION_STEREO_TRIANGULATION_HPP_

#include <Eigen/Core>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief
namespace triangulation {

enum class TriangulationType { Rejected, BearingOnly, Finite };

enum class TriangulationStatus {
  Success,
  InvalidInput,
  InsufficientParallax,
  Singular,
  BehindRay,
  InsufficientBaselineRangeRatio,
  RayMismatch
};

struct TriangulationDiagnostics {
  TriangulationType type = TriangulationType::Rejected;
  TriangulationStatus status = TriangulationStatus::InvalidInput;
  double parallaxAngle = 0.0;
  double minimumParallaxAngle = 0.0;
  double baseline = 0.0;
  double range1 = 0.0;
  double range2 = 0.0;
  double baselineRangeRatio = 0.0;
  double minimumBaselineRangeRatio = 0.0;
  double chi2 = 0.0;
};

/**
 * @brief Triangulate the intersection of two rays.
 * @warning The rays e1 and e2 need to be normalized!
 * @param[in]  p1 Camera center position of frame A in coordinate frame A
 * @param[in]  e1 Ray through keypoint of frame A in coordinate frame A.
 * @param[in]  p2 Camera center position of frame B in coordinate frame A.
 * @param[in]  e2 Ray through keypoint of frame B in coordinate frame A.
 * @param[in]  sigma Ray uncertainty.
 * @param[out] isValid Is the triangulation valid.
 * @param[out] isParallel True for a valid direction-only result. Such a result
 *                        has w=0 and must not be inserted as a landmark until a
 *                        later observation yields finite depth.
 * @return Homogeneous coordinates of triangulated point.
 */
Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1,
                                const Eigen::Vector3d& e1,
                                const Eigen::Vector3d& p2,
                                const Eigen::Vector3d& e2,
                                double sigma,
                                bool& isValid,      // NOLINT
                                bool& isParallel);  // NOLINT

/**
 * @brief Triangulate two rays and report the geometric conditioning diagnostics.
 *
 * This overload preserves the original API while allowing callers which emit
 * diagnostics to distinguish weak geometry from other rejection causes.
 */
Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1,
                                const Eigen::Vector3d& e1,
                                const Eigen::Vector3d& p2,
                                const Eigen::Vector3d& e2,
                                double sigma,
                                bool& isValid,             // NOLINT
                                bool& isParallel,          // NOLINT
                                TriangulationDiagnostics* diagnostics);
}  // namespace triangulation
}  // namespace okvis

#endif /* INCLUDE_OKVIS_TRIANGULATION_STEREO_TRIANGULATION_HPP_ */
