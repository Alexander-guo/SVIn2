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
 * @file stereo_triangulation.cpp
 * @brief Implementation of the triangulateFast function.
 * @author Stefan Leutenegger
 */

#include <algorithm>
#include <cmath>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>
#include <okvis/triangulation/stereo_triangulation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

/// \brief triangulation A namespace for operations related to triangulation.
namespace triangulation {

// Triangulate the intersection of two rays.
Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1,
                                const Eigen::Vector3d& e1,
                                const Eigen::Vector3d& p2,
                                const Eigen::Vector3d& e2,
                                double sigma,
                                bool& isValid,       // NOLINT
                                bool& isParallel) {  // NOLINT
  return triangulateFast(p1, e1, p2, e2, sigma, isValid, isParallel, nullptr);
}

Eigen::Vector4d triangulateFast(const Eigen::Vector3d& p1,
                                const Eigen::Vector3d& e1,
                                const Eigen::Vector3d& p2,
                                const Eigen::Vector3d& e2,
                                double sigma,
                                bool& isValid,                  // NOLINT
                                bool& isParallel,               // NOLINT
                                TriangulationDiagnostics* diagnostics) {
  constexpr double kMinimumParallaxRadians = 0.25 * M_PI / 180.0;
  constexpr double kInputEpsilon = 1.0e-12;
  const Eigen::Vector4d invalidPoint = Eigen::Vector4d::Zero();

  TriangulationDiagnostics localDiagnostics;
  TriangulationDiagnostics& result = diagnostics ? *diagnostics : localDiagnostics;
  result = TriangulationDiagnostics();
  isValid = false;
  isParallel = false;

  if (!p1.allFinite() || !p2.allFinite() || !e1.allFinite() || !e2.allFinite() || !std::isfinite(sigma) ||
      sigma <= 0.0 || e1.norm() <= kInputEpsilon || e2.norm() <= kInputEpsilon) {
    result.status = TriangulationStatus::InvalidInput;
    return invalidPoint;
  }

  const Eigen::Vector3d ray1 = e1.normalized();
  const Eigen::Vector3d ray2 = e2.normalized();
  const Eigen::Vector3d t12 = p2 - p1;
  result.baseline = t12.norm();
  const double rayDot = std::max(-1.0, std::min(1.0, ray1.dot(ray2)));
  const double rayCrossNorm = ray1.cross(ray2).norm();
  result.parallaxAngle = std::atan2(rayCrossNorm, rayDot);
  result.minimumParallaxAngle = std::max(2.0 * sigma, kMinimumParallaxRadians);
  result.minimumBaselineRangeRatio = 2.0 * std::sin(0.5 * result.minimumParallaxAngle);

  const auto bearingOnlyPoint = [&]() {
    const Eigen::Vector3d directionSum = ray1 + ray2;
    if (!directionSum.allFinite() || directionSum.norm() <= kInputEpsilon) {
      return invalidPoint;
    }
    const Eigen::Vector3d direction = directionSum.normalized();
    return Eigen::Vector4d(direction.x(), direction.y(), direction.z(), 0.0);
  };

  if (result.parallaxAngle < result.minimumParallaxAngle) {
    isParallel = true;
    result.status = TriangulationStatus::InsufficientParallax;
    const Eigen::Vector4d direction = bearingOnlyPoint();
    if (direction.isZero()) {
      return invalidPoint;
    }
    isValid = true;
    result.type = TriangulationType::BearingOnly;
    return direction;
  }

  // stolen and adapted from the Kneip toolchain
  // geometric_vision/include/geometric_vision/triangulation/impl/triangulation.hpp
  Eigen::Vector2d b;
  b[0] = t12.dot(ray1);
  b[1] = t12.dot(ray2);
  Eigen::Matrix2d A;
  A(0, 0) = ray1.dot(ray1);
  A(1, 0) = ray1.dot(ray2);
  A(0, 1) = -A(1, 0);
  A(1, 1) = -ray2.dot(ray2);

  bool invertible;
  Eigen::Matrix2d A_inverse;
  A.computeInverseWithCheck(A_inverse, invertible, 1.0e-6);
  if (!invertible) {
    isParallel = true;
    result.status = TriangulationStatus::Singular;
    return invalidPoint;
  }
  const Eigen::Vector2d lambda = A_inverse * b;

  const Eigen::Vector3d xm = lambda[0] * ray1 + p1;
  const Eigen::Vector3d xn = lambda[1] * ray2 + p2;
  const Eigen::Vector3d midpoint = (xm + xn) / 2.0;
  if (!lambda.allFinite() || !midpoint.allFinite()) {
    result.status = TriangulationStatus::InvalidInput;
    return invalidPoint;
  }

  const double signedRange1 = (midpoint - p1).dot(ray1);
  const double signedRange2 = (midpoint - p2).dot(ray2);
  result.range1 = (midpoint - p1).norm();
  result.range2 = (midpoint - p2).norm();
  if (signedRange1 <= 0.0 || signedRange2 <= 0.0) {
    result.status = TriangulationStatus::BehindRay;
    return invalidPoint;
  }

  const double maximumRange = std::max(result.range1, result.range2);
  if (!std::isfinite(maximumRange) || maximumRange <= kInputEpsilon) {
    result.status = TriangulationStatus::InvalidInput;
    return invalidPoint;
  }
  result.baselineRangeRatio = result.baseline / maximumRange;
  if (!std::isfinite(result.baselineRangeRatio) ||
      result.baselineRangeRatio < result.minimumBaselineRangeRatio) {
    result.status = TriangulationStatus::InsufficientBaselineRangeRatio;
    const Eigen::Vector4d direction = bearingOnlyPoint();
    if (direction.isZero()) {
      return invalidPoint;
    }
    isValid = true;
    isParallel = true;
    result.type = TriangulationType::BearingOnly;
    return direction;
  }

  // check it
  const Eigen::Vector3d error = midpoint - xm;
  const Eigen::Vector3d diff = midpoint - (p1 + 0.5 * t12);
  const double diff_sq = diff.dot(diff);
  const double chi2Denominator = diff_sq * sigma * sigma;
  if (!std::isfinite(chi2Denominator) || chi2Denominator <= kInputEpsilon) {
    result.status = TriangulationStatus::InvalidInput;
    return invalidPoint;
  }
  result.chi2 = error.dot(error) / chi2Denominator;

  if (!std::isfinite(result.chi2) || result.chi2 > 9.0) {
    result.status = TriangulationStatus::RayMismatch;
    return invalidPoint;
  }

  isValid = true;
  result.type = TriangulationType::Finite;
  result.status = TriangulationStatus::Success;
  return Eigen::Vector4d(midpoint[0], midpoint[1], midpoint[2], 1.0).normalized();
}

}  // namespace triangulation

}  // namespace okvis
