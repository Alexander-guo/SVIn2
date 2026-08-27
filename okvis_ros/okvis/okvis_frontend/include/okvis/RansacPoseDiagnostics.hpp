#ifndef INCLUDE_OKVIS_RANSACPOSEDIAGNOSTICS_HPP_
#define INCLUDE_OKVIS_RANSACPOSEDIAGNOSTICS_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
namespace ransac_pose_diagnostics {

inline bool enabledFromEnvironmentValue(const char* value) {
  return value != nullptr && std::strcmp(value, "1") == 0;
}

inline double rotationDeltaDegrees(const Eigen::Matrix3d& lhs,
                                   const Eigen::Matrix3d& rhs) {
  const Eigen::Matrix3d delta = lhs.transpose() * rhs;
  const double cosine = std::max(-1.0, std::min(1.0, (delta.trace() - 1.0) * 0.5));
  return std::acos(cosine) * 180.0 / M_PI;
}

// OpenGV's relative-pose model in this frontend is T_CaCb. This is the same
// convention used by the existing (startup-only) pose initialization path.
inline kinematics::Transformation estimatorRelativeCameraPose(
    const kinematics::Transformation& T_WSa,
    const kinematics::Transformation& T_SCa,
    const kinematics::Transformation& T_WSb,
    const kinematics::Transformation& T_SCb) {
  return T_SCa.inverse() * T_WSa.inverse() * T_WSb * T_SCb;
}

inline double directionCosine(const Eigen::Vector3d& lhs,
                              const Eigen::Vector3d& rhs) {
  constexpr double kMinimumNorm = 1.0e-9;
  if (!lhs.allFinite() || !rhs.allFinite() || lhs.norm() < kMinimumNorm ||
      rhs.norm() < kMinimumNorm) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::max(-1.0, std::min(1.0, lhs.normalized().dot(rhs.normalized())));
}

struct Eligibility {
  bool eligible = false;
  const char* reason = "diagnostics_disabled";
};

inline Eligibility relativeRotationEligibility(bool diagnosticsEnabled,
                                                size_t correspondences,
                                                size_t inliers,
                                                double inlierRatio,
                                                double rotationDeltaDeg) {
  if (!diagnosticsEnabled) return {false, "diagnostics_disabled"};
  if (correspondences < 30) return {false, "few_correspondences"};
  if (inliers < 20) return {false, "few_inliers"};
  if (!std::isfinite(inlierRatio) || inlierRatio < 0.6) return {false, "low_inlier_ratio"};
  if (!std::isfinite(rotationDeltaDeg) || rotationDeltaDeg > 15.0) {
    return {false, "rotation_disagreement"};
  }
  return {true, "pass"};
}

inline Eligibility absolutePoseEligibility(bool diagnosticsEnabled,
                                           size_t correspondences,
                                           size_t inliers,
                                           double inlierRatio,
                                           double rotationDeltaDeg,
                                           double translationDeltaMeters) {
  if (!diagnosticsEnabled) return {false, "diagnostics_disabled"};
  if (correspondences < 10) return {false, "few_correspondences"};
  if (inliers < 10) return {false, "few_inliers"};
  if (!std::isfinite(inlierRatio) || inlierRatio < 0.5) return {false, "low_inlier_ratio"};
  if (!std::isfinite(rotationDeltaDeg) || rotationDeltaDeg > 15.0) {
    return {false, "rotation_disagreement"};
  }
  if (!std::isfinite(translationDeltaMeters) || translationDeltaMeters > 1.0) {
    return {false, "translation_disagreement"};
  }
  return {true, "pass"};
}

// Emit every low-support frame and a short recovery burst. At normal support,
// retain startup samples and one sample per diagnostic second.
class RateLimiter {
 public:
  bool shouldEmit(bool diagnosticsEnabled, uint64_t frameId, int64_t second,
                  size_t finiteSupport) {
    if (!diagnosticsEnabled) return false;
    if (finiteSupport <= 15) {
      recoveryFrames_ = 10;
      return true;
    }
    if (recoveryFrames_ > 0) {
      --recoveryFrames_;
      return true;
    }
    if (frameId < 30 || second != lastSecond_) {
      lastSecond_ = second;
      return true;
    }
    return false;
  }

 private:
  int64_t lastSecond_ = std::numeric_limits<int64_t>::min();
  size_t recoveryFrames_ = 0;
};

}  // namespace ransac_pose_diagnostics
}  // namespace okvis

#endif  // INCLUDE_OKVIS_RANSACPOSEDIAGNOSTICS_HPP_
