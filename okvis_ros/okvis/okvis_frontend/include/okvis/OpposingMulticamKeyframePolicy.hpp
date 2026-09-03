#ifndef INCLUDE_OKVIS_OPPOSINGMULTICAMKEYFRAMEPOLICY_HPP_
#define INCLUDE_OKVIS_OPPOSINGMULTICAMKEYFRAMEPOLICY_HPP_

#include <cmath>
#include <cstddef>

#include <okvis/Parameters.hpp>

namespace okvis {

enum class OpposingMulticamKeyframeDecision {
  MinSpacing,
  MaxSpacing,
  QualityDegraded,
  HealthySkip,
  DegradationPending,
};

inline OpposingMulticamKeyframeDecision opposingMulticamKeyframeDecision(
    const KeyframeSelectionParameters& policy,
    size_t processedFramesSinceKeyframe,
    bool timestampAvailable,
    double elapsedSeconds,
    bool rigHealthy,
    size_t unhealthyStreak) {
  const bool elapsedValid = timestampAvailable && std::isfinite(elapsedSeconds) &&
                            elapsedSeconds >= 0.0;
  if (processedFramesSinceKeyframe < static_cast<size_t>(policy.minimumFrames)) {
    return OpposingMulticamKeyframeDecision::MinSpacing;
  }
  // The frame bound remains authoritative if timestamps are unavailable or
  // invalid, preventing permanent keyframe starvation.
  if (processedFramesSinceKeyframe >= static_cast<size_t>(policy.maximumFrames)) {
    return OpposingMulticamKeyframeDecision::MaxSpacing;
  }
  if (!elapsedValid || elapsedSeconds < policy.minimumSeconds) {
    return OpposingMulticamKeyframeDecision::MinSpacing;
  }
  if (elapsedSeconds >= policy.maximumSeconds) {
    return OpposingMulticamKeyframeDecision::MaxSpacing;
  }
  if (unhealthyStreak >= static_cast<size_t>(policy.unhealthyConsecutiveFrames)) {
    return OpposingMulticamKeyframeDecision::QualityDegraded;
  }
  return rigHealthy ? OpposingMulticamKeyframeDecision::HealthySkip
                    : OpposingMulticamKeyframeDecision::DegradationPending;
}

inline const char* opposingMulticamKeyframeDecisionName(
    OpposingMulticamKeyframeDecision decision) {
  switch (decision) {
    case OpposingMulticamKeyframeDecision::MinSpacing: return "min_spacing";
    case OpposingMulticamKeyframeDecision::MaxSpacing: return "max_spacing";
    case OpposingMulticamKeyframeDecision::QualityDegraded: return "quality_degraded";
    case OpposingMulticamKeyframeDecision::HealthySkip: return "healthy_skip";
    case OpposingMulticamKeyframeDecision::DegradationPending: return "degradation_pending";
  }
  return "unknown";
}

inline bool opposingMulticamKeyframeInsert(
    OpposingMulticamKeyframeDecision decision) {
  return decision == OpposingMulticamKeyframeDecision::MaxSpacing ||
         decision == OpposingMulticamKeyframeDecision::QualityDegraded;
}

}  // namespace okvis

#endif  // INCLUDE_OKVIS_OPPOSINGMULTICAMKEYFRAMEPOLICY_HPP_
