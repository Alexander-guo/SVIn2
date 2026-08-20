#include <gtest/gtest.h>
#include <math.h>

#include <chrono>
#include <okvis/DenseMatcher.hpp>
#include <thread>
#include <utility>
#include <vector>

class TestMatchingAlgorithm : public okvis::MatchingAlgorithm {
 public:
  TestMatchingAlgorithm() {}
  virtual ~TestMatchingAlgorithm() {}

  /// \brief this will be called exactly once for each call to DenseMatcher::match()
  virtual void doSetup() {}

  /// \brief what is the size of list A?
  virtual size_t sizeA() const { return listA.size(); }
  /// \brief what is the size of list B?
  virtual size_t sizeB() const { return listB.size(); }

  /// distances above this threshold will not be returned as matches.
  virtual float distanceThreshold() const { return 4.0f; }

  /// by which factor does the first best match has to be better than the second best one.
  virtual float distanceRatioThreshold() const { return 3.0f; }

  /// \brief Should we skip the item in list A? This will be called once for each item in the list
  virtual bool skipA(size_t indexA) const { return indexA == 0; }

  /// \brief Should we skip the item in list B? This will be called many times.
  virtual bool skipB(size_t /* indexB */) const { return false; }

  /// \brief the "distance" between the two points.
  ///        For points that absolutely don't match. Please use float max.
  virtual float distance(size_t indexA, size_t indexB) const {
    double diff = listA[indexA] - listB[indexB];
    return fabs(diff);
  }

  /// \brief a function that tells you how many times setMatching() will be called.
  virtual void reserveMatches(size_t numMatches) {
    matches.clear();
    matches.reserve(numMatches);
  }

  /// \brief At the end of the matching step, this function is called once
  ///        for each pair of matches discovered.
  virtual void setBestMatch(size_t indexA, size_t indexB, double /* distance */) {
    matches.push_back(std::make_pair(indexA, indexB));
  }

  std::vector<double> listA;
  std::vector<double> listB;
  std::vector<std::pair<int, int> > matches;
};

class EqualDistanceConflictMatchingAlgorithm : public okvis::MatchingAlgorithm {
 public:
  EqualDistanceConflictMatchingAlgorithm()
      : distances_(4, std::vector<float>(4, 5.0f)), delayedSource_(0) {
    distances_[0][0] = 1.0f;
    distances_[1][0] = 1.0f;
    distances_[1][1] = 2.0f;
    distances_[2][2] = 1.0f;
    distances_[3][3] = 1.0f;
  }

  void doSetup() override { matches.clear(); }

  size_t sizeA() const override { return distances_.size(); }
  size_t sizeB() const override { return distances_.front().size(); }

  float distanceThreshold() const override { return 4.0f; }
  float distanceRatioThreshold() const override { return 3.0f; }

  float distance(size_t indexA, size_t indexB) const override {
    // Alternate which of the conflicting workers is delayed so both lock
    // acquisition orders are exercised by the repeated test.
    if (indexA == delayedSource_) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return distances_[indexA][indexB];
  }

  void reserveMatches(size_t numMatches) override {
    matches.clear();
    matches.reserve(numMatches);
  }

  void setBestMatch(size_t indexA, size_t indexB, double /* distance */) override {
    matches.push_back(std::make_pair(static_cast<int>(indexA), static_cast<int>(indexB)));
  }

  void setDelayedSource(size_t indexA) { delayedSource_ = indexA; }

  std::vector<std::pair<int, int> > matches;

 private:
  std::vector<std::vector<float> > distances_;
  size_t delayedSource_;
};

TEST(DenseMatcherTestSuite, denseMatcherTest) {
  TestMatchingAlgorithm tma;

  tma.listA.push_back(1.0);
  tma.listA.push_back(3.0);
  tma.listA.push_back(2.0);
  tma.listA.push_back(0.9);

  /// This shouldn't be matched because 18 - 4 > 4.0
  tma.listB.push_back(18.0);
  tma.listB.push_back(2.1);
  tma.listB.push_back(4.0);
  // This shouldn't be matched with listA[0] because of skipping listA[0]
  // So, it will be matches with listA[3]
  tma.listB.push_back(1.0);

  // We should have 1 --> 2 and 2 --> 1

  okvis::DenseMatcher matcher;

  matcher.match(tma);

  ASSERT_EQ(3u, tma.matches.size());

  for (size_t i = 0; i < tma.matches.size(); ++i) {
    switch (tma.matches[i].first) {
      case 1:
        ASSERT_EQ(2, tma.matches[i].second);
        break;
      case 2:
        ASSERT_EQ(1, tma.matches[i].second);
        break;
      case 3:
        ASSERT_EQ(3, tma.matches[i].second);
        break;
      default:
        FAIL() << "Unexpected match " << tma.matches[i].first << " --> " << tma.matches[i].second;
    }
  }
}

TEST(DenseMatcherTestSuite, denseMatcherDistanceRatioTest) {
  TestMatchingAlgorithm tma;

  tma.listA.push_back(8.0);
  tma.listA.push_back(1.0);
  tma.listA.push_back(3.0);
  tma.listA.push_back(2.0);
  tma.listA.push_back(0.9);

  /// This shouldn't be matched because 16/15 < 4
  tma.listB.push_back(18.0);
  tma.listB.push_back(2.1);
  /// This shouldn't be matched because 2/1 < 4
  tma.listB.push_back(4.0);
  tma.listB.push_back(1.0);

  // This shouldn't be matched with listA[0] because of skipping listA[0]
  tma.listB.push_back(7.0);

  // We should have 1 --> 2 and 2 --> 1
  bool useDistanceRatioThreshold = true;
  okvis::DenseMatcher matcher(4, 4, useDistanceRatioThreshold);

  matcher.match(tma);

  ASSERT_EQ(2u, tma.matches.size());

  for (size_t i = 0; i < tma.matches.size(); ++i) {
    switch (tma.matches[i].first) {
      case 1:
        ASSERT_EQ(3, tma.matches[i].second);
        break;
      case 3:
        ASSERT_EQ(1, tma.matches[i].second);
        break;
      default:
        FAIL() << "Unexpected match " << tma.matches[i].first << " --> " << tma.matches[i].second;
    }
  }
}

TEST(DenseMatcherTestSuite, denseMatcherEqualDistanceConflictIsDeterministic) {
  EqualDistanceConflictMatchingAlgorithm tma;
  okvis::DenseMatcher matcher(4);
  const std::vector<std::pair<int, int> > expected{{0, 0}, {1, 1}, {2, 2}, {3, 3}};

  for (size_t repeat = 0; repeat < 128; ++repeat) {
    tma.setDelayedSource(repeat % 2);
    matcher.match(tma);

    ASSERT_EQ(expected.size(), tma.matches.size()) << "repeat " << repeat;
    EXPECT_EQ(expected, tma.matches) << "repeat " << repeat;
  }
}

TEST(DenseMatcherTestSuite, denseMatcherEqualDistanceConflictPreservesRatioThreshold) {
  EqualDistanceConflictMatchingAlgorithm tma;
  okvis::DenseMatcher matcher(4, 4, true);
  const std::vector<std::pair<int, int> > expected{{0, 0}, {2, 2}, {3, 3}};

  for (size_t repeat = 0; repeat < 32; ++repeat) {
    tma.setDelayedSource((repeat + 1) % 2);
    matcher.match(tma);

    ASSERT_EQ(expected.size(), tma.matches.size()) << "repeat " << repeat;
    EXPECT_EQ(expected, tma.matches) << "repeat " << repeat;
  }
}
