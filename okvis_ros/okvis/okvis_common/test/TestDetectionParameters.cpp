#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

#include <okvis/Parameters.hpp>
#include <okvis/VioParametersReader.hpp>

namespace {

class TemporaryConfig {
 public:
  TemporaryConfig() {
    char path[] = "/tmp/okvis_detection_parameters_XXXXXX";
    const int descriptor = mkstemp(path);
    EXPECT_NE(descriptor, -1);
    if (descriptor != -1) close(descriptor);
    path_ = path;
  }

  ~TemporaryConfig() {
    if (!path_.empty()) std::remove(path_.c_str());
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

okvis::VioParameters readParameters(const std::string& path) {
  okvis::VioParametersReader reader(path);
  okvis::VioParameters parameters;
  reader.getParameters(parameters);
  return parameters;
}

void copyWithoutSpatialBalancing(std::istream& input, std::ostream& output) {
  std::string line;
  bool skippingSpatialBlock = false;
  while (std::getline(input, line)) {
    if (line == "    spatialBalancing:") {
      skippingSpatialBlock = true;
      continue;
    }
    if (skippingSpatialBlock && line.rfind("        ", 0) == 0) continue;
    if (skippingSpatialBlock && !line.empty()) skippingSpatialBlock = false;
    output << line << '\n';
  }
}

void copyWithoutRuntimeControls(std::istream& input, std::ostream& output) {
  std::string line;
  bool skippingBlock = false;
  while (std::getline(input, line)) {
    if (line.rfind("finiteLandmarkRetention:", 0) == 0) continue;
    if (line == "threading_options:" || line == "diagnostics_options:") {
      skippingBlock = true;
      continue;
    }
    if (skippingBlock && line.rfind("    ", 0) == 0) continue;
    if (skippingBlock && !line.empty()) skippingBlock = false;
    output << line << '\n';
  }
}

}  // namespace

TEST(DetectionParameters, ReadsExplicitBriskAbsoluteThreshold) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());

  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  bool inserted = false;
  while (std::getline(input, line)) {
    if (line.find("absoluteThreshold:") != std::string::npos) {
      output << "    absoluteThreshold: 750.5\n";
      inserted = true;
    } else {
      output << line << '\n';
      if (!inserted && line.find("    threshold:") != std::string::npos) {
        output << "    absoluteThreshold: 750.5\n";
        inserted = true;
      }
    }
  }
  output.close();
  ASSERT_TRUE(inserted);

  const okvis::VioParameters parameters = readParameters(temporary.path());
  EXPECT_DOUBLE_EQ(parameters.optimization.detectionAbsoluteThreshold, 750.5);
}

TEST(DetectionParameters, UsesHistoricalDefaultWhenAbsoluteThresholdIsOmitted) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());

  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("absoluteThreshold:") != std::string::npos) {
      continue;
    }
    output << line << '\n';
  }
  output.close();
  const okvis::VioParameters parameters = readParameters(temporary.path());
  EXPECT_DOUBLE_EQ(parameters.optimization.detectionAbsoluteThreshold, 800.0);
}

TEST(DetectionParameters, UsesSpatialBalancingDefaultsWhenBlockIsOmitted) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  copyWithoutSpatialBalancing(input, output);
  output.close();

  const okvis::VioParameters vioParameters = readParameters(temporary.path());
  const okvis::SpatialBalancingParams& parameters = vioParameters.optimization.spatialBalancing;
  EXPECT_FALSE(parameters.enable);
  EXPECT_EQ(parameters.rows, 4);
  EXPECT_EQ(parameters.cols, 4);
  EXPECT_EQ(parameters.candidateMultiplier, 3);
  EXPECT_EQ(parameters.minimumPerValidCell, 20);
  EXPECT_EQ(parameters.localPaddingPixels, 32);
}

TEST(DetectionParameters, ReadsEnabledSpatialBalancingAndLocalPadding) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  bool skippingSpatialBlock = false;
  while (std::getline(input, line)) {
    if (line == "    spatialBalancing:") {
      skippingSpatialBlock = true;
      continue;
    }
    if (skippingSpatialBlock && line.rfind("        ", 0) == 0) continue;
    skippingSpatialBlock = false;
    output << line << '\n';
    if (line.find("maxNoKeypoints:") != std::string::npos) {
      output << "    spatialBalancing:\n"
             << "        enable: true\n"
             << "        rows: 2\n"
             << "        cols: 3\n"
             << "        candidateMultiplier: 4\n"
             << "        minimumPerValidCell: 12\n"
             << "        localPaddingPixels: 48\n";
    }
  }
  output.close();

  const okvis::VioParameters vioParameters = readParameters(temporary.path());
  const okvis::SpatialBalancingParams& parameters = vioParameters.optimization.spatialBalancing;
  EXPECT_TRUE(parameters.enable);
  EXPECT_EQ(parameters.rows, 2);
  EXPECT_EQ(parameters.cols, 3);
  EXPECT_EQ(parameters.candidateMultiplier, 4);
  EXPECT_EQ(parameters.minimumPerValidCell, 12);
  EXPECT_EQ(parameters.localPaddingPixels, 48);
}

TEST(DetectionParameters, ReadsRuntimePolicyDiagnosticsAndThreadCounts) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("    matcherThreads:") == 0) {
      output << "    matcherThreads: 8\n";
    } else if (line.find("    estimatorThreads:") == 0) {
      output << "    estimatorThreads: 4\n";
    } else if (line.find("    image:") == 0) {
      output << "    image: true\n";
    } else if (line.find("    imuWindow:") == 0) {
      output << "    imuWindow: true\n";
    } else if (line.find("    landmarkPromotion:") == 0) {
      output << "    landmarkPromotion: true\n";
    } else if (line.find("    bearingTracking:") == 0) {
      output << "    bearingTracking: true\n";
    } else if (line.find("    triangulation:") == 0) {
      output << "    triangulation: true\n";
    } else if (line.find("    reprojection:") == 0) {
      output << "    reprojection: true\n";
    } else {
      output << line << '\n';
    }
  }
  output.close();

  const okvis::VioParameters parameters = readParameters(temporary.path());
  EXPECT_TRUE(parameters.optimization.finiteLandmarkRetention);
  EXPECT_EQ(parameters.threading.matcherThreads, 8);
  EXPECT_EQ(parameters.threading.estimatorThreads, 4);
  EXPECT_FALSE(parameters.diagnostics.drift);
  EXPECT_FALSE(parameters.diagnostics.retention);
  EXPECT_TRUE(parameters.diagnostics.image);
  EXPECT_TRUE(parameters.diagnostics.imuWindow);
  EXPECT_TRUE(parameters.diagnostics.landmarkPromotion);
  EXPECT_TRUE(parameters.diagnostics.bearingTracking);
  EXPECT_TRUE(parameters.diagnostics.triangulation);
  EXPECT_TRUE(parameters.diagnostics.reprojection);
}

TEST(DetectionParameters, UsesSafeRuntimeDefaultsWhenControlsAreOmitted) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  copyWithoutRuntimeControls(input, output);
  output.close();

  const okvis::VioParameters parameters = readParameters(temporary.path());
  EXPECT_TRUE(parameters.optimization.finiteLandmarkRetention);
  EXPECT_EQ(parameters.threading.matcherThreads, 4);
  EXPECT_EQ(parameters.threading.estimatorThreads, 2);
  EXPECT_FALSE(parameters.diagnostics.drift);
  EXPECT_FALSE(parameters.diagnostics.retention);
  EXPECT_FALSE(parameters.diagnostics.image);
  EXPECT_FALSE(parameters.diagnostics.imuWindow);
  EXPECT_FALSE(parameters.diagnostics.landmarkPromotion);
  EXPECT_FALSE(parameters.diagnostics.bearingTracking);
  EXPECT_FALSE(parameters.diagnostics.triangulation);
  EXPECT_FALSE(parameters.diagnostics.reprojection);
}
