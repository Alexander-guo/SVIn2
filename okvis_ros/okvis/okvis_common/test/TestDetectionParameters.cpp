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
    if (line.find("    enable_cross_camera_matching:") == 0) continue;
    if (line == "threading_options:" || line == "diagnostics_options:" ||
        line == "keyframe_selection:") {
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

TEST(DetectionParameters, ReadsNormalizedCameraMaskRectangle) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());

  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  bool inserted = false;
  while (std::getline(input, line)) {
    output << line << '\n';
    if (!inserted && line.find("image_dimension:") != std::string::npos) {
      output << "        mask: [[0.1, 0.2], [0.6, 0.7]],\n";
      inserted = true;
    }
  }
  output.close();
  ASSERT_TRUE(inserted);

  const okvis::VioParameters parameters = readParameters(temporary.path());
  ASSERT_EQ(parameters.nCameraSystem.numCameras(), 1u);
  const auto camera = parameters.nCameraSystem.cameraGeometry(0);
  ASSERT_TRUE(camera->hasMask());
  ASSERT_EQ(camera->mask().size(), cv::Size(832, 832));
  EXPECT_EQ(camera->mask().at<unsigned char>(165, 82), 0);
  EXPECT_EQ(camera->mask().at<unsigned char>(166, 83), 255);
  EXPECT_EQ(camera->mask().at<unsigned char>(582, 499), 255);
  EXPECT_EQ(camera->mask().at<unsigned char>(582, 500), 0);
}

TEST(DetectionParameters, CameraMaskDefaultsToDisabledWhenOmitted) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("        mask:") == 0) continue;
    output << line << '\n';
  }
  output.close();

  const okvis::VioParameters parameters = readParameters(temporary.path());
  ASSERT_EQ(parameters.nCameraSystem.numCameras(), 1u);
  EXPECT_FALSE(parameters.nCameraSystem.cameraGeometry(0)->hasMask());
}

TEST(DetectionParameters, CameraMaskRectanglesAreIndependentPerCamera) {
  std::ifstream input(OKVIS_TEST_X5_DUAL_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  size_t camera = 0;
  while (std::getline(input, line)) {
    output << line << '\n';
    if (line.find("image_dimension:") == std::string::npos) continue;
    if (camera == 0) {
      output << "        mask: [[0.1, 0.2], [0.3, 0.4]],\n";
    } else if (camera == 1) {
      output << "        mask: [[0.6, 0.7], [0.8, 0.9]],\n";
    }
    ++camera;
  }
  output.close();
  ASSERT_EQ(camera, 2u);

  const okvis::VioParameters parameters = readParameters(temporary.path());
  ASSERT_EQ(parameters.nCameraSystem.numCameras(), 2u);
  const auto camera0 = parameters.nCameraSystem.cameraGeometry(0);
  const auto camera1 = parameters.nCameraSystem.cameraGeometry(1);
  ASSERT_TRUE(camera0->hasMask());
  ASSERT_TRUE(camera1->hasMask());
  EXPECT_EQ(camera0->mask().at<unsigned char>(200, 100), 255);
  EXPECT_EQ(camera1->mask().at<unsigned char>(200, 100), 0);
  EXPECT_EQ(camera0->mask().at<unsigned char>(700, 600), 0);
  EXPECT_EQ(camera1->mask().at<unsigned char>(700, 600), 255);
}

TEST(DetectionParameters, RejectsInvalidNormalizedCameraMaskRectangle) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  std::string line;
  bool inserted = false;
  while (std::getline(input, line)) {
    output << line << '\n';
    if (!inserted && line.find("image_dimension:") != std::string::npos) {
      output << "        mask: [[0.7, 0.2], [0.6, 1.1]],\n";
      inserted = true;
    }
  }
  output.close();
  ASSERT_TRUE(inserted);
  EXPECT_THROW(readParameters(temporary.path()), okvis::VioParametersReader::Exception);
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
  bool insertedCrossCameraPolicy = false;
  bool insertedCrossCameraDiagnostic = false;
  while (std::getline(input, line)) {
    if (line.find("    camera_rate:") == 0) {
      output << line << '\n';
      output << "    enable_cross_camera_matching: false\n";
      insertedCrossCameraPolicy = true;
    } else if (line.find("    matcherThreads:") == 0) {
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
      output << "    crossCameraMatching: true\n";
      output << "    keyframeSelection: true\n";
      insertedCrossCameraDiagnostic = true;
    } else {
      output << line << '\n';
    }
  }
  output.close();
  ASSERT_TRUE(insertedCrossCameraPolicy);
  ASSERT_TRUE(insertedCrossCameraDiagnostic);

  const okvis::VioParameters parameters = readParameters(temporary.path());
  EXPECT_TRUE(parameters.optimization.finiteLandmarkRetention);
  EXPECT_EQ(parameters.threading.matcherThreads, 8);
  EXPECT_EQ(parameters.threading.estimatorThreads, 4);
  EXPECT_FALSE(parameters.sensors_information.enableCrossCameraMatching);
  EXPECT_FALSE(parameters.diagnostics.drift);
  EXPECT_FALSE(parameters.diagnostics.retention);
  EXPECT_TRUE(parameters.diagnostics.image);
  EXPECT_TRUE(parameters.diagnostics.imuWindow);
  EXPECT_TRUE(parameters.diagnostics.landmarkPromotion);
  EXPECT_TRUE(parameters.diagnostics.bearingTracking);
  EXPECT_TRUE(parameters.diagnostics.triangulation);
  EXPECT_TRUE(parameters.diagnostics.reprojection);
  EXPECT_TRUE(parameters.diagnostics.crossCameraMatching);
  EXPECT_TRUE(parameters.diagnostics.keyframeSelection);
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
  EXPECT_TRUE(parameters.sensors_information.enableCrossCameraMatching);
  EXPECT_FALSE(parameters.diagnostics.drift);
  EXPECT_FALSE(parameters.diagnostics.retention);
  EXPECT_FALSE(parameters.diagnostics.image);
  EXPECT_FALSE(parameters.diagnostics.imuWindow);
  EXPECT_FALSE(parameters.diagnostics.landmarkPromotion);
  EXPECT_FALSE(parameters.diagnostics.bearingTracking);
  EXPECT_FALSE(parameters.diagnostics.triangulation);
  EXPECT_FALSE(parameters.diagnostics.reprojection);
  EXPECT_FALSE(parameters.diagnostics.crossCameraMatching);
  EXPECT_FALSE(parameters.diagnostics.keyframeSelection);
  EXPECT_FALSE(parameters.keyframeSelection.opposingMulticam);
  EXPECT_EQ(parameters.keyframeSelection.minimumFrames, 3);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumSeconds, 0.10);
  EXPECT_EQ(parameters.keyframeSelection.maximumFrames, 15);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.maximumSeconds, 0.75);
  EXPECT_EQ(parameters.keyframeSelection.unhealthyConsecutiveFrames, 2);
  EXPECT_EQ(parameters.keyframeSelection.minimumAssociated, 30);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumBearingCoverage, 0.30);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumBearingScatter, 0.15);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumPersistentThreeFraction, 0.10);
}

TEST(DetectionParameters, ReadsOpposingMulticamKeyframePolicy) {
  std::ifstream input(OKVIS_TEST_X5_CONFIG);
  ASSERT_TRUE(input.good());
  TemporaryConfig temporary;
  std::ofstream output(temporary.path());
  ASSERT_TRUE(output.good());
  output << input.rdbuf();
  output << "\nkeyframe_selection:\n"
         << "    opposingMulticam: true\n"
         << "    minimumFrames: 4\n"
         << "    minimumSeconds: 0.2\n"
         << "    maximumFrames: 18\n"
         << "    maximumSeconds: 0.9\n"
         << "    unhealthyConsecutiveFrames: 3\n"
         << "    minimumAssociated: 35\n"
         << "    minimumBearingCoverage: 0.32\n"
         << "    minimumBearingScatter: 0.17\n"
         << "    minimumPersistentThreeFraction: 0.12\n";
  output.close();

  const okvis::VioParameters parameters = readParameters(temporary.path());
  EXPECT_TRUE(parameters.keyframeSelection.opposingMulticam);
  EXPECT_EQ(parameters.keyframeSelection.minimumFrames, 4);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumSeconds, 0.2);
  EXPECT_EQ(parameters.keyframeSelection.maximumFrames, 18);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.maximumSeconds, 0.9);
  EXPECT_EQ(parameters.keyframeSelection.unhealthyConsecutiveFrames, 3);
  EXPECT_EQ(parameters.keyframeSelection.minimumAssociated, 35);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumBearingCoverage, 0.32);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumBearingScatter, 0.17);
  EXPECT_DOUBLE_EQ(parameters.keyframeSelection.minimumPersistentThreeFraction, 0.12);
}
