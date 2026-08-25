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
