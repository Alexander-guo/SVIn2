cmake_minimum_required(VERSION 3.8.11)
project(okvis_frontend)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()    

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# build the library
add_library(${PROJECT_NAME} 
  src/Frontend.cpp
  src/VioKeyframeWindowMatchingAlgorithm.cpp
  src/stereo_triangulation.cpp
  src/ProbabilisticStereoTriangulator.cpp
  src/FrameNoncentralAbsoluteAdapter.cpp
  src/FrameRelativeAdapter.cpp
  src/SpatialKeypointBalancer.cpp
  include/okvis/Frontend.hpp
  include/okvis/VioKeyframeWindowMatchingAlgorithm.hpp
  include/okvis/triangulation/stereo_triangulation.hpp
  include/okvis/triangulation/ProbabilisticStereoTriangulator.hpp
  include/okvis/SpatialKeypointBalancer.hpp
  include/opengv/absolute_pose/FrameNoncentralAbsoluteAdapter.hpp
  include/opengv/relative_pose/FrameRelativeAdapter.hpp
  include/opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp
  include/opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp)

# and link it                 
target_link_libraries(${PROJECT_NAME}
  PRIVATE ${BRISK_LIBRARIES} 
  PRIVATE ${OpenGV_LIBRARIES} 
  PRIVATE ${CERES_LIBRARIES}
  PUBLIC okvis_util
  PUBLIC okvis_cv 
  PUBLIC okvis_ceres 
  PUBLIC okvis_timing 
  PUBLIC okvis_matcher)

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")

# testing
if(BUILD_TESTS)
  if(APPLE)
    add_definitions(-DGTEST_HAS_TR1_TUPLE=1)
  else()
    add_definitions(-DGTEST_HAS_TR1_TUPLE=0)
  endif(APPLE)
  enable_testing()
  set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
  add_executable(${PROJECT_TEST_NAME}
    test/runTests.cpp
    test/TestDeferredBearingTracks.cpp
    test/TestDeterministicFeatureProduction.cpp
    test/TestDeterministicOpenGVRansac.cpp
    test/TestOpposingMulticamKeyframePolicy.cpp
    test/TestRansacPoseDiagnostics.cpp
    test/TestSpatialKeypointBalancer.cpp
    test/TestStereoTriangulation.cpp
  )
  target_link_libraries(${PROJECT_TEST_NAME}
    ${PROJECT_NAME}
    gtest
    gtest_main
    pthread)
  add_test(test ${PROJECT_TEST_NAME})
endif()
