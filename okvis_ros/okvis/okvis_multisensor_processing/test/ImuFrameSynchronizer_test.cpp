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
 *  Created on: Sep 13, 2014
 *      Author: Pascal Gohl
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

#include "okvis/ImuFrameSynchronizer.hpp"

#include <chrono>
#include <future>
#include <thread>

/// \brief okvis Main namespace of this package.
namespace okvis {} /* namespace okvis */

TEST(ImuFrameSynchronizerTest, AlreadyPublishedBoundaryIsObservedWithoutASecondNotification) {
  okvis::ImuFrameSynchronizer synchronizer;
  const okvis::Time frameStamp(10, 0);
  synchronizer.gotImuData(frameStamp);

  const std::future<bool> result =
      std::async(std::launch::async, [&synchronizer, &frameStamp]() {
        return synchronizer.waitForUpToDateImuData(frameStamp);
      });

  ASSERT_EQ(result.wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
  EXPECT_TRUE(result.get());
}

TEST(ImuFrameSynchronizerTest, ConcurrentWaitersObserveTheSameMonotonicTimestamp) {
  okvis::ImuFrameSynchronizer synchronizer;
  const okvis::Time frameStamp(20, 0);
  std::future<bool> first = std::async(std::launch::async, [&synchronizer, &frameStamp]() {
    return synchronizer.waitForUpToDateImuData(frameStamp);
  });
  std::future<bool> second = std::async(std::launch::async, [&synchronizer, &frameStamp]() {
    return synchronizer.waitForUpToDateImuData(frameStamp);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  synchronizer.gotImuData(frameStamp);

  ASSERT_EQ(first.wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
  ASSERT_EQ(second.wait_for(std::chrono::milliseconds(100)), std::future_status::ready);
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
}
