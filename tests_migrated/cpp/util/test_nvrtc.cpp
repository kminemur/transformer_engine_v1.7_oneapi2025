/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "util/rtc.h"

using namespace transformer_engine;

TEST(UtilTest, NVRTC) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (!rtc::is_enabled()) {
    GTEST_SKIP() << "NVRTC not enabled, skipping tests";
  }

  // GPU data buffer
  int *device_buffer;
  std::vector<int> host_buffer(2);
  device_buffer = sycl::malloc_device<int>(2, q_ct1); // NOLINT(*)
  q_ct1.memset(device_buffer, 0, 2 * sizeof(int)).wait();

  // CUDA kernel implementations
  const char code1[] = R"code(
#include <cuda_runtime.h>
__global__ void my_kernel(int2 *data) {
  data->x = 123;
  data->y = -456;
}
)code";
  const char code2[] = R"code(
#include "utils.cuh"
__global__ void my_kernel(uint32_t *data) {
  data[0] = 789;
  data[1] = 12;
}
)code";

  // Make sure kernels are not available
  auto& nvrtc_manager = rtc::KernelManager::instance();
  EXPECT_FALSE(nvrtc_manager.is_compiled("my gtest kernel1"));
  EXPECT_FALSE(nvrtc_manager.is_compiled("my gtest kernel2"));
  EXPECT_THROW(nvrtc_manager.launch("my gtest kernel1", 1, 1, 0, 0,
                                    device_buffer),
               std::runtime_error);
  EXPECT_THROW(nvrtc_manager.launch("my gtest kernel2", 1, 1, 0, 0,
                                    device_buffer),
               std::runtime_error);

  // Compile and run first kernel
  EXPECT_NO_THROW(nvrtc_manager.compile("my gtest kernel1",
                                        "my_kernel",
                                        code1,
                                        "test_nvrtc_kernel1.cu"));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel1"));
  EXPECT_FALSE(nvrtc_manager.is_compiled("my gtest kernel2"));
  EXPECT_NO_THROW(nvrtc_manager.launch("my gtest kernel1", 1, 1, 0, 0,
                                       device_buffer));
  EXPECT_EQ(DPCT_CHECK_ERROR(
                q_ct1.memcpy(host_buffer.data(), device_buffer, 2 * sizeof(int))
                    .wait()),
            0);
  EXPECT_EQ(host_buffer[0], 123);
  EXPECT_EQ(host_buffer[1], -456);

  // Compile and run second kernel
  EXPECT_NO_THROW(nvrtc_manager.compile("my gtest kernel2",
                                        "my_kernel",
                                        code2,
                                        "test_nvrtc_kernel2.cu"));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel1"));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel2"));
  EXPECT_NO_THROW(nvrtc_manager.launch("my gtest kernel2", 1, 1, 0, 0, device_buffer));
  EXPECT_EQ(DPCT_CHECK_ERROR(
                q_ct1.memcpy(host_buffer.data(), device_buffer, 2 * sizeof(int))
                    .wait()),
            0);
  EXPECT_EQ(host_buffer[0], 789);
  EXPECT_EQ(host_buffer[1], 12);
}
