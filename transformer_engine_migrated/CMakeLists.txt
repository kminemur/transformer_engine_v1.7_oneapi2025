# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_SYCL_ARCHITECTURES)
  set(CMAKE_SYCL_ARCHITECTURES 70 80 89 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(transformer_engine LANGUAGES  CXX)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64_gen,nvptx64-nvidia-cuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=spir64")
# include(dpct.cmake)

# set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} --threads 4 -fsycl -fsycl-targets=spir64_gen,nvptx64-nvidia-cuda")
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} --threads 4 -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

# find_package(IntelSYCL REQUIRED)
find_package(MKL REQUIRED)

# Check for cuDNN frontend API
#set(CUDNN_FRONTEND_INCLUDE_DIR
#    "${CMAKE_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
#if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
#    message(FATAL_ERROR
#            "Could not find cuDNN frontend API. "
#            "Try running 'git submodule update --init --recursive' "
#            "within the Transformer Engine source.")
#endif()
#include(${CMAKE_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)

# find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(common)
if(NVTE_WITH_USERBUFFERS)
    message(STATUS "userbuffers support enabled")
    add_subdirectory(pytorch/csrc/userbuffers)
endif()


# option(ENABLE_JAX "Enable JAX in the building workflow." OFF)
# message(STATUS "JAX support: ${ENABLE_JAX}")
# if(ENABLE_JAX)
#   find_package(pybind11 CONFIG REQUIRED)
#   add_subdirectory(jax)
# endif()
