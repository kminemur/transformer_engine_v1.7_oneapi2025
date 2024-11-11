cmake_minimum_required(VERSION 3.24)

find_package(Catch2 QUIET)

if(NOT Catch2_FOUND)
    Include(FetchContent)

    # Fetch and build catch2
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Find cudnn
include(${CMAKE_SOURCE_DIR}/cmake/cuDNN.cmake)

add_executable(
    samples
    
    cpp/mha.cpp.dp.cpp
    cpp/convolutions.cpp.dp.cpp
    cpp/dgrads.cpp.dp.cpp
    cpp/matmuls.cpp.dp.cpp
    cpp/batchnorm.cpp.dp.cpp
    cpp/layernorm.cpp.dp.cpp
    cpp/rmsnorm.cpp.dp.cpp
    cpp/wgrads.cpp.dp.cpp
    cpp/serialization.cpp.dp.cpp
    cpp/autotuning.cpp.dp.cpp
    cpp/pointwise.cpp.dp.cpp
    cpp/resample.cpp.dp.cpp

    legacy_samples/conv_sample.cpp.dp.cpp 
    legacy_samples/resnet_test_list.cpp.dp.cpp
    legacy_samples/resnet_sample.cpp.dp.cpp
    legacy_samples/test_list.cpp.dp.cpp 
    legacy_samples/fp16_emu.cpp.dp.cpp 
    legacy_samples/helpers.cpp.dp.cpp 
    legacy_samples/fusion_sample.cpp.dp.cpp 
    legacy_samples/fp8_sample.cpp.dp.cpp 
    legacy_samples/norm_samples.cpp.dp.cpp
    legacy_samples/fused_mha_sample.cpp.dp.cpp
    legacy_samples/f16_flash_mha_sample.cpp.dp.cpp
    legacy_samples/fp8_flash_mha_sample.cpp.dp.cpp
)

if(DEFINED ENV{NO_DEFAULT_IN_SWITCH})
    message("Default case in the switch is disabled")
    add_compile_definitions(NO_DEFAULT_IN_SWITCH)
endif()

if (MSVC)
    target_compile_options(
        samples PRIVATE
        /W4 /WX # warning level 3 and all warnings as errors
        /wd4100 # allow unused parameters
        /wd4458 # local hides class member (currently a problem for all inline setters)
        /wd4505 # unreferenced function with internal linkage has been removed
        /wd4101 /wd4189 # unreferenced local
        /bigobj # increase number of sections in .Obj file
    )
else()
    target_compile_options(
        samples PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wno-unused-function
    )
endif()

target_link_libraries(
    samples

    cudnn_frontend
    _cudnn_frontend_pch
    Catch2::Catch2WithMain

    CUDNN::cudnn_all
)

# cuDNN dlopen's its libraries
# Add all libraries in link line as NEEDED
set_target_properties(
    samples PROPERTIES LINK_WHAT_YOU_USE TRUE
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_source_files_properties(
    legacy_samples/resnet_test_list.cpp.dp.cpp legacy_samples/resnet_sample.cpp.dp.cpp
    PROPERTIES 
    include_directories "${PROJECT_SOURCE_DIR}/samples/legacy_samples/resnet_block/include"
)
