# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Configure Transformer Engine library
set(transformer_engine_SOURCES)
list(APPEND transformer_engine_SOURCES
     transformer_engine.dp.cpp # ok 
    # #  transpose/cast_transpose.dp.cpp # ok but requires rtc files
    # #  transpose/transpose.dp.cpp # ok but requires rtc files
    #  transpose/cast_transpose_fusion.dp.cpp # ok 
    #  transpose/transpose_fusion.dp.cpp # ok 
    #  transpose/multi_cast_transpose.dp.cpp # ok
    # activation/gelu.dp.cpp # ok
    # activation/relu.dp.cpp # ok
    # activation/swiglu.dp.cpp # ok
    # # #  fused_attn/fused_attn_f16_max512_seqlen.dp.cpp # need to fix cuda-front ends
    # # #  fused_attn/fused_attn_f16_arbitrary_seqlen.dp.cpp # need to fix cuda-front ends
    # # #  fused_attn/fused_attn_fp8.dp.cpp # need to fix cuda-front ends
    # # #  fused_attn/fused_attn.dp.cpp # need to fix cuda-front ends
    # # #  fused_attn/utils.dp.cpp # need to fix cuda-front ends
    # #  gemm/cublaslt_gemm.dp.cpp # fail due to some updates happening
    #  layer_norm/ln_api.dp.cpp # ok
    # # #  layer_norm/ln_bwd_semi_cuda_kernel.dp.cpp # data type issuess
    # # #  layer_norm/ln_fwd_cuda_kernel.dp.cpp # data type issues
    #  rmsnorm/rmsnorm_api.dp.cpp # ok
    # # #  rmsnorm/rmsnorm_bwd_semi_cuda_kernel.dp.cpp # data type issues
    # # #  rmsnorm/rmsnorm_fwd_cuda_kernel.dp.cpp # data type issues
    #  util/cast.dp.cpp # ok
    #  util/cuda_driver.dp.cpp # ok
    #  util/cuda_runtime.dp.cpp # ok
    # #  util/rtc.dp.cpp # rtc is not supported yet
    #  util/system.dp.cpp # ok
    # fused_softmax/scaled_masked_softmax.dp.cpp # ok
    # fused_softmax/scaled_upper_triang_masked_softmax.dp.cpp # ok
    # fused_softmax/scaled_aligned_causal_masked_softmax.dp.cpp # ok
    #  fused_rope/fused_rope.dp.cpp # ok
    #  recipe/delayed_scaling.dp.cpp #ok
     )
add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Configure dependencies
target_link_libraries(transformer_engine PUBLIC
                        ${MKL_LIB})
target_link_libraries(transformer_engine PUBLIC
                        ${DNN_LIB})
target_include_directories(transformer_engine PRIVATE
                           ${SYCL_INCLUDE_DIR})
# target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
# target_include_directories(transformer_engine PRIVATE
#                         "~/SYCLomatic/clang/runtime/dpct-rt/include")


# Make header files with C++ strings
function(make_string_header STRING STRING_NAME)
# DPCT3001:567: "configure_file" is used to copy an input file to an output file while performing transformations of the input file content. You may need to update the name of the input and output file if the input file contains CUDA syntax, and make sure the input file exists by copying or migrating the original input file from in-root folder.
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
    file(READ "${file_}" STRING)
# DPCT3001:569: "configure_file" is used to copy an input file to an output file while performing transformations of the input file content. You may need to update the name of the input and output file if the input file contains CUDA syntax, and make sure the input file exists by copying or migrating the original input file from in-root folder.
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
make_string_header("${cuda_include_path}"
                   string_path_cuda_include)
# make_string_header_from_file(transpose/rtc/cast_transpose.dp.cpp
#                              string_code_transpose_rtc_cast_transpose_sycl)
# make_string_header_from_file(transpose/rtc/transpose.dp.cpp
#                              string_code_transpose_rtc_transpose_sycl)
make_string_header_from_file(utils.dp.hpp
                             string_code_utils_sycl)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.dp.cpp
                            fused_softmax/scaled_upper_triang_masked_softmax.dp.cpp
                            fused_softmax/scaled_aligned_causal_masked_softmax.dp.cpp
                            PROPERTIES
                            COMPILE_OPTIONS "-ffast-math")
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS} -O3")

# Install library
install(TARGETS transformer_engine DESTINATION .)
