# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(test_operator
               test_qdq.dp.cpp
               test_cast_transpose.dp.cpp
               test_transpose.dp.cpp
               test_cast_transpose_dbias.dp.cpp
               test_cast_transpose_dbias_dgelu.dp.cpp
               test_cast_transpose_dgeglu.dp.cpp
               test_gelu.dp.cpp
               test_geglu.dp.cpp
               test_dgeglu.dp.cpp
               test_layernorm.dp.cpp
               test_rmsnorm.dp.cpp
               test_multi_cast_transpose.dp.cpp
               test_causal_softmax.dp.cpp
               ../test_common.dp.cpp)

list(APPEND test_operator_LINKER_LIBS  GTest::gtest_main ${TE_LIB})

target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS})
target_compile_options(test_operator PRIVATE -O2)

include(GoogleTest)
gtest_discover_tests(test_operator)
