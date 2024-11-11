# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

add_executable(
	#test_operator
	#test_qdq.cpp
	#test_cast_transpose.cpp
	#test_transpose.cpp
	#test_cast_transpose_dbias.cpp
	#test_cast_transpose_dbias_dgelu.cpp
	#test_cast_transpose_dgeglu.cpp
               test_gelu.cpp
               test_geglu.cpp
               test_dgeglu.cpp
	       #test_layernorm.cpp
	       #test_rmsnorm.cpp
	       #test_multi_cast_transpose.cpp
	       #test_causal_softmax.cpp
               ../test_common.cpp)

list(APPEND test_operator_LINKER_LIBS  GTest::gtest_main ${TE_LIB})

target_link_libraries(test_operator PUBLIC ${test_operator_LINKER_LIBS})
target_compile_options(test_operator PRIVATE -O2)

include(GoogleTest)
gtest_discover_tests(test_operator)
