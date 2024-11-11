#include <dpct/dnnl_utils.hpp>
/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Matmul autotuning", "[matmul][graph][autotuning]") {
    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }
    namespace fe = cudnn_frontend;

    // matmul problem size
    int64_t const b = 16;
    int64_t const m = 32;
    int64_t const n = 64;
    int64_t const k = 128;

    // Initialize input tensors
    Surface<sycl::half> A_gpu(b * m * k, false);
    Surface<sycl::half> B_gpu(b * k * n, false);
    Surface<sycl::half> C_gpu(b * m * n, false);

    int64_t a_uid = 0, b_uid = 1, c_uid = 2;

    dpct::dnnl::engine_ext handle;
    checkCudnnErr(DPCT_CHECK_ERROR(handle.create_engine()));

    auto create_graph = [&]() -> fe::graph::Graph {
        // Make cudnn graph
        fe::graph::Graph graph{};

        // Create the two non-virtual input tensors A and B.
        // There are read from global memory.
        auto A_attributes = fe::graph::Tensor_attributes()
                                .set_name("A")
                                .set_dim({b, m, k})
                                .set_stride({m * k, k, 1})
                                .set_uid(a_uid)
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto A            = graph.tensor(A_attributes);
        auto B_attributes = fe::graph::Tensor_attributes()
                                .set_name("B")
                                .set_dim({b, k, n})
                                .set_stride({k * n, n, 1})
                                .set_uid(b_uid)
                                .set_data_type(fe::DataType_t::BFLOAT16);
        auto B = graph.tensor(B_attributes);

        auto matmul_attributes =
            fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
        auto C = graph.matmul(A, B, matmul_attributes);
        C->set_output(true).set_uid(c_uid).set_data_type(fe::DataType_t::BFLOAT16);

        REQUIRE(graph.validate().is_good());

        REQUIRE(graph.build_operation_graph(dpct::dnnl::engine_ext).is_good());

        REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph.check_support(handle).is_good());

        return graph;
    };

    auto graph = create_graph();

    graph.deselect_workspace_greater_than(0);
    auto plan_count = graph.get_execution_plan_count();
    std::cout << "Graph has " << plan_count << " plan candidates." << std::endl;

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::ALL).is_good());

    std::unordered_map<int64_t, void*> variant_pack = {
        {a_uid, A_gpu.devPtr}, {b_uid, B_gpu.devPtr}, {c_uid, C_gpu.devPtr}};

    auto autotune = [&]() -> int64_t {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
        try {
    const int iter_count = 10;
        dpct::event_ptr start, stop;
        start = new sycl::event();
        stop  = new sycl::event();
        dev_ct1.queues_wait_and_throw();

        dpct::queue_ptr stream = &dpct::get_in_order_queue();
        stream                 = handle.get_queue();

        std::vector<float> execution_times;
        execution_times.resize(plan_count, 10.0f);  // Some arbitrary high time

        int64_t workspace_size = 0;
        for (auto i = 0; i < plan_count; i++) {
            workspace_size = std::max(workspace_size, graph.get_workspace_size_plan_at_index(i));
        }

        Surface<int8_t> workspace(workspace_size, false);

        for (auto i = 0; i < plan_count; i++) {
            float time_ms = 0.0f;

            auto warmup_status = graph.execute_plan_at_index(handle, variant_pack, workspace.devPtr, i);

            if (warmup_status.is_bad()) {
                std::cout << "Plan at index " << i << " failed execution " << warmup_status.get_message() << std::endl;
                continue;
            }
            dev_ct1.queues_wait_and_throw();

            dpct::sync_barrier(start, stream);
            for (int iter = 0; iter < iter_count; iter++) {
                auto status = graph.execute_plan_at_index(handle, variant_pack, workspace.devPtr, i);
                (void)status;
            }
            dpct::sync_barrier(stop, stream);
            stop->wait_and_throw();
            time_ms = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() -
                       start->get_profiling_info<sycl::info::event_profiling::command_start>()) /
                      1000000.0f;

            std::cout << "Plan at index " << i << " took " << time_ms / iter_count << " ms." << std::endl;
            execution_times[i] = time_ms / iter_count;
        }

        return std::distance(std::begin(execution_times),
                             std::min_element(std::begin(execution_times), std::end(execution_times)));
    }
    catch (sycl::exception const &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }
    };
    // Run cudnn graph

    auto candidate_index = autotune();

    std::cout << "Successful candidate is at index " << candidate_index << std::endl;

    REQUIRE(graph.build_plan_at_index(handle, candidate_index).is_good());

    Surface<int8_t> workspace(graph.get_workspace_size_plan_at_index(candidate_index), false);

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
    /*
    DPCT1027:1171: The call to cudnnDestroy was replaced with 0 because this functionality is redundant in SYCL.
    */
    checkCudnnErr(0);
}