#include "megakernel.cuh"

#include "config.cuh"

#include <iostream>
#include "pyutils/pyutils.cuh" // Kittens Python utilities
#include <pybind11/pybind11.h>

using namespace kittens;

struct globals {
    using instruction_layout = megakernel::instruction_layout<{{PROJECT_NAME_LOWER}}_config>;
    using timing_layout = megakernel::timing_layout<{{PROJECT_NAME_LOWER}}_config>;
    instruction_layout instructions;
    timing_layout timings;
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3({{PROJECT_NAME_LOWER}}_config::NUM_THREADS); }
    int dynamic_shared_memory() { return {{PROJECT_NAME_LOWER}}_config::DYNAMIC_SHARED_MEMORY; }
};

using state = megakernel::state<{{PROJECT_NAME_LOWER}}_config>;

struct TestOp {
    static constexpr int opcode = 1;
    struct controller {
        static __device__ int init_semaphores(const globals &g, state &s) {
            return 0;
        }
        static __device__ int release_lid(const globals &g, typename {{PROJECT_NAME_LOWER}}_config::instruction_t &instruction, int &query) {
            return query;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state &s) {
            if(laneid() == 0) { printf("Hello, world from {{PROJECT_NAME_LOWER}}!\n"); }
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state &s) {
            // Wait and release pages
            if(laneid() < {{PROJECT_NAME_LOWER}}_config::NUM_PAGES) {
                s.wait_page_ready(laneid());
                s.finish_page(laneid(), {{PROJECT_NAME_LOWER}}_config::NUM_CONSUMER_WARPS);
            }
#ifdef KITTENS_BLACKWELL
            else if(laneid() == {{PROJECT_NAME_LOWER}}_config::NUM_PAGES) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, {{PROJECT_NAME_LOWER}}_config::NUM_CONSUMER_WARPS);
            }
#endif
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state &s) {}
    };
    struct storer {
        static __device__ void run(const globals &g, state &s) {}
    };
};

PYBIND11_MODULE({{PROJECT_NAME_LOWER}}, m)
{
    m.doc() = "";
    kittens::py::bind_kernel<megakernel::mk<{{PROJECT_NAME_LOWER}}_config, globals, TestOp>>(m, "example_megakernel",
                                                  &globals::instructions,
                                                  &globals::timings);
}