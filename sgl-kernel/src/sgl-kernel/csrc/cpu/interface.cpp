// TODO: this is initially taken from shm_interface.cpp in deepspeed
// Seems we need to adapt it to be same as ccl.cpp
#include <torch/extension.h>

#include "shm.h"

// Communication settings
static int world_rank = -1;
static int world_size = -1;

static bool is_initialized = 0;

static bool all_ranks_local_p = false;

void initialize(int size, int rank)
{
    if (is_initialized) return;

    // Check whether all ranks is on the same physical machine.
    // If true, we will use an SHM based low latency allreduce

    auto ls_string = std::getenv("LOCAL_SIZE");
    int ls = 0;
    if (ls_string != NULL) { ls = std::stoi(std::getenv("LOCAL_SIZE")); }

    if (size >= 1 && size == ls) { all_ranks_local_p = true; }

    world_size = size;
    world_rank = rank;
    is_initialized = 1;

    auto addr_string = std::getenv("MASTER_ADDR");
    if (addr_string == NULL) { addr_string = ""; }
    auto port_string = std::getenv("MASTER_PORT");
    if (port_string == NULL) { port_string = ""; }

    if (all_ranks_local_p) { shm_initialize(size, rank, addr_string, port_string); }
}

void inference_all_reduce(torch::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group) {
    // TODO: check ReduceOp is SUM
    
    auto numel = data.numel();

    int data_size = 0;
    bool data_type_fallback = false;

    switch (data.scalar_type()) {
        case c10::ScalarType::BFloat16: data_size = numel * 2; break;
        case c10::ScalarType::Float: data_size = numel * 4; break;
        default: data_type_fallback = true;
    }

    if (data_type_fallback || !all_ranks_local_p) {
        // TODO: add fallback
        TORCH_CHECK(false, "Data type not supported or ranks are not local.");
    } else {
        all_reduce_outer_loop(data, numel, data_size);

        // TODO: impl the above func and remove this call
        std::vector<torch::Tensor> tensors = {data};
        process_group->allreduce(tensors)->wait();

    }

    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("initialize", &initialize, "shm initialize");
    // m.def("inference_all_reduce", &inference_all_reduce, "low latency all_reduce implementation");
    m.def("all_reduce", &all_reduce, "All reduce (CPU)");
    m.def("inference_all_reduce", &inference_all_reduce, "low latency all_reduce implementation");
}