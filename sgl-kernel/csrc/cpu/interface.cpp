#include <ATen/record_function.h>
#include <torch/all.h>

#include <string>

#include "shm.h"

// Communication settings
static int world_rank = -1;
static int world_size = -1;

static bool is_initialized = false;

static bool all_ranks_local_p = false;

void initialize(int64_t size, int64_t rank) {
  if (is_initialized) {
    return;
  }

  // Check whether all ranks is on the same physical machine.
  // If true, we will use an SHM based low latency allreduce

  auto ls_string = std::getenv("LOCAL_SIZE");
  int ls = 0;
  if (ls_string != NULL) {
    ls = std::stoi(std::getenv("LOCAL_SIZE"));
  }

  if (size >= 1 && size == ls) {
    all_ranks_local_p = true;
  }

  world_size = size;
  world_rank = rank;
  is_initialized = true;

  const char* addr_string = std::getenv("MASTER_ADDR");
  if (addr_string == NULL) {
    addr_string = "";
  }
  const char* port_string = std::getenv("MASTER_PORT");
  if (port_string == NULL) {
    port_string = "";
  }
  // When several sglang replicas run on one host (sglang_router.launch_server
  // --dp-size N spawns N tp groups in the same process family), MASTER_ADDR /
  // MASTER_PORT are often equal or unset across replicas, and shm.cpp derives
  // its /dev/shm segment names from those two strings plus rank. Two replicas'
  // rank 0 workers then map the *same* segment, four TP ranks share two
  // buffers, the internal collective sequence counter desyncs, and decode
  // hangs. Fold SGLANG_DP_RANK (set by sglang_router.launch_server per
  // replica) into port_string so each replica gets its own segment. Harmless
  // when unset (single-replica) — port_string stays as-is.
  const char* dp_rank_string = std::getenv("SGLANG_DP_RANK");
  std::string port_with_dp;
  if (dp_rank_string != NULL) {
    port_with_dp.assign(port_string);
    port_with_dp.push_back('_');
    port_with_dp.append(dp_rank_string);
    port_string = port_with_dp.c_str();
  }

  if (all_ranks_local_p) {
    shm_initialize(size, rank, addr_string, port_string);
  }
}

void shm_allreduce(torch::Tensor& data, int64_t op) {
  TORCH_CHECK(op == c10d::ReduceOp::SUM, "Only torch.distributed.ReduceOp.SUM is supported");

  auto numel = data.numel();
  int data_size = numel * data.element_size();
  all_reduce_outer_loop(data, numel, data_size);

  return;
}

torch::Tensor shm_allgather(torch::Tensor& data, int64_t dim) {
  auto numel = data.numel();
  int data_size = numel * data.element_size();
  if (dim < 0) {
    dim += data.dim();
  }
  std::vector<int64_t> result_shape = data.sizes().vec();
  result_shape[dim] *= world_size;
  torch::Tensor result_tensor = torch::empty(result_shape, data.options());
  return all_gather(result_tensor, data, dim, numel, data_size);
}
