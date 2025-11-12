#include <torch/all.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#ifndef __SHM_COLLECTIVES__
#define __SHM_COLLECTIVES__
#define VECTOR_LENGTH_IN_BYTES 32

// TODO: put this in .h and let shm.cpp also use it?
constexpr int STATE_GROUP_ALL_GATHER = 2;
constexpr int STATE_GROUP_ALL_GATHER_INTO_TENSOR = 3;

void shm_initialize(int size, int rank, const char* addr_string, const char* port_string);
void all_reduce_outer_loop(torch::Tensor& data, size_t numel, int data_size);
template <int STATE_GROUP>
torch::Tensor& all_gather(torch::Tensor& result, torch::Tensor& data, int dim, size_t numel, int data_size);
void reduce_scatter_outer_loop(torch::Tensor& output, torch::Tensor& data, size_t numel, int data_size);
#endif
