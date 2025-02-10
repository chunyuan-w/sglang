#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

void all_reduce(
    torch::Tensor input,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group) {
  std::vector<torch::Tensor> tensors = {input};
  process_group->allreduce(tensors)->wait();
  return;
}
