from sgl_kernel.ops.cpu._kernels import all_reduce as _all_reduce


def custom_reduce_cpu(inp, group):
    _all_reduce(inp, group)