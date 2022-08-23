#include <tuple>
#include <string>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> run_align(
    torch::Tensor &indices,
    const torch::Tensor &lx);

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x)          \
    TORCH_CHECK((x).is_contiguous(), \
                #x " must be contiguous")

#define CHECK_CUDA(x)                   \
    TORCH_CHECK((x).device().is_cuda(), \
                #x " must be located in the CUDA")

#define CHECK_INT(x)                                \
    TORCH_CHECK((x).scalar_type() == torch::kInt32, \
                #x " must be a Int tensor")

std::tuple<torch::Tensor, torch::Tensor> align(
    torch::Tensor &indices,
    const torch::Tensor &lengths)
{
    // check contiguous
    CHECK_CONTIGUOUS(indices);
    CHECK_CONTIGUOUS(lengths);
    // check types
    CHECK_INT(lengths);
    TORCH_CHECK(
        (indices.scalar_type() == torch::kInt32) or
            (indices.scalar_type() == torch::kInt64),
        "indices must be a tensor of torch.int16/32/64 dtype.")
    // check device
    CHECK_CUDA(indices);
    CHECK_CUDA(lengths);
    // check input dimensions and shapes
    TORCH_CHECK(indices.dim() == 2, "indices must have 2 dimensions (N, T)")
    TORCH_CHECK(indices.size(0) == lengths.size(0), "indices and lengths must be of equal size in dim 0.")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(indices));

    return run_align(indices, lengths);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "align_",
        &align,
        "CUDA based CTC alignment",
        pybind11::arg("indices"),
        pybind11::arg("lengths"));
}
