#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#include <torch/types.h>
#include <torch/extension.h>

#define None torch::indexing::None
#define Slice torch::indexing::Slice

#define W 64
#define H 16
// length of each slice
#define S 4

template <typename scalar_t>
__global__ void kernel_process_slice(
    scalar_t *indices,
    const scalar_t *endSlices,
    int *sizeSlice,
    const unsigned int *lx,
    const unsigned int N,
    const unsigned int T)
{
    // index of the slice
    unsigned int ns = blockIdx.x * W + threadIdx.x;
    unsigned int start = ns * S;
    unsigned int n = blockIdx.y * H + threadIdx.y;

    // skip kernel out of boundary.
    if (n >= N || start >= lx[n])
        return;

    indices += n * T;
    sizeSlice += n * ((T + S - 1) / S) + ns;
    scalar_t last = 0xFFFFFFFF;
    if (ns > 0)
    {
        last = endSlices[n * (T / S) + ns - 1];
    }
    unsigned int p_cur = start;
    unsigned int end = (start + S > lx[n]) ? lx[n] : start + S;
    for (; start < end; start++)
    {
        if (indices[start] != last)
        {
            if (indices[start] == 0)
            {
                last = 0;
                continue;
            }
            if (p_cur != start)
            {
                indices[p_cur] = indices[start];
            }
            p_cur++;
            last = indices[start];
        }
    }
    (*sizeSlice) = p_cur - (ns * S);
    return;
}

template <typename scalar_t>
void run_process_slice(
    scalar_t *indices,
    const scalar_t *endSlices,
    int *sizeSlice,
    const unsigned int *lx,
    const unsigned int N, const unsigned int T)
{
    dim3 threads(W, H);
    dim3 blocks(((T + S - 1) / S + W - 1) / W, (N + H - 1) / H);

    kernel_process_slice<scalar_t><<<blocks, threads>>>(indices, endSlices, sizeSlice, lx, N, T);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CTC align: slice proceesing error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return;
}

template <typename scalar_t>
__global__ void kernel_shift(
    scalar_t *src,
    scalar_t *dst,
    int *sliceSize,
    int *cumSize,
    const unsigned int N,
    const unsigned int T,
    const unsigned int N_Slice)
{

    unsigned int t = blockIdx.x * W + threadIdx.x;
    unsigned int n = blockIdx.y * H + threadIdx.y;
    unsigned int ns = t / S;

    // skip kernel out of boundary.
    if (n >= N || ns >= N_Slice)
        return;

    cumSize += n * N_Slice + ns;
    sliceSize += n * N_Slice + ns;
    if ((*sliceSize) == 0 || t >= (ns * S) + *sliceSize)
        return;

    src += n * T + t;
    dst += n * T + (*cumSize) + (t - ns * S);
    *dst = *src;
    return;
}

template <typename scalar_t>
void run_units_shift(
    scalar_t *src,
    scalar_t *dst,
    int *sliceSize,
    int *cumSize,
    const unsigned int N,
    const unsigned int T)
{
    dim3 threads(W, H);
    dim3 blocks((T + W - 1) / W, (N + H - 1) / H);

    kernel_shift<scalar_t><<<blocks, threads>>>(src, dst, sliceSize, cumSize, N, T, (T + S - 1) / S);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CTC align: shifting error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    return;
}

std::tuple<torch::Tensor, torch::Tensor> run_align(
    torch::Tensor &indices,
    const torch::Tensor &lx)
{
    const auto N = indices.size(0);
    const auto T = indices.size(1);
    const auto device = indices.device();
    auto temp = indices.clone();

    // store last element of each slice
    auto endSlices = indices.index({"...", Slice(S - 1, None, S)}).contiguous();
    auto slicesize = torch::zeros({N, (T + S - 1) / S}, torch::device(device).dtype(torch::kInt32));

    // process slices, get valid data in each slices
    switch (indices.scalar_type())
    {
    case torch::kInt32:
        run_process_slice<int>(
            (int *)temp.data_ptr<int>(),
            (const int *)endSlices.data_ptr<int>(),
            (int *)slicesize.data_ptr<int>(),
            (const unsigned int *)lx.data_ptr<int>(),
            N, T);
        break;
    case torch::kInt64:
        run_process_slice<long>(
            (long *)temp.data_ptr<long>(),
            (const long *)endSlices.data_ptr<long>(),
            (int *)slicesize.data_ptr<int>(),
            (const unsigned int *)lx.data_ptr<int>(),
            N, T);
        break;
    default:
        break;
    }

    // squeeze and assign
    // get begin of memory location of each slice
    auto cumSize = torch::roll(slicesize.cumsum(1, torch::kInt32), 1, 1).contiguous();
    cumSize.index_put_({"...", 0}, 0);

    // zero out indices for debugging
    // indices.zero_();

    switch (indices.scalar_type())
    {
    case torch::kInt32:
        run_units_shift<int>(
            (int *)temp.data_ptr<int>(),
            (int *)indices.data_ptr<int>(),
            (int *)slicesize.data_ptr<int>(),
            (int *)cumSize.data_ptr<int>(),
            N, T);
        break;
    case torch::kInt64:
        run_units_shift<long>(
            (long *)temp.data_ptr<long>(),
            (long *)indices.data_ptr<long>(),
            (int *)slicesize.data_ptr<int>(),
            (int *)cumSize.data_ptr<int>(),
            N, T);
        break;
    default:
        break;
    }
    auto lens = slicesize.sum(1);
    return std::make_tuple(indices, lens);
}
