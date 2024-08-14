#include <torch/extension.h>


std::vector<torch::Tensor> unique_argmin_forward_cuda(torch::Tensor input);
{
    auto outputs = torch::empty(input.size(0), torch::dtype(input.dtype()).requires_grad(false));
    auto rows = torch::empty(input.size(1), torch::dtype(torch::kInt64).requires_grad(false));
    auto cols = torch::empty(input.size(2), torch::dtype(torch::kInt64).requires_grad(false));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "unique_argmin_forward_cuda", ([&] {
        unique_argmin_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
        );
    }));
}

template <typename scalar_t>
__global__ unique_argmin_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> input)
{

}
