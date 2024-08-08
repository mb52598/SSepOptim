#include <torch/extension.h>


template<typename T>
std::vector<torch::Tensor> unique_argmin_forward_cpu(torch::Tensor input)
{
    assert(input.size(0) == input.size(1));
    const auto input_a = input.accessor<T, 2>();

    const auto channels = input.size(0);
    T lowest_sum = 0;
    auto best_row_indexes = torch::empty(channels,
        torch::dtype(torch::kInt64).requires_grad(false));
    auto best_column_indexes = torch::empty(channels,
        torch::dtype(torch::kInt64).requires_grad(false));
    
    auto row_indexes = torch::empty(channels,
        torch::dtype(torch::kInt64).requires_grad(false));
    auto row_indexes_a = row_indexes.accessor<int64_t, 1>();

    auto column_indexes = torch::empty(channels,
        torch::dtype(torch::kInt64).requires_grad(false));
    auto column_indexes_a = column_indexes.accessor<int64_t, 1>();

    std::vector<bool> indexes(channels, false);

    for (int64_t i = 0; i < channels; ++i)
    {
        T sum = 0;
        auto row_index = i;
        for (int64_t j = 0; j < channels; ++j)
        {
            T lowest_value = 0;
            int64_t lowest_index = channels;
            for (int64_t k = 0; k < channels; ++k)
            {
                if (indexes[k])
                {
                    continue;
                }
                auto value = input_a[row_index][k];
                if (lowest_index == channels || value < lowest_value)
                {
                    lowest_value = value;
                    lowest_index = k;
                }
            }

            sum += lowest_value;
            indexes[lowest_index] = true;
            row_indexes_a[j] = row_index;
            column_indexes_a[j] = lowest_index;

            if (++row_index >= channels)
            {
                row_index = 0;
            }
        }
        if (i == 0 || sum < lowest_sum)
        {
            lowest_sum = sum;
            best_row_indexes.copy_(row_indexes);
            best_column_indexes.copy_(column_indexes);
        }
        std::fill(indexes.begin(), indexes.end(), false);
    }

    return {
        torch::tensor(lowest_sum, torch::dtype(input.dtype()).requires_grad(false)),
        best_row_indexes,
        best_column_indexes
    };
}

/**
 * @brief Calculate argmin of tensor where all the indexes are unique
 * 
 * @param input Input tensor of shape [channel, channel]
 * @return torch::Tensor Resulting tensor of shape [1]
 */
std::vector<torch::Tensor> unique_argmin_forward(torch::Tensor input)
{
    if (input.device().is_cuda())
    {

    }
    if (input.dtype() == torch::kFloat16)
    {
        return unique_argmin_forward_cpu<torch::Half>(input);
    }
    else if (input.dtype() == torch::kFloat32)
    {
        return unique_argmin_forward_cpu<float>(input);
    }
    else if (input.dtype() == torch::kFloat64)
    {
        return unique_argmin_forward_cpu<double>(input);
    }
    else
    {
        throw std::runtime_error("Invalid tensor dtype");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unique_argmin_forward, "Unique argmin forward");
}
