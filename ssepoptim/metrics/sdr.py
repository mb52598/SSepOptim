import torch


@torch.compile
def scale_invariant_signal_to_distortion_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool,
) -> torch.Tensor:
    eps = torch.finfo(prediction.dtype).eps

    if zero_mean:
        prediction = prediction - torch.mean(prediction, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

    optimal_scaling = torch.sum(target * prediction, dim=-1, keepdim=True) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )

    projection = optimal_scaling * target

    noise = prediction - projection

    ratio = torch.sum(projection**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps)

    return 10 * torch.log10(ratio + eps)
