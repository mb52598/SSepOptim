import torch


def synchronize_device(device: torch.device):
    if device.type.startswith("cuda"):
        torch.cuda.synchronize(device)
