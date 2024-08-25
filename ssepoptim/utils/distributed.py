import os

import torch.distributed as dist


def get_global_rank() -> int:
    return int(os.environ["RANK"])


def get_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def is_distributed() -> bool:
    return dist.is_torchelastic_launched()
