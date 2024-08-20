import os


def get_global_rank() -> int:
    return int(os.environ["RANK"])


def get_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])
