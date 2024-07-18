import argparse
from typing import cast

import ssepoptim.datasets as _
from ssepoptim.dataset import SpeechSeparationDatasetFactory


class Arguments:
    download_all: bool
    dataset: str


def main():
    parser = argparse.ArgumentParser(
        description="Speech Separation Optimization Framework - Dataset downloader"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        dest="download_all",
        action="store_true",
        required=False,
    )
    group.add_argument(
        "dataset",
        choices=SpeechSeparationDatasetFactory.list_entries(),
        help="name of the dataset",
        nargs="?"
    )
    args: Arguments = cast(Arguments, parser.parse_args())
    SpeechSeparationDatasetFactory.download(args.dataset)


if __name__ == "__main__":
    main()
