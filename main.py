import argparse
import os
from typing import cast

from ssepoptim.main import main as ssepoptim_main


class Arguments:
    config_path: str


def main():
    parser = argparse.ArgumentParser(
        description="Speech Separation Optimization Framework"
    )
    parser.add_argument(
        "-cfg",
        "--configuration-path",
        dest="config_path",
        type=str,
        default="config/default.ini",
        help="path to the configuration file or folder",
    )
    args = cast(Arguments, parser.parse_args())
    if os.path.isdir(args.config_path):
        config_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(args.config_path)
            for file in files
        ]
    else:
        config_paths = [args.config_path]
    for i, config_path in enumerate(config_paths):
        ssepoptim_main(config_path, i, len(config_paths))


if __name__ == "__main__":
    main()
