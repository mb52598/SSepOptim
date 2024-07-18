import argparse
from typing import cast

from ssepoptim.main import main as ssepoptim_main


class Arguments:
    config_paths: list[str]


def main():
    parser = argparse.ArgumentParser(
        description="Speech Separation Optimization Framework"
    )
    parser.add_argument(
        "-cfg",
        "--configuration-path",
        dest="config_paths",
        type=str,
        default=["config/default.ini"],
        help="path to the configuration file",
        nargs="+",
    )
    args = cast(Arguments, parser.parse_args())
    for config_path in args.config_paths:
        ssepoptim_main(config_path)


if __name__ == "__main__":
    main()
