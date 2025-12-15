import sys

import argparse

import argparse
import sys


from .cuda import _wait_gpu_run_register


def main():
    parser = argparse.ArgumentParser(
        description='PhoenixCat CLI',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title='command',
        dest='command',  # 必须有，用于判断用户输入了哪个子命令
        required=True,
    )

    _wait_gpu_run_register(subparsers)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
