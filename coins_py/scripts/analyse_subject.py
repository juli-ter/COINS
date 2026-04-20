from __future__ import annotations

import argparse
from coins_py.options import coins_options
from coins_py.subject_analysis import coins_analyse_subject_behaviour


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('sub_id', type=int)
    parser.add_argument('--main-dir', type=str, default=None)
    args = parser.parse_args()
    options = coins_options(args.main_dir)
    coins_analyse_subject_behaviour(args.sub_id, options)


if __name__ == '__main__':
    main()
