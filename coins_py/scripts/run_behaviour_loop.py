from __future__ import annotations

import argparse
from coins_py.options import coins_options
from coins_py.subject_analysis import loop_coins_analyse_behaviour
from coins_py.group_analysis import (
    coins_group_post_jump_adjustments,
    coins_group_reaction_times,
    coins_group_regression_kernels_session_wise,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-dir', type=str, default=None)
    args = parser.parse_args()
    options = coins_options(args.main_dir)
    loop_coins_analyse_behaviour(options)
    coins_group_post_jump_adjustments(options)
    coins_group_reaction_times(options)
    coins_group_regression_kernels_session_wise(options)


if __name__ == '__main__':
    main()
