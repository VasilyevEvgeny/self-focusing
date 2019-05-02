from core.libs import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_root_dir")
    parser.add_argument("--gif")
    parser.add_argument("--global_results_dir_name")

    return parser.parse_args()