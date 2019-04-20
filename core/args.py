from core.libs import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_root_dir")

    return parser.parse_args()