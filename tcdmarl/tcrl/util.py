from pathlib import Path
import random

import numpy as np

# from transformers import pipeline, set_seed

# from consts import *


def random_from(xs: list):
    num = len(xs)
    i = np.random.randint(num)
    return xs[i]


def set_all_seeds(seed: int):
    # set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_lines(path: Path, lines: list[str]):
    with open(path, "w") as f:
        f.writelines(lines)
        print(f"wrote {len(lines)} lines to {path}")


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)
