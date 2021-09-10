import argparse

RESULT_DIR = "../data/results/"
DATADIR = "../data/"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="lastfm-small",
    help="dataset name: same as the directory name of your dataset (lastfm-small/delicious/douban)",
)
parser.add_argument(
    "--method",
    default="userCF",
    help="itemAvg | userCF | friendsCF | communityCF | privaCTCF | all",
)
parser.add_argument("--eps_l", type=float, default=0.5, help="privacy loss for HCT")
parser.add_argument("--seed", type=int, default=-1, help="seed for the HCT")
parser.add_argument(
    "--seedSplit", type=int, default=42, help="seed for the test train split"
)
parser.add_argument(
    "--tSplitRatio", type=float, default=0.1, help="test train split ratio"
)
parser.add_argument("--K", type=int, default=100, help="top K for evaluation")
parser.add_argument(
    "--max_r", type=float, default=None, help="maximum rating in the dataset"
)
parser.add_argument(
    "--cold",
    type=str2bool,
    nargs="?",
    const=True,
    default=True,
    help="Evaluate for only cold start users",
)
args = parser.parse_args()
