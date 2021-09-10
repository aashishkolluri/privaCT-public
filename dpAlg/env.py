import argparse
import dill as pickle
import sys

pickle.HIGHEST_PROTOCOL = 2

import resource
# May segfault without this line. 0x100 is a guess at the size of each stack frame.
max_rec = 0x1000000
resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)


# Give the location of the data (with the '/' at the end)
DATA_HOME = "../data/"

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
    default="facebook",
    help="dataset name: same as the directory name of your dataset (facebook/pgp)",
)
parser.add_argument(
    "--dataFormat", default="txt", help="the format of the datafile(edgefile)"
)
parser.add_argument("--dfPath", default=None, help="The path to the edgefile")
parser.add_argument("--treePath", default=None, help="The path to the saved HCT")
parser.add_argument(
    "--all",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Find HRG models for all epsilons",
)
parser.add_argument(
    "--createHCT",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="create a HCT model",
)
parser.add_argument("--eps_l", type=float, default=0.5, help="privacy loss for HCT")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument(
    "--seed", type=int, default=1234, help="Level at which to find communities"
)
parser.add_argument(
    "--dp",
    type=str2bool,
    nargs="?",
    const=True,
    default=True,
    help="apply DP on counts",
)
parser.add_argument(
    "--onlydp",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="do not create HCT for non dp",
)
parser.add_argument(
    "--globalHRG",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Create HRG for global",
)

args = parser.parse_args()