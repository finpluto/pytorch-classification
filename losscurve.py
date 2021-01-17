'''
Loss curve visualization from training log file.
Author: Finpluto
'''
from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Loss curve visualization util")
parser.add_argument("--path", "-p", default="./checkpoint/log.txt",
                    help="path to the log.txt file.", metavar="SRC")
parser.add_argument("--filename", "-f", default="save",
                    help="name of saving figure.", metavar="NAME")
parser.add_argument("--save-path", "-s", default=".",
                    help="destination path of saving figure.", metavar="DEST")
parser.add_argument("--dpi", "-d", type=int, default=300,
                    help="dpi of saving figure.", metavar="NUM")

args = parser.parse_args()

data = np.loadtxt(args.path, skiprows=1, delimiter="\t", usecols=(1, 2))
plt.plot(data)
plt.title("Loss Curve")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["Training Loss", "Validating Loss"])
plt.savefig(f"{args.save_path}/{args.filename}.png", dpi=args.dpi)