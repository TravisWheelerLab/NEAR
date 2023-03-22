"""Write paths to training and evaluation data to a file"""
import os

TRAIN_DIR = "/xdisk/twheeler/daphnedemekas/train-alignments"

with open(
    "/xdisk/twheeler/daphnedemekas/train_paths2.txt", "w", encoding="utf-8"
) as tpaths:

    print("Writing train paths")
    for Q in range(1, 5):
        print(Q)
        for T in range(45):
            files = os.listdir(f"{TRAIN_DIR}/{Q}/{T}")
            for f in files:
                tpaths.write(f"{TRAIN_DIR}/{Q}/{T}/{f}" + "\n")

VAL_DIR = "/xdisk/twheeler/daphnedemekas/eval-alignments"

with open(
    "/xdisk/twheeler/daphnedemekas/valpaths2.txt", "w", encoding="utf-8"
) as valpaths:

    print("Writing val paths")
    Q = 0
    for T in range(45):
        files = os.listdir(f"{VAL_DIR}/{Q}/{T}")
        for f in files:
            valpaths.write(
                f"/xdisk/twheeler/daphnedemekas/eval-alignments/{Q}/{T}/{f}"
                + "\n"
            )
