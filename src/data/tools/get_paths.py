"""Write paths to training and evaluation data to a file"""
import os

# TRAIN_DIR = "/xdisk/twheeler/daphnedemekas/train-alignments-multipos"

# with open(
#     "/xdisk/twheeler/daphnedemekas/train_paths-multipos.txt", "w", encoding="utf-8"
# ) as tpaths:

#     print("Writing train paths")
#     for Q in range(4):
#         print(Q)
#         for T in range(45):
#             files = os.listdir(f"{TRAIN_DIR}/{Q}/{T}")
#             for f in files:
#                 tpaths.write(f"{TRAIN_DIR}/{Q}/{T}/{f}" + "\n")

VAL_DIR = "/xdisk/twheeler/daphnedemekas/eval-alignments-multipos"

with open("/xdisk/twheeler/daphnedemekas/valpaths-multipos.txt", "w", encoding="utf-8") as valpaths:

    print("Writing val paths")
    for Q in [0, 1, 2]:
        for T in range(45):
            files = os.listdir(f"{VAL_DIR}/{Q}/{T}")
            for f in files:
                valpaths.write(
                    f"/xdisk/twheeler/daphnedemekas/eval-alignments-multipos/{Q}/{T}/{f}" + "\n"
                )
