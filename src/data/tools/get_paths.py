"""Write paths to training and evaluation data to a file"""
import os

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

t_alignments_multipos = config["trainalignmentsmultipos"]
e_alignments_multipos = config["evalalignmentsmultipos"]
t_alignments = config["trainalignmentspath"]
e_alignments = config["evalalignmentspath"]
queryfastasdir = config["queryfastasdir"]
targetfastasdir = config["targetfastasdir"]


def single_positive_paths():

    TRAIN_DIR = t_alignments

    with open("/xdisk/twheeler/daphnedemekas/train_paths.txt", "w", encoding="utf-8") as tpaths:

        print("Writing train paths")
        for Q in range(4):
            print(Q)
            for T in range(45):
                files = os.listdir(f"{TRAIN_DIR}/{Q}/{T}")
                for f in files:
                    tpaths.write(f"{TRAIN_DIR}/{Q}/{T}/{f}" + "\n")

    VAL_DIR = "/xdisk/twheeler/daphnedemekas/eval-alignments"

    with open("/xdisk/twheeler/daphnedemekas/valpaths.txt", "w", encoding="utf-8") as valpaths:

        print("Writing val paths")
        for Q in [0, 1]:
            for T in range(45):
                files = os.listdir(f"{VAL_DIR}/{Q}/{T}")
                for f in files:
                    valpaths.write(f"{VAL_DIR}/{Q}/{T}/{f}" + "\n")


def multi_positive_paths():
    TRAIN_DIR = "/xdisk/twheeler/daphnedemekas/train-alignments-multipos2"

    with open(
        "/xdisk/twheeler/daphnedemekas/train_paths-multipos.txt", "w"
    ) as tpaths:

        print("Writing train paths")
        for Q in range(4):
            print(Q)
            for T in range(45):
                files = os.listdir(f"{TRAIN_DIR}/{Q}/{T}")
                for f in files:
                    tpaths.write(f"{TRAIN_DIR}/{Q}/{T}/{f}" + "\n")

    VAL_DIR = "/xdisk/twheeler/daphnedemekas/eval-alignments-multipos2"

    with open(
        "/xdisk/twheeler/daphnedemekas/valpaths-multipos.txt", "w", encoding="utf-8"
    ) as valpaths:

        print("Writing val paths")
        for Q in [0, 1]:
            for T in range(45):
                files = os.listdir(f"{VAL_DIR}/{Q}/{T}")
                for f in files:
                    valpaths.write(f"{VAL_DIR}/{Q}/{T}/{f}" + "\n")


#single_positive_paths()
multi_positive_paths()
