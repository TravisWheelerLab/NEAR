import glob
import os

train_directory = "/xdisk/twheeler/daphnedemekas/train-alignments"

trainpaths = open("/xdisk/twheeler/daphnedemekas/train_paths2.txt", "w")

print("Writing train paths")
for Q in range(1, 5):
    print(Q)
    for T in range(45):
        files = os.listdir(f"{train_directory}/{Q}/{T}")
        for f in files:
            trainpaths.write(f"/xdisk/twheeler/daphnedemekas/train-alignments/{Q}/{T}/{f}" + "\n")

trainpaths.close()


val = "/xdisk/twheeler/daphnedemekas/eval-alignments"

valpaths = open("/xdisk/twheeler/daphnedemekas/valpaths2.txt", "w")

print("Writing val paths")
Q = 0
for T in range(45):
    files = os.listdir(f"{val}/{Q}/{T}")
    for f in files:
        valpaths.write(f"/xdisk/twheeler/daphnedemekas/eval-alignments/{Q}/{T}/{f}" + "\n")

valpaths.close()
