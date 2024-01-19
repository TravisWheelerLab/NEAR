"""Write paths to training and evaluation data to a file"""
import os
import argparse
import tqdm


def main(train_dir, eval_dir, train_path, eval_path):
    TRAIN_DIR = train_dir

    with open(train_path, "w") as tpaths:
        print("Writing train paths")
        for Q in range(4):
            print(Q)
            for T in range(45):
                if os.path.exists(f"{TRAIN_DIR}/{Q}/{T}"):
                    files = os.listdir(f"{TRAIN_DIR}/{Q}/{T}")
                    for f in files:
                        tpaths.write(f"{TRAIN_DIR}/{Q}/{T}/{f}" + "\n")

    VAL_DIR = eval_dir

    with open(eval_path, "w", encoding="utf-8") as valpaths:
        print("Writing val paths")
        for Q in [3]:
            for T in tqdm.tqdm(range(45)):
                files = os.listdir(f"{VAL_DIR}/{Q}/{T}")
                for f in files:
                    valpaths.write(f"{VAL_DIR}/{Q}/{T}/{f}" + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir")
    parser.add_argument("--eval_dir")
    parser.add_argument("--train_path")
    parser.add_argument("--eval_path")

    args = parser.parse_args()

    main(args.train_dir, args.eval_dir, args.train_path, args.eval_path)
