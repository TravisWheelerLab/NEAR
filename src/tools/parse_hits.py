from src.data.hmmerhits import HmmerHits
import pickle
import argparse


def main(hmmer_dir_path: str, save_dir=None):

    hmmerhits = HmmerHits(dir_path=hmmer_dir_path)
    target_hits = hmmerhits.get_hits(hmmer_dir_path)
    if save_dir is not None:
        print(f"Saving hits to {save_dir}")
        with open(f"{save_dir}.pkl", "wb") as evalhitsfile:
            pickle.dump(target_hits, evalhitsfile)
        return target_hits


parser = argparse.ArgumentParser()
parser.add_argument("hmmer_dir_path")
parser.add_argument("save_dir")

args = parser.parse_args()

if __name__ == "__main__":
    main(args.hmmer_dir_path, args.save_dir)
