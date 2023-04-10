from src.data.hmmerhits import FastaFile, HmmerHits
import os
import pickle
import argparse
import tqdm

HOME = os.environ["HOME"]


def update(d1, d2):
    c = d1.copy()
    for key in d2:
        if key in d1:
            c[key].update(d2[key])
        else:
            c[key] = d2[key]
    return c


def parse_eval_hmmer_hits(
    hmmer_dirpath,
    query_id,
    save_dir=None,
    evaltargetfastafile="/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargets.fa",
):
    query_id = str(query_id)

    hmmerhits = HmmerHits(dir_path=hmmer_dirpath)
    all_target_hits = {}

    targetsequencefasta = FastaFile(evaltargetfastafile)
    targetsequencedata = targetsequencefasta.data

    for tfile in tqdm.tqdm(range(45)):

        target_hits = hmmerhits.get_hits(
            hmmer_dirpath,
            tfile,
            query_num=query_id,
            filtered_targets=list(targetsequencedata.keys()),
        )  # {'target_dirnum' :{'query_dirnum': {qname: {tname: data} }  } }
        all_target_hits = update(all_target_hits, target_hits)

    print("Finished parsing. Returning dictionary")
    print(f"Saving hits to {save_dir}.pkl..")
    with open(f"{save_dir}.pkl", "wb") as evalhitsfile:
        pickle.dump(all_target_hits, evalhitsfile)
    return all_target_hits


# parser = argparse.ArgumentParser()
# parser.add_argument("task_id")
# args = parser.parse_args()

# parse("/xdisk/twheeler/daphnedemekas/phmmer_normal_results", task_id = args.task_id, query_id=4, save_dir = '/xdisk/twheeler/daphnedemekas/phmmer_normal_query_4_results/evaltargethmmerhits')
# parse("/xdisk/twheeler/daphnedemekas/phmmer_max_results", task_id = args.task_id, save_dir = '/xdisk/twheeler/daphnedemekas/query_0_results/evaltargethmmerhits')

# parse_full("/xdisk/twheeler/daphnedemekas/phmmer_normal_results", query_id=0, save_dir = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERNORMAL_QUERY0_RESULTS')
# parse_full("/xdisk/twheeler/daphnedemekas/phmmer_max_results", query_id = 0, save_dir = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERMAX_QUERY0_RESULTS')


# parse_full("/xdisk/twheeler/daphnedemekas/phmmer_normal_results", query_id=4, save_dir = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERNORMAL_QUERY4_RESULTS')
if __name__ == "__main__":
    parse_eval_hmmer_hits(
        "/xdisk/twheeler/daphnedemekas/phmmer_max_results",
        query_id=4,
        save_dir="/xdisk/twheeler/daphnedemekas/prefilter/data/evaluationtargetdict",
    )
