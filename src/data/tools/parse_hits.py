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


def parse(dirpath, query_id=0, task_id=1, save_dir=None):
    query_id = str(query_id)

    hmmerhits = HmmerHits(dir_path=dirpath)

    with open("evaltargetsequences.pkl", "rb") as file:
        eval_targets = pickle.load(file)

    tfile = int(task_id) - 1

    print(f"Parsing tfile {tfile}")

    # targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{tfile}.fa")
    # targetdata = targetfasta.data
    # eval_targets = {}

    # for target, values in targetdata.items():
    #     if target in val_targets:
    #         eval_targets.update({target: values})
    # targetsequences.update(eval_targets)
    target_hits = hmmerhits.get_hits(
        dirpath, tfile, query_num=query_id, filtered_targets=list(eval_targets)
    )  # {'target_dirnum' :{'query_dirnum': {qname: {tname: data} }  } }
    print(f"Saving hits to {save_dir}_{tfile}..")
    with open(f"{save_dir}_{tfile}", "wb") as evalhitsfile:
        pickle.dump(target_hits, evalhitsfile)


def parse_full(dirpath, query_id=0, save_dir=None):
    query_id = str(query_id)

    hmmerhits = HmmerHits(dir_path=dirpath)
    all_target_hits = {}

    for tfile in tqdm.tqdm(range(45)):

        # targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{tfile}.fa")
        # targetdata = targetfasta.data
        # eval_targets = {}

        # for target, values in targetdata.items():
        #     if target in val_targets:
        #         eval_targets.update({target: values})
        # targetsequences.update(eval_targets)
        target_hits = hmmerhits.get_hits(
            dirpath, tfile, query_num=query_id
        )  # {'target_dirnum' :{'query_dirnum': {qname: {tname: data} }  } }
        all_target_hits = update(all_target_hits, target_hits)
        # if query_id == '0' and tfile == 40:
        #     print("Im here")
        #     print(all_target_hits['UniRef90_UPI001F15798D'])
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
# parse_full("/xdisk/twheeler/daphnedemekas/phmmer_max_results", query_id = 4, save_dir = '/xdisk/twheeler/daphnedemekas/ALL_PHMMERMAX_QUERY4_RESULTS_test')
