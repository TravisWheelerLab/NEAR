import os
import pdb
import tqdm
from src.data.hmmerhits import FastaFile
import argparse
import itertools


def get_full_sequences(query_num, target_num, query, target):
    queryfile = f"uniref/split_subset/queries/queries_{query_num}.fa"
    queryfasta = FastaFile(queryfile)
    # all_hits = {}
    querysequences = queryfasta.data
    targetsequences = {}

    targetfasta = FastaFile(f"uniref/split_subset/targets/targets_{target_num}.fa")
    targetdata = targetfasta.data
    targetsequences.update(targetdata)

    if query not in querysequences or target not in targetsequences:
        return None, None

    query_sequence = querysequences[query]
    target_sequence = targetsequences[target]

    return query_sequence, target_sequence


def get_training_data(target_num, query_num, path):

    files = os.listdir(path)

    training_data = {}
    for file in tqdm.tqdm(files):
        f = open(f"{path}/{file}", "r")
        text = f.readlines()
        f.close()
        if len(text) == 0:
            print(f"File {f'{path}/{file}'} is empty.")
            os.remove(f"{path}/{file}")
            continue

        query_and_target = text[0]
        query = query_and_target.split()[0].strip(">").strip("\n")
        target = query_and_target.split()[-1].strip("\n")
        queryseq = text[1].strip("\n")
        targetseq = text[2].strip("\n")
        # if len(text) < 4:
        #     fullquery, fulltarget = get_full_sequences(query_num, target_num, query, target)
        #     if fullquery is None:
        #         continue
        # else:
        fullquery = text[3].strip("\n")
        fulltarget = text[4].strip("\n")

        if not query in training_data.keys():
            training_data[query] = {}

        training_data[query][queryseq] = {
            "target": target,
            "targetsequence": targetseq,
            "fullqseq": fullquery,
            "fulltseq": fulltarget,
        }

    return training_data


def write_new_data_with_matches(new_training_data_path, training_data):

    i = 0
    if len(os.listdir(new_training_data_path)) != 0:
        i = max([int(f.strip(".txt")) for f in os.listdir(new_training_data_path)])
    print("Writing training data...")
    for query in tqdm.tqdm(list(training_data.keys())[i:]):
        sequences = training_data[query].keys()
        for sequence in sequences:
            querymatches = [
                (s, s.find(sequence)) for s in sequences if sequence in s and sequence != s
            ]
            querytarget = training_data[query][sequence]
            if len(querymatches) > 0:
                targetmatches = []
                f = open(f"{new_training_data_path}/{i}.txt", "w")
                line1 = f'{query} & {querytarget["target"]} '
                for match in querymatches:
                    seq, idx = match
                    targetmatch = training_data[query][seq].copy()
                    targetmatch.update(
                        {
                            "targetsequence": training_data.copy()[query][seq]["targetsequence"][
                                idx : idx + len(sequence)
                            ]
                        }
                    )
                    targetmatches.append(targetmatch)

                for t in targetmatches:
                    line1 += f'& {t["target"]}'
                f.write(line1 + "\n")
                f.write("<" + sequence + "\n")
                f.write("<" + querytarget["targetsequence"] + "\n")
                for t in targetmatches:
                    f.write("<" + t["targetsequence"] + "\n")
                f.write(querytarget["fullqseq"] + "\n")
                f.write(querytarget["fulltseq"] + "\n")
                for t in targetmatches:
                    f.write(t["fulltseq"] + "\n")
                f.close()
            else:
                f = open(f"{new_training_data_path}/{i}.txt", "w")
                f.write(f'{query} & {querytarget["target"]} ' + "\n")
                f.write("<" + sequence + "\n")
                f.write("<" + querytarget["targetsequence"] + "\n")
                f.write(querytarget["fullqseq"] + "\n")
                f.write(querytarget["fulltseq"] + "\n")
                f.close()
            i += 1


def main_train(task_id):
    targets = list(range(45))
    train_queries = list(range(4))

    train_target_queries = list(itertools.product(targets, train_queries))

    target_num = train_target_queries[int(task_id) - 1][0]
    query_num = train_target_queries[int(task_id) - 1][1]

    print(f"Collecting training data for target num {target_num} and query num {query_num}")

    training_data_path = f"/xdisk/twheeler/daphnedemekas/train-alignments/{query_num}/{target_num}"
    new_training_data_path = (
        f"/xdisk/twheeler/daphnedemekas/train-alignments-multipos2/{query_num}/{target_num}"
    )

    if not os.path.exists(f"/xdisk/twheeler/daphnedemekas/train-alignments-multipos2/{query_num}"):
        os.mkdir(f"/xdisk/twheeler/daphnedemekas/train-alignments-multipos2/{query_num}")

    if not os.path.exists(
        f"/xdisk/twheeler/daphnedemekas/train-alignments-multipos2/{query_num}/{target_num}"
    ):
        os.mkdir(
            f"/xdisk/twheeler/daphnedemekas/train-alignments-multipos2/{query_num}/{target_num}"
        )

    training_data = get_training_data(target_num, query_num, training_data_path)
    write_new_data_with_matches(new_training_data_path, training_data)


def main_eval(task_id):
    targets = list(range(45))
    eval_queries = [0, 1]

    eval_target_queries = list(itertools.product(targets, eval_queries))

    target_num = eval_target_queries[int(task_id) - 1][0]
    query_num = eval_target_queries[int(task_id) - 1][1]

    eval_data_path = f"/xdisk/twheeler/daphnedemekas/eval-alignments/{query_num}/{target_num}"
    new_training_data_path = (
        f"/xdisk/twheeler/daphnedemekas/eval-alignments-multipos2/{query_num}/{target_num}"
    )
    if not os.path.exists(f"/xdisk/twheeler/daphnedemekas/eval-alignments-multipos2/{query_num}"):
        os.mkdir(f"/xdisk/twheeler/daphnedemekas/eval-alignments-multipos2/{query_num}")

    if not os.path.exists(
        f"/xdisk/twheeler/daphnedemekas/eval-alignments-multipos2/{query_num}/{target_num}"
    ):
        os.mkdir(
            f"/xdisk/twheeler/daphnedemekas/eval-alignments-multipos2/{query_num}/{target_num}"
        )

    eval_data = get_training_data(target_num, query_num, eval_data_path)
    write_new_data_with_matches(new_training_data_path, eval_data)


parser = argparse.ArgumentParser()
parser.add_argument("task_id")
args = parser.parse_args()

task_id = args.task_id

main_train(args.task_id)

if int(args.task_id) <= 45 * 2:
    main_eval(args.task_id)

print("Done")
